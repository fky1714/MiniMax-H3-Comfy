"""Thin client for the ComfyUI HTTP/WebSocket API.

Talks to the ComfyUI server this worker starts as a subprocess on
127.0.0.1 (see src/handler.py). Mirrors the flow runpod-workers/worker-comfyui
uses (confirmed against its handler.py source): upload any input images ->
POST /prompt to queue the graph -> watch the WebSocket for the `executing`
message with node=None (that is ComfyUI's own "this prompt is done" signal,
not a guess) -> GET /history/{id} for the output node's saved filename ->
GET /view to fetch the bytes.

No workflow-JSON handling here — src/workflow.py already returns the graph in
the exact dict shape POST /prompt expects.
"""

from __future__ import annotations

import json
import logging
import time
import uuid

import requests
import websocket  # websocket-client package

logger = logging.getLogger("minimax-h3-comfy")


class ComfyError(Exception):
    """Raised when ComfyUI reports a prompt validation or execution error."""


class ComfyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188, timeout_s: float = 30.0):
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.timeout_s = timeout_s

    # -- Health --------------------------------------------------------------

    def is_alive(self) -> bool:
        """True if the ComfyUI HTTP server is answering. Does not imply a
        model is loaded — ComfyUI itself has no such concept; models are
        resolved lazily per-prompt from the node graph."""
        try:
            r = requests.get(f"{self.base_url}/", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def wait_until_alive(self, max_wait_s: float, interval_s: float = 1.0) -> None:
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            if self.is_alive():
                return
            time.sleep(interval_s)
        raise ComfyError(f"ComfyUI server did not respond within {max_wait_s:.0f}s")

    # -- Image upload ----------------------------------------------------------

    def upload_image(self, image_bytes: bytes, filename: str) -> str:
        """POST /upload/image. Returns the filename ComfyUI stored it under —
        pass this straight to workflow.build_workflow()'s
        first_frame_name/last_frame_name, not the original filename, since
        ComfyUI may rename on collision."""
        r = requests.post(
            f"{self.base_url}/upload/image",
            files={"image": (filename, image_bytes)},
            data={"overwrite": "true"},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        # ComfyUI returns {"name": ..., "subfolder": ..., "type": "input"}.
        # LoadImage's `image` widget wants "name" alone when subfolder is
        # empty (the common case for a flat upload), "subfolder/name" otherwise.
        name = data["name"]
        if data.get("subfolder"):
            name = f"{data['subfolder']}/{name}"
        return name

    # -- Queue + wait ------------------------------------------------------

    def queue(self, workflow: dict, client_id: str) -> str:
        """POST /prompt. Returns the prompt_id."""
        r = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=self.timeout_s,
        )
        if r.status_code != 200:
            # ComfyUI's validation errors come back as 400 with a structured
            # body naming the offending node/input — surface that instead of
            # a bare HTTPError, it is the only useful debugging signal a
            # graph-construction bug leaves behind.
            try:
                detail = r.json()
            except ValueError:
                detail = r.text
            raise ComfyError(f"POST /prompt failed ({r.status_code}): {detail}")
        return r.json()["prompt_id"]

    def wait_for_completion(self, prompt_id: str, client_id: str, timeout_s: float) -> None:
        """Block on the WebSocket until ComfyUI signals this prompt is done.

        ComfyUI's own completion signal is an `executing` message whose
        `data.node` is null AND whose `data.prompt_id` matches — not a
        generic "no more messages" heuristic. An `execution_error` message
        for this prompt_id raises ComfyError with whatever detail ComfyUI
        gave (usually points straight at the failing node).
        """
        ws = websocket.create_connection(
            f"{self.ws_url}?clientId={client_id}", timeout=timeout_s,
        )
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ComfyError(f"Timed out after {timeout_s:.0f}s waiting for prompt {prompt_id}")
                ws.settimeout(min(remaining, 30.0))
                try:
                    raw = ws.recv()
                except Exception as e:
                    # Binary frames (preview images) and read timeouts both
                    # land here; a timeout just means "check the deadline and
                    # keep listening", not failure.
                    if "timed out" in str(e).lower():
                        continue
                    raise
                if not isinstance(raw, str):
                    continue  # binary preview frame, not a status message
                msg = json.loads(raw)
                mtype = msg.get("type")
                data = msg.get("data", {})
                if data.get("prompt_id") not in (None, prompt_id):
                    continue  # message about a different queued prompt

                if mtype == "executing" and data.get("node") is None:
                    return
                if mtype == "execution_error":
                    raise ComfyError(f"ComfyUI execution error for {prompt_id}: {data}")
        finally:
            ws.close()

    # -- Fetch output ----------------------------------------------------------

    def fetch_output(self, prompt_id: str, output_node_id: str) -> bytes:
        """GET /history/{id}, find output_node_id's saved file, GET /view it."""
        r = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=self.timeout_s)
        r.raise_for_status()
        history = r.json()
        if prompt_id not in history:
            raise ComfyError(f"prompt {prompt_id} not found in /history")

        entry = history[prompt_id]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            raise ComfyError(f"ComfyUI reported an error for {prompt_id}: {status}")

        outputs = entry.get("outputs", {}).get(output_node_id)
        if not outputs:
            raise ComfyError(
                f"No outputs for node {output_node_id!r} in prompt {prompt_id}. "
                f"Nodes with outputs: {sorted(entry.get('outputs', {}))}"
            )

        # SaveVideo (and SaveImage) report files under a type-named key —
        # "video"/"images"/"gifs" depending on ComfyUI version and node.
        # Take the first file list present rather than hardcoding one key.
        file_list = None
        for key in ("video", "videos", "images", "gifs"):
            if key in outputs and outputs[key]:
                file_list = outputs[key]
                break
        if not file_list:
            raise ComfyError(f"Output node {output_node_id!r} produced no files: {outputs}")

        f = file_list[0]
        params = {
            "filename": f["filename"],
            "subfolder": f.get("subfolder", ""),
            "type": f.get("type", "output"),
        }
        r = requests.get(f"{self.base_url}/view", params=params, timeout=self.timeout_s)
        r.raise_for_status()
        return r.content


def new_client_id() -> str:
    return uuid.uuid4().hex
