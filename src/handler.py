"""RunPod Serverless handler for the MiniMax-H3 ComfyUI worker.

Unlike the diffusers sibling, this worker does not load model weights into
its own process. It launches ComfyUI as a subprocess and talks to it over
HTTP/WebSocket (src/comfy_client.py); ComfyUI resolves and loads weights
lazily, per node, the first time a prompt graph references them. What this
handler owns is the RunPod job contract: validating input (src/validator.py,
same field set as the diffusers sibling's API.md), fetching keyframes
(src/image.py), building the node graph (src/workflow.py), running it, and
uploading the resulting mp4 (src/storage.py) — no video encoding of its own,
ComfyUI's CreateVideo/SaveVideo nodes already produce a muxed mp4.

Mode is implicit, same as the sibling: image_url/image_base64/last_image_*
present -> I2V (fl2va); otherwise -> T2V (t2va). A WORKER_MODE=r2v
deployment serves Reference-to-Video (ref2va) instead: every job takes up
to 9 `ref_images` and no keyframes.
"""

import io
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import runpod

from src import config, watchdog
from src.comfy_client import ComfyClient, ComfyError, new_client_id
from src.errors import InferenceError, ValidationError, format_error
from src.image import fetch_image, fetch_last_image, fetch_ref_images, preprocess_image
from src.model_link import ModelLinkError, link_model_cache
from src.storage import upload_video
from src.validator import validate_input
from src.workflow import OUTPUT_NODE_ID, VAE_VIDEO_FILES, build_r2v_workflow, build_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s",
)
logger = logging.getLogger("minimax-h3-comfy")

_COMFY_READY = threading.Event()
_COMFY_ERROR: Exception | None = None
_comfy_process: subprocess.Popen | None = None
_client = ComfyClient(host=config.COMFY_HOST, port=config.COMFY_PORT)


# -- ComfyUI subprocess boot ───────────────────────────────────────────────────

def _boot_comfy_bg() -> None:
    """Symlink the Model Cache snapshot into ComfyUI's models/ tree, launch
    ComfyUI, and wait for its HTTP server to answer. Runs on a background
    thread so request validation can overlap the boot, same pattern as the
    diffusers sibling's background model load."""
    global _COMFY_ERROR, _comfy_process
    try:
        logger.info("Linking Model Cache snapshot into ComfyUI models/...")
        snapshot_root = link_model_cache(os.path.join(config.COMFY_DIR, "models"))
        logger.info("Model weights linked from %s", snapshot_root)

        logger.info("Starting ComfyUI subprocess in %s", config.COMFY_DIR)
        comfy_args = [
            sys.executable, "main.py",
            "--listen", config.COMFY_HOST,
            "--port", str(config.COMFY_PORT),
        ]
        if config.COMFY_DISABLE_PINNED_MEMORY:
            comfy_args.append("--disable-pinned-memory")
        if config.COMFY_DISABLE_COMPILER:
            comfy_args.append("--disable-comfy-compiler")
        _comfy_process = subprocess.Popen(
            comfy_args,
            cwd=config.COMFY_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(target=_pipe_comfy_logs, daemon=True).start()

        _client.wait_until_alive(config.COMFY_BOOT_TIMEOUT_S)
        logger.info("ComfyUI is up.")
    except (ModelLinkError, ComfyError) as e:
        _COMFY_ERROR = e
        logger.exception("ComfyUI boot failed")
    except BaseException as e:
        _COMFY_ERROR = e
        logger.exception("ComfyUI boot failed (unexpected)")
    finally:
        _COMFY_READY.set()


def _pipe_comfy_logs() -> None:
    """Forward the ComfyUI subprocess's own stdout/stderr into this worker's
    log stream — otherwise its cold-start progress and any node errors are
    invisible in RunPod's worker log."""
    if _comfy_process is None or _comfy_process.stdout is None:
        return
    for line in _comfy_process.stdout:
        logger.info("[comfyui] %s", line.rstrip())


def _wait_for_comfy() -> None:
    _COMFY_READY.wait()
    if _COMFY_ERROR is not None:
        raise InferenceError(f"ComfyUI failed to start: {_COMFY_ERROR}")
    if _comfy_process is not None and _comfy_process.poll() is not None:
        raise InferenceError(
            f"ComfyUI process exited early (code {_comfy_process.returncode})."
        )


# -- Keyframe upload ───────────────────────────────────────────────────────────

def _upload_keyframe(img, filename: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return _client.upload_image(buf.getvalue(), filename)


def _prepare_images(params, job_id: str):
    """Returns (first_name, last_name) — filenames already uploaded to
    ComfyUI's input/ directory, or None. Mirrors the diffusers sibling's
    _prepare_images canvas-alignment behavior: the first keyframe (or the
    last, if that's all there is) sets the canvas, the other is resized onto
    it — MiniMax-H3 stretches the first frame and cover-crops the last."""
    first = fetch_image(params)
    last = fetch_last_image(params)

    if first is None and last is None:
        return None, None

    anchor = first if first is not None else last
    anchor = preprocess_image(anchor, params["width"], params["height"], config.MOD_VALUE)
    width, height = anchor.size

    other = last if first is not None else None
    if other is not None:
        other = preprocess_image(other, width, height, config.MOD_VALUE)

    if first is not None:
        first_name = _upload_keyframe(anchor, f"{job_id}_first.png")
        last_name = _upload_keyframe(other, f"{job_id}_last.png") if other is not None else None
    else:
        first_name = None
        last_name = _upload_keyframe(anchor, f"{job_id}_last.png")

    return first_name, last_name


def _prepare_ref_images(params, job_id: str) -> list[str]:
    """Upload R2V references to ComfyUI's input/ dir, in prompt order.
    Uploaded at their own size — MiniMaxH3ReferenceToVideo rescales them
    itself according to ref_image_size, so no canvas alignment here."""
    return [
        _upload_keyframe(img, f"{job_id}_ref{i}.png")
        for i, img in enumerate(fetch_ref_images(params))
    ]


# -- Single job ────────────────────────────────────────────────────────────────

def _run_job(job_id: str, params: dict) -> dict:
    timings = {}
    is_r2v = params["mode"] == "r2v"

    # Fail before any GPU work if the Model Cache snapshot predates the
    # INT8 VAE: ComfyUI would only reject the graph after preprocessing.
    vae_file = VAE_VIDEO_FILES[params["video_vae"]]
    if not os.path.exists(os.path.join(config.COMFY_DIR, "models", "vae", vae_file)):
        raise ValidationError(
            f"video_vae={params['video_vae']!r} is not available on this worker ({vae_file} missing).",
            hint="The endpoint's Model Cache snapshot predates it; use video_vae='fp16' or refresh the cache.",
        )

    t0 = time.perf_counter()
    if is_r2v:
        ref_names = _prepare_ref_images(params, job_id)
    else:
        first_name, last_name = _prepare_images(params, job_id)
    timings["preprocess_s"] = round(time.perf_counter() - t0, 3)

    seed = params["seed"]
    if seed < 0:
        import random
        seed = random.randint(0, 2**32 - 1)

    common = dict(
        prompt=params["prompt"],
        width=params["width"],
        height=params["height"],
        seconds=params["seconds"],
        seed=seed,
        steps=params["steps"],
        mute=params["mute"],
        attention=config.ATTENTION_BACKENDS[params["attention"]],
        video_vae=params["video_vae"],
        filename_prefix=f"minimax_h3/{job_id}",
    )
    if is_r2v:
        workflow = build_r2v_workflow(
            ref_image_names=ref_names,
            turbo=params["turbo"],
            ref_image_size=params["ref_image_size"],
            scheduler=params["scheduler"],
            **common,
        )
    else:
        workflow = build_workflow(
            turbo_mode=config.TURBO_MODE,
            first_frame_name=first_name,
            last_frame_name=last_name,
            **common,
        )

    client_id = new_client_id()
    t0 = time.perf_counter()
    try:
        prompt_id = _client.queue(workflow, client_id)
        _client.wait_for_completion(prompt_id, client_id, config.COMFY_INFERENCE_TIMEOUT_S)
        video_bytes = _client.fetch_output(prompt_id, OUTPUT_NODE_ID)
    except ComfyError as e:
        raise InferenceError(str(e))
    timings["inference_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        mp4_path = f.name
    try:
        out_cfg = params["output"]
        video_url = upload_video(
            local_path=mp4_path,
            job_id=job_id,
            bucket=out_cfg["bucket"],
            prefix=out_cfg["prefix"],
            public=out_cfg["public"],
            ttl_seconds=out_cfg["ttl_seconds"],
        )
    finally:
        try:
            os.remove(mp4_path)
        except OSError:
            pass
    timings["upload_s"] = round(time.perf_counter() - t0, 3)
    timings["total_s"] = round(sum(timings.values()), 3)

    meta = {
        "mode": params["mode"],
        "seconds": params["seconds"],
        "fps": params["fps"],
        "frames": params["num_frames"],
        "seed": seed,
        "steps": workflow["scheduler"]["inputs"]["steps"],
        "cfg_scale": params["cfg_scale"],
        "width": params["width"],
        "height": params["height"],
        "model": config.model_tag(params["turbo"]),
        "has_audio": not params["mute"],
        "audio_sampling_rate": None if params["mute"] else 32000,
        "attention": params["attention"],
        "video_vae": params["video_vae"],
        "ignored_params": params["ignored_params"],
    }
    if is_r2v:
        meta.update(
            ref_count=len(ref_names),
            turbo=params["turbo"],
            ref_image_size=params["ref_image_size"],
            scheduler=params["scheduler"],
        )

    return {
        "status": "completed",
        "video_url": video_url,
        "meta": meta,
        "timing": timings,
    }


# -- Entry point ──────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """RunPod serverless handler.

    Single T2V:   {"input": {"prompt": "..."}}
    Single I2V:   {"input": {"image_url": "...", "prompt": "..."}}
    Single R2V:   {"input": {"ref_images": [{"url": "..."}], "prompt": "<Picture 1> ..."}}
                  (WORKER_MODE=r2v deployments only)
    Batch:        {"input": {"batch": [{...}, {...}], "resolution": "480p"}}
    Health:       {"input": {"op": "health"}}  (SERVE_MODE=api / Pod deploy path)
    """
    job_id = job.get("id", "unknown")
    inp = job.get("input", {})

    watchdog.touch()

    if inp.get("op") == "health":
        if _COMFY_ERROR is not None:
            return {"status": "error", "error": str(_COMFY_ERROR)}
        if not _COMFY_READY.is_set():
            return {"status": "loading"}
        return {"status": "ok"}

    logger.info("Job %s started", job_id)

    try:
        _wait_for_comfy()

        if "batch" in inp:
            batch_items = inp["batch"]
            if not isinstance(batch_items, list) or len(batch_items) == 0:
                return {"status": "failed", "error": "batch must be a non-empty list."}
            if len(batch_items) > config.MAX_BATCH_SIZE:
                return {
                    "status": "failed",
                    "error": f"batch exceeds max size ({config.MAX_BATCH_SIZE}).",
                }
            shared_defaults = {k: v for k, v in inp.items() if k != "batch"}
            results = []
            for i, item in enumerate(batch_items):
                merged = {**shared_defaults, **item}
                item_id = f"{job_id}_b{i}"
                try:
                    params = validate_input(merged)
                    results.append(_run_job(item_id, params))
                except Exception as e:
                    logger.exception("Batch item %d failed: %s", i, e)
                    results.append(format_error(e))
            result = {
                "status": "completed",
                "results": results,
                "meta": {"model": config.model_tag()},
            }
        else:
            params = validate_input(inp)
            result = _run_job(job_id, params)

        logger.info("Job %s completed", job_id)
        return result

    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        return format_error(exc)


# -- Startup ───────────────────────────────────────────────────────────────────

def _probe(msg: str) -> None:
    sys.stderr.write(f"[startup] {msg}\n")
    sys.stderr.flush()


try:
    _probe("entrypoint reached")
    threading.Thread(target=_boot_comfy_bg, name="comfy-boot", daemon=True).start()

    # No-op unless SERVE_MODE=api and RUNPOD_POD_ID/RUNPOD_API_KEY are present.
    watchdog.start()

    if config.SERVE_MODE == "api":
        sys.argv += [
            "--rp_serve_api",
            "--rp_api_host", "0.0.0.0",
            "--rp_api_port", config.POD_API_PORT,
        ]
        _probe(f"SERVE_MODE=api: rp_serve_api on port {config.POD_API_PORT}")
except BaseException as _e:
    sys.stderr.write(f"[startup] FATAL during warm-up: {type(_e).__name__}: {_e}\n")
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    raise

# KEEP THIS CALL AT MODULE LEVEL, UNINDENTED — RunPod's GitHub integration
# statically scans the repo for it and refuses to deploy ("handler not
# found") when it is wrapped in a try/except or a function.
runpod.serverless.start({"handler": handler})
