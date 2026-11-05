"""Build the MiniMax-H3 ComfyUI prompt graph directly, in API format.

Why this exists rather than a JSON template
--------------------------------------------
Comfy-Org's official T2V/I2V templates (`Comfy-Org/workflow_templates`,
`video_minimax_h3_t2v.json` / `video_minimax_h3_i2v.json`) turn out to be the
SAME subgraph — "Image to Video (MiniMax H3)" — used both ways: whether it
runs t2va or fl2va is decided purely by whether `first_frame`/`last_frame`
are connected, per the subgraph's own note. That answers the "one workflow
or two" question decisively: one graph, and this module reproduces it.

The subgraph's UI-convenience nodes (`ComfySwitchNode`, `ComfyMathExpression`,
`PrimitiveInt`/`Float`/`Boolean`, `ResolutionSelector`) exist only to let a
human toggle things in the ComfyUI editor. None of them are needed to drive
the graph over the API — the Turbo-LoRA on/off branch, the step-count switch,
and the duration -> frame-grid math they implement are all trivial to redo in
plain Python, which is what this module does. That drops the worker's
runtime dependency to ComfyUI's own bundled nodes: `MiniMaxH3ImageToVideo` is
native as of ComfyUI 0.30.0, and everything else (UNETLoader, CLIPLoader,
VAELoader, LoraLoaderModelOnly, RandomNoise, KSamplerSelect, BasicScheduler,
BasicGuider, SamplerCustomAdvanced, VAEDecode, VAEDecodeAudio, CreateVideo,
SaveVideo, LoadImage) ships with ComfyUI core. The one exception is opt-in:
`attention` adds AttentionBackendOverride, this repo's own custom node
(comfy_nodes/, copied into ComfyUI by the Dockerfile).

Node graph (cross-checked against ComfyUI's own source —
comfy_extras/nodes_minimax_h3.py, nodes_video.py — not just the template):

    UNETLoader ────────────────┐
    CLIPLoader ─────────────┐  │
    VAELoader (video) ───┐  │  │
    VAELoader (audio) ─┐ │  │  │
                       │ │  │  ├─(turbo_mode)──> LoraLoaderModelOnly ─┐
                       │ │  │  └────────────────────────(no turbo)───┤
                       │ │  │                                        │
                       │ │  └──> MiniMaxH3ImageToVideo <──(prompt,   │
                       │ │       (first/last_frame optional)  size,  │
                       │ │              │         │           length)│
                       │ │        positive      LATENT               │
                       │ │              │         │                  │
                       │ │              v         v                  │
                       │ │         BasicGuider  SamplerCustomAdvanced │
                       │ │              │         ^  ^  ^   (model <─┘ here too)
                       │ │              └─GUIDER──┘  │  │
                       │ │      RandomNoise ──NOISE──┘  │
                       │ │      KSamplerSelect ─SAMPLER─┤
                       │ │      BasicScheduler ─SIGMAS──┘  (model <─ same branch)
                       │ │              │
                       │ │           output (LATENT, reused for both decodes)
                       │ │        ┌─────┴──────┐
                       │ │        v             v
                       │ └──> VAEDecode    VAEDecodeAudio <─┘
                       └───────────(vae)        (vae)
                                   │             │
                                   v             v
                                 CreateVideo(images, audio, fps=24)
                                   │
                                   v
                                 SaveVideo

`MiniMaxH3ImageToVideo.length` does NOT self-snap to the model's 17k+5 frame
grid (its schema just declares step=17 as a UI hint) — `frame_length()` below
reproduces the grid math the official template does via ComfyMathExpression:
`max(5, round(seconds*24)) + (5 - (that % 17)) % 17`.

Reference-to-Video (WORKER_MODE=r2v) is a second graph, build_r2v_workflow():
the same loaders and sampler chain around `MiniMaxH3ReferenceToVideo`
(ref2va checkpoint, up to 9 reference images, no first/last frame), after
Comfy-Org's video_minimax_h3_r2v.json template.

Open question: whether
BasicScheduler's steps needs a diffusers-style off-by-one adjustment, and
whether skipping MiniMaxH3SigmaShift (present in ComfyUI core but unused by
the official template) is actually fine. The second is settled by reading
ComfyUI 0.34.0: supported_models.MiniMaxH3 already defaults to shift 12 /
audio_shift 3, the node's own defaults, so it would be a no-op.
"""

from __future__ import annotations

# Model filenames — must match fkyyy/MiniMax-H3-comfy-slim exactly (see
# tools/make_comfy_slim_fork.py). These are also the official
# Comfy-Org filenames, unchanged, which is why the slim fork kept them as-is:
# whatever ComfyUI ships next expects these names in these directories.
UNET_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
# Video VAE per request (`video_vae`). int8 is Comfy-Org's INT8+ConvRot
# decoder, which ComfyUI 0.36 detects from its quantization metadata and runs
# on comfy_kitchen's fused kernels; the encoder side (I2V first/last frames,
# R2V references) is unchanged. Both live in the slim forks so either can be
# chosen without a redeploy.
VAE_VIDEO_FILES = {
    "fp16": "minimax_h3_video_vae_fp16.safetensors",
    "int8": "minimax_h3_video_vae_int8_convrot.safetensors",
}
VAE_VIDEO_NAME = VAE_VIDEO_FILES["fp16"]
VAE_AUDIO_NAME = "minimax_h3_audio_vae_fp32.safetensors"
LORA_NAME = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

# R2V (ref2va) — must match fkyyy/minimax-h3-comfy-r2v-slim (see
# tools/make_comfy_slim_fork.py --workflow r2v). CLIP and both
# VAEs are shared with fl2va. The official R2V template loads the int8 video
# VAE instead of fp16; the fp16 one is the same VAE at higher precision and
# is already proven on the i2v endpoint, so the r2v fork reuses it.
REF_UNET_NAME = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
# lightx2v's 8-step v1.1 (768p) ref2v Turbo LoRA. Comfy-Org only re-hosts the
# older 4-step v0.1 (trained at 544p); 8 steps is the better fidelity/speed
# point for reference-heavy prompts. Its pairing with the *pruned* ref2va
# build is unverified on this worker (ComfyUI's docs warn full-build LoRAs
# lose adaLN keys on pruned builds) — turbo is opt-in per request for now.
REF_LORA_NAME = "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"

# The official template runs the turbo branch at 8 steps. 4 was compared
# side by side against 8 (2026-09-24, same seeds, T2V and I2V at 480p/5s):
# same composition, motion and audio, slightly softer fur detail, ~40% less
# time — chosen as the default.
TURBO_STEPS = 4
BASE_STEPS = 20
# R2V: reference tokens ride along at every step, so few-step sampling
# under-applies them. 8 is what the LoRA is distilled for; 20 is the official
# R2V template's non-turbo default.
R2V_TURBO_STEPS = 8
R2V_BASE_STEPS = 20

FPS = 24
FRAME_GRID = 17
FRAME_OFFSET = 5
MIN_LENGTH = FRAME_OFFSET  # MiniMaxH3ImageToVideo.length schema: min=5
SECONDS_CEILING = 15       # model hard limit (matches the diffusers worker's config.py)


def frame_length(seconds: float, fps: int = FPS) -> int:
    """Seconds -> frame count on the model's `17k + 5` grid.

    The identity is the official ComfyMathExpression node's:
    `max(5, round(seconds*fps)) + (5 - (that % 17)) % 17`. Reproduced here in
    Python since the UI-convenience node isn't needed to drive the API.

    One deliberate deviation from the official node: that formula only rounds
    UP, so 15s lands on 362 frames (15.08s) — past the model's 15s ceiling,
    even though it is inside the "trained range 124-362" the node's own
    tooltip claims. The sibling diffusers worker's derive_num_frames()
    instead steps back to the nearest grid point at-or-under the ceiling
    (345 frames, 14.38s) so meta.frames matches API.md's table across both
    workers. This function does the same, on top of the official math.
    """
    base = max(MIN_LENGTH, round(seconds * fps))
    length = base + (FRAME_OFFSET - (base % FRAME_GRID)) % FRAME_GRID
    while length / fps > SECONDS_CEILING and length > MIN_LENGTH:
        length -= FRAME_GRID
    return length


def _shared_loaders(unet_name: str, video_vae: str = "fp16") -> dict[str, dict]:
    """Loader nodes common to both workflows: the diffusion model (fl2va or
    ref2va), the Qwen3-VL text encoder and both VAEs."""
    return {
        "unet_loader": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "clip_loader": {
            "class_type": "CLIPLoader",
            # type="minimax" selects the Qwen3-VL-derived conditioner path;
            # this is a fixed value for this model, not a user-facing choice.
            "inputs": {"clip_name": CLIP_NAME, "type": "minimax", "device": "default"},
        },
        "vae_loader_video": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_VIDEO_FILES[video_vae]},
        },
        "vae_loader_audio": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_AUDIO_NAME},
        },
    }


def _model_branch(nodes: dict, turbo: bool, lora_name: str) -> list:
    """Adds the Turbo LoRA node when `turbo` and returns the MODEL ref the
    scheduler and guider read from."""
    if not turbo:
        return ["unet_loader", 0]
    nodes["lora"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["unet_loader", 0],
            "lora_name": lora_name,
            "strength_model": 1.0,
        },
    }
    return ["lora", 0]


def _sampling_and_output(
    nodes: dict,
    model_ref: list,
    cond_node: str,
    seed: int,
    steps: int,
    scheduler: str,
    mute: bool,
    filename_prefix: str,
    attention: str | None = None,
) -> dict:
    """Sampler chain + decode + mux, identical for fl2va and ref2va: both
    conditioning nodes output [positive, LATENT] and the official templates
    sample them the same way."""
    # Model patches chain after the LoRA. `attention` is a ComfyUI-registered
    # backend name ("sage", "comfy_kitchen_int8"); None keeps the default.
    if attention:
        nodes["attention"] = {
            "class_type": "AttentionBackendOverride",
            "inputs": {"model": model_ref, "backend": attention},
        }
        model_ref = ["attention", 0]
    nodes["noise"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": seed},
    }
    nodes["sampler_select"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "res_multistep"},
    }
    nodes["scheduler"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": model_ref,
            "scheduler": scheduler,
            "steps": steps,
            "denoise": 1.0,
        },
    }
    nodes["guider"] = {
        "class_type": "BasicGuider",
        "inputs": {
            "model": model_ref,
            # MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo outputs are
            # [positive, LATENT] in that order (confirmed from the templates'
            # node.outputs arrays) — index 0 is the conditioning, not index 1.
            "conditioning": [cond_node, 0],
        },
    }
    nodes["sampler"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["noise", 0],
            "guider": ["guider", 0],
            "sampler": ["sampler_select", 0],
            "sigmas": ["scheduler", 0],
            "latent_image": [cond_node, 1],
        },
    }
    # Both decodes read the SAME sampler output — MiniMax-H3 denoises video
    # and audio latents in one packed sequence, so there is one LATENT to
    # split, not two separate sampling passes.
    nodes["vae_decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["vae_loader_video", 0]},
    }
    create_video_inputs: dict = {
        "images": ["vae_decode", 0],
        "fps": float(FPS),
        "bit_depth": 8,
    }
    # mute skips only the audio VAE decode + mux — the packed video+audio
    # latent from `sampler` is identical either way (see the module docstring:
    # one denoising pass produces both), so this saves no diffusion time, only
    # the decode/encode tail. CreateVideo's `audio` input is optional in
    # ComfyUI core (comfy_extras/nodes_video.py), so omitting the key here is
    # enough — no silent-audio placeholder needed.
    if not mute:
        nodes["vae_decode_audio"] = {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sampler", 0], "vae": ["vae_loader_audio", 0]},
        }
        create_video_inputs["audio"] = ["vae_decode_audio", 0]
    nodes["create_video"] = {"class_type": "CreateVideo", "inputs": create_video_inputs}
    nodes["save_video"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["create_video", 0],
            "filename_prefix": filename_prefix,
            "format": "auto",
        },
    }
    return nodes


def build_workflow(
    prompt: str,
    width: int,
    height: int,
    seconds: float,
    seed: int,
    turbo_mode: bool = True,
    steps: int | None = None,
    first_frame_name: str | None = None,
    last_frame_name: str | None = None,
    mute: bool = False,
    filename_prefix: str = "minimax_h3/output",
    attention: str | None = None,
    video_vae: str = "fp16",
) -> dict:
    """Build the ComfyUI API-format prompt dict for one T2V/I2V job.

    `first_frame_name` / `last_frame_name` are filenames already uploaded to
    ComfyUI's `input/` directory (via POST /upload/image — see
    src/comfy_client.py) — NOT raw image bytes. Passing neither selects t2va;
    passing either or both selects fl2va, exactly as the official template's
    subgraph does by connection presence. `steps` overrides the branch
    default (TURBO_STEPS/BASE_STEPS) when the caller wants a specific NFE.

    Returns a plain dict of `{node_id: {"class_type": ..., "inputs": {...}}}`
    — the same shape POST /prompt expects as `{"prompt": <this>, ...}`. No
    JSON template is read; every node is built directly.
    """
    length = frame_length(seconds)

    nodes = _shared_loaders(UNET_NAME, video_vae)
    model_ref = _model_branch(nodes, turbo_mode, LORA_NAME)
    if steps is None:
        steps = TURBO_STEPS if turbo_mode else BASE_STEPS

    h3_inputs: dict = {
        "clip": ["clip_loader", 0],
        "vae": ["vae_loader_video", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": length,
    }
    # Optional IMAGE inputs: omit the key entirely rather than sending null —
    # ComfyUI's API format treats a missing key as "not connected", which is
    # what selects t2va over fl2va inside MiniMaxH3ImageToVideo.
    if first_frame_name:
        nodes["load_first_frame"] = {
            "class_type": "LoadImage",
            "inputs": {"image": first_frame_name},
        }
        h3_inputs["first_frame"] = ["load_first_frame", 0]
    if last_frame_name:
        nodes["load_last_frame"] = {
            "class_type": "LoadImage",
            "inputs": {"image": last_frame_name},
        }
        h3_inputs["last_frame"] = ["load_last_frame", 0]

    nodes["h3"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": h3_inputs}

    return _sampling_and_output(
        nodes, model_ref, "h3", seed, steps, "simple", mute, filename_prefix,
        attention,
    )


def ref_input_key(index: int) -> str:
    """API-format input key for the index-th reference image (0-based).

    MiniMaxH3ReferenceToVideo declares `ref_images` as an Autogrow input
    (TemplatePrefix, prefix="ref_image_", max=9); ComfyUI flattens Autogrow
    children into dotted keys, `ref_images.ref_image_0` .. `.ref_image_8`
    (comfy_api/latest/_io.py finalize_prefix, and the official
    video_minimax_h3_r2v.json template uses the same names). Not yet
    exercised against a live ComfyUI — if the first GPU run rejects the
    graph, this key format is the first thing to check.
    """
    return f"ref_images.ref_image_{index}"


def build_r2v_workflow(
    prompt: str,
    width: int,
    height: int,
    seconds: float,
    seed: int,
    ref_image_names: list[str],
    turbo: bool = False,
    steps: int | None = None,
    ref_image_size: str = "match",
    scheduler: str = "simple",
    mute: bool = False,
    filename_prefix: str = "minimax_h3/output",
    attention: str | None = None,
    video_vae: str = "fp16",
) -> dict:
    """Build the ComfyUI API-format prompt dict for one Reference-to-Video job.

    Mirrors Comfy-Org's video_minimax_h3_r2v.json: ref2va checkpoint, the
    same text encoder and VAEs as fl2va, MiniMaxH3ReferenceToVideo instead of
    MiniMaxH3ImageToVideo, and the same sampler chain. `ref_image_names` are
    filenames already uploaded to ComfyUI's input/ dir, in prompt order — the
    i-th one is what the prompt calls `<Picture i+1>`. The node has no
    first/last-frame inputs; continuity from a previous clip is done by
    passing its last frame as one of the references.

    `ref_image_size="match"` scales each reference to the generation's area;
    "max" keeps up to a 2048px short edge, which the node's own tooltip
    warns can be several times slower. `scheduler` defaults to the
    template's "simple"; its note says "beta"/"normal" tend to do better on
    reference-heavy prompts, so callers may pick them per request.
    """
    length = frame_length(seconds)

    nodes = _shared_loaders(REF_UNET_NAME, video_vae)
    model_ref = _model_branch(nodes, turbo, REF_LORA_NAME)
    if steps is None:
        steps = R2V_TURBO_STEPS if turbo else R2V_BASE_STEPS

    r2v_inputs: dict = {
        "clip": ["clip_loader", 0],
        # Without the video VAE the references only condition the text
        # encoder, not the latent — the node's own docs say so. The audio
        # VAE only encodes reference audio, which this worker doesn't take,
        # but wiring it costs nothing since it is loaded for decode anyway.
        "vae": ["vae_loader_video", 0],
        "audio_vae": ["vae_loader_audio", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": length,
        "ref_image_size": ref_image_size,
    }
    for i, name in enumerate(ref_image_names):
        node_id = f"load_ref_{i}"
        nodes[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        r2v_inputs[ref_input_key(i)] = [node_id, 0]

    nodes["h3"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": r2v_inputs}

    return _sampling_and_output(
        nodes, model_ref, "h3", seed, steps, scheduler, mute, filename_prefix,
        attention,
    )


# The node whose output IS the job result — src/comfy_client.py looks this
# key up in the /history response to find the saved file.
OUTPUT_NODE_ID = "save_video"
