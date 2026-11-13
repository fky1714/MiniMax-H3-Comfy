"""Configuration and defaults for the MiniMax-H3 ComfyUI worker.

Mirrors the earlier diffusers worker's field names and defaults wherever the two
workers share a concept (validation limits, S3, Pod on-demand mode), so
API.md's contract and the on-demand deploy path both carry over unchanged.
Fields that only make sense for the diffusers Modular Pipeline (FP8 flags,
ComponentsManager offload, LORA_ALPHA/LORA_SCALE, sigma shifts) are dropped —
this worker's checkpoint is pre-quantized (int8/nvfp4) and its Turbo LoRA
pairing is fixed, both baked into fkyyy/MiniMax-H3-comfy-slim itself rather
than chosen at runtime.
"""

import os


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# -- Model / ComfyUI process ──────────────────────────────────────────────────
# Lowercase deliberately — see src/model_link.py's REPO_ID comment. Must match
# the literal string configured in the RunPod endpoint's Model field, not the
# Hub's mixed-case canonical id.
HF_REPO = os.getenv(
    "HF_REPO",
    "fkyyy/minimax-h3-comfy-r2v-slim"
    if os.getenv("WORKER_MODE", "i2v").strip().lower() == "r2v"
    else "fkyyy/minimax-h3-comfy-slim",
)
# Where the Dockerfile installs ComfyUI itself (core only, no models baked
# in — weights come from Model Cache via src/model_link.py at runtime).
COMFY_DIR = os.getenv("COMFY_DIR", "/opt/ComfyUI")
COMFY_HOST = os.getenv("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.getenv("COMFY_PORT", "8188"))
# Cold start: ComfyUI's own process boot (before any model weights load —
# those load lazily per-prompt) plus this worker's model_link.py symlinking.
COMFY_BOOT_TIMEOUT_S = int(os.getenv("COMFY_BOOT_TIMEOUT_S", "120"))
# Per-job ceiling: first prompt after boot pays weight-load time on top of
# inference, so this is generous relative to steady-state generation time.
COMFY_INFERENCE_TIMEOUT_S = int(os.getenv(
    "COMFY_INFERENCE_TIMEOUT_S",
    # R2V defaults to 20 non-turbo steps plus reference tokens; a 768p/10s
    # clip can plausibly run past 10 minutes.
    "1800" if os.getenv("WORKER_MODE", "i2v").strip().lower() == "r2v" else "600",
))
# A known ComfyUI 0.30.x regression around pinned-memory allocation has been
# reported fixed by --disable-pinned-memory. Off by default until confirmed
# either way against the pinned COMFYUI_VERSION — flip via env var without
# a rebuild if cold starts show it.
COMFY_DISABLE_PINNED_MEMORY = _env_bool("COMFY_DISABLE_PINNED_MEMORY", "0")

# ComfyUI 0.36's model compiler (malloc graphs + CUDA graphs) is on by
# default. Escape hatch, without a rebuild, if it misbehaves with this
# worker's model patch (the attention override).
COMFY_DISABLE_COMPILER = _env_bool("COMFY_DISABLE_COMPILER", "0")

# Whether to route through the unofficial 8-step Turbo LoRA baked into the
# slim fork. Off would mean 20 NFE at the base checkpoint's own schedule —
# unvalidated on this worker and
# not exposed to clients; this is a deploy-time toggle, not a request field.
TURBO_MODE = _env_bool("TURBO_MODE", "1")

# Which workflow this deployment serves. One image, two endpoints:
#   i2v — T2V/I2V via fl2va (the original endpoint, Model field
#         fkyyy/minimax-h3-comfy-slim)
#   r2v — Reference-to-Video via ref2va (a separate endpoint whose Model
#         field is fkyyy/minimax-h3-comfy-r2v-slim, ~41 GiB)
# The two are kept apart rather than merged into one 64 GB repo: the
# diffusers sibling's 75 GB fork stalled forever in Model Cache's "pending
# download" state, and one endpoint serving both would swap a ~20 GB
# diffusion model in and out of VRAM whenever requests alternate.
WORKER_MODE = os.getenv("WORKER_MODE", "i2v").strip().lower()
if WORKER_MODE not in ("i2v", "r2v"):
    raise ValueError(f"WORKER_MODE must be 'i2v' or 'r2v', got {WORKER_MODE!r}")

# R2V's Turbo LoRA is a per-request choice (the graph is rebuilt per job, so
# it costs nothing to switch), unlike TURBO_MODE above. Off by default, same
# as the official R2V template: ComfyUI's own docs warn that a few-step
# turbo "can barely apply references" — faces and poses drift — so fidelity
# wins over speed unless the caller opts in.
R2V_TURBO_DEFAULT = _env_bool("R2V_TURBO_DEFAULT", "0")
MAX_REF_IMAGES = 9          # MiniMaxH3ReferenceToVideo's ref_images max
REF_IMAGE_SIZES = ("match", "max")
R2V_SCHEDULERS = ("simple", "beta", "normal")

# Per-request `attention` (comfy_nodes/attention_backend). "default" is
# whatever ComfyUI picks at startup (PyTorch SDPA on this image); the others
# map to ComfyUI's registered backend names. Off until a same-seed A/B
# confirms quality holds; flip with the env var rather than a rebuild.
ATTENTION_BACKENDS = {
    "default": None,
    "sage": "sage",
    "ck_int8": "comfy_kitchen_int8",
}
ATTENTION_DEFAULT = os.getenv("ATTENTION_DEFAULT", "default").strip().lower()
if ATTENTION_DEFAULT not in ATTENTION_BACKENDS:
    raise ValueError(f"ATTENTION_DEFAULT must be one of {list(ATTENTION_BACKENDS)}, got {ATTENTION_DEFAULT!r}")

# Per-request `video_vae` default ("fp16" or "int8", workflow.VAE_VIDEO_FILES).
# int8 needs a slim-fork snapshot that carries the INT8 VAE file; the handler
# rejects the job up front if the linked models/vae/ dir lacks it.
VIDEO_VAES = ("fp16", "int8")
VIDEO_VAE_DEFAULT = os.getenv("VIDEO_VAE_DEFAULT", "fp16").strip().lower()
if VIDEO_VAE_DEFAULT not in VIDEO_VAES:
    raise ValueError(f"VIDEO_VAE_DEFAULT must be one of {list(VIDEO_VAES)}, got {VIDEO_VAE_DEFAULT!r}")

MODEL_TAG = os.getenv("MODEL_TAG", "minimax-h3-comfy")


def model_tag(turbo: bool | None = None) -> str:
    parts = [MODEL_TAG]
    if WORKER_MODE == "r2v":
        parts.append("r2v")
        if turbo:
            parts.append("turbo-8step")
    elif TURBO_MODE:
        parts.append("turbo-8step")
    return "-".join(parts)


# -- Inference defaults ───────────────────────────────────────────────────────
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "960"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "544"))
DEFAULT_FPS = 24
DEFAULT_SECONDS = 5
# 4 with the 8-step Turbo LoRA: compared against 8 on 2026-09-24 and judged
# good enough for ~40% less time per clip (see workflow.TURBO_STEPS).
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "4" if TURBO_MODE else "20"))
DEFAULT_CFG_SCALE = 1.0   # accepted for interface parity, NOT used
DEFAULT_SEED = -1

MOD_VALUE = 32
FRAME_MOD = 17
FRAME_OFFSET = 5

# -- Resolution presets — identical to the sibling diffusers worker's table --
RESOLUTION_PRESETS = {
    "768p":          (1344, 768),
    "768p-portrait": (768, 1344),
    "720p":          (1344, 768),   # alias
    "720p-portrait": (768, 1344),   # alias
    "square":        (992, 992),
    "480p":          (960, 544),
    "480p-portrait": (544, 960),
}

# -- Limits ───────────────────────────────────────────────────────────────────
SECONDS_CEILING = 15
MAX_SECONDS = min(int(os.getenv("MAX_SECONDS", "10")), SECONDS_CEILING)
# The model is trained on 124-362 frames (5.17-15 s), so 5 is the default
# floor. The node itself accepts any length on the 17n+5 grid from 5 frames
# up; MIN_SECONDS lowers the floor for experiments below the trained range
# (1 s -> 39 frames, 2 s -> 56, 3 s -> 73, 4 s -> 107). Quality there is not
# guaranteed.
TRAINED_MIN_SECONDS = 5
MIN_SECONDS = max(1, min(int(os.getenv("MIN_SECONDS", str(TRAINED_MIN_SECONDS))), MAX_SECONDS))
ALLOWED_FPS = {24}
MAX_STEPS = 60
MAX_PROMPT_LENGTH = 2000
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 8192 * 8192
MAX_RESOLUTION_AREA = int(os.getenv("MAX_RESOLUTION_AREA", str(1032192)))
MIN_DIMENSION = 128
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "4"))
IMAGE_DOWNLOAD_TIMEOUT = 10
UPLOAD_TIMEOUT = 60

# Same rationale as the sibling worker: the checkpoint is guidance-distilled,
# so these are validated for client-compatibility then dropped.
IGNORED_PARAMS = ("negative_prompt", "cfg_scale", "image_strength")
# R2V-only fields an i2v endpoint accepts and drops (ref_images itself is
# rejected outright there — silently ignoring references would hand back a
# video that looks nothing like what was asked for).
R2V_ONLY_PARAMS = ("turbo", "ref_image_size", "scheduler")

# -- S3 / object storage ──────────────────────────────────────────────────────
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_REGION = os.getenv("S3_REGION", "auto")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")
S3_BUCKET_DEFAULT = os.getenv("S3_BUCKET_DEFAULT", "")
S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL", "")

# -- On-demand Pod mode (SERVE_MODE=api, see src/watchdog.py) ─────────────────
SERVE_MODE = os.getenv("SERVE_MODE", "serverless")
POD_API_PORT = os.getenv("POD_API_PORT", "8000")
POD_IDLE_TIMEOUT_S = int(os.getenv("POD_IDLE_TIMEOUT_S", "180"))
POD_MAX_LIFETIME_S = int(os.getenv("POD_MAX_LIFETIME_S", "10800"))
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
