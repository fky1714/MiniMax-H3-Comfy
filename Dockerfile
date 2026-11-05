FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Model Cache mounts the endpoint's Model repo under
# /runpod-volume/huggingface-cache/hub/ but does NOT set HF_HOME for us —
# confirmed on a live worker (both HF_HOME and HF_HUB_CACHE unset). The
# Flux/Krea/Wan2.2 workers set it here themselves; same here.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/runpod-volume/huggingface-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ComfyUI itself, via comfy-cli (its torch is replaced with a cu130 build in
# the next step).
#
# 0.30.0 is the floor (MiniMaxH3ImageToVideo landed there, Comfy-Org/ComfyUI
# #15224); pinned to a specific release rather than "latest" so a future
# ComfyUI release cannot change node behavior under this worker without a
# deliberate version bump here. Bump only after re-verifying
# src/workflow.py's node graph against the new release's node schemas.
#
# 0.36.0 (from 0.34.0): MiniMax-H3 video VAE fused kernels (faster decode,
# automatic for the fp16 VAE this worker loads) and the Comfy model compiler
# (on by default; COMFY_DISABLE_COMPILER=1 turns it off). Re-checked against
# 0.36.0's source: MiniMaxH3ImageToVideo / CreateVideo / SaveVideo schemas
# are unchanged; MiniMaxH3ReferenceToVideo's vae/audio_vae became optional
# (we still wire both); MiniMaxH3 model_base is unchanged.
ARG COMFYUI_VERSION=0.36.0
ARG COMFY_DIR=/opt/ComfyUI
ENV COMFY_DIR=${COMFY_DIR}
RUN pip install --no-cache-dir comfy-cli && \
    comfy --workspace ${COMFY_DIR} --skip-prompt install \
        --version ${COMFYUI_VERSION} --cuda-version 12.8 --nvidia && \
    test -f ${COMFY_DIR}/main.py

# Swap torch for a cu130 build. On cu128 torch, ComfyUI's comfy_kitchen
# disables its CUDA and Triton kernel backends ("You need pytorch with cu130
# or higher to use optimized CUDA operations") and runs the int8_convrot
# transformer and nvfp4 text encoder through the eager fallback — measured
# ~12 s per sampling step at 480p/5s on an H100 NVL. torchaudio's newest
# cu130 build is 2.11.0, so the trio is pinned to the matching 2.11 set.
# cu130 wheels bundle their own CUDA runtime but need a host driver that
# supports CUDA 13.0 (>= 580) — restrict the endpoint's allowed CUDA
# versions accordingly.
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
# Explicit +cu130 local versions (PyPI also publishes torch==2.11.0, as a
# different CUDA build); no --force-reinstall, so ComfyUI's other pinned deps
# are only touched if torch actually requires it.
RUN python -m pip install --no-cache-dir --timeout 300 --retries 5 \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
        torch==${TORCH_VERSION}+cu130 torchaudio==${TORCHAUDIO_VERSION}+cu130 \
        torchvision==${TORCHVISION_VERSION}+cu130 && \
    python -c "import torch, torchaudio, torchvision; \
print(f'torch={torch.__version__} cuda={torch.version.cuda} torchaudio={torchaudio.__version__} torchvision={torchvision.__version__}'); \
assert torch.version.cuda and torch.version.cuda.startswith('13'), torch.version.cuda"

# SageAttention for the per-request `attention: "sage"` option (see
# comfy_nodes/attention_backend). 1.0.6 is the PyPI release: Triton kernels
# (INT8 QK, fp16 PV), no CUDA build step, head_dim 64/128 (H3's is 128).
# SageAttention 2's sm90 CUDA kernels would be faster on H100 but must be
# compiled against torch's CUDA 13 -- this image's nvcc is 12.8 -- so not here.
# ComfyUI registers the "sage" backend only if this imports; a failing import
# makes `attention: "sage"` jobs fail, not the worker.
ARG SAGEATTENTION_VERSION=1.0.6
RUN python -m pip install --no-cache-dir sageattention==${SAGEATTENTION_VERSION}

# This worker's own deps (runpod, requests, websocket-client for
# src/comfy_client.py, boto3 for src/storage.py) — separate from whatever
# comfy-cli installed into the same environment for ComfyUI itself.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --timeout 180 --retries 5 \
    -r requirements.txt

# Application code
COPY src/ src/

# Our own ComfyUI custom node (opt-in per request via `attention`, see
# src/workflow.py). Pure Python on ComfyUI's own modules — nothing to pip
# install; ComfyUI picks it up from custom_nodes/ at startup.
COPY comfy_nodes/attention_backend ${COMFY_DIR}/custom_nodes/attention_backend

# Verify imports at build time. handler.py is excluded — importing it starts
# the ComfyUI subprocess and blocks on model linking. Everything else is
# pure Python with no GPU/process dependency, so any dependency mismatch or
# typo fails the build here instead of at the next worker cold start.
RUN python -c "\
from src.config import MODEL_TAG, HF_REPO, COMFY_DIR, model_tag; \
from src.validator import validate_input; \
from src.workflow import build_workflow, build_r2v_workflow, frame_length, OUTPUT_NODE_ID; \
from src.comfy_client import ComfyClient, ComfyError; \
from src.model_link import link_model_cache, ModelLinkError; \
from src.image import fetch_image, fetch_last_image, fetch_ref_images, preprocess_image; \
from src.storage import upload_video; \
from src.errors import format_error; \
from src.security import validate_url; \
import runpod, boto3, websocket; \
assert frame_length(5, 24) == 124, frame_length(5, 24); \
wf = build_workflow(prompt='a cat', width=960, height=544, seconds=5, seed=1); \
assert OUTPUT_NODE_ID in wf, wf.keys(); \
assert 'attention' not in wf, wf.keys(); \
wf = build_workflow(prompt='a cat', width=960, height=544, seconds=5, seed=1, attention='sage'); \
assert wf['attention']['inputs']['backend'] == 'sage', wf['attention']; \
wf = build_workflow(prompt='a cat', width=960, height=544, seconds=5, seed=1, video_vae='int8'); \
assert wf['vae_loader_video']['inputs']['vae_name'].endswith('int8_convrot.safetensors'), wf['vae_loader_video']; \
import os; assert all(os.path.isfile(os.path.join(COMFY_DIR, 'custom_nodes', n, '__init__.py')) for n in ('attention_backend',)); \
wf = build_r2v_workflow(prompt='<Picture 1>', width=768, height=1344, seconds=5, seed=1, ref_image_names=['r.png']); \
assert wf['h3']['class_type'] == 'MiniMaxH3ReferenceToVideo', wf['h3']; \
print(f'All imports OK (tag={MODEL_TAG} model={model_tag()} repo={HF_REPO} comfy_dir={COMFY_DIR})')"

# Only relevant when SERVE_MODE=api (the on-demand Pod deploy path, see
# src/watchdog.py). Serverless (the default) never
# binds this — EXPOSE is documentation for `docker run -p`/Pod port
# mapping, not a switch.
EXPOSE 8000

CMD ["python", "-u", "src/handler.py"]
