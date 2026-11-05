#!/usr/bin/env python3
"""Build a slim fork of Comfy-Org/MiniMax-H3 for RunPod Model Cache.

Why this exists
----------------
This is the ComfyUI-native path for MiniMax-H3. An earlier diffusers Modular
Pipeline worker kept failing on RunPod: repeated "image ready, model pending
download" stalls with no RunPod-side fix, then a diffusers API break in
load_components(), then a 30-minute build-time cap on baking the model into
the image.

ComfyUI got native day-0 support for H3 on 2026-08-03 (Comfy-Org/ComfyUI
#15224), and Comfy-Org publishes ready-made pruned/quantized checkpoints in
ComfyUI's own key layout, which diffusers cannot read. Using them natively
means adopting Comfy-Org's maintained loading code instead of carrying a
hand-rolled float8 quantizer and a from-scratch Modular Pipeline
integration.

Same problem as ever, though: Comfy-Org/MiniMax-H3 is 447 GB (it carries
every precision tier for every workflow side by side — full bf16, int8,
fp8_scaled, pruned and non-pruned, t2va and ref2va, three text-encoder
precisions). RunPod Model Cache mirrors a repo WHOLE, so pointing it at that
repo directly reproduces the exact failure this script exists to avoid. This
copies out only the T2V/I2V (fl2va) file set, at the smallest precision tier
Comfy-Org ships:

    diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors   19.53 GB
    text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors         14.61 GB
    vae/minimax_h3_video_vae_fp16.safetensors                          4.85 GB
    vae/minimax_h3_video_vae_int8_convrot.safetensors                  2.62 GB
    vae/minimax_h3_audio_vae_fp32.safetensors                          0.56 GB
    loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors    1.82 GB
                                                                       --------
                                                                       ~44.0 GB

The LoRA is Comfy-Org's own ComfyUI-format re-export of lightx2v's 8-step
Turbo distillation — not an official H3 release, but published by Comfy-Org
itself rather than a third party, and it is what takes this from ~20
diffusion-model NFE down to 8. Comfy-Org ships no native step-distillation;
without this LoRA the ComfyUI default is a much slower ~20-step bake.

ref2va (the reference-to-video workflow) is kept out of this fork: it would
add another ~23 GB. `--workflow r2v` builds a separate fork for it instead
(ref2va + the shared text encoder/VAEs + lightx2v's ref2v Turbo LoRA),
served by its own WORKER_MODE=r2v endpoint.

How it works
-------------
One file at a time: download -> upload -> delete. Peak local disk stays at
roughly one file (~20 GB, the diffusion model) instead of 41 GB, and the run
is resumable — files already present in the destination repo are skipped.

    pip install -U huggingface_hub hf_transfer
    export HF_TOKEN=hf_...            # needs write access to the destination
    python tools/make_comfy_slim_fork.py --dst <your-hf-name>/MiniMax-H3-comfy-slim --dry-run
    python tools/make_comfy_slim_fork.py --dst <your-hf-name>/MiniMax-H3-comfy-slim --public

    # Reference-to-Video fork for the r2v endpoint (~41.4 GiB, same as the i2v fork):
    python tools/make_comfy_slim_fork.py --workflow r2v --dst <your-hf-name>/MiniMax-H3-comfy-r2v-slim --dry-run
    python tools/make_comfy_slim_fork.py --workflow r2v --dst <your-hf-name>/MiniMax-H3-comfy-r2v-slim --public

Visibility: public, because a free HF account's private-storage quota does
not reliably clear repos of this size. Check the MiniMax H3 Community License
(and the LoRA's terms) before redistributing the weights publicly.
"""

import argparse
import os
import sys
import time

try:
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass

from huggingface_hub import HfApi, get_token, hf_hub_download

SRC_REPO = "Comfy-Org/MiniMax-H3"

# path in source repo -> nothing needed here; ComfyUI resolves loaders by
# directory (diffusion_models/, text_encoders/, vae/, loras/), so the
# destination repo keeps the exact same paths and a worker just points
# ComfyUI's model search paths (or symlinks) at the Model Cache snapshot.
LIGHTX2V_TURBO_REPO = "lightx2v/Minimax-h3-Turbo"

# Per-workflow file sets: (source repo, path in source, path in destination).
# ComfyUI resolves loaders by directory (diffusion_models/, text_encoders/,
# vae/, loras/), so destination paths keep that layout.
_SHARED = [
    (SRC_REPO, "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
     "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    (SRC_REPO, "vae/minimax_h3_video_vae_fp16.safetensors",
     "vae/minimax_h3_video_vae_fp16.safetensors"),
    # INT8+ConvRot video VAE (2.62 GB): ComfyUI 0.36's fused decoder kernels
    # run it faster than the fp16 one. Kept alongside fp16 so the worker can
    # pick either per request (`video_vae`) — a repo that loses fp16 would
    # strand any worker image still loading it.
    (SRC_REPO, "vae/minimax_h3_video_vae_int8_convrot.safetensors",
     "vae/minimax_h3_video_vae_int8_convrot.safetensors"),
    (SRC_REPO, "vae/minimax_h3_audio_vae_fp32.safetensors",
     "vae/minimax_h3_audio_vae_fp32.safetensors"),
    (SRC_REPO, "README.md", "README.md"),
    (SRC_REPO, ".gitattributes", ".gitattributes"),
]
WORKFLOW_FILES = {
    # fkyyy/MiniMax-H3-comfy-slim, ~44.0 GB — the original T2V/I2V fork.
    "i2v": [
        (SRC_REPO, "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
         "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
        (SRC_REPO, "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
         "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
    ] + _SHARED,
    # fkyyy/MiniMax-H3-comfy-r2v-slim, ~44 GiB (about the same as i2v) — Reference-to-Video for the
    # MiniMax-H3-Comfy worker's WORKER_MODE=r2v endpoint. A separate repo
    # rather than adding ~23 GB to the i2v fork: 64 GB is close to the 75 GB
    # fork that never left Model Cache's "pending download" state. The ref2v
    # Turbo LoRA comes from lightx2v (8-step v1.1, 768p); Comfy-Org only
    # re-hosts the older 4-step v0.1. Its file sits at that repo's root, so
    # it is placed under loras/ here.
    "r2v": [
        (SRC_REPO, "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
         "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
        (LIGHTX2V_TURBO_REPO, "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors",
         "loras/minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"),
    ] + _SHARED,
}
_OPTIONAL = ("README.md", ".gitattributes")


def select_files(api: HfApi, token: str | None, workflow: str) -> list[tuple[str, str, str, int]]:
    """Returns (src_repo, src_path, dst_path, size) for every file of the
    workflow's set that exists upstream."""
    sizes: dict[str, dict[str, int]] = {}
    out, missing = [], []
    for repo, src_path, dst_path in WORKFLOW_FILES[workflow]:
        if repo not in sizes:
            info = api.repo_info(repo, files_metadata=True, token=token)
            sizes[repo] = {s.rfilename: (s.size or 0) for s in info.siblings}
        if src_path in sizes[repo]:
            out.append((repo, src_path, dst_path, sizes[repo][src_path]))
        elif src_path not in _OPTIONAL:
            missing.append(f"{repo}/{src_path}")
    if missing:
        print(f"WARNING: expected file(s) not found upstream: {missing}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dst", required=True, help="Destination repo id, e.g. <your-hf-name>/MiniMax-H3-comfy-slim")
    ap.add_argument("--workflow", choices=sorted(WORKFLOW_FILES), default="i2v",
                    help="Which file set to copy: i2v (fl2va, default) or r2v (ref2va)")
    ap.add_argument("--public", action="store_true", help="Create the destination public (default: private)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--work-dir", default=None,
                    help="Staging dir for the one-file-at-a-time relay (default: tools/_stage)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token and not args.dry_run:
        print(
            "No Hugging Face token. Set HF_TOKEN or run `huggingface-cli login` "
            "with a token that has write access to the destination.",
            file=sys.stderr,
        )
        return 2

    api = HfApi()
    files = select_files(api, token, args.workflow)
    total = sum(f[3] for f in files)
    print(f"{len(files)} files selected for --workflow {args.workflow}, {total / 1024**3:.2f} GB total")

    if args.dry_run:
        for repo, src_path, dst_path, size in sorted(files, key=lambda f: f[2]):
            print(f"  {size / 1024**3:9.3f} GB  {dst_path}  <- {repo}")
        return 0

    api.create_repo(args.dst, repo_type="model", private=not args.public, exist_ok=True, token=token)

    existing = {s.rfilename for s in api.repo_info(args.dst, token=token).siblings}
    todo = [f for f in files if f[2] not in existing]
    print(f"{len(existing)} already present, {len(todo)} to copy")

    work = args.work_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_stage")
    os.makedirs(work, exist_ok=True)
    largest = max((f[3] for f in todo), default=0)
    print(f"staging in {work} (needs ~{largest / 1024**3:.1f} GB free for the largest file)")

    done_bytes = 0
    for i, (repo, src_path, path, size) in enumerate(sorted(todo, key=lambda f: f[2]), 1):
        t0 = time.perf_counter()
        print(f"[{i}/{len(todo)}] {path} ({size / 1024**3:.3f} GB) <- {repo}", flush=True)

        local = hf_hub_download(repo, filename=src_path, local_dir=work, token=token)
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=path,
                repo_id=args.dst,
                repo_type="model",
                token=token,
                commit_message=f"Add {path}",
            )
        finally:
            try:
                os.remove(local)
            except OSError:
                pass

        done_bytes += size
        elapsed = time.perf_counter() - t0
        print(
            f"      done in {elapsed:.1f}s - {done_bytes / 1024**3:.2f} GB / "
            f"{sum(f[3] for f in todo) / 1024**3:.2f} GB",
            flush=True,
        )

    print(f"\nSlim ComfyUI fork ready: https://huggingface.co/{args.dst}")
    print("Set it, in lowercase, as the endpoint's Model field and as HF_REPO on the worker.")
    if not args.public:
        print("It is private, so the endpoint also needs an HF_TOKEN with read access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
