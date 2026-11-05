"""Link RunPod Model Cache's pre-downloaded snapshot into ComfyUI's models/ tree.

Model Cache mirrors the endpoint's Model repo in HF-hub cache layout under
/runpod-volume/huggingface-cache/hub/models--<org>--<name>/snapshots/<sha>/.
It does NOT set HF_HOME/HF_HUB_CACHE (verified on a live worker: both unset),
so the Dockerfile sets HF_HOME itself, and the lookup below also tries the
known mount path directly.

The cache directory is matched case-insensitively instead of going through
huggingface_hub.snapshot_download: the Model field is typed lowercase while
the Hub id is mixed case, and snapshot_download would only look for one exact
spelling. This never downloads anything — if the snapshot is not on disk it
fails immediately, listing what IS there, rather than falling back to a live
download.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src import config

logger = logging.getLogger("minimax-h3-comfy")

# From config.HF_REPO (itself from WORKER_MODE / HF_REPO) so the r2v
# endpoint links fkyyy/minimax-h3-comfy-r2v-slim instead of the i2v fork.
# Lowercase — see config.HF_REPO's comment.
REPO_ID = config.HF_REPO

# Must match the slim fork's top-level layout (these are also ComfyUI's own
# models/ category names).
SUBDIRS = ("diffusion_models", "text_encoders", "vae", "loras")

MODEL_CACHE_HUB = "/runpod-volume/huggingface-cache/hub"


class ModelLinkError(RuntimeError):
    pass


def _hub_dirs() -> list[Path]:
    candidates = []
    if os.environ.get("HF_HUB_CACHE"):
        candidates.append(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        candidates.append(os.path.join(os.environ["HF_HOME"], "hub"))
    candidates += [MODEL_CACHE_HUB, os.path.expanduser("~/.cache/huggingface/hub")]
    seen, out = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(Path(c))
    return out


def _find_snapshot(repo_id: str) -> Path | None:
    want = "models--" + repo_id.replace("/", "--").lower()
    for hub in _hub_dirs():
        if not hub.is_dir():
            continue
        for entry in hub.iterdir():
            if entry.name.lower() != want:
                continue
            snapshots = entry / "snapshots"
            if not snapshots.is_dir():
                continue
            for snap in sorted(snapshots.iterdir()):
                if any((snap / s).is_dir() for s in SUBDIRS):
                    return snap
    return None


def _describe_volume() -> str:
    lines = [f"HF_HOME={os.environ.get('HF_HOME', '<unset>')}",
             f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE', '<unset>')}"]
    for hub in _hub_dirs():
        if hub.is_dir():
            lines.append(f"{hub}: {sorted(p.name for p in hub.iterdir())}")
        else:
            lines.append(f"{hub}: <missing>")
    rv = Path("/runpod-volume")
    if rv.is_dir():
        lines.append(f"/runpod-volume: {sorted(p.name for p in rv.iterdir())}")
    else:
        lines.append("/runpod-volume: <missing>")
    return "; ".join(lines)


def link_model_cache(comfy_models_dir: str, repo_id: str = REPO_ID) -> str:
    """Symlink each weight category from the Model Cache snapshot into
    comfy_models_dir. Returns the snapshot root. Raises ModelLinkError if the
    snapshot isn't on disk."""
    logger.info("Model Cache lookup for %s: %s", repo_id, _describe_volume())

    snapshot = _find_snapshot(repo_id)
    if snapshot is None:
        raise ModelLinkError(
            f"{repo_id} snapshot not found in any HF hub cache. "
            f"Check the endpoint's Model field. Seen: {_describe_volume()}"
        )
    logger.info("Snapshot root: %s", snapshot)

    # Per-file links into ComfyUI's existing category dirs: a fresh ComfyUI
    # install already has models/<category>/ with put_*_here placeholders,
    # so the dirs themselves can't simply be replaced by symlinks.
    linked = 0
    for subdir in SUBDIRS:
        src_dir = snapshot / subdir
        if not src_dir.is_dir():
            logger.warning("Snapshot missing expected subdir %s, skipping", subdir)
            continue
        dst_dir = Path(comfy_models_dir) / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.iterdir():
            dst = dst_dir / src.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            # Snapshot entries are themselves symlinks into ../../blobs;
            # resolve so the link doesn't depend on a relative hop.
            dst.symlink_to(src.resolve())
            logger.info("Linked %s -> %s", dst, src.resolve())
            linked += 1

    if linked == 0:
        raise ModelLinkError(f"No model files found under snapshot {snapshot}")

    return str(snapshot)
