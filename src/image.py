"""Image download, decode, and preprocessing.

Both keyframes are optional for MiniMax-H3 (it supports t2va), so the fetch
helpers return None when nothing was provided. `fetch_image` reads the first
frame, `fetch_last_image` the frame the video ends on.
"""

import base64
import io

import requests
from PIL import Image

from src import config
from src.errors import DownloadError, ValidationError
from src.security import validate_url


def _fetch(validated_input: dict, url_key: str, b64_key: str) -> Image.Image | None:
    if validated_input.get(url_key):
        return _download_image(validated_input[url_key])
    if validated_input.get(b64_key):
        return _decode_base64(validated_input[b64_key])
    return None


def fetch_image(validated_input: dict) -> Image.Image | None:
    """Fetch the first keyframe from URL or base64. Returns None for t2va."""
    return _fetch(validated_input, "image_url", "image_base64")


def fetch_last_image(validated_input: dict) -> Image.Image | None:
    """Fetch the closing keyframe from URL or base64. Returns None when unset."""
    return _fetch(validated_input, "last_image_url", "last_image_base64")


def fetch_ref_images(validated_input: dict) -> list[Image.Image]:
    """Fetch R2V reference images in order. validator.py has already
    normalized each entry to {"url": ...} or {"base64": ...}."""
    return [_fetch(ref, "url", "base64") for ref in validated_input.get("ref_images", [])]


def _download_image(url: str) -> Image.Image:
    validate_url(url)

    try:
        resp = requests.get(
            url,
            timeout=config.IMAGE_DOWNLOAD_TIMEOUT,
            stream=True,
            headers={"User-Agent": "MiniMaxH3-Worker/1.0"},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise DownloadError(
            f"Failed to download image: {e}",
            hint="Check that the URL is accessible and returns an image.",
        )

    content_type = resp.headers.get("Content-Type", "")
    if content_type and not content_type.startswith("image/"):
        raise DownloadError(
            f"URL returned non-image Content-Type: {content_type}",
            hint="Provide a URL that returns an image (png/jpg/webp).",
        )

    chunks = []
    total = 0
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        total += len(chunk)
        if total > config.MAX_IMAGE_BYTES:
            raise DownloadError(
                f"Image exceeds max size ({config.MAX_IMAGE_BYTES // (1024*1024)} MB).",
                hint="Use a smaller image.",
            )
        chunks.append(chunk)

    return _open_image(b"".join(chunks))


def _decode_base64(b64_str: str) -> Image.Image:
    # Accept data-URL prefix as well as raw base64.
    if b64_str.startswith("data:"):
        _, _, b64_str = b64_str.partition(",")

    try:
        data = base64.b64decode(b64_str)
    except Exception:
        raise ValidationError(
            "Invalid base64 encoding for image_base64.",
            hint="Send raw base64 or a data: URL.",
        )

    if len(data) > config.MAX_IMAGE_BYTES:
        raise ValidationError(
            f"Decoded image exceeds max size ({config.MAX_IMAGE_BYTES // (1024*1024)} MB).",
            hint="Use a smaller image.",
        )

    return _open_image(data)


def _open_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise DownloadError(
            "Cannot decode image data.",
            hint="Ensure the image is a valid PNG, JPEG, or WebP file.",
        )

    if img.format and img.format.upper() not in ("PNG", "JPEG", "JPG", "WEBP"):
        raise ValidationError(
            f"Unsupported image format: {img.format}",
            hint="Supported formats: PNG, JPEG, WebP.",
        )

    w, h = img.size
    if w * h > config.MAX_IMAGE_PIXELS:
        raise ValidationError(
            f"Image too large: {w}x{h} ({w*h} pixels).",
            hint=f"Max pixels: {config.MAX_IMAGE_PIXELS}.",
        )

    return img.convert("RGB")


def preprocess_image(img: Image.Image, target_width: int, target_height: int, mod_value: int) -> Image.Image:
    """Resize image to model input size, preserving aspect ratio and aligning to mod_value."""
    import numpy as np

    max_area = target_height * target_width
    aspect_ratio = img.height / img.width

    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

    height = max(height, mod_value)
    width = max(width, mod_value)

    return img.resize((width, height), Image.LANCZOS)
