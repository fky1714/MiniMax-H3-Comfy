"""Input validation for the MiniMax-H3 ComfyUI worker.

Field-for-field mirror of the earlier diffusers worker's validator so
existing clients and API.md keep working unchanged. The only behavioral
difference: frame-count snapping is delegated to
workflow.frame_length() here rather than reimplemented, since that module
already has to reproduce the same `17n + 5` grid math for the ComfyUI node
graph and the two must never drift apart.
"""

from src import config
from src.errors import ValidationError
from src.workflow import frame_length


def _validate_image_source(inp: dict, url_key: str, b64_key: str) -> bool:
    has_url = bool(inp.get(url_key))
    has_b64 = bool(inp.get(b64_key))
    if has_url and has_b64:
        raise ValidationError(
            f"Provide either {url_key} or {b64_key}, not both.",
            hint="Remove one of the image source fields.",
        )
    return has_url or has_b64


def _normalize_ref_images(raw) -> list[dict]:
    """ref_images -> [{"url": ...} | {"base64": ...}, ...], prompt order.

    Each entry may be an object with exactly one of `url` / `base64`, or a
    bare string: http(s) URLs are taken as `url`, anything else (raw base64
    or a data: URL) as `base64`."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValidationError(
            "ref_images must be an array.",
            hint='e.g. [{"url": "https://..."}, {"base64": "..."}]',
        )
    if len(raw) > config.MAX_REF_IMAGES:
        raise ValidationError(
            f"ref_images accepts at most {config.MAX_REF_IMAGES} images.",
            hint=f"You sent {len(raw)}.",
        )
    refs = []
    for i, item in enumerate(raw):
        if isinstance(item, str) and item:
            if item.startswith(("http://", "https://")):
                refs.append({"url": item})
            else:
                refs.append({"base64": item})
            continue
        if isinstance(item, dict):
            url, b64 = item.get("url"), item.get("base64")
            if bool(url) != bool(b64) and isinstance(url or b64, str):
                refs.append({"url": url} if url else {"base64": b64})
                continue
        raise ValidationError(
            f"ref_images[{i}] must be a URL/base64 string or an object with exactly one of 'url' or 'base64'.",
        )
    return refs


def validate_input(inp: dict) -> dict:
    """Validate and normalize job input. Returns a cleaned dict with defaults applied."""
    if not isinstance(inp, dict):
        raise ValidationError(
            "Input must be a JSON object.",
            hint="Send a JSON object with at least a 'prompt' field.",
        )

    has_first = _validate_image_source(inp, "image_url", "image_base64")
    has_last = _validate_image_source(inp, "last_image_url", "last_image_base64")
    is_i2v = has_first or has_last
    mode = "i2v" if is_i2v else "t2v"

    ref_images = _normalize_ref_images(inp.get("ref_images"))
    if config.WORKER_MODE == "r2v":
        if is_i2v:
            raise ValidationError(
                "This R2V endpoint does not take image_url/last_image_* keyframes.",
                hint="Pass the frame as one of ref_images and refer to it in the prompt as <Picture N>.",
            )
        mode = "r2v"
    elif ref_images:
        raise ValidationError(
            "ref_images requires the R2V endpoint.",
            hint="Send this request to the MiniMax-H3 R2V endpoint (WORKER_MODE=r2v).",
        )

    prompt = inp.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValidationError("prompt must be a string.")
    prompt = prompt.strip()
    if not prompt:
        raise ValidationError(
            "prompt is required.",
            hint="Provide a text description of the desired video.",
        )
    if len(prompt) > config.MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"prompt exceeds max length ({config.MAX_PROMPT_LENGTH} chars).",
            hint="Shorten your prompt.",
        )

    # Validated for parity with the other workers, then ignored: the
    # checkpoint is guidance-distilled and has no negative prompt.
    negative_prompt = inp.get("negative_prompt", "")
    if not isinstance(negative_prompt, str):
        raise ValidationError("negative_prompt must be a string.")
    if len(negative_prompt) > config.MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"negative_prompt exceeds max length ({config.MAX_PROMPT_LENGTH} chars).",
        )

    seconds = inp.get("seconds", config.DEFAULT_SECONDS)
    if not isinstance(seconds, (int, float)) or seconds != int(seconds):
        raise ValidationError(
            "seconds must be an integer.",
            hint=f"Allowed range: {config.MIN_SECONDS}-{config.MAX_SECONDS}",
        )
    seconds = int(seconds)
    if seconds < config.MIN_SECONDS or seconds > config.MAX_SECONDS:
        raise ValidationError(
            f"seconds must be between {config.MIN_SECONDS} and {config.MAX_SECONDS}.",
            hint=(
                f"You sent seconds={seconds}. This endpoint accepts "
                f"{config.MIN_SECONDS}-{config.MAX_SECONDS} (env MIN_SECONDS / MAX_SECONDS); "
                f"MiniMax-H3 is trained on {config.TRAINED_MIN_SECONDS}-{config.SECONDS_CEILING} second clips."
            ),
        )

    fps = inp.get("fps", config.DEFAULT_FPS)
    if fps not in config.ALLOWED_FPS:
        raise ValidationError(
            f"fps must be one of {sorted(config.ALLOWED_FPS)}.",
            hint=f"You sent fps={fps}. MiniMax-H3 generates at a fixed 24 fps.",
        )

    num_frames = frame_length(seconds, fps)

    turbo = None
    ref_image_size = "match"
    scheduler = "simple"
    if mode == "r2v":
        turbo = inp.get("turbo", config.R2V_TURBO_DEFAULT)
        if not isinstance(turbo, bool):
            raise ValidationError("turbo must be a boolean.")
        ref_image_size = inp.get("ref_image_size", ref_image_size)
        if ref_image_size not in config.REF_IMAGE_SIZES:
            raise ValidationError(
                f"ref_image_size must be one of {list(config.REF_IMAGE_SIZES)}.",
                hint="'max' keeps more reference detail but can be several times slower.",
            )
        scheduler = inp.get("scheduler", scheduler)
        if scheduler not in config.R2V_SCHEDULERS:
            raise ValidationError(f"scheduler must be one of {list(config.R2V_SCHEDULERS)}.")

    # R2V has no single default: None lets workflow.build_r2v_workflow pick
    # 8 (turbo) or 20 (base) to match whichever branch the request chose.
    default_steps = None if mode == "r2v" else config.DEFAULT_STEPS
    steps = inp.get("steps", default_steps)
    if steps is not None and (not isinstance(steps, int) or steps < 1 or steps > config.MAX_STEPS):
        raise ValidationError(
            f"steps must be an integer between 1 and {config.MAX_STEPS}.",
            hint=f"You sent steps={steps}.",
        )

    cfg_scale = inp.get("cfg_scale", config.DEFAULT_CFG_SCALE)
    if not isinstance(cfg_scale, (int, float)) or cfg_scale < 0 or cfg_scale > 20:
        raise ValidationError(
            "cfg_scale must be a number between 0 and 20.",
            hint=f"You sent cfg_scale={cfg_scale}.",
        )

    seed = inp.get("seed", config.DEFAULT_SEED)
    if not isinstance(seed, int):
        raise ValidationError("seed must be an integer.")

    resolution = inp.get("resolution")
    if resolution:
        if resolution not in config.RESOLUTION_PRESETS:
            presets = ", ".join(sorted(config.RESOLUTION_PRESETS.keys()))
            raise ValidationError(
                f"Unknown resolution preset: {resolution}",
                hint=f"Available presets: {presets}. Or specify width/height directly.",
            )
        width, height = config.RESOLUTION_PRESETS[resolution]
    else:
        width = inp.get("width", config.DEFAULT_WIDTH)
        height = inp.get("height", config.DEFAULT_HEIGHT)

    if not isinstance(width, int) or not isinstance(height, int):
        raise ValidationError("width and height must be integers.")
    if width < config.MIN_DIMENSION or height < config.MIN_DIMENSION:
        raise ValidationError(
            f"width and height must be at least {config.MIN_DIMENSION}.",
            hint=f"You sent {width}x{height}.",
        )
    if width % config.MOD_VALUE != 0 or height % config.MOD_VALUE != 0:
        snapped_w = width // config.MOD_VALUE * config.MOD_VALUE
        snapped_h = height // config.MOD_VALUE * config.MOD_VALUE
        raise ValidationError(
            f"width and height must be divisible by {config.MOD_VALUE}.",
            hint=f"You sent {width}x{height}. Try {snapped_w}x{snapped_h}.",
        )
    if width * height > config.MAX_RESOLUTION_AREA:
        raise ValidationError(
            f"Total pixels ({width}x{height}={width*height}) exceeds max ({config.MAX_RESOLUTION_AREA}).",
            hint="Use a smaller resolution or increase MAX_RESOLUTION_AREA.",
        )

    # Accepted, ignored: MiniMax-H3's keyframes have no strength knob.
    image_strength = float(inp.get("image_strength", 1.0))
    if image_strength < 0 or image_strength > 1.0:
        raise ValidationError(
            "image_strength must be between 0 and 1.",
            hint=f"You sent image_strength={image_strength}.",
        )

    mute = inp.get("mute", False)
    if not isinstance(mute, bool):
        raise ValidationError("mute must be a boolean.")

    video_vae = inp.get("video_vae", config.VIDEO_VAE_DEFAULT)
    if video_vae not in config.VIDEO_VAES:
        raise ValidationError(
            f"video_vae must be one of {list(config.VIDEO_VAES)}.",
            hint=f"You sent video_vae={video_vae!r}.",
        )

    attention = inp.get("attention", config.ATTENTION_DEFAULT)
    if attention not in config.ATTENTION_BACKENDS:
        raise ValidationError(
            f"attention must be one of {list(config.ATTENTION_BACKENDS)}.",
            hint=f"You sent attention={attention!r}.",
        )

    output_cfg = inp.get("output", {})
    if not isinstance(output_cfg, dict):
        raise ValidationError("output must be a JSON object.")

    ignored = [p for p in config.IGNORED_PARAMS if p in inp]
    if mode != "r2v":
        ignored += [p for p in config.R2V_ONLY_PARAMS if p in inp]

    return {
        "mode": mode,
        "image_url": inp.get("image_url"),
        "image_base64": inp.get("image_base64"),
        "last_image_url": inp.get("last_image_url"),
        "last_image_base64": inp.get("last_image_base64"),
        "ref_images": ref_images,
        "turbo": turbo,
        "ref_image_size": ref_image_size,
        "scheduler": scheduler,
        "image_strength": image_strength,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seconds": seconds,
        "fps": fps,
        "num_frames": num_frames,
        "steps": steps,
        "cfg_scale": float(cfg_scale),
        "seed": seed,
        "width": width,
        "height": height,
        "mute": mute,
        "attention": attention,
        "video_vae": video_vae,
        "ignored_params": ignored,
        "output": {
            "bucket": output_cfg.get("bucket", config.S3_BUCKET_DEFAULT),
            "prefix": output_cfg.get("prefix", "minimax-h3-comfy/"),
            "public": output_cfg.get("public", True),
            "ttl_seconds": output_cfg.get("ttl_seconds", 0),
        },
    }
