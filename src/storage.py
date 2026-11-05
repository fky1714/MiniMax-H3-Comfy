"""S3-compatible object storage upload."""

import logging
import os
from datetime import datetime, timezone

import boto3
from botocore.config import Config as BotoConfig

from src import config
from src.errors import UploadError

logger = logging.getLogger("minimax-h3-comfy")


def _get_s3_client():
    kwargs = {
        "service_name": "s3",
        "region_name": config.S3_REGION,
        "aws_access_key_id": config.S3_ACCESS_KEY_ID,
        "aws_secret_access_key": config.S3_SECRET_ACCESS_KEY,
        "config": BotoConfig(
            connect_timeout=10,
            read_timeout=config.UPLOAD_TIMEOUT,
            retries={"max_attempts": 2},
        ),
    }
    if config.S3_ENDPOINT:
        kwargs["endpoint_url"] = config.S3_ENDPOINT
    return boto3.client(**kwargs)


def upload_video(
    local_path: str,
    job_id: str,
    bucket: str,
    prefix: str,
    public: bool = True,
    ttl_seconds: int = 0,
) -> str:
    """Upload an mp4 to S3-compatible storage. Returns the public or signed URL."""
    if not bucket:
        raise UploadError(
            "No S3 bucket configured.",
            hint="Set S3_BUCKET_DEFAULT or pass output.bucket in the request.",
        )

    now = datetime.now(timezone.utc)
    date_prefix = now.strftime("%Y/%m/%d")
    key = f"{prefix}{date_prefix}/{job_id}.mp4"

    client = _get_s3_client()
    file_size = os.path.getsize(local_path)
    logger.info("Uploading %s (%.1f MB) -> s3://%s/%s", local_path, file_size / (1024 * 1024), bucket, key)

    try:
        client.upload_file(
            Filename=local_path,
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
    except Exception as e:
        raise UploadError(
            f"S3 upload failed: {e}",
            hint="Check S3 credentials, bucket name, and network connectivity.",
        )

    if public and config.S3_PUBLIC_BASE_URL:
        url = f"{config.S3_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    elif not public and ttl_seconds > 0:
        try:
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=ttl_seconds,
            )
        except Exception as e:
            raise UploadError(
                f"Failed to generate presigned URL: {e}",
                hint="Check S3 credentials and permissions.",
            )
    elif config.S3_ENDPOINT:
        url = f"{config.S3_ENDPOINT.rstrip('/')}/{bucket}/{key}"
    else:
        url = f"https://{bucket}.s3.{config.S3_REGION}.amazonaws.com/{key}"

    logger.info("Upload complete: %s", url)
    return url
