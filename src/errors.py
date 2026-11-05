"""Error classification for the MiniMax-H3 T2V/I2V worker."""

import traceback
import uuid


class WorkerError(Exception):
    """Base error with structured output for HTTP handler."""

    error_type: str = "InternalError"

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.trace_id = uuid.uuid4().hex


class ValidationError(WorkerError):
    error_type = "ValidationError"


class DownloadError(WorkerError):
    error_type = "DownloadError"


class InferenceError(WorkerError):
    error_type = "InferenceError"


class EncodeError(WorkerError):
    error_type = "EncodeError"


class UploadError(WorkerError):
    error_type = "UploadError"


def format_error(exc: Exception) -> dict:
    """Format any exception into the standard error response dict.

    `error` must be a plain string: the runpod SDK pops a top-level `error`
    key and sends it to /job-done as-is, and a dict there is rejected with
    400 Bad Request — the whole result is lost and the client only sees
    {"status": "failed"}. Structured details go in `error_detail` instead,
    which stays in `output`.
    """
    if isinstance(exc, WorkerError):
        error_type, message = exc.error_type, exc.message
        details = {"hint": exc.hint, "trace_id": exc.trace_id}
    else:
        error_type, message = "InternalError", str(exc)
        details = {
            "hint": "An unexpected error occurred.",
            "trace_id": uuid.uuid4().hex,
            "traceback": traceback.format_exc(),
        }
    return {
        "status": "failed",
        "error": f"{error_type}: {message}",
        "error_detail": {"type": error_type, "message": message, "details": details},
    }
