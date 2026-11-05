"""Self-termination for the on-demand Pod deploy path (SERVE_MODE=api).

The design puts the *client* in charge of
keeping a Pod alive during a batch (light touches between requests) and
explicitly terminating it when done — this module is the safety net under
that, not the primary control. Its only job is to guarantee a forgotten or
crashed client's Pod does not run, and bill, forever.

Two independent timers, either one fires termination:
  - idle timeout: no handler activity for POD_IDLE_TIMEOUT_S
  - max lifetime: POD_MAX_LIFETIME_S since this process started, regardless
    of activity — the hard cap for a stuck client or a runaway batch

A no-op everywhere else: `start()` only arms anything when SERVE_MODE=api,
so the Serverless deploy path (the default) never imports runpod's Pod API
or spends a thread on this.
"""

import logging
import os
import threading
import time

from src import config

logger = logging.getLogger("minimax-h3-comfy")

_last_activity = time.monotonic()
_start_time = time.monotonic()
_lock = threading.Lock()
_armed = False


def touch() -> None:
    """Record activity. Call on every request the handler processes,
    including health checks — a client polling readiness during a long
    cold start should not have the idle timer fire out from under it."""
    global _last_activity
    with _lock:
        _last_activity = time.monotonic()


def _terminate_self(reason: str) -> None:
    logger.warning("Watchdog: terminating pod %s (%s)", config.RUNPOD_POD_ID, reason)
    try:
        import runpod
        runpod.api_key = config.RUNPOD_API_KEY
        runpod.terminate_pod(config.RUNPOD_POD_ID)
        logger.warning("Watchdog: terminate_pod call succeeded")
    except Exception:
        logger.exception("Watchdog: terminate_pod call failed")
    finally:
        # Belt-and-suspenders: whether or not the API call actually stopped
        # billing, exiting this process takes it out of serving traffic
        # immediately. os._exit skips cleanup on purpose — there is nothing
        # here worth a graceful shutdown for, and the point is to stop fast.
        os._exit(1)


def _loop() -> None:
    check_interval = max(1, min(config.POD_IDLE_TIMEOUT_S // 4, 30))
    while True:
        time.sleep(check_interval)
        now = time.monotonic()
        with _lock:
            idle_s = now - _last_activity
        lifetime_s = now - _start_time

        if lifetime_s > config.POD_MAX_LIFETIME_S:
            _terminate_self(f"max lifetime {config.POD_MAX_LIFETIME_S}s exceeded "
                            f"(ran {lifetime_s:.0f}s)")
            return
        if idle_s > config.POD_IDLE_TIMEOUT_S:
            _terminate_self(f"idle {idle_s:.0f}s > {config.POD_IDLE_TIMEOUT_S}s timeout")
            return


def start() -> None:
    """Arm the watchdog. Safe to call unconditionally; no-ops unless
    SERVE_MODE=api and the Pod identity/credentials it needs are present."""
    global _armed
    if config.SERVE_MODE != "api":
        return
    if _armed:
        return
    if not config.RUNPOD_POD_ID:
        logger.warning("Watchdog: SERVE_MODE=api but RUNPOD_POD_ID is not set "
                        "(not running on a Pod?); self-termination disabled.")
        return
    if not config.RUNPOD_API_KEY:
        logger.warning("Watchdog: SERVE_MODE=api but RUNPOD_API_KEY is not set; "
                        "self-termination disabled. This Pod will NOT stop itself "
                        "on idle or max lifetime — pass RUNPOD_API_KEY in its env.")
        return

    touch()
    _armed = True
    threading.Thread(target=_loop, name="watchdog", daemon=True).start()
    logger.info("Watchdog armed: idle_timeout=%ds max_lifetime=%ds pod_id=%s",
                config.POD_IDLE_TIMEOUT_S, config.POD_MAX_LIFETIME_S, config.RUNPOD_POD_ID)
