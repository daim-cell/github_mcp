import inspect
import json
import sys
import time
import functools
from pathlib import Path

_LOG_FILE = Path(__file__).parent.parent / "logs" / "tool_calls.jsonl"
_SUMMARY_LIMIT = 500


def _summarise(value: object) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    if len(text) > _SUMMARY_LIMIT:
        return text[:_SUMMARY_LIMIT] + f"... [{len(text) - _SUMMARY_LIMIT} chars truncated]"
    return text


def _rate_limit_remaining() -> int | None:
    try:
        from utils.auth import get_github_client
        return get_github_client().get_rate_limit().resources.core.remaining
    except Exception:
        return None


def log_tool_call(func):
    """Append a JSON record to tool_calls.jsonl for every tool invocation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Normalise positional + keyword args into a single named dict
        sig = inspect.signature(func)
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = dict(bound.arguments)
        except TypeError:
            inputs = {"args": list(args), **kwargs}

        start = time.time()
        error: str | None = None
        result = None

        print(f"[tool] {func.__name__} ← {json.dumps(inputs, default=str)}", file=sys.stderr, flush=True)
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            latency = round((time.time() - start) * 1000, 2)
            remaining = _rate_limit_remaining()
            status = "OK" if error is None else f"ERROR: {error}"
            print(f"[tool] {func.__name__} → {status} ({latency}ms, {remaining} requests remaining)", file=sys.stderr, flush=True)

            record = {
                "timestamp": start,
                "tool": func.__name__,
                "inputs": inputs,
                "output_summary": _summarise(result) if error is None else None,
                "latency_ms": latency,
                "success": error is None,
                "error": error,
                "rate_limit_remaining": remaining,
            }
            _LOG_FILE.parent.mkdir(exist_ok=True)
            with _LOG_FILE.open("a") as fh:
                fh.write(json.dumps(record) + "\n")

    return wrapper
