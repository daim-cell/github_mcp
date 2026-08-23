import json
import time
import functools
from pathlib import Path

_LOG_FILE = Path(__file__).parent.parent / "logs" / "tool_calls.jsonl"


def log_tool_call(func):
    """Append a JSON record to tool_calls.jsonl for every tool invocation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        error: str | None = None
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            record = {
                "tool": func.__name__,
                "kwargs": kwargs,
                "duration_ms": round((time.time() - start) * 1000, 2),
                "timestamp": time.time(),
                "error": error,
            }
            _LOG_FILE.parent.mkdir(exist_ok=True)
            with _LOG_FILE.open("a") as fh:
                fh.write(json.dumps(record) + "\n")

    return wrapper
