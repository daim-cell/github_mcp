import os
import time
from github import Github
from dotenv import load_dotenv

load_dotenv()

_client: Github | None = None


def get_github_client() -> Github:
    global _client
    if _client is None:
        # Accept either key name so .env flexibility is preserved
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_KEY")
        if not token:
            raise RuntimeError("GITHUB_TOKEN is not set in the environment")
        _client = Github(token)
    return _client


def guard_rate_limit() -> None:
    """Block until the GitHub core rate limit has enough remaining calls."""
    core = get_github_client().get_rate_limit().core
    if core.remaining < 10:
        wait = max(0.0, core.reset.timestamp() - time.time()) + 1
        time.sleep(wait)
