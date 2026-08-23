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
    """Raise if the GitHub core rate limit is too low to make a safe API call."""
    core = get_github_client().get_rate_limit().resources.core
    if core.remaining < 10:
        reset_in = max(0, int(core.reset.timestamp() - time.time()))
        raise RuntimeError(
            f"GitHub rate limit too low ({core.remaining} requests remaining). "
            f"Limit resets in {reset_in}s. Please wait before retrying."
        )
