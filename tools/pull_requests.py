from utils.auth import get_github_client, guard_rate_limit
from utils.logger import log_tool_call


@log_tool_call
def get_pull_requests(repo: str, state: str = "open", base: str = "", max_results: int = 10) -> list[dict]:
    """List pull requests for a GitHub repository, optionally filtered by state and base branch."""
    guard_rate_limit()
    try:
        gh = get_github_client()
        repository = gh.get_repo(repo)
        kwargs = {"state": state}
        if base:
            kwargs["base"] = base
        prs = repository.get_pulls(**kwargs)
        result = []
        for pr in prs[:max_results]:
            result.append({
                "number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "author": pr.user.login if pr.user else "",
                "base": pr.base.ref,
                "head": pr.head.ref,
                "mergeable": pr.mergeable,
                "created_at": pr.created_at.isoformat(),
                "url": pr.html_url,
                "diff_url": pr.diff_url,
            })
        return result
    except Exception as exc:
        return f"Error fetching pull requests: {exc}"
