from utils.auth import get_github_client, guard_rate_limit
from utils.logger import log_tool_call


@log_tool_call
def list_issues(repo: str, state: str = "open", label: str = "", max_results: int = 10) -> list[dict]:
    """List issues for a GitHub repository, optionally filtered by state and label."""
    guard_rate_limit()
    try:
        gh = get_github_client()
        repository = gh.get_repo(repo)
        kwargs = {"state": state}
        if label:
            kwargs["labels"] = [repository.get_label(label)]
        issues = repository.get_issues(**kwargs)
        result = []
        for issue in issues[:max_results]:
            # get_issues returns PRs too unless filtered; skip them
            if issue.pull_request:
                continue
            result.append({
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "author": issue.user.login if issue.user else "",
                "labels": [lbl.name for lbl in issue.labels],
                "created_at": issue.created_at.isoformat(),
                "url": issue.html_url,
                "body": (issue.body or "")[:500],
            })
        return result
    except Exception as exc:
        return f"Error listing issues: {exc}"


@log_tool_call
def create_issue(repo: str, title: str, body: str = "", labels: list[str] | None = None) -> dict:
    """Open a new issue on a GitHub repository. Only use against personal test repos."""
    if not title or not title.strip():
        return {"error": "title must not be empty"}
    guard_rate_limit()
    try:
        gh = get_github_client()
        repository = gh.get_repo(repo)
        label_objs = []
        for name in (labels or []):
            try:
                label_objs.append(repository.get_label(name))
            except Exception:
                # Skip labels that don't exist rather than aborting
                pass
        issue = repository.create_issue(
            title=title.strip(),
            body=body or "",
            labels=label_objs,
        )
        return {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "url": issue.html_url,
            "labels": [lbl.name for lbl in issue.labels],
        }
    except Exception as exc:
        return {"error": f"Error creating issue: {exc}"}
