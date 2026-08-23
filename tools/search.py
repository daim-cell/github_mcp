from utils.auth import get_github_client, guard_rate_limit
from utils.logger import log_tool_call


@log_tool_call
def search_repositories(query: str, max_results: int = 5) -> list[dict]:
    """Search GitHub repositories and return key metadata for each match."""
    guard_rate_limit()
    try:
        gh = get_github_client()
        results = gh.search_repositories(query=query)
        repos = []
        for repo in results[:max_results]:
            repos.append({
                "full_name": repo.full_name,
                "description": repo.description or "",
                "url": repo.html_url,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "language": repo.language or "",
                "open_issues": repo.open_issues_count,
                "topics": repo.get_topics(),
            })
        return repos
    except Exception as exc:
        return f"Error searching repositories: {exc}"
