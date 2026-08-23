import base64

from utils.auth import get_github_client, guard_rate_limit
from utils.logger import log_tool_call


@log_tool_call
def get_file_contents(repo: str, path: str, branch: str = "main") -> dict:
    """Fetch the contents of a file from a GitHub repository at a given path and branch."""
    guard_rate_limit()
    try:
        gh = get_github_client()
        repository = gh.get_repo(repo)
        file_obj = repository.get_contents(path, ref=branch)
        # get_contents can return a list when path is a directory
        if isinstance(file_obj, list):
            return {"error": f"'{path}' is a directory, not a file"}
        content = base64.b64decode(file_obj.content).decode("utf-8", errors="replace")
        return {
            "repo": repo,
            "path": file_obj.path,
            "branch": branch,
            "sha": file_obj.sha,
            "size": file_obj.size,
            "content": content,
        }
    except Exception as exc:
        return {"error": f"Error fetching file contents: {exc}"}
