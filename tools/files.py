import base64

from utils.auth import get_github_client, guard_rate_limit
from utils.logger import log_tool_call


@log_tool_call
def list_directory(repo: str, path: str = "", branch: str = "main") -> list[dict]:
    """List files and directories at a given path in a GitHub repository."""
    guard_rate_limit()
    try:
        gh = get_github_client()
        repository = gh.get_repo(repo)
        contents = repository.get_contents(path or "", ref=branch)
        if not isinstance(contents, list):
            contents = [contents]
        return [
            {
                "name": item.name,
                "path": item.path,
                "type": item.type,   # "file" or "dir"
                "size": item.size,
                "url": item.html_url,
            }
            for item in sorted(contents, key=lambda x: (x.type != "dir", x.name))
        ]
    except Exception as exc:
        return {"error": f"Error listing directory: {exc}"}


@log_tool_call
def get_file_contents(repo: str, path: str, branch: str = "main") -> dict:
    """Fetch the contents of a single file from a GitHub repository."""
    guard_rate_limit()
    try:
        gh = get_github_client()
        repository = gh.get_repo(repo)
        file_obj = repository.get_contents(path, ref=branch)
        if isinstance(file_obj, list):
            return {"error": f"'{path}' is a directory — use list_directory instead"}
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
