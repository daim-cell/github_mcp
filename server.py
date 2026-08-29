import sys
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tools.search import search_repositories
from tools.files import list_directory, get_file_contents
from tools.issues import list_issues, create_issue
from tools.pull_requests import get_pull_requests

load_dotenv()

mcp = FastMCP("github-mcp")

mcp.tool()(search_repositories)
mcp.tool()(list_directory)
mcp.tool()(get_file_contents)
mcp.tool()(list_issues)
mcp.tool()(create_issue)
mcp.tool()(get_pull_requests)


def _validate_auth() -> None:
    """Confirm the GitHub token is valid before the server accepts connections."""
    from utils.auth import get_github_client
    try:
        login = get_github_client().get_user().login
        print(f"[server] Authenticated as GitHub user: {login}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[server] GitHub authentication failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    _validate_auth()
    # Default transport is stdio — stdout/stdin carry MCP JSON-RPC messages.
    # All application logging must go to stderr (see _validate_auth above).
    mcp.run()
