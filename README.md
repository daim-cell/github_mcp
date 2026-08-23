# GitHub Research MCP

An MCP server that gives AI assistants structured access to GitHub repository research.

GitHub Research MCP lets an AI agent search repositories, inspect project metadata, search code, and analyze repository activity using a small set of focused MCP tools backed by the GitHub API.

## Features

* Search GitHub repositories using keywords and filters
* Retrieve detailed repository information
* Search code across repositories
* Inspect repository activity and maintenance signals
* Use GitHub data directly from MCP-compatible AI clients
* Simple, focused toolset designed for research workflows

## Tools

### `search_repos`

Search GitHub repositories matching a query.

Example use cases:

* Find popular Rust vector databases
* Discover open-source alternatives to a product
* Find repositories related to a specific technology
* Filter projects by language, stars, or topic

Example input:

```json
{
  "query": "vector database",
  "language": "Rust",
  "min_stars": 1000
}
```

---

### `get_repo`

Get detailed information about a GitHub repository.

Returns information such as:

* Repository name
* Description
* Stars
* Forks
* Primary language
* Topics
* License
* Open issues
* Default branch
* Last updated date
* Repository URL

Example input:

```json
{
  "owner": "owner-name",
  "repo": "repository-name"
}
```

---

### `search_code`

Search GitHub code for a keyword, function, library, configuration, or implementation pattern.

Example use cases:

* Find usage examples for a library
* Search for a specific function
* Discover implementations of an algorithm
* Find configuration examples across repositories

Example input:

```json
{
  "query": "createClient",
  "repo": "owner/repository"
}
```

The repository parameter can be optional if global code search is supported.

---

### `get_repo_activity`

Inspect how actively a repository is maintained.

Possible signals include:

* Recent commits
* Pull request activity
* Issue activity
* Latest release
* Contributor activity
* Last push date

Example input:

```json
{
  "owner": "owner-name",
  "repo": "repository-name"
}
```

This tool can help an AI assistant answer questions such as:

> Is this project still actively maintained?

or:

> Which of these open-source projects appears healthiest?


## Architecture

```text
MCP Client
    |
    v
GitHub Research MCP
    |
    +-- search_repos
    +-- get_repo
    +-- search_code
    +-- get_repo_activity
    |
    v
GitHub API
```

The MCP server acts as a focused interface between an AI assistant and GitHub's API.

Instead of exposing dozens of GitHub endpoints, it provides a small set of tools optimized for repository research.


## MCP Configuration

Add the server to your MCP client configuration.

Example:

```json
{
  "mcpServers": {
    "github-research": {
      "command": "node",
      "args": [
        "/absolute/path/to/github-research-mcp/dist/index.js"
      ],
      "env": {
        "GITHUB_TOKEN": "YOUR_GITHUB_TOKEN"
      }
    }
  }
}
```

The exact configuration may vary depending on the MCP client you use.

## Example Workflow

Suppose a user asks:

> Find three actively maintained open-source feature flag platforms.

The AI agent could perform:

```text
search_repos
    ↓
get_repo
    ↓
get_repo_activity
    ↓
compare results
```

For implementation research:

> How are popular Node.js projects implementing request rate limiting?

The agent could use:

```text
search_repos
    ↓
search_code
    ↓
get_repo
```

The MCP server handles data retrieval while the AI model handles comparison, summarization, and reasoning.

