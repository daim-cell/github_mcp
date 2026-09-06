import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters, stdio_client
from opentelemetry import trace

from agent import _make_langchain_tool


@asynccontextmanager
async def mcp_tools(tracer: trace.Tracer):
    """Yield a list of LangChain StructuredTools backed by the MCP server."""
    server_params = StdioServerParameters(
        command=sys.executable, args=["server.py"], env=None
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_list = await session.list_tools()
            tools = [_make_langchain_tool(session, t, tracer) for t in tool_list.tools]
            yield tools
