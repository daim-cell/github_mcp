import asyncio
import sys
from typing import Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters, stdio_client

load_dotenv()

_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_pydantic(schema: dict) -> type[BaseModel]:
    """Convert a JSON Schema object to a Pydantic model for LangChain tool validation."""
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}
    for name, prop in props.items():
        py_type = _JSON_TYPE_MAP.get(prop.get("type", "string"), str)
        desc = prop.get("description", "")
        if name in required:
            fields[name] = (py_type, Field(description=desc))
        else:
            fields[name] = (Optional[py_type], Field(default=None, description=desc))
    # Pydantic requires at least one field; add a dummy if the schema is empty
    if not fields:
        fields["_placeholder"] = (Optional[str], Field(default=None, exclude=True))
    return create_model("ToolArgs", **fields)


def _make_langchain_tool(session: ClientSession, mcp_tool: Any) -> StructuredTool:
    """Wrap a single MCP tool definition as a LangChain StructuredTool."""
    tool_name = mcp_tool.name
    args_schema = _schema_to_pydantic(mcp_tool.input_schema or {})

    async def _call(**kwargs: Any) -> str:
        # Drop the dummy placeholder before forwarding
        kwargs.pop("_placeholder", None)
        result = await session.call_tool(tool_name, arguments=kwargs or None)
        if result.is_error:
            first = result.content[0] if result.content else None
            return f"Error: {getattr(first, 'text', str(first))}"
        return "\n".join(
            getattr(block, "text", str(block)) for block in result.content
        )

    return StructuredTool.from_function(
        coroutine=_call,
        name=tool_name,
        description=mcp_tool.description or "",
        args_schema=args_schema,
    )


async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tool_list = await session.list_tools()
            tools = [_make_langchain_tool(session, t) for t in tool_list.tools]

            print(f"[agent] Connected. {len(tools)} tool(s) available.", flush=True)
            for t in tools:
                print(f"  • {t.name}: {t.description}", flush=True)

            llm = ChatOllama(model="llama3.2:3b", temperature=0)
            system_prompt = (
                "You are a GitHub research assistant. "
                "You help users explore GitHub repositories by searching for projects, "
                "analyzing their metadata, and answering questions about them. "
                "Always use the available tools to fetch live data rather than relying on prior knowledge."
            )
            agent = create_react_agent(llm, tools, prompt=system_prompt)

            print("\n[agent] Ready. Type a query or 'quit' to exit.\n")
            while True:
                try:
                    query = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[agent] Exiting.")
                    break
                if not query:
                    continue
                if query.lower() in ("quit", "exit", "q"):
                    break

                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": query}]}
                )
                print(f"\n{result['messages'][-1].content}\n")


if __name__ == "__main__":
    asyncio.run(main())
