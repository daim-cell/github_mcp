import asyncio
import json
import sys
from typing import Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters, stdio_client

load_dotenv()


def _patch_tool_calls(msg: AIMessage) -> AIMessage:
    """Convert text-formatted JSON tool calls to structured tool_calls.

    llama3.2:3b sometimes emits {"name": "...", "parameters": {...}} as plain
    text instead of using the structured function-calling format. When that
    happens LangGraph sees no tool_calls and routes to END immediately. This
    interceptor runs after the LLM and converts the text JSON to a real
    tool_call so the graph continues normally.
    """
    if msg.tool_calls or not isinstance(msg.content, str) or "{" not in msg.content:
        return msg
    try:
        start = msg.content.index("{")
        obj, _ = json.JSONDecoder().raw_decode(msg.content, start)
        if not isinstance(obj, dict) or "name" not in obj:
            return msg
        params = obj.get("parameters") or obj.get("arguments") or {}
        if not isinstance(params, dict):
            return msg
        return AIMessage(
            content="",
            tool_calls=[{
                "id": f"fallback_{obj['name']}",
                "name": obj["name"],
                "args": params,
                "type": "tool_call",
            }],
        )
    except (ValueError, json.JSONDecodeError, KeyError):
        return msg


def _wrap_llm(llm):
    """Attach _patch_tool_calls after bind_tools so text JSON becomes real tool calls."""
    class _Wrapped:
        def bind_tools(self, tools, **kwargs):
            return llm.bind_tools(tools, **kwargs) | RunnableLambda(_patch_tool_calls)
        def __getattr__(self, name):
            return getattr(llm, name)
    return _Wrapped()


_MAX_CALLS_PER_TOOL = 3
_call_counts: dict[str, int] = {}


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
        # anyOf is generated for union types like `list[str] | None`.
        # Extract the first non-null branch so we get the real type.
        if "anyOf" in prop:
            non_null = [s for s in prop["anyOf"] if s.get("type") != "null"]
            prop = non_null[0] if non_null else {"type": "string"}
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
        count = _call_counts.get(tool_name, 0)
        if count >= _MAX_CALLS_PER_TOOL:
            return (
                f"Error: '{tool_name}' has already been called {count} time(s) "
                f"this query (limit: {_MAX_CALLS_PER_TOOL}). "
                "Use results you already have or try a different approach."
            )
        _call_counts[tool_name] = count + 1

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

            llm = _wrap_llm(ChatOllama(model="qwen2.5:7b", temperature=0))
            system_prompt = (
                "You are a GitHub research assistant with access to live GitHub API tools.\n\n"
                "STRICT RULES — follow these every single time:\n"
                "1. You MUST call a tool before answering ANY question about GitHub repositories, "
                "issues, files, or pull requests. No exceptions.\n"
                "2. NEVER answer from your training data — it is outdated and will be wrong.\n"
                "3. For multi-step questions (e.g. 'find the top repo then list its issues'), "
                "call tools one at a time, using each result to inform the next call.\n"
                "4. If a tool returns an error, report it clearly — do not fall back to guessing.\n"
                "5. Only give a final answer after you have tool results in hand."
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

                _call_counts.clear()
                # Stream graph steps so tool calls are visible in real time
                async for chunk in agent.astream(
                    {"messages": [{"role": "user", "content": query}]},
                    stream_mode="updates",
                ):
                    for node, update in chunk.items():
                        if node == "tools":
                            for msg in update.get("messages", []):
                                print(f"  [tool result: {getattr(msg, 'name', '?')}]", flush=True)
                        elif node == "agent":
                            for msg in update.get("messages", []):
                                tool_calls = getattr(msg, "tool_calls", [])
                                for tc in tool_calls:
                                    print(f"  [calling: {tc['name']}({tc['args']})]", flush=True)
                                if not tool_calls and getattr(msg, "content", ""):
                                    print(f"\n{msg.content}\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
