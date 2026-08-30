import asyncio
import json
import os
import sys
import time
from typing import Any, Optional
import tiktoken
from dotenv import load_dotenv
from opentelemetry import trace
import phoenix as px
from phoenix.otel import register
from pydantic import BaseModel, Field, create_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters, stdio_client
from constants import SUMMARY_LIMIT, INJECTION_PATTERNS, MAX_QUERY_LENGTH, ISSUE_QUERY_PATTERNS, ISSUE_RESPONSE_PATTERN, OUTPUT_FALLBACK, CLASSIFIER_SYSTEM_PROMPT, SAFETY_SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
load_dotenv()

# cl100k_base is close enough for token estimation across most models
_tokenizer = tiktoken.get_encoding("cl100k_base")

_COST_PER_1K_INPUT = float(os.getenv("COST_PER_1K_INPUT", "0.0"))
_COST_PER_1K_OUTPUT = float(os.getenv("COST_PER_1K_OUTPUT", "0.0"))
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "llama3.2:3b")
BASE_MODEL = os.getenv("BASE_MODEL", "qwen2.5:7b")


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


class TokenCostCallbackHandler(BaseCallbackHandler):
    """Captures real token counts from LLMResult and attaches them to the active OTel span."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        span = trace.get_current_span()
        if not span.is_recording():
            return

        prompt_tokens = 0
        completion_tokens = 0

        # Prefer usage_metadata on the AIMessage (Ollama, OpenAI via langchain-ollama)
        for gen_row in response.generations:
            for gen in gen_row:
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "usage_metadata", None) or {}
                if meta:
                    prompt_tokens += int(meta.get("input_tokens", 0))
                    completion_tokens += int(meta.get("output_tokens", 0))

        # Fallback: llm_output["token_usage"] for OpenAI-style responses
        if not prompt_tokens and not completion_tokens:
            llm_output = response.llm_output or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage") or llm_output
            prompt_tokens = int(usage.get("prompt_tokens", usage.get("prompt_eval_count", 0)))
            completion_tokens = int(usage.get("completion_tokens", usage.get("eval_count", 0)))

        total_tokens = prompt_tokens + completion_tokens

        cost_usd = (
            prompt_tokens * _COST_PER_1K_INPUT / 1000
            + completion_tokens * _COST_PER_1K_OUTPUT / 1000
        )

        span.set_attribute("llm.prompt_tokens", prompt_tokens)
        span.set_attribute("llm.completion_tokens", completion_tokens)
        span.set_attribute("llm.total_tokens", total_tokens)
        span.set_attribute("llm.cost_usd", round(cost_usd, 8))


def _setup_tracing() -> trace.Tracer:
    """Start a local Phoenix session and wire up an OTLP tracer pointing to it."""
    px.launch_app()
    register(project_name="github-mcp-agent", batch=False, verbose=False)
    return trace.get_tracer("github-mcp-agent")


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




def _make_langchain_tool(
    session: ClientSession, mcp_tool: Any, tracer: trace.Tracer
) -> StructuredTool:
    """Wrap a single MCP tool definition as a LangChain StructuredTool."""
    tool_name = mcp_tool.name
    args_schema = _schema_to_pydantic(mcp_tool.inputSchema or {})

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

        output = ""
        error: str | None = None
        t0 = time.perf_counter()
        input_text = json.dumps(kwargs, default=str)

        with tracer.start_as_current_span(tool_name) as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.inputs", input_text)
            span.set_attribute("tool.input_tokens", _count_tokens(input_text))
            try:
                result = await session.call_tool(tool_name, arguments=kwargs or None)
                if result.isError:
                    first = result.content[0] if result.content else None
                    output = f"Error: {getattr(first, 'text', str(first))}"
                    error = output
                else:
                    output = "\n".join(
                        getattr(block, "text", str(block)) for block in result.content
                    )
            except Exception as exc:
                output = f"Error: {exc}"
                error = str(exc)
                raise
            finally:
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                summary = output[:SUMMARY_LIMIT] + "..." if len(output) > SUMMARY_LIMIT else output
                output_tokens = _count_tokens(output)
                span.set_attribute("tool.output_summary", summary)
                span.set_attribute("tool.output_tokens", output_tokens)
                span.set_attribute("tool.total_tokens", _count_tokens(input_text) + output_tokens)
                span.set_attribute("tool.latency_ms", latency_ms)
                span.set_attribute("tool.success", error is None)
                if error:
                    span.set_attribute("tool.error", error)
                    span.set_status(trace.StatusCode.ERROR, error)
                else:
                    span.set_status(trace.StatusCode.OK)

        return output

    return StructuredTool.from_function(
        coroutine=_call,
        name=tool_name,
        description=mcp_tool.description or "",
        args_schema=args_schema,
    )


async def _run_agent_and_collect(
    agent: Any,
    messages: list,
    token_cb: TokenCostCallbackHandler,
) -> str:
    """Stream the agent and return the final answer string.

    Prints tool-call status lines in real time; buffers and returns the final
    answer text without printing so the caller can validate it first.
    """
    final_answer = ""
    async for chunk in agent.astream(
        {"messages": messages},
        stream_mode="updates",
        config={"callbacks": [token_cb]},
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
                    content = getattr(msg, "content", "") or ""
                    if not tool_calls and content:
                        final_answer = content
    return final_answer


async def plan_query(
    query: str,
    base_llm: ChatOllama,
    tracer: trace.Tracer,
) -> SystemMessage:
    """Produce an explicit numbered execution plan for the query.

    Opens a 'plan' child span within the active agent.query span.
    The LLM has no tool access here — pure reasoning only.
    Returns a SystemMessage that is prepended to the react agent's message list.
    """
    with tracer.start_as_current_span("plan") as span:
        span.set_attribute("plan.query", query)
        t0 = time.perf_counter()
        try:
            result = await base_llm.ainvoke([
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ])
            plan_text = (result.content or "").strip()
        except Exception as exc:
            plan_text = "Step 1: Answer the user's GitHub question using available tools."
            span.add_event("plan_llm_error", {"error": str(exc)})

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        span.set_attribute("plan.text", plan_text)
        span.set_attribute("plan.tokens", _count_tokens(plan_text))
        span.set_attribute("plan.latency_ms", latency_ms)
        span.set_status(trace.StatusCode.OK)

    print(f"  [plan]\n{plan_text}\n", flush=True)
    return SystemMessage(content=f"Execution plan:\n{plan_text}")


async def validate_input(
    query: str,
    classifier_llm: ChatOllama,
    tracer: trace.Tracer,
) -> tuple[bool, str]:
    """Gate the query before the agent graph runs.

    Returns (True, "") if the query passes all checks.
    Returns (False, user-facing error message) on the first failure.
    """
    with tracer.start_as_current_span("input.validation") as span:
        # Check 1 — length (O(1), run first)
        if len(query) > MAX_QUERY_LENGTH:
            reason = f"Query exceeds {MAX_QUERY_LENGTH} characters ({len(query)} submitted)"
            span.set_attribute("validation.check", "length")
            span.set_attribute("validation.passed", False)
            span.set_attribute("validation.rejection_reason", reason)
            span.set_status(trace.StatusCode.ERROR, "length_exceeded")
            return (False, f"Query too long ({len(query)} chars). Maximum is {MAX_QUERY_LENGTH} characters.")

        # Check 2 — regex injection patterns
        for pattern in INJECTION_PATTERNS:
            if pattern.search(query):
                reason = f"Matched injection pattern: {pattern.pattern}"
                span.set_attribute("validation.check", "injection_regex")
                span.set_attribute("validation.passed", False)
                span.set_attribute("validation.rejection_reason", reason)
                span.set_status(trace.StatusCode.ERROR, "injection_detected")
                return (False, "Query rejected: potential prompt injection detected.")

        # Check 3 — single combined LLM classifier (injection + scope)
        try:
            result = await classifier_llm.ainvoke([
                SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
                HumanMessage(content=f"User query: {query}\n\nResponse:"),
            ])
            verdict = (result.content or "").strip().lower()
            if verdict.startswith("injection"):
                reason = "LLM classifier detected prompt injection"
                span.set_attribute("validation.check", "injection_llm")
                span.set_attribute("validation.passed", False)
                span.set_attribute("validation.rejection_reason", reason)
                span.set_status(trace.StatusCode.ERROR, "injection_detected")
                return (False, "Query rejected: potential prompt injection detected.")
            if not verdict.startswith("allowed"):
                reason = f"LLM classifier returned '{verdict}' — query is out of scope"
                span.set_attribute("validation.check", "scope")
                span.set_attribute("validation.passed", False)
                span.set_attribute("validation.rejection_reason", reason)
                span.set_status(trace.StatusCode.ERROR, "out_of_scope")
                return (False, "Query rejected: this assistant only handles GitHub-related tasks.")
        except Exception as exc:
            # Classifier LLM unavailable — fail open so legitimate queries aren't blocked
            span.add_event("classifier_llm_error", {"error": str(exc)})

        span.set_attribute("validation.check", "all_passed")
        span.set_attribute("validation.passed", True)
        span.set_attribute("validation.rejection_reason", "")
        span.set_status(trace.StatusCode.OK)
        return (True, "")


async def validate_output(
    response: str,
    query: str,
    classifier_llm: ChatOllama,
    agent: Any,
    tracer: trace.Tracer,
    token_cb: TokenCostCallbackHandler,
) -> tuple[bool, str]:
    """Validate the agent's final answer before returning it to the user.

    Returns (True, validated_response) on success.
    Returns (False, fallback_message) if both the original and retry fail.
    """
    with tracer.start_as_current_span("output.validation") as span:
        if not response:
            span.set_attribute("validation.passed", False)
            span.set_attribute("validation.retry_fired", False)
            span.set_attribute("validation.rejection_reason", "Empty response from agent")
            span.set_status(trace.StatusCode.ERROR, "empty_response")
            return (False, OUTPUT_FALLBACK)

        rejection_reason = ""
        specific_issue = "a complete, accurate answer"

        # Check 1 — schema heuristic for issue-list queries
        needs_structured = any(p.search(query) for p in ISSUE_QUERY_PATTERNS)
        if needs_structured and not ISSUE_RESPONSE_PATTERN.search(response):
            rejection_reason = "Issue-list query returned no issue numbers"
            specific_issue = "a structured list including issue numbers (e.g., #123), titles, and URLs"

        # Check 2 — safety classifier (skip if already flagged by schema check)
        if not rejection_reason:
            try:
                result = await classifier_llm.ainvoke([
                    SystemMessage(content=SAFETY_SYSTEM_PROMPT),
                    HumanMessage(content=f"Response to review: {response}\n\nAssessment:"),
                ])
                verdict = (result.content or "").strip().lower()
                if verdict.startswith("unsafe"):
                    rejection_reason = "Safety classifier flagged the response"
                    specific_issue = "a safe, factual response that avoids harmful or misleading content"
            except Exception as exc:
                # Safety classifier unavailable — fail open
                span.add_event("classifier_llm_error", {"error": str(exc)})

        # All checks passed — return as-is
        if not rejection_reason:
            span.set_attribute("validation.passed", True)
            span.set_attribute("validation.retry_fired", False)
            span.set_attribute("validation.rejection_reason", "")
            span.set_status(trace.StatusCode.OK)
            return (True, response)

        # Retry once with an explicit correction instruction
        span.set_attribute("validation.retry_fired", True)
        print(f"  [output validation failed: {rejection_reason} — retrying]", flush=True)

        _call_counts.clear()
        retry_query = (
            f"{query}\n\n"
            f"[System: Your previous response was rejected. "
            f"Please provide {specific_issue}. Try again with a complete answer.]"
        )
        retry_response = await _run_agent_and_collect(
            agent, [{"role": "user", "content": retry_query}], token_cb
        )

        # Re-run schema + safety checks on the retry response
        retry_ok = True
        if needs_structured and not ISSUE_RESPONSE_PATTERN.search(retry_response):
            retry_ok = False
        if retry_ok and retry_response:
            try:
                result = await classifier_llm.ainvoke([
                    SystemMessage(content=SAFETY_SYSTEM_PROMPT),
                    HumanMessage(content=f"Response to review: {retry_response}\n\nAssessment:"),
                ])
                if (result.content or "").strip().lower().startswith("unsafe"):
                    retry_ok = False
            except Exception:
                pass  # fail open on classifier error

        if retry_ok and retry_response:
            span.set_attribute("validation.passed", True)
            span.set_attribute("validation.rejection_reason", rejection_reason)
            span.set_status(trace.StatusCode.OK)
            return (True, retry_response)

        span.set_attribute("validation.passed", False)
        span.set_attribute("validation.rejection_reason", rejection_reason)
        span.set_status(trace.StatusCode.ERROR, rejection_reason)
        return (False, OUTPUT_FALLBACK)


async def main() -> None:
    tracer = _setup_tracing()

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tool_list = await session.list_tools()
            tools = [_make_langchain_tool(session, t, tracer) for t in tool_list.tools]

            print(f"[agent] Connected. {len(tools)} tool(s) available.", flush=True)
            for t in tools:
                print(f"  • {t.name}: {t.description}", flush=True)

            _base_llm = ChatOllama(model=BASE_MODEL, temperature=0)
            llm = _wrap_llm(_base_llm)
            classifier_llm = ChatOllama(model=CLASSIFIER_MODEL, temperature=0)

            token_cb = TokenCostCallbackHandler()
            agent = create_react_agent(llm, tools, prompt=AGENT_SYSTEM_PROMPT)

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

                ok, err = await validate_input(query, classifier_llm, tracer)
                if not ok:
                    print(f"[rejected] {err}", flush=True)
                    continue

                with tracer.start_as_current_span("agent.query") as query_span:
                    query_span.set_attribute("query.text", query)
                    query_span.set_attribute("query.input_tokens", _count_tokens(query))

                    plan_msg = await plan_query(query, _base_llm, tracer)
                    response = await _run_agent_and_collect(
                        agent, [plan_msg, {"role": "user", "content": query}], token_cb
                    )

                    ok, final = await validate_output(
                        response, query, classifier_llm, agent, tracer, token_cb
                    )
                    print(f"\n{final}\n", flush=True)

                    query_span.set_status(trace.StatusCode.OK)


if __name__ == "__main__":
    asyncio.run(main())
