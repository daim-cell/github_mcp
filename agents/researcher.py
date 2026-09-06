import asyncio
import os
import sys
from collections.abc import AsyncGenerator

import uvicorn.config as _uvc
if not hasattr(_uvc, "LoopSetupType"):
    _uvc.LoopSetupType = _uvc.LoopFactoryType

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, Server
from acp_sdk.server.app import create_app
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from opentelemetry import trace
from pydantic import BaseModel
from tavily import TavilyClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import (
    _wrap_llm,
    validate_input,
    TokenCostCallbackHandler,
    _call_counts,
)
from agents.planner import PlannerOutput
from constants import AGENT_SYSTEM_PROMPT
from memory.shared_store import write_findings
from utils.mcp_session import mcp_tools


class ResearcherInput(BaseModel):
    brief: PlannerOutput


class ResearcherOutput(BaseModel):
    topic: str
    questions_answered: list[str]
    questions_unanswered: list[str]
    findings_stored: int


server = Server()

_base_llm = ChatOllama(model=os.getenv("BASE_MODEL", "qwen2.5:7b"), temperature=0)
_llm = _wrap_llm(_base_llm)
_classifier_llm = ChatOllama(model=os.getenv("CLASSIFIER_MODEL", "llama3.2:3b"), temperature=0)
_token_cb = TokenCostCallbackHandler()
_tracer = trace.get_tracer("researcher")


_MAX_TOOL_STEPS = 8      # recursion_limit passed to LangGraph
_QUESTION_TIMEOUT = 120  # seconds per question before giving up


async def _collect(agent, question: str) -> str:
    """Run the ReAct agent for one question with a hard step cap and time limit.

    Returns the final answer string, or empty string if the agent exhausted
    its step budget or timed out without producing a text answer.
    """
    final_answer = ""
    try:
        async with asyncio.timeout(_QUESTION_TIMEOUT):
            async for chunk in agent.astream(
                {"messages": [HumanMessage(content=question)]},
                stream_mode="updates",
                config={"callbacks": [_token_cb], "recursion_limit": _MAX_TOOL_STEPS * 2},
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
    except TimeoutError:
        print(f"  [timeout] question exceeded {_QUESTION_TIMEOUT}s — skipping", flush=True)
    except Exception as e:
        # GraphRecursionError and other LangGraph errors surface here
        err = str(e)
        if "recursion" in err.lower() or "graph" in err.lower():
            print(f"  [step limit] agent hit {_MAX_TOOL_STEPS} tool steps — stopping", flush=True)
        else:
            print(f"  [error] {err}", flush=True)
    return final_answer


def _make_tavily_tool() -> Tool:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
    def search(query: str) -> str:
        results = client.search(query, max_results=5)
        return "\n\n".join(r["content"] for r in results.get("results", []))
    return Tool(
        name="web_search",
        func=search,
        description="Search the web for current information on any topic.",
    )


@server.agent(
    name="researcher",
    description="Executes a research brief using GitHub MCP tools and web search.",
)
async def researcher_handler(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, None]:
    # Parse the PlannerOutput brief from the first message
    try:
        raw = input[0].parts[0].content
        brief = PlannerOutput.model_validate_json(raw)
    except Exception as e:
        yield f"Error parsing brief: {e}"
        return

    questions_answered: list[str] = []
    questions_unanswered: list[str] = []
    total_chunks = 0

    use_github = "github" in brief.required_sources
    use_web = "web" in brief.required_sources

    tavily_tool = _make_tavily_tool() if use_web else None

    async with mcp_tools(_tracer) as github_tools:
        for question in brief.key_questions:
            _call_counts.clear()

            # Guardrail — skip questions the classifier blocks
            ok, err = await validate_input(question, _classifier_llm, _tracer)
            if not ok:
                yield f"[skipped] {question} — {err}"
                questions_unanswered.append(question)
                continue

            # Select tools based on required_sources
            tools: list = []
            if use_github:
                tools.extend(github_tools)
            if use_web and tavily_tool:
                tools.append(tavily_tool)

            agent = create_react_agent(_llm, tools, prompt=AGENT_SYSTEM_PROMPT)

            yield f"[researching] {question}"

            answer = await _collect(agent, question)

            if answer.strip():
                chunks = write_findings(
                    topic=brief.topic,
                    question=question,
                    content=answer,
                )
                total_chunks += chunks
                questions_answered.append(question)
            else:
                yield f"[unanswered] Could not find an answer for: {question}"
                questions_unanswered.append(question)

    result = ResearcherOutput(
        topic=brief.topic,
        questions_answered=questions_answered,
        questions_unanswered=questions_unanswered,
        findings_stored=total_chunks,
    )
    yield result.model_dump_json()


if __name__ == "__main__":
    import uvicorn
    app = create_app(*server.agents)
    uvicorn.run(app, host="127.0.0.1", port=8002)
