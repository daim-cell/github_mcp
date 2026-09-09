"""
LangGraph StateGraph orchestrating the Planner → Researcher → Writer pipeline.

Start each agent service before running this:
    python -m agents.planner       # port 8001
    python -m agents.researcher    # port 8002
    python -m agents.writer        # port 8003

Then run:
    python supervisor.py
"""
import asyncio
import json
import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

load_dotenv()

# acp_sdk / uvicorn compat shim must be imported before any acp_sdk import
import uvicorn.config as _uvc
if not hasattr(_uvc, "LoopSetupType"):
    _uvc.LoopSetupType = _uvc.LoopFactoryType

from acp_sdk.client import Client
from acp_sdk.models import Message, MessageAwaitResume, MessagePart

from langchain_ollama import ChatOllama
from opentelemetry import trace

sys.path.insert(0, os.path.dirname(__file__))
from agent import validate_input, TokenCostCallbackHandler, _call_counts
from agents.planner import PlannerOutput
from agents.researcher import ResearcherOutput
from agents.writer import WriterOutput

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

PLANNER_URL = "http://127.0.0.1:8001"
RESEARCHER_URL = "http://127.0.0.1:8002"
WRITER_URL = "http://127.0.0.1:8003"
_HEADERS = {"Content-Type": "application/json"}

_MIN_ANSWERED = 1          # fewer than this → retry researcher
_MAX_RESEARCHER_RETRIES = 1

_classifier_llm = ChatOllama(model=os.getenv("CLASSIFIER_MODEL", "llama3.2:3b"), temperature=0)
_tracer = trace.get_tracer("supervisor")


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

class PipelineState(TypedDict):
    topic: str
    brief: PlannerOutput | None
    researcher_summary: ResearcherOutput | None
    final_document: WriterOutput | None
    retry_count: int
    error: str | None


# ──────────────────────────────────────────────
# Helper — extract last valid JSON from ACP output
# ──────────────────────────────────────────────

def _last_json(run) -> str:
    for msg in reversed(run.output):
        for part in reversed(msg.parts):
            try:
                json.loads(part.content or "")
                return part.content
            except (json.JSONDecodeError, TypeError):
                continue
    return ""


# ──────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────

async def node_validate_input(state: PipelineState) -> PipelineState:
    print(f"\n[supervisor] validate_input: {state['topic']!r}", flush=True)
    _call_counts.clear()
    ok, err = await validate_input(state["topic"], _classifier_llm, _tracer)
    if not ok:
        print(f"  [blocked] {err}", flush=True)
        return {**state, "error": err}
    return state


async def node_run_planner(state: PipelineState) -> PipelineState:
    print("\n[supervisor] run_planner", flush=True)

    async with Client(base_url=PLANNER_URL, headers=_HEADERS) as c:
        run = await c.run_sync(
            [Message(parts=[MessagePart(content=state["topic"])])],
            agent="planner",
        )

        # Planner may pause for human approval
        if run.status.value == "awaiting":
            for msg in run.output:
                for part in msg.parts:
                    if part.content:
                        print(f"\n  [planner brief]\n{part.content}", flush=True)
            print("\n  Type 'approve' or 'revise: <feedback>':", flush=True)
            reply = await asyncio.get_event_loop().run_in_executor(None, input, "  > ")
            run = await c.run_resume_sync(
                MessageAwaitResume(
                    message=Message(parts=[MessagePart(content=reply.strip())])
                ),
                run_id=run.run_id,
            )

    raw = _last_json(run)
    if not raw:
        return {**state, "error": "Planner produced no structured brief."}

    try:
        brief = PlannerOutput.model_validate_json(raw)
    except Exception as e:
        return {**state, "error": f"Planner output parse error: {e}"}

    print(f"  [brief approved] topic={brief.topic!r}, {len(brief.key_questions)} questions", flush=True)
    return {**state, "brief": brief}


async def node_run_researcher(state: PipelineState) -> PipelineState:
    retry = state["retry_count"]
    print(f"\n[supervisor] run_researcher (attempt {retry + 1})", flush=True)

    brief_json = state["brief"].model_dump_json()
    # On retry, append a note asking the researcher to dig deeper
    if retry > 0:
        data = json.loads(brief_json)
        data["_retry_note"] = (
            "Previous run answered fewer than the required questions. "
            "Please search more thoroughly or try alternative search terms."
        )
        brief_json = json.dumps(data)

    async with Client(base_url=RESEARCHER_URL, headers=_HEADERS) as c:
        run = await c.run_sync(
            [Message(parts=[MessagePart(content=brief_json)])],
            agent="researcher",
        )

    raw = _last_json(run)
    if not raw:
        # Still increment so we don't retry forever
        return {**state, "retry_count": retry + 1,
                "error": "Researcher produced no output."}

    try:
        summary = ResearcherOutput.model_validate_json(raw)
    except Exception as e:
        return {**state, "retry_count": retry + 1,
                "error": f"Researcher output parse error: {e}"}

    print(
        f"  [researcher done] answered={len(summary.questions_answered)}, "
        f"unanswered={len(summary.questions_unanswered)}, "
        f"chunks_stored={summary.findings_stored}",
        flush=True,
    )
    return {**state, "researcher_summary": summary, "retry_count": retry + 1}


async def node_run_writer(state: PipelineState) -> PipelineState:
    print("\n[supervisor] run_writer", flush=True)

    payload = json.dumps({
        "brief": json.loads(state["brief"].model_dump_json()),
        "researcher_summary": json.loads(state["researcher_summary"].model_dump_json()),
    })

    async with Client(base_url=WRITER_URL, headers=_HEADERS) as c:
        run = await c.run_sync(
            [Message(parts=[MessagePart(content=payload)])],
            agent="writer",
        )

    raw = _last_json(run)
    if not raw:
        return {**state, "error": "Writer produced no output."}

    try:
        doc = WriterOutput.model_validate_json(raw)
    except Exception as e:
        return {**state, "error": f"Writer output parse error: {e}"}

    print(f"  [writer done] validation_passed={doc.validation_passed}", flush=True)
    return {**state, "final_document": doc}


async def node_human_checkpoint(state: PipelineState) -> PipelineState:
    """Show the final document; let the user publish or discard."""
    doc = state["final_document"]
    if not doc:
        return state

    print(f"\n{'='*60}", flush=True)
    print("FINAL DOCUMENT", flush=True)
    print("="*60, flush=True)
    print(doc.document, flush=True)
    print(f"\n  sources_used: {doc.sources_used}", flush=True)
    print(f"  validation_passed: {doc.validation_passed}", flush=True)
    print(f"\n  Type 'publish' to accept or 'discard' to reject:", flush=True)

    reply = await asyncio.get_event_loop().run_in_executor(None, input, "  > ")
    reply = reply.strip().lower()

    _log_run(state, published=(reply == "publish"))

    if reply != "publish":
        print("  [discarded]", flush=True)
    else:
        print("  [published]", flush=True)

    return state


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def _log_run(state: PipelineState, *, published: bool) -> None:
    import datetime
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "topic": state["topic"],
        "questions_answered": (
            state["researcher_summary"].questions_answered
            if state.get("researcher_summary") else []
        ),
        "questions_unanswered": (
            state["researcher_summary"].questions_unanswered
            if state.get("researcher_summary") else []
        ),
        "findings_stored": (
            state["researcher_summary"].findings_stored
            if state.get("researcher_summary") else 0
        ),
        "validation_passed": (
            state["final_document"].validation_passed
            if state.get("final_document") else False
        ),
        "published": published,
        "error": state.get("error"),
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/pipeline_runs.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


# ──────────────────────────────────────────────
# Routing
# ──────────────────────────────────────────────

def route_after_validate(state: PipelineState) -> str:
    return "error" if state.get("error") else "run_planner"


def route_after_planner(state: PipelineState) -> str:
    return "error" if state.get("error") else "run_researcher"


def route_after_researcher(state: PipelineState) -> str:
    if state.get("error"):
        return "error"
    summary = state.get("researcher_summary")
    answered = len(summary.questions_answered) if summary else 0
    # Retry if too few answers AND we haven't hit the retry cap
    if answered < _MIN_ANSWERED and state["retry_count"] <= _MAX_RESEARCHER_RETRIES:
        print(
            f"  [supervisor] only {answered} question(s) answered "
            f"(need {_MIN_ANSWERED}) — retrying researcher",
            flush=True,
        )
        return "run_researcher"
    return "run_writer"


def route_after_writer(state: PipelineState) -> str:
    return "error" if state.get("error") else "human_checkpoint"


async def node_error(state: PipelineState) -> PipelineState:
    print(f"\n[supervisor] PIPELINE ERROR: {state.get('error')}", flush=True)
    _log_run(state, published=False)
    return state


# ──────────────────────────────────────────────
# Graph assembly
# ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("validate_input", node_validate_input)
    g.add_node("run_planner", node_run_planner)
    g.add_node("run_researcher", node_run_researcher)
    g.add_node("run_writer", node_run_writer)
    g.add_node("human_checkpoint", node_human_checkpoint)
    g.add_node("error", node_error)

    g.set_entry_point("validate_input")

    g.add_conditional_edges("validate_input", route_after_validate,
                            {"run_planner": "run_planner", "error": "error"})

    g.add_conditional_edges("run_planner", route_after_planner,
                            {"run_researcher": "run_researcher", "error": "error"})

    g.add_conditional_edges("run_researcher", route_after_researcher,
                            {"run_researcher": "run_researcher",
                             "run_writer": "run_writer",
                             "error": "error"})

    g.add_conditional_edges("run_writer", route_after_writer,
                            {"human_checkpoint": "human_checkpoint", "error": "error"})

    g.add_edge("human_checkpoint", END)
    g.add_edge("error", END)

    return g.compile()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

async def run(topic: str) -> None:
    graph = build_graph()
    initial: PipelineState = {
        "topic": topic,
        "brief": None,
        "researcher_summary": None,
        "final_document": None,
        "retry_count": 0,
        "error": None,
    }
    print(f"\n[supervisor] Starting pipeline for topic: {topic!r}", flush=True)
    await graph.ainvoke(initial)
    print(f"\n[supervisor] Pipeline complete.", flush=True)


if __name__ == "__main__":
    initial_topic = " ".join(sys.argv[1:]).strip()

    async def _loop() -> None:
        graph = build_graph()
        first = True
        while True:
            if first and initial_topic:
                topic = initial_topic
                first = False
            else:
                try:
                    topic = input("\nEnter a topic (or 'quit' to exit): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[supervisor] Exiting.")
                    break
                if not topic or topic.lower() in ("quit", "exit", "q"):
                    print("[supervisor] Exiting.")
                    break

            initial: PipelineState = {
                "topic": topic,
                "brief": None,
                "researcher_summary": None,
                "final_document": None,
                "retry_count": 0,
                "error": None,
            }
            print(f"\n[supervisor] Starting pipeline for topic: {topic!r}", flush=True)
            await graph.ainvoke(initial)
            print(f"\n[supervisor] Pipeline complete.", flush=True)

    asyncio.run(_loop())
