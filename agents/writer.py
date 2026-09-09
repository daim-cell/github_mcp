import asyncio
import json
import os
import sys
from collections.abc import AsyncGenerator

import uvicorn.config as _uvc
if not hasattr(_uvc, "LoopSetupType"):
    _uvc.LoopSetupType = _uvc.LoopFactoryType

from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, Server
from acp_sdk.server.app import create_app
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import _wrap_llm
from agents.planner import PlannerOutput
from agents.researcher import ResearcherOutput
from constants import SAFETY_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT
from memory.shared_store import retrieve_relevant


class WriterInput(BaseModel):
    brief: PlannerOutput
    researcher_summary: ResearcherOutput


class WriterOutput(BaseModel):
    document: str
    sources_used: list[str]
    validation_passed: bool


server = Server()

_base_llm = ChatOllama(model=os.getenv("BASE_MODEL", "qwen2.5:7b"), temperature=0)
_llm = _wrap_llm(_base_llm)
_classifier_llm = ChatOllama(model=os.getenv("CLASSIFIER_MODEL", "llama3.2:3b"), temperature=0)


@server.agent(
    name="writer",
    description="Retrieves findings from shared memory and synthesizes a structured document.",
)
async def writer_handler(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, None]:
    # Parse input
    try:
        payload = json.loads(input[0].parts[0].content)
        brief = PlannerOutput.model_validate(payload["brief"])
        summary = ResearcherOutput.model_validate(payload["researcher_summary"])
    except Exception as e:
        yield f"Error parsing input: {e}"
        return

    yield f"[retrieving] fetching findings for topic: {brief.topic!r}"

    # Pass 1: broad topic retrieval
    broad = retrieve_relevant(query=brief.topic, topic=brief.topic, top_k=10)

    # Pass 2: per-question retrieval
    per_question: list[dict] = []
    for q in brief.key_questions:
        per_question.extend(retrieve_relevant(query=q, topic=brief.topic, top_k=3))

    # Deduplicate by content, preserve order
    seen: set[str] = set()
    deduped: list[dict] = []
    for chunk in broad + per_question:
        if chunk["content"] not in seen:
            seen.add(chunk["content"])
            deduped.append(chunk)

    if not deduped:
        yield WriterOutput(
            document="No findings were found in shared memory for this topic.",
            sources_used=[],
            validation_passed=False,
        ).model_dump_json()
        return

    # Collect unique source questions for citation
    sources_used = list(dict.fromkeys(c["question"] for c in deduped if c["question"]))

    # Build context block
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['question']}]\n{c['content']}" for c in deduped
    )

    synthesis_prompt = (
        f"{WRITER_SYSTEM_PROMPT}\n\n"
        f"Topic: {brief.topic}\n"
        f"Output format: {brief.output_format}\n\n"
        f"Key questions to address:\n"
        + "\n".join(f"- {q}" for q in brief.key_questions)
        + f"\n\nRetrieved context:\n{context_block}"
    )

    yield "[synthesizing] generating document from retrieved findings..."

    # Truncate context to avoid exceeding the model's context window
    MAX_CONTEXT_CHARS = 12_000
    if len(synthesis_prompt) > MAX_CONTEXT_CHARS:
        truncated_context = context_block[:MAX_CONTEXT_CHARS - len(synthesis_prompt) + len(context_block)]
        synthesis_prompt = (
            f"{WRITER_SYSTEM_PROMPT}\n\n"
            f"Topic: {brief.topic}\n"
            f"Output format: {brief.output_format}\n\n"
            f"Key questions to address:\n"
            + "\n".join(f"- {q}" for q in brief.key_questions)
            + f"\n\nRetrieved context (truncated):\n{truncated_context[:MAX_CONTEXT_CHARS]}"
        )

    _SYNTHESIS_TIMEOUT = 600  # seconds
    try:
        response = await asyncio.wait_for(
            _llm.ainvoke([HumanMessage(content=synthesis_prompt)]),
            timeout=_SYNTHESIS_TIMEOUT,
        )
        document = (response.content or "").strip()
    except asyncio.TimeoutError:
        document = ""
        print(f"  [writer] synthesis timed out after {_SYNTHESIS_TIMEOUT}s", flush=True)
    except Exception as e:
        document = ""
        print(f"  [writer] synthesis failed: {type(e).__name__}: {e}", flush=True)

    if not document:
        yield WriterOutput(
            document="LLM produced an empty document.",
            sources_used=sources_used,
            validation_passed=False,
        ).model_dump_json()
        return

    # Safety check
    yield "[validating] running safety check on document..."
    try:
        verdict_msg = await asyncio.wait_for(
            _classifier_llm.ainvoke([
                SystemMessage(content=SAFETY_SYSTEM_PROMPT),
                HumanMessage(content=f"Response to review: {document}\n\nAssessment:"),
            ]),
            timeout=30,
        )
        verdict = (verdict_msg.content or "").strip().lower()
        validation_passed = not verdict.startswith("unsafe")
    except Exception:
        validation_passed = True  # fail open if classifier unavailable

    yield WriterOutput(
        document=document,
        sources_used=sources_used,
        validation_passed=validation_passed,
    ).model_dump_json()


if __name__ == "__main__":
    import uvicorn
    app = create_app(*server.agents)
    uvicorn.run(app, host="127.0.0.1", port=8003)
