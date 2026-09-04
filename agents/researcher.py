from collections.abc import AsyncGenerator

import uvicorn.config as _uvc
if not hasattr(_uvc, "LoopSetupType"):
    _uvc.LoopSetupType = _uvc.LoopFactoryType

from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, Server
from acp_sdk.server.app import create_app
from pydantic import BaseModel

from planner import PlannerOutput

# Guardrail functions imported from agent.py — not called in stub, import verified here.
# Full implementation will call: validate_input(question, classifier_llm, tracer)
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import validate_input, validate_output


class ResearcherInput(BaseModel):
    brief: PlannerOutput


class ResearcherOutput(BaseModel):
    topic: str
    questions_answered: list[str]
    questions_unanswered: list[str]
    findings_stored: int


server = Server()

_STUB = ResearcherOutput(
    topic="stub topic",
    questions_answered=["What is the stub question?"],
    questions_unanswered=[],
    findings_stored=0,
)


@server.agent(
    name="researcher",
    description="Executes a research brief using GitHub MCP tools and web search.",
)
async def researcher_handler(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, None]:
    """Researcher stub — returns hardcoded ResearcherOutput JSON."""
    yield _STUB.model_dump_json()


if __name__ == "__main__":
    import uvicorn
    app = create_app(*server.agents)
    uvicorn.run(app, host="127.0.0.1", port=8002)
