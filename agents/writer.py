from collections.abc import AsyncGenerator

import uvicorn.config as _uvc
if not hasattr(_uvc, "LoopSetupType"):
    _uvc.LoopSetupType = _uvc.LoopFactoryType

from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, Server
from acp_sdk.server.app import create_app
from pydantic import BaseModel

from planner import PlannerOutput
from researcher import ResearcherOutput


class WriterInput(BaseModel):
    brief: PlannerOutput
    researcher_summary: ResearcherOutput


class WriterOutput(BaseModel):
    document: str
    sources_used: list[str]
    validation_passed: bool


server = Server()

_STUB = WriterOutput(
    document="Stub document content.",
    sources_used=[],
    validation_passed=True,
)


@server.agent(
    name="writer",
    description="Retrieves findings from shared memory and synthesizes a structured document.",
)
async def writer_handler(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, None]:
    """Writer stub — returns hardcoded WriterOutput JSON."""
    yield _STUB.model_dump_json()


if __name__ == "__main__":
    import uvicorn
    app = create_app(*server.agents)
    uvicorn.run(app, host="127.0.0.1", port=8003)
