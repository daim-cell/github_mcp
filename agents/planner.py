import json
import os
from collections.abc import AsyncGenerator

import uvicorn.config as _uvc
if not hasattr(_uvc, "LoopSetupType"):
    _uvc.LoopSetupType = _uvc.LoopFactoryType

from acp_sdk.models import Message, MessageAwaitRequest, MessagePart
from acp_sdk.server import Context, RunYield, Server
from acp_sdk.server.app import create_app
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from constants import BRIEF_SYSTEM_PROMPT


class PlannerInput(BaseModel):
    topic: str
    context: str = ""


class PlannerOutput(BaseModel):
    topic: str
    key_questions: list[str]
    required_sources: list[str]
    output_format: str
    approved: bool


MAX_REVISIONS = 3

server = Server()

_llm = ChatOllama(model=os.getenv("BASE_MODEL", "qwen2.5:7b"), temperature=0)


@server.agent(
    name="planner",
    description="Breaks a user topic into a structured research brief.",
)
async def planner_handler(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, None]:
    topic = input[0].parts[0].content if input else "unknown topic"
    extra_context = ""

    for attempt in range(MAX_REVISIONS + 1):
        prompt = f"Topic: {topic}"
        if extra_context:
            prompt += f"\n\nRevision feedback: {extra_context}"

        response = await _llm.ainvoke([
            SystemMessage(content=BRIEF_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        try:
            data = json.loads(response.content)
            data["approved"] = False
            brief = PlannerOutput(**data)
        except Exception as e:
            if attempt == MAX_REVISIONS:
                yield f"Failed to generate a valid brief after {MAX_REVISIONS} attempts: {e}"
                return
            extra_context = f"Previous output was not valid JSON: {e}. Fix it."
            continue

        brief_text = brief.model_dump_json(indent=2)
        yield (
            f"Research brief (attempt {attempt + 1}):\n{brief_text}\n\n"
            "Reply 'approve' to proceed, or 'revise: <feedback>' to request changes."
        )

        if attempt == MAX_REVISIONS:
            brief.approved = True
            yield brief.model_dump_json()
            return

        resume = yield MessageAwaitRequest(
            message=Message(parts=[MessagePart(content="Awaiting your approval...")])
        )

        reply = resume.message.parts[0].content.strip().lower()

        if reply == "approve":
            brief.approved = True
            yield brief.model_dump_json()
            return
        elif reply.startswith("revise:"):
            extra_context = reply[len("revise:"):].strip()
        else:
            brief.approved = True
            yield brief.model_dump_json()
            return


if __name__ == "__main__":
    import uvicorn
    app = create_app(*server.agents)
    uvicorn.run(app, host="127.0.0.1", port=8001)
