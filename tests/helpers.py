"""Fixture data shared by the tests: two small repos shaped like ApiAgentService1 and ApiAgentService2."""

import json
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from agentkit import AgentSpec, RepoRef
from agentkit.agent import Agent
from agentkit.codemap import DESCRIBE_PROMPT
from agentkit.registry import HttpFactory, default_http_factory
from agentkit.server import create_app
from agentkit.testing import FakeLLM, http_factory_for, make_settings, parse_sse
from agentkit.tools.email import send_email

SERVICE1 = RepoRef("acme/ApiAgentService1")
SERVICE2 = RepoRef("acme/ApiAgentService2")

OPERATION_APP = '''from fastapi import APIRouter, FastAPI

from operation.models import NumericOpRequest, NumericOpResponse

app = FastAPI(title="operation", version="1.0.0")

v1 = APIRouter(prefix="/v1/operation", tags=["operation"])


@v1.post("/numeric_op", response_model=NumericOpResponse)
def numeric_op(request: NumericOpRequest) -> NumericOpResponse:
    return NumericOpResponse(result=request.a - request.b)


app.include_router(v1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
'''

OPERATION_MODELS = '''from pydantic import BaseModel


class NumericOpRequest(BaseModel):
    a: float
    b: float


class NumericOpResponse(BaseModel):
    result: float
'''

OPERATION_CLIENT = '''import httpx

from operation.models import NumericOpRequest, NumericOpResponse


class OperationClient:
    def __init__(self, base_url: str) -> None:
        self._http = httpx.Client(base_url=base_url)

    def numeric_op(self, a: float, b: float) -> NumericOpResponse:
        response = self._http.post("/v1/operation/numeric_op", json=NumericOpRequest(a=a, b=b).model_dump())
        return NumericOpResponse.model_validate(response.json())

    def health(self) -> bool:
        return self._http.get("/health").is_success

    def close(self) -> None:
        self._http.close()
'''

OPERATION_INIT = '''from operation.client import OperationClient
from operation.models import NumericOpRequest, NumericOpResponse

__all__ = ["NumericOpRequest", "NumericOpResponse", "OperationClient"]
'''

OPERATION_FILES = {
    "pyproject.toml": '''[project]
name = "operation"
version = "1.0.0"
description = "Subtracts two numbers"
requires-python = ">=3.13"
dependencies = ["fastapi>=0.115", "httpx>=0.27"]

[tool.hatch.build.targets.wheel]
packages = ["src/operation"]
''',
    "README.md": "# operation\n\n```sh\nuv sync\nuv run uvicorn operation.app:app --port 8001\n```\n",
    ".github/workflows/tests.yml": "name: Tests\n\non:\n  push:\n    branches: [main]\n  pull_request:\n  workflow_dispatch:\n\njobs: {}\n",
    "src/operation/__init__.py": OPERATION_INIT,
    "src/operation/app.py": OPERATION_APP,
    "src/operation/models.py": OPERATION_MODELS,
    "src/operation/client.py": OPERATION_CLIENT,
    "tests/test_app.py": "def test_ok():\n    assert True\n",
}

CONSUMER_CONFIG = 'import os\n\nOPERATION_BASE_URL = os.environ.get("OPERATION_BASE_URL", "http://localhost:8001")\n'

CONSUMER_FILES = {
    "pyproject.toml": '''[project]
name = "consumer"
version = "0.1.0"
description = "Calls the operation service"
requires-python = ">=3.13"
dependencies = ["fastapi>=0.115", "operation"]

[tool.uv.sources]
operation = { path = "../ApiAgentService1", editable = true }

[tool.hatch.build.targets.wheel]
packages = ["src/consumer"]
''',
    "src/consumer/__init__.py": "",
    "src/consumer/config.py": CONSUMER_CONFIG,
    "src/consumer/app.py": '''from consumer.config import OPERATION_BASE_URL
from operation import OperationClient

client = OperationClient(OPERATION_BASE_URL)


def compute(a: float, b: float) -> float:
    return client.numeric_op(a, b).result


def healthy() -> bool:
    return client.health()
''',
    "tests/test_app.py": "def test_ok():\n    assert True\n",
}


def service1_spec(**overrides: Any) -> AgentSpec:
    values: dict[str, Any] = {
        "id": "service1",
        "name": "Operation",
        "repo": SERVICE1,
        "reads": [SERVICE2],
        "system_prompt": "You maintain the operation service.",
    }
    return AgentSpec(**(values | overrides))


def service2_spec(**overrides: Any) -> AgentSpec:
    values: dict[str, Any] = {
        "id": "service2",
        "name": "Consumer",
        "repo": SERVICE2,
        "reads": [SERVICE1],
        "system_prompt": "You maintain the consumer service.",
        "tools": [send_email],
        "features": {"emails"},
    }
    return AgentSpec(**(values | overrides))


def build_agent(
    tmp_path: Path,
    remotes: dict[str, Path],
    spec: AgentSpec,
    *,
    llm: FakeLLM | None = None,
    clone: bool = True,
    http_factory: HttpFactory = default_http_factory,
    **settings: Any,
) -> Agent:
    agent = Agent(
        spec,
        make_settings(tmp_path / spec.id, git_base_url=str(remotes["root"]), **settings),
        llm=llm or FakeLLM(),
        http_factory=http_factory,
    )
    if clone:
        agent.workspace.ensure_cloned()
    return agent


def build_app(
    tmp_path: Path,
    remotes: dict[str, Path],
    spec: AgentSpec,
    *,
    llm: FakeLLM | None = None,
    clone: bool = True,
    http_factory: HttpFactory = default_http_factory,
    **settings: Any,
):
    app = create_app(
        spec,
        make_settings(tmp_path / spec.id, git_base_url=str(remotes["root"]), **settings),
        llm=llm or FakeLLM(),
        http_factory=http_factory,
    )
    if clone:
        app.state.agent.workspace.ensure_cloned()
    return app


def describer(system: str, user: str) -> str:
    """Answers codebase-map requests with a purpose for each requested file; summaries otherwise."""
    if system != DESCRIBE_PROMPT:
        return "Summary of the conversation."
    section = user.split("# Describe these files\n", 1)[1].split("\n# Architecture summary requested", 1)[0]
    paths = [line for line in section.splitlines() if line and line != "(none)"]
    return json.dumps({"files": {path: f"Purpose of {path}" for path in paths}, "architecture": "Requests flow through FastAPI."})


def github_factory(handler) -> HttpFactory:
    return http_factory_for({}, fallback=lambda url: httpx.Client(base_url=url, transport=httpx.MockTransport(handler)))


def start(client: TestClient) -> str:
    return client.post("/api/conversations", json={}).json()["id"]


def send(client: TestClient, conversation_id: str, text: str) -> list[dict[str, Any]]:
    response = client.post(f"/api/conversations/{conversation_id}/messages", json={"message": text})
    assert response.status_code == 200, response.text
    return parse_sse(response.text)


def decide(client: TestClient, conversation_id: str, approve: bool, reason: str = "") -> list[dict[str, Any]]:
    response = client.post(f"/api/conversations/{conversation_id}/confirm", json={"approve": approve, "reason": reason})
    assert response.status_code == 200, response.text
    return parse_sse(response.text)
