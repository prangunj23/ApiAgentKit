from fastapi.testclient import TestClient

from agentkit.server import create_app
from agentkit.testing import FakeLLM, make_settings
from helpers import service1_spec


def test_foreign_hosts_and_origins_are_rejected(tmp_path):
    client = TestClient(create_app(service1_spec(), make_settings(tmp_path), llm=FakeLLM()))

    assert client.get("/api/info").status_code == 200
    assert client.get("/api/info", headers={"Host": "evil.example"}).status_code == 403
    assert client.post("/api/conversations", json={}, headers={"Origin": "https://evil.example"}).status_code == 403
    # Reads from other origins are harmless: the browser won't let that page see the response.
    assert client.get("/api/info", headers={"Origin": "https://evil.example"}).status_code == 200

    allowed = client.post("/api/conversations", json={}, headers={"Origin": "http://localhost:5173"})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    preflight = client.options(
        "/api/conversations", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
    )
    assert preflight.status_code == 200


def test_agent_routes_require_the_shared_token(tmp_path):
    client = TestClient(create_app(service1_spec(), make_settings(tmp_path / "a"), llm=FakeLLM()))
    requests = [
        ("/api/agent-messages", {"from_agent": "x", "thread_id": "t", "message": "hi"}),
        ("/api/agent-messages/reply", {"from_agent": "x", "thread_id": "t", "reply": "hi"}),
        ("/api/events/inbound", {"from_agent": "x", "summary": "s"}),
    ]
    for path, body in requests:
        assert client.post(path, json=body).status_code == 401
        assert client.post(path, json=body, headers={"X-Agent-Token": "wrong"}).status_code == 401

    unconfigured = TestClient(create_app(service1_spec(), make_settings(tmp_path / "b", shared_token=""), llm=FakeLLM()))
    response = unconfigured.post("/api/events/inbound", json={"from_agent": "x", "summary": "s"}, headers={"X-Agent-Token": "x"})
    assert response.status_code == 503


def test_research_files_cannot_escape_the_folder(tmp_path):
    client = TestClient(create_app(service1_spec(), make_settings(tmp_path), llm=FakeLLM()))
    assert client.get("/api/research/.codebase-map.json").status_code == 404
    assert client.get("/api/research/..%2Fagent.db").status_code == 404
