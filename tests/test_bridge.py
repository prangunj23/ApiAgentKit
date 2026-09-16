"""The bridge Claude Code uses to talk to an agent, and the MCP server around it."""

import json

import anyio
import pytest
from fastapi.testclient import TestClient

from agentkit.bridge import AgentBridge, BridgeError
from agentkit.testing import FakeLLM
from helpers import build_app, service1_spec


def bridge_for(tmp_path, remotes, llm) -> tuple[AgentBridge, TestClient]:
    app = build_app(tmp_path, remotes, service1_spec(), llm=llm)
    client = TestClient(app, base_url="http://127.0.0.1:9000")
    return AgentBridge(client, agent_id="service1"), client


def test_ask_starts_a_labelled_chat_and_returns_the_tools_and_reply(tmp_path, remotes):
    llm = FakeLLM([{"tool_calls": [("list_files", {"path": "src"})]}, "There are four modules.", "Yes, app.py."])
    bridge, client = bridge_for(tmp_path, remotes, llm)

    answer = bridge.ask("What's in src?")
    conversation_id = answer.splitlines()[0].removeprefix("conversation_id: ")
    assert f"UI: http://localhost:5173/#/agents/service1/chats/{conversation_id}" in answer
    assert '→ list_files({"path": "src"})' in answer
    assert "↳ src/operation/__init__.py" in answer
    assert answer.endswith("Agent's reply:\nThere are four modules.")
    assert client.get(f"/api/conversations/{conversation_id}").json()["conversation"]["title"] == "Claude Code: What's in src?"

    follow_up = bridge.ask("Is one the app?", conversation_id)
    assert follow_up.endswith("Yes, app.py.")
    assert f"- {conversation_id} · " in bridge.conversations()

    transcript = bridge.read(conversation_id)
    assert "**you:** What's in src?" in transcript and "**agent:** Yes, app.py." in transcript
    assert "  → list_files(" in transcript


def test_approvals_are_left_to_the_user(tmp_path, remotes):
    llm = FakeLLM([{"tool_calls": [("open_pull_request", {"title": "Nothing", "body": "x"})]}, "Opened."])
    bridge, client = bridge_for(tmp_path, remotes, llm)
    answer = bridge.ask("Open a PR")
    assert "WAITING FOR APPROVAL: the agent wants to run `open_pull_request`." in answer
    assert "Only the user can approve this, in the UI" in answer
    assert not hasattr(bridge, "approve")
    assert "Waiting for approval (1)" in bridge.status()

    conversation_id = answer.splitlines()[0].removeprefix("conversation_id: ")
    client.post(f"/api/conversations/{conversation_id}/confirm", json={"approve": False, "reason": "not now"})
    assert "**agent:** Opened." in bridge.read(conversation_id, wait=True)


def test_an_unreachable_agent_is_a_clear_error():
    import httpx

    bridge = AgentBridge(httpx.Client(base_url="http://127.0.0.1:1"))
    with pytest.raises(BridgeError, match="Is it running"):
        bridge.ask("hi")


def test_the_mcp_server_exposes_the_bridge(tmp_path, remotes):
    pytest.importorskip("mcp")
    from agentkit.mcp_server import build_server

    bridge, _ = bridge_for(tmp_path, remotes, FakeLLM(["Hello from service1."]))
    server = build_server(bridge, name="Operation", agent_id="service1")

    async def run():
        tools = {tool.name for tool in await server.list_tools()}
        result = await server.call_tool("ask_agent", {"message": "Say hello"})
        return tools, result

    tools, result = anyio.run(run)
    assert tools == {"ask_agent", "list_conversations", "read_conversation", "agent_status"}
    text = json.dumps(result.model_dump() if hasattr(result, "model_dump") else result, default=str)
    assert "Hello from service1." in text
