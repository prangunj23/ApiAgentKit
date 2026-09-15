import json

import httpx
from fastapi.testclient import TestClient

from agentkit.learning import REFLECT_PROMPT
from agentkit.testing import FakeLLM, git
from helpers import build_app, decide, github_factory, send, service1_spec, start

PR_SCRIPT = [
    {"tool_calls": [("write_file", {"path": "src/operation/models.py", "content": "# changed\n"})]},
    {"tool_calls": [("open_pull_request", {"title": "Change models", "body": "Tests pass."})]},
]


def test_plain_reply_is_streamed_and_saved(tmp_path, remotes):
    app = build_app(tmp_path, remotes, service1_spec(), llm=FakeLLM(["Hello from the agent."]))
    client = TestClient(app)
    conversation_id = start(client)

    events = send(client, conversation_id, "Hi there")
    assert [event["type"] for event in events] == ["message", "token", "message", "done"]

    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert detail["conversation"]["title"] == "Hi there"
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [("user", "Hi there"), ("assistant", "Hello from the agent.")]
    assert "chat with user · Summary of the conversation." in app.state.agent.memory.recent_path.read_text()


def test_tool_results_feed_the_next_model_call(tmp_path, remotes):
    llm = FakeLLM([{"tool_calls": [("read_file", {"path": "src/operation/models.py"})]}, "There are two models."])
    client = TestClient(build_app(tmp_path, remotes, service1_spec(), llm=llm))

    events = send(client, start(client), "What models exist?")
    result = next(event for event in events if event["type"] == "tool_result")
    assert "class NumericOpRequest" in result["content"]
    assert events[-1]["type"] == "done"

    tool_messages = [message for message in llm.calls[1]["messages"] if message["role"] == "tool"]
    assert "class NumericOpRequest" in tool_messages[0]["content"]
    assert "# Operating context" in llm.calls[0]["messages"][0]["content"]


def test_reasoning_is_sent_back_with_tool_calls(tmp_path, remotes):
    llm = FakeLLM([{"reasoning": "I should read models.py.", "tool_calls": [("read_file", {"path": "src/operation/models.py"})]}, "Two models."])
    client = TestClient(build_app(tmp_path, remotes, service1_spec(), llm=llm))

    send(client, start(client), "What models exist?")
    [assistant] = [message for message in llm.calls[1]["messages"] if message["role"] == "assistant"]
    assert assistant["reasoning_content"] == "I should read models.py."
    assert assistant["tool_calls"][0]["function"]["name"] == "read_file"


def test_bad_tool_calls_become_errors_the_model_sees(tmp_path, remotes):
    llm = FakeLLM([{"tool_calls": [("write_file", {"path": "agent/evil.py", "content": "x"}), ("no_such_tool", {})]}, "Sorry."])
    client = TestClient(build_app(tmp_path, remotes, service1_spec(), llm=llm))

    results = [event["content"] for event in send(client, start(client), "Break things") if event["type"] == "tool_result"]
    assert results[0].startswith("Error: Edits in ApiAgentService1 are limited to src/, tests/")
    assert results[1] == "Error: unknown tool 'no_such_tool'"


def test_denied_action_is_recorded_and_starts_a_reflection(tmp_path, remotes):
    proposal = json.dumps(
        {"proposals": [{"kind": "lesson", "title": "Explain before PRs", "body": "Summarize the change first.", "why": "PR denied."}]}
    )
    llm = FakeLLM([*PR_SCRIPT, "Understood."], responder=lambda system, user: proposal if system == REFLECT_PROMPT else "Summary.")
    app = build_app(tmp_path, remotes, service1_spec(), llm=llm, github_token="ghp_testtoken123")
    client = TestClient(app)
    conversation_id = start(client)

    events = send(client, conversation_id, "Change the models")
    assert events[-1]["type"] == "confirm_required"
    assert events[-1]["action"]["name"] == "open_pull_request"
    assert "models.py" in events[-1]["action"]["preview"]
    assert [c["id"] for c in client.get("/api/pending").json()] == [conversation_id]
    assert send(client, conversation_id, "hello?") == [{"type": "error", "message": "Approve or deny the pending action first."}]

    events = decide(client, conversation_id, False, "Explain the change first")
    assert events[0] == {"type": "decision", "approved": False, "name": "open_pull_request"}
    assert events[1]["content"] == "The user denied this action. Reason: Explain the change first"
    assert events[-1]["type"] == "done"

    agent = app.state.agent
    assert agent.store.list_feedback(conversation_id)[0]["source"] == "deny_reason"
    assert [p["title"] for p in client.get("/api/learnings", params={"status": "proposed"}).json()] == ["Explain before PRs"]
    assert client.get("/api/pending").json() == []
    assert agent.workspace.own.is_dirty()
    assert "agent/chat" not in git(tmp_path, "ls-remote", str(remotes["root"] / "acme/ApiAgentService1.git"))


def test_approved_pull_request_is_pushed_and_the_checkout_reset(tmp_path, remotes):
    requests: list[httpx.Request] = []

    def github(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path == "/repos/acme/ApiAgentService1/pulls":
            return httpx.Response(201, json={"html_url": "https://github.com/acme/ApiAgentService1/pull/7"})
        return httpx.Response(404)

    app = build_app(
        tmp_path, remotes, service1_spec(), llm=FakeLLM([*PR_SCRIPT, "Opened."]), http_factory=github_factory(github), github_token="ghp_testtoken123"
    )
    client = TestClient(app)
    conversation_id = start(client)
    send(client, conversation_id, "Change the models")

    events = decide(client, conversation_id, True)
    assert events[1]["content"] == "Opened pull request: https://github.com/acme/ApiAgentService1/pull/7"
    body = json.loads(requests[0].content)
    assert body["head"] == f"agent/chat-{conversation_id[:8]}" and body["base"] == "main"
    assert requests[0].headers["Authorization"] == "Bearer ghp_testtoken123"
    assert f"refs/heads/{body['head']}" in git(tmp_path, "ls-remote", str(remotes["root"] / "acme/ApiAgentService1.git"))

    agent = app.state.agent
    assert agent.workspace.own.branch() == "main" and not agent.workspace.own.is_dirty()
    assert agent.workspace.holder is None
    assert agent.store.list_pr_outcomes()[0]["state"] == "open"
    assert any(event["type"] == "pr_opened" for event in agent.store.list_events())
