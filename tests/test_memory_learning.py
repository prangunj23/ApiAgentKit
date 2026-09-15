import json

import httpx
from fastapi.testclient import TestClient

from agentkit.learning import REFLECT_PROMPT
from agentkit.testing import FakeLLM
from helpers import build_agent, build_app, github_factory, send, service1_spec, start


def test_conversation_summaries_replace_their_own_line(tmp_path, remotes):
    memory = build_agent(tmp_path, remotes, service1_spec(), clone=False).memory
    memory.record_conversation("c1", "first")
    memory.record_event("PR opened")
    memory.record_conversation("c1", "second")
    lines = memory.recent()
    assert len(lines) == 2 and "second" in lines[0] and "PR opened" in lines[1]

    for index in range(50):
        memory.record_event(f"event {index}")
    assert len(memory.recent()) == 40


def test_system_prompt_includes_memory_and_only_active_lessons(tmp_path, remotes):
    agent = build_agent(tmp_path, remotes, service1_spec(), clone=False)
    agent.memory.remember("Service1 returns a - b")
    agent.store.add_learning(kind="lesson", title="Run tests first", body="Always run tests before a PR.", status="active")
    agent.store.add_learning(kind="lesson", title="Unapproved", body="Should not appear.")
    agent.store.add_learning(kind="skill", title="Assess a contract change", body="When to use: upstream API changed\n\n1. Diff", status="active")

    prompt = agent.system_prompt(0)
    assert "Service1 returns a - b" in prompt
    assert "**Run tests first**: Always run tests before a PR." in prompt
    assert "Should not appear" not in prompt
    assert "- Assess a contract change: When to use: upstream API changed" in prompt
    assert "isn't cloned yet" in prompt

    agent.memory.remember("x" * 50_000)
    assert len(agent.system_prompt(0)) < 20_000


def test_feedback_reflection_approval_and_mirroring(tmp_path, remotes):
    proposals = json.dumps(
        {
            "proposals": [
                {"kind": "lesson", "title": "Cite files", "body": "Name the files you read.", "why": "User asked for sources."},
                {"kind": "skill", "title": "Trace a request", "body": "When to use: explaining flow\n\n1. Read app.py", "why": "Worked."},
            ]
        }
    )
    llm = FakeLLM(["The endpoint subtracts."], responder=lambda system, user: proposals if system == REFLECT_PROMPT else "Summary.")
    app = build_app(tmp_path, remotes, service1_spec(), llm=llm, clone=False)
    client = TestClient(app)
    events = send(client, start(client), "What does numeric_op do?")
    message_id = next(e["message"]["id"] for e in events if e["type"] == "message" and e["message"]["role"] == "assistant")

    assert client.post(f"/api/messages/{message_id}/feedback", json={"rating": -1, "comment": "Say which file"}).status_code == 200
    client.post(f"/api/messages/{message_id}/feedback", json={"rating": -1})  # same trigger: no second reflection
    proposed = client.get("/api/learnings", params={"status": "proposed"}).json()
    assert sorted(item["title"] for item in proposed) == ["Cite files", "Trace a request"]
    lesson = next(item for item in proposed if item["kind"] == "lesson")
    skill = next(item for item in proposed if item["kind"] == "skill")

    approved = client.post(f"/api/learnings/{lesson['id']}", json={"action": "approve", "body": "Name every file you read."}).json()
    assert approved["status"] == "active"
    client.post(f"/api/learnings/{skill['id']}", json={"action": "approve"})
    agent = app.state.agent
    assert "Name every file you read." in agent.memory.lessons_path.read_text()
    assert (agent.learning.skills_dir / "trace-a-request.md").exists()
    assert "Name every file you read." in agent.system_prompt(0)

    client.post(f"/api/learnings/{skill['id']}", json={"action": "retire"})
    assert not (agent.learning.skills_dir / "trace-a-request.md").exists()
    assert client.post(f"/api/learnings/{lesson['id']}", json={"action": "bogus"}).status_code == 422
    assert client.post("/api/messages/missing/feedback", json={"rating": 1}).status_code == 404


def test_pr_outcomes_are_tracked_and_trigger_reflection(tmp_path, remotes):
    pr = {"state": "open", "merged": False, "comments": []}

    def github(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/acme/ApiAgentService1/pulls/3":
            return httpx.Response(200, json={"state": pr["state"], "merged": pr["merged"]})
        if path == "/repos/acme/ApiAgentService1/pulls/3/comments":
            return httpx.Response(200, json=pr["comments"])
        if path == "/repos/acme/ApiAgentService1/issues/3/comments":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    reflections: list[str] = []

    def responder(system: str, user: str) -> str:
        if system == REFLECT_PROMPT:
            reflections.append(user)
            return '{"proposals": []}'
        return "Summary."

    agent = build_agent(
        tmp_path, remotes, service1_spec(), llm=FakeLLM(responder=responder), clone=False, http_factory=github_factory(github), github_token="ghp_testtoken123"
    )
    url = "https://github.com/acme/ApiAgentService1/pull/3"
    conversation = agent.store.create_conversation()
    agent.store.upsert_pr_outcome(url, state="open", conversation_id=conversation["id"])

    agent.learning.poll_prs()
    assert reflections == []

    pr["comments"] = [{"id": 11, "user": {"login": "reviewer"}, "body": "Please add a test", "path": "src/x.py"}]
    agent.learning.poll_prs()
    agent.learning.poll_prs()
    assert len(reflections) == 1 and "reviewer: Please add a test" in reflections[0]

    pr["state"] = "closed"
    agent.learning.poll_prs()
    assert len(reflections) == 2 and "closed without being merged" in reflections[1]
    assert agent.store.list_pr_outcomes()[0]["state"] == "closed"

    agent.learning.poll_prs()
    assert len(reflections) == 2
