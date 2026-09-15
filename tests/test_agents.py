import json

from fastapi.testclient import TestClient

from agentkit.testing import FakeLLM, commit_files, http_factory_for
from helpers import OPERATION_APP, build_app, decide, describer, send, service1_spec, service2_spec, start

URL1, URL2 = "http://127.0.0.1:9001", "http://127.0.0.1:9002"


def two_agents(tmp_path, remotes, llm1, llm2):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps([{"id": "service1", "name": "Operation", "url": URL1}, {"id": "service2", "name": "Consumer", "url": URL2}]))
    apps: dict = {}
    factory = http_factory_for(apps)
    apps[URL1] = build_app(tmp_path, remotes, service1_spec(), llm=llm1, port=9001, http_factory=factory, registry=str(registry))
    apps[URL2] = build_app(tmp_path, remotes, service2_spec(), llm=llm2, port=9002, http_factory=factory, registry=str(registry))
    return apps[URL1], apps[URL2]


def test_agents_talk_and_both_keep_the_thread(tmp_path, remotes):
    llm1 = FakeLLM([{"tool_calls": [("message_agent", {"to": "service2", "message": "Would renaming b break you?"})]}, "It's safe."])
    llm2 = FakeLLM(["No: consumer passes a and b positionally."])
    app1, app2 = two_agents(tmp_path, remotes, llm1, llm2)
    client1, client2 = TestClient(app1, base_url=URL1), TestClient(app2, base_url=URL2)
    conversation_id = start(client1)

    events = send(client1, conversation_id, "Ask service2 about renaming b")
    result = next(event for event in events if event["type"] == "tool_result")
    assert result["content"] == "Consumer replied:\n\nNo: consumer passes a and b positionally."

    [thread1] = client1.get("/api/conversations", params={"kind": "agent"}).json()
    [thread2] = client2.get("/api/conversations", params={"kind": "agent"}).json()
    assert thread1["parent_conversation_id"] == conversation_id and thread1["peer_agent"] == "service2"
    assert thread2["thread_id"] == thread1["thread_id"] and thread2["peer_agent"] == "service1"
    messages2 = client2.get(f"/api/conversations/{thread2['id']}").json()["messages"]
    assert [(m["sender"], m["content"]) for m in messages2] == [
        ("service1", "Would renaming b break you?"),
        ("service2", "No: consumer passes a and b positionally."),
    ]

    assert "message_agent" not in [tool["function"]["name"] for tool in llm2.calls[0]["tools"]]
    assert llm2.calls[0]["messages"][-1]["content"].startswith("[Message from the service1 agent]")
    assert "`service2` (Consumer)" in llm1.calls[0]["messages"][0]["content"]
    assert "with agent service2" in app1.state.agent.memory.recent_path.read_text()
    assert "with agent service1" in app2.state.agent.memory.recent_path.read_text()
    assert client2.post(f"/api/conversations/{thread2['id']}/messages", json={"message": "hi"}).status_code == 409


def test_reply_after_approval_is_forwarded_to_the_asking_agent(tmp_path, remotes, monkeypatch):
    sent = []
    monkeypatch.setattr("agentkit.tools.email.send", lambda to, subject, text, html: sent.append((to, subject)))
    monkeypatch.setenv("EMAIL_TO", "dev@example.com")
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    llm1 = FakeLLM([{"tool_calls": [("message_agent", {"to": "service2", "message": "Email the team about the rename."})]}, "Waiting."])
    llm2 = FakeLLM([{"tool_calls": [("send_email", {"subject": "Rename coming", "body_markdown": "**b** becomes **y**."})]}, "Emailed the team."])
    app1, app2 = two_agents(tmp_path, remotes, llm1, llm2)
    client1, client2 = TestClient(app1, base_url=URL1), TestClient(app2, base_url=URL2)

    events = send(client1, start(client1), "Have service2 email the team")
    assert "needs the user's approval" in next(e for e in events if e["type"] == "tool_result")["content"]

    [pending] = client2.get("/api/pending").json()
    assert pending["pending_action"]["name"] == "send_email"
    events2 = decide(client2, pending["id"], True)
    assert events2[1]["content"] == "Emailed dev@example.com: Rename coming"
    assert sent == [(["dev@example.com"], "Rename coming")]
    assert client2.get("/api/emails").json()[0]["status"] == "sent"

    [thread1] = client1.get("/api/conversations", params={"kind": "agent"}).json()
    last = client1.get(f"/api/conversations/{thread1['id']}").json()["messages"][-1]
    assert (last["sender"], last["content"]) == ("service2", "Emailed the team.")


def test_contract_change_reaches_the_dependent_agent(tmp_path, remotes):
    app1, app2 = two_agents(tmp_path, remotes, FakeLLM(responder=describer), FakeLLM(responder=describer))
    agent1, agent2 = app1.state.agent, app2.state.agent
    agent1.codemap.update()

    commit_files(remotes["service1"], {"src/operation/app.py": OPERATION_APP.replace('"/numeric_op"', '"/subtract"')}, "Rename endpoint")
    agent1.sync_workspace()

    [event] = [e for e in agent2.store.list_events() if e["type"] == "contract_changed"]
    assert event["summary"].startswith("[from service1] Public contract of acme/ApiAgentService1 changed")
    assert "/v1/operation/subtract" in event["summary"]
    assert "Public contract of acme/ApiAgentService1 changed" in agent2.memory.recent_path.read_text()
