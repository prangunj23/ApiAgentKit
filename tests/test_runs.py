"""Turns run on their own thread, so they finish however their listeners come and go."""

import threading

from fastapi.testclient import TestClient

from agentkit import loop
from agentkit.runs import Run, Runs
from agentkit.testing import FakeLLM, parse_sse
from helpers import build_app, send, service1_spec, start


class GatedLLM(FakeLLM):
    """A FakeLLM whose model calls wait for the test to open the gate, like a slow NIM call."""

    def __init__(self, replies):
        super().__init__(replies)
        self.entered = threading.Event()
        self.gate = threading.Event()

    def stream(self, messages, tools):
        self.entered.set()
        assert self.gate.wait(10), "the test never opened the gate"
        yield from super().stream(messages, tools)


def running_turn(tmp_path, remotes, replies):
    """An app with a turn in progress that no one is streaming, as after the page that started it closed."""
    llm = GatedLLM(replies)
    app = build_app(tmp_path, remotes, service1_spec(), llm=llm)
    client = TestClient(app)
    conversation_id = start(client)
    agent = app.state.agent
    agent.runs.start(conversation_id, loop.run_turn(agent, conversation_id, "Summarize service2's reply"))
    assert llm.entered.wait(10)
    return app, client, llm, conversation_id


def test_turn_finishes_with_no_one_listening(tmp_path, remotes):
    app, client, llm, conversation_id = running_turn(tmp_path, remotes, ["Here is the summary."])
    assert client.get(f"/api/conversations/{conversation_id}").json()["conversation"]["running"] is True
    assert [conversation["running"] for conversation in client.get("/api/conversations").json()] == [True]

    llm.gate.set()
    assert app.state.agent.runs.wait(conversation_id, timeout=10)

    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert detail["conversation"]["running"] is False
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [
        ("user", "Summarize service2's reply"),
        ("assistant", "Here is the summary."),
    ]


def test_closing_a_listener_leaves_the_turn_running(tmp_path, remotes):
    app, client, llm, conversation_id = running_turn(tmp_path, remotes, ["Done."])
    listener = app.state.agent.runs.get(conversation_id).follow(heartbeat_seconds=0.05)
    assert next(listener)["type"] == "message"
    listener.close()  # what the server does when the page reloads mid-stream

    llm.gate.set()
    assert app.state.agent.runs.wait(conversation_id, timeout=10)
    assert client.get(f"/api/conversations/{conversation_id}").json()["messages"][-1]["content"] == "Done."


def test_following_a_turn_replays_it_from_the_start(tmp_path, remotes, monkeypatch):
    app, client, llm, conversation_id = running_turn(tmp_path, remotes, ["All done."])
    followed = threading.Event()
    original_follow = Run.follow

    def follow(self, *args, **kwargs):
        followed.set()
        yield from original_follow(self, *args, **kwargs)

    monkeypatch.setattr(Run, "follow", follow)
    responses = []
    follower = threading.Thread(target=lambda: responses.append(client.get(f"/api/conversations/{conversation_id}/stream")))
    follower.start()
    assert followed.wait(10)
    llm.gate.set()
    follower.join(10)

    events = parse_sse(responses[0].text)
    assert [event["type"] for event in events] == ["message", "token", "message", "done"]
    assert events[0]["message"]["content"] == "Summarize service2's reply"
    assert client.get(f"/api/conversations/{conversation_id}/stream").text == ""


def test_a_second_message_while_running_is_refused(tmp_path, remotes):
    app, client, llm, conversation_id = running_turn(tmp_path, remotes, ["First reply."])
    assert send(client, conversation_id, "Another question") == [{"type": "error", "message": "This conversation is already running."}]

    llm.gate.set()
    assert app.state.agent.runs.wait(conversation_id, timeout=10)
    assert len(llm.calls) == 1
    contents = [m["content"] for m in client.get(f"/api/conversations/{conversation_id}").json()["messages"]]
    assert "Another question" not in contents


def test_a_crashing_turn_reports_a_redacted_error_and_stops_running():
    runs = Runs(secrets=["sekrit-token-123"])

    def events():
        yield {"type": "message", "message": {}}
        raise RuntimeError("model exploded with sekrit-token-123")

    run = runs.start("c1", events())
    assert runs.wait("c1", timeout=10)
    assert list(run.follow()) == [
        {"type": "message", "message": {}},
        {"type": "error", "message": "RuntimeError: model exploded with ***"},
    ]
    assert not runs.is_running("c1")


def test_idle_listeners_get_keep_alives():
    runs = Runs()
    gate = threading.Event()

    def events():
        gate.wait(10)
        yield {"type": "done", "message_id": "m1"}

    listener = runs.start("c1", events()).follow(heartbeat_seconds=0.01)
    assert next(listener) is None
    gate.set()
    assert [event for event in listener if event is not None] == [{"type": "done", "message_id": "m1"}]
