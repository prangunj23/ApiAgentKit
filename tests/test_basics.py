import sqlite3

from agentkit.config import Settings
from agentkit.frontmatter import dump, parse
from agentkit.llm import ThinkFilter, extract_json
from agentkit.security import redact
from agentkit.store import Store


def test_think_blocks_are_removed_even_when_split_across_chunks():
    text = ThinkFilter()
    out = "".join(text.feed(chunk) for chunk in ["Hel", "lo <thi", "nk>secret</th", "ink> world", " <"]) + text.flush()
    assert out == "Hello  world <"


def test_extract_json_ignores_reasoning_and_fences():
    assert extract_json('<think>{"no": 1}</think>```json\n{"a": 1}\n```') == {"a": 1}


def test_frontmatter_round_trip():
    meta = {"title": "Map", "commit": "1234567", "dirty": False, "sources": ["a"]}
    assert parse(dump(meta, "Body")) == (meta, "Body")


def test_redact_hides_tokens():
    assert redact("https://x-access-token:ghp_secret@github.com/a/b.git") == "https://***@github.com/a/b.git"
    assert redact("key is nvapi-12345678", ["nvapi-12345678", ""]) == "key is ***"


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_DATA_DIR", raising=False)
    monkeypatch.setenv("AGENT_HOME", str(tmp_path))
    monkeypatch.setenv("UI_ORIGINS", "http://localhost:5173, http://example.test")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-key\n")
    settings = Settings.from_env("service1", 9001)
    assert settings.data_dir == tmp_path / "service1"
    assert settings.ui_origins == ["http://localhost:5173", "http://example.test"]
    assert settings.nvidia_api_key == "nvapi-key"
    assert "127.0.0.1:9001" in settings.allowed_hosts

    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path / "custom"))
    assert Settings.from_env("service1", 9001).data_dir == tmp_path / "custom"


def test_conversations_and_messages_round_trip(tmp_path):
    store = Store(tmp_path / "agent.db")
    conversation = store.create_conversation(title="Hello")
    assert conversation["kind"] == "user" and conversation["pending_action"] is None

    store.add_message(conversation["id"], "user", "hi", sender="user")
    calls = [{"id": "call_1", "name": "read_file", "arguments": "{}"}]
    assistant = store.add_message(conversation["id"], "assistant", "", sender="service1", tool_calls=calls)
    store.add_message(conversation["id"], "tool", "file text", sender="read_file", tool_call_id="call_1")

    messages = store.list_messages(conversation["id"])
    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"] == calls and messages[1]["id"] == assistant["id"]
    assert store.count_tool_rounds(conversation["id"]) == 1
    assert store.list_conversations()[0]["preview"] == "hi"

    store.set_pending_action(conversation["id"], {"name": "open_pull_request", "arguments": {"title": "x"}})
    assert store.pending_conversations()[0]["pending_action"]["arguments"] == {"title": "x"}
    store.set_pending_action(conversation["id"], None)
    assert store.pending_conversations() == []


def test_threads_emails_events_learnings_and_pr_outcomes(tmp_path):
    store = Store(tmp_path / "agent.db")
    parent = store.create_conversation()
    thread = store.create_conversation(
        kind="agent", channel="agent_http", peer_agent="service2", thread_id="t1", parent_conversation_id=parent["id"]
    )
    assert store.find_thread("t1")["id"] == thread["id"]
    assert store.find_agent_conversation(parent["id"], "service2", "agent_http")["id"] == thread["id"]
    assert [c["id"] for c in store.list_conversations(kind="agent")] == [thread["id"]]

    store.add_email(
        conversation_id=parent["id"],
        recipients=["a@example.com"],
        subject="S",
        body_markdown="**b**",
        body_html="<p><strong>b</strong></p>",
        provider="resend",
        status="sent",
    )
    assert store.list_emails()[0]["recipients"] == ["a@example.com"]

    store.add_event("pr_opened", "first")
    store.add_event("tests_run", "second")
    assert [event["summary"] for event in store.list_events()] == ["second", "first"]

    learning = store.add_learning(kind="lesson", title="T", body="B")
    assert store.update_learning(learning["id"], status="active", body="B2")["body"] == "B2"
    assert store.list_learnings(status="active", kind="lesson")[0]["id"] == learning["id"]
    assert store.claim_reflection(parent["id"], "denied:1")
    assert not store.claim_reflection(parent["id"], "denied:1")

    url = "https://github.com/a/b/pull/1"
    store.upsert_pr_outcome(url, state="open", conversation_id=parent["id"])
    store.upsert_pr_outcome(url, state="merged", merged=True)
    outcome = store.list_pr_outcomes()[0]
    assert outcome["merged"] is True and outcome["state"] == "merged" and outcome["conversation_id"] == parent["id"]


def test_existing_databases_gain_the_reasoning_column(tmp_path):
    path = tmp_path / "agent.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE messages (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, conversation_id TEXT NOT NULL,"
        " role TEXT NOT NULL, sender TEXT NOT NULL DEFAULT '', content TEXT NOT NULL DEFAULT '', tool_calls_json TEXT,"
        " tool_call_id TEXT, created_at TEXT NOT NULL)"
    )
    db.commit()
    db.close()

    store = Store(path)
    conversation = store.create_conversation()
    assert store.add_message(conversation["id"], "assistant", "hi", reasoning="because")["reasoning"] == "because"
