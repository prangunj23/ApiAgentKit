"""SQLite storage for one agent: conversations, messages, emails, events, feedback, learnings, PR outcomes.

Columns ending in `_json` hold JSON; rows come back with the suffix dropped and the value decoded.
"""

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    thread_id TEXT,
    kind TEXT NOT NULL,
    peer_agent TEXT,
    channel TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    parent_conversation_id TEXT,
    pending_action_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversations_thread ON conversations(thread_id);

CREATE TABLE IF NOT EXISTS messages (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL,
    sender TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT,
    tool_call_id TEXT,
    reasoning TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id, seq);

CREATE TABLE IF NOT EXISTS emails (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    conversation_id TEXT,
    recipients_json TEXT NOT NULL,
    subject TEXT NOT NULL,
    body_markdown TEXT NOT NULL,
    body_html TEXT NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    summary TEXT NOT NULL,
    url TEXT,
    conversation_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    conversation_id TEXT NOT NULL,
    message_id TEXT,
    source TEXT NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learnings (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    why TEXT NOT NULL DEFAULT '',
    evidence_conversation_id TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT
);

CREATE TABLE IF NOT EXISTS pr_outcomes (
    pr_url TEXT PRIMARY KEY,
    conversation_id TEXT,
    state TEXT NOT NULL,
    merged INTEGER NOT NULL DEFAULT 0,
    review_comments_json TEXT NOT NULL DEFAULT '[]',
    checked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reflections (
    conversation_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (conversation_id, trigger)
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_id() -> str:
    return uuid.uuid4().hex


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if key.endswith("_json"):
            data[key.removesuffix("_json")] = json.loads(value) if value else None
        elif key != "seq":
            data[key] = value
    return data


def _encode(key: str, value: Any) -> Any:
    if key.endswith("_json"):
        return None if value is None else json.dumps(value)
    return value


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            # Databases created before the model's reasoning was stored lack this column.
            if "reasoning" not in {row["name"] for row in db.execute("PRAGMA table_info(messages)")}:
                db.execute("ALTER TABLE messages ADD COLUMN reasoning TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _rows(self, sql: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        with self._connect() as db:
            return [_decode(row) for row in db.execute(sql, params).fetchall()]

    def _row(self, sql: str, params: tuple | list = ()) -> dict[str, Any] | None:
        rows = self._rows(sql, params)
        return rows[0] if rows else None

    def _insert(self, table: str, row: dict[str, Any]) -> None:
        columns = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._connect() as db:
            db.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", [_encode(k, v) for k, v in row.items()])

    def _execute(self, sql: str, params: tuple | list = ()) -> int:
        with self._connect() as db:
            return db.execute(sql, params).rowcount

    # Conversations

    def create_conversation(
        self,
        *,
        kind: str = "user",
        channel: str = "chat",
        title: str = "",
        peer_agent: str | None = None,
        thread_id: str | None = None,
        parent_conversation_id: str | None = None,
    ) -> dict[str, Any]:
        conversation_id = new_id()
        timestamp = now()
        self._insert(
            "conversations",
            {
                "id": conversation_id,
                "thread_id": thread_id,
                "kind": kind,
                "peer_agent": peer_agent,
                "channel": channel,
                "title": title,
                "parent_conversation_id": parent_conversation_id,
                "pending_action_json": None,
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )
        return self.get_conversation(conversation_id)  # type: ignore[return-value]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        return self._row("SELECT * FROM conversations WHERE id = ?", (conversation_id,))

    def find_thread(self, thread_id: str) -> dict[str, Any] | None:
        return self._row("SELECT * FROM conversations WHERE thread_id = ? ORDER BY created_at LIMIT 1", (thread_id,))

    def find_agent_conversation(self, parent_conversation_id: str, peer_agent: str, channel: str) -> dict[str, Any] | None:
        return self._row(
            "SELECT * FROM conversations WHERE parent_conversation_id = ? AND peer_agent = ? AND channel = ?"
            " ORDER BY created_at DESC LIMIT 1",
            (parent_conversation_id, peer_agent, channel),
        )

    def list_conversations(self, kind: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        return self._rows(
            """
            SELECT c.*,
                   (SELECT m.content FROM messages m
                     WHERE m.conversation_id = c.id AND m.role IN ('user', 'assistant') AND m.content != ''
                     ORDER BY m.seq DESC LIMIT 1) AS preview
              FROM conversations c
             WHERE (? IS NULL OR c.kind = ?)
             ORDER BY c.updated_at DESC
             LIMIT ?
            """,
            (kind, kind, limit),
        )

    def pending_conversations(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM conversations WHERE pending_action_json IS NOT NULL ORDER BY updated_at DESC")

    def set_title(self, conversation_id: str, title: str) -> None:
        self._execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))

    def set_pending_action(self, conversation_id: str, action: dict[str, Any] | None) -> None:
        self._execute(
            "UPDATE conversations SET pending_action_json = ?, updated_at = ? WHERE id = ?",
            (_encode("pending_action_json", action), now(), conversation_id),
        )

    # Messages

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str = "",
        *,
        sender: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        reasoning: str | None = None,
    ) -> dict[str, Any]:
        message = {
            "id": new_id(),
            "conversation_id": conversation_id,
            "role": role,
            "sender": sender,
            "content": content,
            "tool_calls_json": tool_calls or None,
            "tool_call_id": tool_call_id,
            "reasoning": reasoning or None,
            "created_at": now(),
        }
        self._insert("messages", message)
        self._execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (message["created_at"], conversation_id))
        return self.get_message(message["id"])  # type: ignore[return-value]

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        return self._row("SELECT * FROM messages WHERE id = ?", (message_id,))

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM messages WHERE conversation_id = ? ORDER BY seq", (conversation_id,))

    def count_tool_rounds(self, conversation_id: str) -> int:
        row = self._row(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ? AND role = 'assistant' AND tool_calls_json IS NOT NULL",
            (conversation_id,),
        )
        return int(row["n"]) if row else 0

    # Emails

    def add_email(
        self,
        *,
        conversation_id: str | None,
        recipients: list[str],
        subject: str,
        body_markdown: str,
        body_html: str,
        provider: str,
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        email_id = new_id()
        self._insert(
            "emails",
            {
                "id": email_id,
                "conversation_id": conversation_id,
                "recipients_json": recipients,
                "subject": subject,
                "body_markdown": body_markdown,
                "body_html": body_html,
                "provider": provider,
                "status": status,
                "error": error,
                "created_at": now(),
            },
        )
        return self.get_email(email_id)  # type: ignore[return-value]

    def get_email(self, email_id: str) -> dict[str, Any] | None:
        return self._row("SELECT * FROM emails WHERE id = ?", (email_id,))

    def list_emails(self, limit: int = 200) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM emails ORDER BY seq DESC LIMIT ?", (limit,))

    # Events

    def add_event(self, type: str, summary: str, *, url: str | None = None, conversation_id: str | None = None) -> dict[str, Any]:
        event = {
            "id": new_id(),
            "type": type,
            "summary": summary,
            "url": url,
            "conversation_id": conversation_id,
            "created_at": now(),
        }
        self._insert("events", event)
        return event

    def list_events(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM events ORDER BY seq DESC LIMIT ?", (limit,))

    # Feedback

    def add_feedback(
        self, *, conversation_id: str, source: str, rating: int, comment: str = "", message_id: str | None = None
    ) -> dict[str, Any]:
        row = {
            "id": new_id(),
            "conversation_id": conversation_id,
            "message_id": message_id,
            "source": source,
            "rating": rating,
            "comment": comment,
            "created_at": now(),
        }
        self._insert("feedback", row)
        return row

    def list_feedback(self, conversation_id: str | None = None) -> list[dict[str, Any]]:
        if conversation_id:
            return self._rows("SELECT * FROM feedback WHERE conversation_id = ? ORDER BY seq", (conversation_id,))
        return self._rows("SELECT * FROM feedback ORDER BY seq DESC")

    # Learnings

    def add_learning(
        self, *, kind: str, title: str, body: str, why: str = "", evidence_conversation_id: str | None = None, status: str = "proposed"
    ) -> dict[str, Any]:
        learning_id = new_id()
        self._insert(
            "learnings",
            {
                "id": learning_id,
                "kind": kind,
                "title": title,
                "body": body,
                "why": why,
                "evidence_conversation_id": evidence_conversation_id,
                "status": status,
                "created_at": now(),
                "decided_at": None,
            },
        )
        return self.get_learning(learning_id)  # type: ignore[return-value]

    def get_learning(self, learning_id: str) -> dict[str, Any] | None:
        return self._row("SELECT * FROM learnings WHERE id = ?", (learning_id,))

    def list_learnings(self, status: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM learnings WHERE (? IS NULL OR status = ?) AND (? IS NULL OR kind = ?) ORDER BY seq DESC",
            (status, status, kind, kind),
        )

    def update_learning(self, learning_id: str, *, status: str, title: str | None = None, body: str | None = None) -> dict[str, Any] | None:
        self._execute(
            "UPDATE learnings SET status = ?, title = COALESCE(?, title), body = COALESCE(?, body), decided_at = ? WHERE id = ?",
            (status, title, body, now(), learning_id),
        )
        return self.get_learning(learning_id)

    def claim_reflection(self, conversation_id: str, trigger: str) -> bool:
        """Record that a reflection ran for this trigger. False if it already did."""
        try:
            self._insert("reflections", {"conversation_id": conversation_id, "trigger": trigger, "created_at": now()})
        except sqlite3.IntegrityError:
            return False
        return True

    # PR outcomes

    def upsert_pr_outcome(
        self, pr_url: str, *, state: str, merged: bool = False, review_comments: list[dict[str, Any]] | None = None, conversation_id: str | None = None
    ) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO pr_outcomes (pr_url, conversation_id, state, merged, review_comments_json, checked_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pr_url) DO UPDATE SET
                    conversation_id = COALESCE(excluded.conversation_id, pr_outcomes.conversation_id),
                    state = excluded.state, merged = excluded.merged,
                    review_comments_json = excluded.review_comments_json, checked_at = excluded.checked_at
                """,
                (pr_url, conversation_id, state, int(merged), json.dumps(review_comments or []), now()),
            )

    def get_pr_outcome(self, pr_url: str) -> dict[str, Any] | None:
        return self._row("SELECT * FROM pr_outcomes WHERE pr_url = ?", (pr_url,))

    def list_pr_outcomes(self, state: str | None = None) -> list[dict[str, Any]]:
        rows = self._rows(
            "SELECT * FROM pr_outcomes WHERE (? IS NULL OR state = ?) ORDER BY checked_at DESC", (state, state)
        )
        for row in rows:
            row["merged"] = bool(row["merged"])
        return rows
