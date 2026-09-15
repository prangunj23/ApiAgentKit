"""Agent memory files: notes the agent keeps, and a recent-activity log that updates itself."""

import logging
import threading
from typing import TYPE_CHECKING, Any

from agentkit.research import today

if TYPE_CHECKING:
    from agentkit.agent import Agent

log = logging.getLogger(__name__)

RECENT_LIMIT = 40
SUMMARY_PROMPT = """You maintain an AI agent's activity log. Summarize the conversation below in ONE line of at most 160 characters: what was asked or discussed, what the agent did, and any outcome (files changed, tests, PRs, emails). No preamble and no quotes."""


def format_transcript(messages: list[dict[str, Any]], max_chars: int = 12_000) -> str:
    lines = []
    for message in messages:
        if message["role"] == "tool":
            lines.append(f"[{message['sender'] or 'tool'} result] {message['content'][:300]}")
            continue
        label = message["sender"] or message["role"]
        if message["content"]:
            lines.append(f"{label}: {message['content']}")
        if message.get("tool_calls"):
            lines.append(f"{label} called: {', '.join(call['name'] for call in message['tool_calls'])}")
    return "\n".join(lines)[-max_chars:]


def marker(conversation_id: str) -> str:
    return f"<!-- conversation:{conversation_id} -->"


def strip_markers(lines: list[str]) -> list[str]:
    return [line.split(" <!-- ", 1)[0] for line in lines]


class Memory:
    def __init__(self, agent: "Agent") -> None:
        self.agent = agent
        self.dir = agent.settings.data_dir / "memory"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.notes_path = self.dir / "notes.md"
        self.recent_path = self.dir / "recent.md"
        self.lessons_path = self.dir / "lessons.md"
        self._lock = threading.Lock()
        self._timers: dict[str, threading.Timer] = {}

    # Notes

    def notes(self) -> str:
        return self.notes_path.read_text() if self.notes_path.exists() else ""

    def remember(self, note: str) -> None:
        with self._lock, self.notes_path.open("a") as file:
            file.write(f"- {' '.join(note.split())} ({today()})\n")

    def forget(self, text: str) -> int:
        with self._lock:
            lines = self.notes().splitlines()
            kept = [line for line in lines if text.lower() not in line.lower()]
            self.notes_path.write_text("\n".join(kept) + ("\n" if kept else ""))
            return len(lines) - len(kept)

    # Recent activity

    def recent(self) -> list[str]:
        return [line for line in self.recent_path.read_text().splitlines() if line.strip()] if self.recent_path.exists() else []

    def _prepend(self, line: str, replace_marker: str | None = None) -> None:
        with self._lock:
            lines = [existing for existing in self.recent() if not (replace_marker and replace_marker in existing)]
            self.recent_path.write_text("\n".join([line, *lines][:RECENT_LIMIT]) + "\n")

    def record_conversation(self, conversation_id: str, summary: str) -> None:
        self._prepend(f"- {today()} · {summary} {marker(conversation_id)}", replace_marker=marker(conversation_id))

    def record_event(self, summary: str) -> None:
        self._prepend(f"- {today()} · {summary}")

    # Summaries

    def schedule_summary(self, conversation_id: str) -> None:
        delay = self.agent.settings.summary_delay_seconds
        if not self.agent.settings.background or delay <= 0:
            self.summarize(conversation_id)
            return
        with self._lock:
            if conversation_id in self._timers:
                return
            timer = threading.Timer(delay, self._run_scheduled, args=(conversation_id,))
            timer.daemon = True
            self._timers[conversation_id] = timer
            timer.start()

    def _run_scheduled(self, conversation_id: str) -> None:
        with self._lock:
            self._timers.pop(conversation_id, None)
        self.summarize(conversation_id)

    def summarize(self, conversation_id: str) -> str | None:
        store = self.agent.store
        conversation = store.get_conversation(conversation_id)
        if conversation is None:
            return None
        transcript = format_transcript(store.list_messages(conversation_id))
        if not transcript.strip():
            return None
        try:
            line = " ".join(self.agent.llm.complete(SUMMARY_PROMPT, transcript).split())[:200]
        except Exception:
            log.exception("Couldn't summarize conversation %s", conversation_id)
            return None
        if not line:
            return None
        if conversation["kind"] == "user":
            who = "chat with user"
        elif conversation["channel"] == "github_dispatch":
            who = f"GitHub dispatch to {conversation['peer_agent']}"
        else:
            who = f"with agent {conversation['peer_agent']}"
        self.record_conversation(conversation_id, f"{who} · {line}")
        return line

    def view(self) -> dict[str, Any]:
        return {
            "notes": self.notes(),
            "recent": "\n".join(strip_markers(self.recent())),
            "lessons": self.lessons_path.read_text() if self.lessons_path.exists() else "",
        }
