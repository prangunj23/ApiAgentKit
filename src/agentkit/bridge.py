"""A client for one agent's HTTP API, for use by another assistant such as Claude Code (see agentkit/mcp_server.py).

It does what the UI does: start and continue chats, read conversations, and list what's waiting for approval.
It deliberately can't approve anything. An approval card is for the agent's human, in the UI.
"""

import json
from collections.abc import Iterator
from typing import Any

import httpx

TITLE_PREFIX = "Claude Code: "
RESULT_CHARS = 600


class BridgeError(Exception):
    pass


def iter_sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Events from a server-sent event stream, read line by line."""
    for line in lines:
        if line.startswith("data: "):
            yield json.loads(line[len("data: ") :])


def _short(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


class AgentBridge:
    def __init__(self, client: httpx.Client, *, ui_url: str = "http://localhost:5173", agent_id: str = "") -> None:
        self.client = client
        self.ui_url = ui_url.rstrip("/")
        self.agent_id = agent_id

    def _get(self, path: str, **params: Any) -> Any:
        try:
            response = self.client.get(path, params=params or None)
        except httpx.TransportError as err:
            raise BridgeError(f"Couldn't reach the agent at {self.client.base_url}: {err}. Is it running?") from err
        if response.is_error:
            raise BridgeError(f"GET {path} returned {response.status_code}: {response.text[:300]}")
        return response.json()

    def _link(self, conversation_id: str) -> str:
        return f"{self.ui_url}/#/agents/{self.agent_id}/chats/{conversation_id}" if self.agent_id else conversation_id

    # Reading

    def status(self) -> str:
        info = self._get("/api/info")
        pending = self._get("/api/pending")
        events = self._get("/api/events", limit=10)
        emails = self._get("/api/emails")[:5]
        lines = [
            f"# {info['name']} (`{info['id']}`, {info.get('kind', 'service')})",
            f"Repo: {(info.get('repo') or {}).get('slug') or 'none'}; reads: {', '.join(r['slug'] for r in info['reads']) or 'none'}",
            "",
            f"## Waiting for approval ({len(pending)})",
            *[f"- {c['id']}: `{c['pending_action']['name']}` in “{c['title']}”" for c in pending],
            "",
            "## Recent events",
            *[f"- {e['created_at'][:16]} {e['type']}: {e['summary'].splitlines()[0]}" for e in events],
            "",
            "## Recent emails",
            *[f"- {m['created_at'][:16]} [{m['status']}] to {', '.join(m['recipients'])}: {m['subject']}" for m in emails],
        ]
        return "\n".join(lines)

    def conversations(self, limit: int = 20) -> str:
        rows = self._get("/api/conversations", kind="user")[:limit]
        if not rows:
            return "No conversations yet."
        return "\n".join(
            f"- {c['id']} · {c['updated_at'][:16]} · {c['title'] or '(untitled)'}"
            + (" · RUNNING" if c.get("running") else "")
            + (f" · WAITING FOR APPROVAL ({c['pending_action']['name']})" if c.get("pending_action") else "")
            for c in rows
        )

    def read(self, conversation_id: str, last: int = 20, wait: bool = False) -> str:
        if wait:
            # Follow a turn in progress (for example one resumed after an approval) to its end.
            self._follow("GET", f"/api/conversations/{conversation_id}/stream")
        detail = self._get(f"/api/conversations/{conversation_id}")
        conversation, messages = detail["conversation"], detail["messages"]
        lines = [f"# {conversation['title'] or conversation_id}", f"UI: {self._link(conversation_id)}"]
        if conversation.get("running"):
            lines.append("_A turn is still running; read again with wait=true to get its end._")
        for message in messages[-last:]:
            if message["role"] == "tool":
                lines.append(f"  ↳ {message['sender']}: {_short(message['content'], RESULT_CHARS)}")
            elif message["role"] == "assistant":
                for call in message.get("tool_calls") or []:
                    lines.append(f"  → {call['name']}({_short(call['arguments'], 200)})")
                if message["content"]:
                    lines.append(f"**agent:** {message['content']}")
            else:
                who = "you" if message["sender"] in {"", "user"} else message["sender"]
                lines.append(f"**{who}:** {message['content']}")
        if conversation.get("pending_action"):
            lines.append(self._approval_note(conversation_id, conversation["pending_action"]))
        return "\n".join(lines)

    # Chatting

    def ask(self, message: str, conversation_id: str = "") -> str:
        """Send a message and wait for the agent's turn to end. Starts a new conversation unless one is given."""
        if not conversation_id:
            title = TITLE_PREFIX + (message.strip().splitlines()[0][:48] if message.strip() else "chat")
            try:
                response = self.client.post("/api/conversations", json={"title": title})
            except httpx.TransportError as err:
                raise BridgeError(f"Couldn't reach the agent at {self.client.base_url}: {err}. Is it running?") from err
            if response.is_error:
                raise BridgeError(f"Couldn't start a conversation: {response.status_code} {response.text[:300]}")
            conversation_id = response.json()["id"]
        events = self._follow("POST", f"/api/conversations/{conversation_id}/messages", json={"message": message})
        return self._summarize(conversation_id, events)

    def _follow(self, method: str, path: str, **kwargs: Any) -> list[dict[str, Any]]:
        try:
            with self.client.stream(method, path, timeout=httpx.Timeout(30, read=None), **kwargs) as response:
                if response.is_error:
                    response.read()
                    raise BridgeError(f"{method} {path} returned {response.status_code}: {response.text[:300]}")
                return list(iter_sse(response.iter_lines()))
        except httpx.TransportError as err:
            raise BridgeError(f"Lost the connection to the agent: {err}. The turn keeps running; read the conversation with wait=true.") from err

    def _summarize(self, conversation_id: str, events: list[dict[str, Any]]) -> str:
        lines = [f"conversation_id: {conversation_id}", f"UI: {self._link(conversation_id)}", ""]
        reply = ""
        for event in events:
            kind = event["type"]
            if kind == "tool_call":
                lines.append(f"→ {event['name']}({_short(event['arguments'], 200)})")
            elif kind == "tool_result":
                lines.append(f"  ↳ {_short(event['content'], RESULT_CHARS)}")
            elif kind == "message" and event["message"]["role"] == "assistant" and event["message"]["content"]:
                reply = event["message"]["content"]
            elif kind == "error":
                lines.append(f"ERROR: {event['message']}")
            elif kind == "confirm_required":
                lines.append(self._approval_note(conversation_id, event["action"]))
        if reply:
            lines += ["", "Agent's reply:", reply]
        elif not any(event["type"] in {"error", "confirm_required"} for event in events):
            lines.append("(The turn ended without a reply.)")
        return "\n".join(lines)

    def _approval_note(self, conversation_id: str, action: dict[str, Any]) -> str:
        preview = action.get("preview") or _short(action.get("arguments", {}), 800)
        return (
            f"\nWAITING FOR APPROVAL: the agent wants to run `{action['name']}`.\n{preview}\n\n"
            f"Only the user can approve this, in the UI: {self._link(conversation_id)}. "
            "Tell them, then read the conversation with wait=true once they have decided."
        )
