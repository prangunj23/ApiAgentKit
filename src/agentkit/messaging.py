"""Sending a message to another agent and keeping this agent's copy of the thread."""

from typing import TYPE_CHECKING, Any

from agentkit.registry import PeerError
from agentkit.store import new_id

if TYPE_CHECKING:
    from agentkit.agent import Agent


def title_of(message: str, fallback: str) -> str:
    return message.strip().splitlines()[0][:60] if message.strip() else fallback


def deliver(
    agent: "Agent",
    to: str,
    message: str,
    *,
    parent_conversation_id: str | None = None,
    conversation: dict[str, Any] | None = None,
    thread_id: str | None = None,
    wait: bool = True,
    timeout: float = 600,
) -> dict[str, Any]:
    """POST `message` to agent `to`. Returns {status, reply, error, conversation_id}.

    With wait, returns when the other agent's turn ends; without, as soon as it starts (status "accepted").
    Both sides of the exchange are stored in an agent thread here (a new one unless `conversation` is given).
    A given `thread_id` makes a retry safe: the other agent ignores a message its thread already has.
    Raises PeerError if the agent isn't reachable or this agent isn't linked to it.
    """
    agent.peers.get(to)
    store = agent.store
    if conversation is None and thread_id:
        conversation = store.find_thread(thread_id)
    if conversation is None:
        conversation = store.create_conversation(
            kind="agent",
            channel="agent_http",
            peer_agent=to,
            thread_id=thread_id or new_id(),
            parent_conversation_id=parent_conversation_id,
            title=title_of(message, f"Message to {to}"),
        )
    already_sent = any(m["role"] == "assistant" and m["content"] == message for m in store.list_messages(conversation["id"]))
    if not already_sent:
        store.add_message(conversation["id"], "assistant", message, sender=agent.spec.id)
    try:
        try:
            result = agent.peers.request(
                to,
                "POST",
                "/api/agent-messages",
                json={"from_agent": agent.spec.id, "thread_id": conversation["thread_id"], "message": message, "wait": wait},
                timeout=timeout,
            )
        except PeerError as err:
            store.add_message(conversation["id"], "user", f"(Message not delivered: {err})", sender=to)
            raise
        if result["status"] == "replied" or (result["status"] == "duplicate" and result.get("reply") and not already_sent):
            store.add_message(conversation["id"], "user", result["reply"], sender=to)
        return result
    finally:
        agent.memory.schedule_summary(conversation["id"])
