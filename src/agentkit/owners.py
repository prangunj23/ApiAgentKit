"""Telling a service's owners (their developer agents) about something.

`urgent` and `needs_attention` go to the on-call owner only; `fyi` goes to every owner. A service with owners
but no on-call sends everything to every owner.

Notifications are delivered right away when the agent is free to message others (depth 0). While it is answering
another agent (depth 1) they wait in the outbox, and a background worker delivers them after the reply. The worker
only makes HTTP calls; it never runs a model turn, so queued notifications can't start a chain of messages.
"""

import logging
from typing import TYPE_CHECKING, Any

from agentkit.messaging import deliver
from agentkit.registry import PeerError, RegistryEntry

if TYPE_CHECKING:
    from agentkit.agent import Agent

log = logging.getLogger(__name__)

URGENCIES = ("fyi", "needs_attention", "urgent")
MAX_ATTEMPTS = 3


def recipients_for(agent: "Agent", urgency: str) -> list[RegistryEntry]:
    owners = agent.peers.owners()
    if urgency == "fyi":
        return owners
    oncall = agent.peers.oncall()
    return [oncall] if oncall else owners


def owner_message(
    agent: "Agent", recipient: RegistryEntry, *, headline: str, detail: str, urgency: str, url: str = "", reported_by: str | None = None
) -> str:
    """The message a developer agent receives. Only headline, detail, urgency and url come from the model."""
    repo = agent.spec.repo.slug if agent.spec.repo else agent.spec.name
    oncall = agent.peers.oncall()
    role = "on-call owner" if oncall and oncall.id == recipient.id else "owner"
    person = (recipient.developer or {}).get("name") or "your developer"
    lines = [f"[{agent.spec.id}] News about {repo}.", f"You are receiving this as: {role}."]
    if reported_by:
        lines.append(f"Reported by: {reported_by}.")
    lines += ["", f"Urgency: {urgency}", f"Headline: {headline.strip()}", "", detail.strip()]
    if url:
        lines += ["", f"Link: {url}"]
    lines += [
        "",
        f"{person} owns this service. Decide whether they need to hear about this, and if so tell them with notify_developer.",
    ]
    return "\n".join(lines)


def queue(
    agent: "Agent",
    *,
    headline: str,
    detail: str,
    urgency: str,
    url: str = "",
    reported_by: str | None = None,
    source_conversation_id: str | None = None,
) -> list[RegistryEntry]:
    """Put a notification for each recipient in the outbox. Returns the recipients."""
    recipients = recipients_for(agent, urgency)
    for recipient in recipients:
        message = owner_message(agent, recipient, headline=headline, detail=detail, urgency=urgency, url=url, reported_by=reported_by)
        agent.store.add_outbox(
            recipient=recipient.id,
            headline=headline,
            message=message,
            urgency=urgency,
            url=url or None,
            source_conversation_id=source_conversation_id,
        )
    if recipients:
        agent.kick_outbox()
    return recipients


def send_now(
    agent: "Agent",
    recipient: RegistryEntry,
    message: str,
    *,
    headline: str,
    urgency: str,
    url: str,
    conversation_id: str | None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Hand the notification over without waiting for the owner's agent to act on it, which can take minutes."""
    result = deliver(agent, recipient.id, message, parent_conversation_id=conversation_id, thread_id=thread_id, wait=False, timeout=60)
    agent.log_event("owner_notified", f"Notified {recipient.id} ({urgency}): {headline}", url=url or None, conversation_id=conversation_id)
    return result


def deliver_outbox(agent: "Agent") -> int:
    """Try every queued notification once. Returns how many were delivered."""
    delivered = 0
    for row in agent.store.queued_outbox():
        attempts = row["attempts"] + 1
        try:
            recipient = agent.peers.get(row["recipient"])
            send_now(
                agent,
                recipient,
                row["message"],
                headline=row["headline"],
                urgency=row["urgency"],
                url=row["url"] or "",
                conversation_id=row["source_conversation_id"],
                # The row id names the thread, so a redelivery after a crash is recognized and ignored.
                thread_id=row["id"],
            )
        except Exception as err:
            if not isinstance(err, PeerError):
                log.exception("Outbox delivery failed")
            if attempts >= MAX_ATTEMPTS:
                agent.store.update_outbox(row["id"], status="failed", attempts=attempts, error=str(err))
                agent.log_event("owner_notify_failed", f"Couldn't notify {row['recipient']} after {attempts} attempts: {row['headline']}\n{err}")
            else:
                agent.store.update_outbox(row["id"], status="queued", attempts=attempts, error=str(err))
            continue
        agent.store.update_outbox(row["id"], status="sent", attempts=attempts)
        delivered += 1
    return delivered
