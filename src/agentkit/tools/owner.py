"""Routing news to people: a service tells its owners' agents, and a developer's agent emails its developer.

Neither tool needs approval. notify_owners only reaches the owners listed in the registry, and notify_developer
only reaches the one developer the agent works for.
"""

from agentkit import owners
from agentkit.registry import PeerError
from agentkit.spec import ToolContext, ToolError, string, tool
from agentkit.store import now
from agentkit.tools.email import deliver_email


@tool(
    "notify_owners",
    "Tell the developers who own this service about something, through their agents. `urgent` and `needs_attention` "
    "go to the on-call owner; `fyi` goes to every owner. Their agents decide whether to email them. No approval needed.",
    {
        "headline": string("One line: what happened."),
        "detail_markdown": string("What the owner needs to know: what changed, what it affects, and what to do."),
        "urgency": {"type": "string", "enum": list(owners.URGENCIES), "description": "fyi, needs_attention, or urgent."},
        "url": string("A link to the change, PR, or run, if there is one."),
    },
    required=("headline", "detail_markdown", "urgency"),
)
def notify_owners(ctx: ToolContext, headline: str, detail_markdown: str, urgency: str, url: str = "") -> str:
    agent = ctx.agent
    if urgency not in owners.URGENCIES:
        raise ToolError(f"urgency must be one of {', '.join(owners.URGENCIES)}")
    recipients = owners.recipients_for(agent, urgency)
    if not recipients:
        raise ToolError(f"The registry lists no owners for {agent.spec.id} that you can reach.")

    if ctx.depth > 0:
        # Answering another agent: we can't message anyone else until this turn ends.
        conversation = agent.store.get_conversation(ctx.conversation_id) or {}
        owners.queue(
            agent,
            headline=headline,
            detail=detail_markdown,
            urgency=urgency,
            url=url,
            reported_by=conversation.get("peer_agent"),
            source_conversation_id=ctx.conversation_id,
        )
        names = ", ".join(recipient.id for recipient in recipients)
        return f"Queued for {names}; it will be delivered after this reply."

    lines = []
    for recipient in recipients:
        message = owners.owner_message(agent, recipient, headline=headline, detail=detail_markdown, urgency=urgency, url=url)
        try:
            result = owners.send_now(agent, recipient, message, headline=headline, urgency=urgency, url=url, conversation_id=ctx.conversation_id)
        except PeerError as err:
            agent.store.add_outbox(
                recipient=recipient.id,
                headline=headline,
                message=message,
                urgency=urgency,
                url=url or None,
                source_conversation_id=ctx.conversation_id,
            )
            agent.kick_outbox()
            lines.append(f"- {recipient.id}: not reachable ({err}); queued to retry.")
            continue
        if result["status"] in {"accepted", "duplicate", "replied"}:
            lines.append(f"- {recipient.id} received it and is deciding whether to email its developer.")
        elif result["status"] == "awaiting_approval":
            lines.append(f"- {recipient.id} received it; its thread is waiting for its developer's approval.")
        else:
            lines.append(f"- {recipient.id} couldn't take it: {result.get('error')}")
    return "\n".join(lines)


def developer_footer(ctx: ToolContext, url: str = "") -> str:
    agent = ctx.agent
    lines = ["", "---", f"Sent by {agent.spec.id} ({agent.spec.name}) at {now()}"]
    if url:
        lines.append(f"Link: {url}")
    if agent.settings.ui_origins:
        lines.append(f"Open the agent: {agent.settings.ui_origins[0].rstrip('/')}/#/agents/{agent.spec.id}")
    return "\n".join(lines)


@tool(
    "notify_developer",
    "Email the developer you work for. It can't reach anyone else, and it needs no approval. "
    "A footer saying which agent sent it, and the link, is added for you.",
    {
        "subject": string("Subject line."),
        "body_markdown": string("Email body in markdown."),
        "url": string("The PR, change, or run the email is about, if any."),
    },
    required=("subject", "body_markdown"),
)
def notify_developer(ctx: ToolContext, subject: str, body_markdown: str, url: str = "") -> str:
    agent = ctx.agent
    developer = agent.peers.developer_of(agent.spec.id) or {}
    address = str(developer.get("email") or "").strip()
    if not address:
        raise ToolError(f"The registry has no developer.email for {agent.spec.id}.")
    return deliver_email(ctx, [address], subject, body_markdown.rstrip() + "\n" + developer_footer(ctx, url))
