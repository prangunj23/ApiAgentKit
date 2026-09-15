from agentkit.registry import PeerError
from agentkit.spec import ToolContext, ToolError, string, tool
from agentkit.store import new_id


@tool(
    "message_agent",
    "Send a message to another agent and wait for its reply, for example to ask how a change affects its repo.",
    {"to": string("The other agent's id."), "message": string("Your message.")},
    required=("to", "message"),
)
def message_agent(ctx: ToolContext, to: str, message: str) -> str:
    agent = ctx.agent
    if ctx.depth > 0:
        raise ToolError("You can't message other agents while answering one.")
    try:
        peer = agent.peers.get(to)
    except PeerError as err:
        raise ToolError(str(err)) from err

    store = agent.store
    conversation = store.find_agent_conversation(ctx.conversation_id, to, "agent_http") or store.create_conversation(
        kind="agent",
        channel="agent_http",
        peer_agent=to,
        thread_id=new_id(),
        parent_conversation_id=ctx.conversation_id,
        title=message.strip().splitlines()[0][:60] if message.strip() else f"Message to {to}",
    )
    store.add_message(conversation["id"], "assistant", message, sender=agent.spec.id)
    try:
        result = agent.peers.request(
            to,
            "POST",
            "/api/agent-messages",
            json={"from_agent": agent.spec.id, "thread_id": conversation["thread_id"], "message": message},
            timeout=600,
        )
    except PeerError as err:
        store.add_message(conversation["id"], "user", f"(Message not delivered: {err})", sender=to)
        raise ToolError(str(err)) from err

    try:
        if result["status"] == "replied":
            store.add_message(conversation["id"], "user", result["reply"], sender=to)
            return f"{peer.name} replied:\n\n{result['reply']}"
        if result["status"] == "awaiting_approval":
            return (
                f"{peer.name} needs the user's approval for an action before it can finish. "
                "Its reply will appear in your thread with it once the user decides."
            )
        return f"{peer.name} couldn't answer: {result.get('error') or 'unknown error'}"
    finally:
        agent.memory.schedule_summary(conversation["id"])
