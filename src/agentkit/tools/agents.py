from agentkit.messaging import deliver
from agentkit.registry import PeerError
from agentkit.spec import ToolContext, ToolError, string, tool


@tool(
    "message_agent",
    "Send a message to another agent and wait for its reply, for example to ask how a change affects its repo.",
    {"to": string("The other agent's id."), "message": string("Your message.")},
    required=("to", "message"),
    top_level_only=True,
)
def message_agent(ctx: ToolContext, to: str, message: str) -> str:
    agent = ctx.agent
    if ctx.depth > 0:
        raise ToolError("You can't message other agents while answering one.")
    try:
        peer = agent.peers.get(to)
    except PeerError as err:
        raise ToolError(str(err)) from err

    existing = agent.store.find_agent_conversation(ctx.conversation_id, to, "agent_http")
    try:
        result = deliver(agent, to, message, parent_conversation_id=ctx.conversation_id, conversation=existing)
    except PeerError as err:
        raise ToolError(str(err)) from err

    if result["status"] in {"replied", "duplicate"}:
        return f"{peer.name} replied:\n\n{result['reply']}"
    if result["status"] == "awaiting_approval":
        return (
            f"{peer.name} needs the user's approval for an action before it can finish. "
            "Its reply will appear in your thread with it once the user decides."
        )
    return f"{peer.name} couldn't answer: {result.get('error') or 'unknown error'}"
