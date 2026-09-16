"""An MCP server that lets Claude Code talk to one agent, e.g. your developer agent.

    claude mcp add apiagent -- uv run --project <ApiAgentKit> --extra mcp agentkit mcp --agent dev-pranit --registry <registry.json>

Needs the `mcp` extra. The tools block while the agent works, on a worker thread so the MCP session stays responsive.
"""

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from agentkit.bridge import AgentBridge, BridgeError

INSTRUCTIONS = """These tools talk to {name} (`{agent_id}`), an agentkit agent that works for the user. Use ask_agent to
give it tasks or questions; it can read the repos it has access to, request and write PR reviews, and email the user.
A turn can take minutes. The agent may stop at an approval card: only the user can approve, in the agent UI, so tell
them and then use read_conversation with wait=true. Never pretend an approval happened."""


def build_server(bridge: AgentBridge, *, name: str, agent_id: str) -> MCPServer:
    server = MCPServer(name="apiagent", instructions=INSTRUCTIONS.format(name=name, agent_id=agent_id))

    async def call(fn, *args):
        try:
            return await anyio.to_thread.run_sync(fn, *args)
        except BridgeError as err:
            raise ToolError(str(err)) from err

    @server.tool()
    async def ask_agent(message: str, conversation_id: str = "") -> str:
        """Send a message to the agent and wait for its reply. Leave conversation_id empty to start a new chat;
        pass the id from an earlier reply to continue that one. Returns the tools it ran and its reply."""
        return await call(bridge.ask, message, conversation_id)

    @server.tool()
    async def list_conversations(limit: int = 20) -> str:
        """List the agent's chats with the user, newest first, with their ids and whether they're running or waiting for approval."""
        return await call(bridge.conversations, limit)

    @server.tool()
    async def read_conversation(conversation_id: str, last: int = 20, wait: bool = False) -> str:
        """Read the last messages of a chat. With wait=true, first wait for a turn that is still running, e.g. after the user approved something."""
        return await call(bridge.read, conversation_id, last, wait)

    @server.tool()
    async def agent_status() -> str:
        """The agent's identity, what's waiting for the user's approval, and its recent events and emails."""
        return await call(bridge.status)

    return server
