from agentkit.registry import PeerError
from agentkit.spec import ToolContext, ToolError, string, string_list, tool


@tool(
    "save_research",
    "Save findings to your research folder as a markdown file, so you and other agents can use them later.",
    {
        "title": string("Short title."),
        "content_markdown": string("The findings, in markdown."),
        "sources": string_list("Files, commits, or URLs the findings are based on."),
    },
    required=("title", "content_markdown"),
)
def save_research(ctx: ToolContext, title: str, content_markdown: str, sources: list[str] | None = None) -> str:
    name = ctx.agent.research.save(title, content_markdown, conversation_id=ctx.conversation_id, sources=sources)
    ctx.agent.log_event("research_saved", f"Saved research: {title}", conversation_id=ctx.conversation_id)
    return f"Saved research/{name}"


@tool("list_research", "List research files. Pass another agent's id to list theirs.", {"agent": string("Agent id; defaults to you.")})
def list_research(ctx: ToolContext, agent: str = "") -> str:
    if agent and agent != ctx.agent.spec.id:
        try:
            items = ctx.agent.peers.request(agent, "GET", "/api/research")
        except PeerError as err:
            raise ToolError(str(err)) from err
    else:
        items = ctx.agent.research.list()
    return "\n".join(f"- {item['file']}: {item['title']} (updated {item['updated_at']})" for item in items) or "No research files."


@tool(
    "read_research",
    "Read a research file. Pass another agent's id to read theirs, e.g. their codebase-map.md.",
    {"file": string("File name, e.g. codebase-map.md."), "agent": string("Agent id; defaults to you.")},
    required=("file",),
    max_result_chars=40_000,
)
def read_research(ctx: ToolContext, file: str, agent: str = "") -> str:
    if agent and agent != ctx.agent.spec.id:
        ctx.agent.research.path(file)
        try:
            return ctx.agent.peers.request(agent, "GET", f"/api/research/{file}")["content"]
        except PeerError as err:
            raise ToolError(str(err)) from err
    return ctx.agent.research.read(file)
