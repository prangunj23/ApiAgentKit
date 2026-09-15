from agentkit.spec import ToolContext, string, tool


@tool("remember", "Add a durable fact to your memory notes. They are shown to you in every conversation.", {"note": string("One fact.")}, required=("note",))
def remember(ctx: ToolContext, note: str) -> str:
    ctx.agent.memory.remember(note)
    return "Remembered."


@tool("forget", "Remove memory notes that contain the given text.", {"text": string("Text to match.")}, required=("text",))
def forget(ctx: ToolContext, text: str) -> str:
    return f"Removed {ctx.agent.memory.forget(text)} note(s)."


@tool(
    "update_codebase_notes",
    "Replace the 'Notes & gotchas' section of your codebase map. It survives map regeneration.",
    {"markdown": string("The complete new notes section.")},
    required=("markdown",),
)
def update_codebase_notes(ctx: ToolContext, markdown: str) -> str:
    ctx.agent.codemap.set_notes(markdown)
    return "Updated the codebase map notes."
