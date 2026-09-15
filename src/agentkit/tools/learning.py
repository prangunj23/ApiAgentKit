from agentkit.spec import ToolContext, ToolError, string, tool


@tool(
    "propose_lesson",
    "Propose a general rule for how you should work in future. The user approves it before it takes effect.",
    {"title": string("Short title."), "rule": string("The rule."), "why": string("The evidence for it.")},
    required=("title", "rule", "why"),
)
def propose_lesson(ctx: ToolContext, title: str, rule: str, why: str) -> str:
    ctx.agent.learning.propose("lesson", title, rule, why, ctx.conversation_id)
    return "Proposed. It takes effect once the user approves it."


@tool(
    "propose_skill",
    "Propose a reusable step-by-step procedure for a kind of task. The user approves it before it takes effect.",
    {
        "name": string("Short name."),
        "when_to_use": string("When this skill applies."),
        "steps_markdown": string("Numbered steps in markdown."),
    },
    required=("name", "when_to_use", "steps_markdown"),
)
def propose_skill(ctx: ToolContext, name: str, when_to_use: str, steps_markdown: str) -> str:
    body = f"When to use: {when_to_use}\n\n{steps_markdown}"
    ctx.agent.learning.propose("skill", name, body, f"Proposed during conversation {ctx.conversation_id}", ctx.conversation_id)
    return "Proposed. It takes effect once the user approves it."


@tool("read_skill", "Load the full steps of one of your active skills.", {"name": string("Skill name.")}, required=("name",))
def read_skill(ctx: ToolContext, name: str) -> str:
    skill = ctx.agent.learning.find_skill(name)
    if skill is None:
        names = ", ".join(item["title"] for item in ctx.agent.learning.active("skill")) or "none"
        raise ToolError(f"No active skill named {name!r}. Active skills: {names}")
    return f"# {skill['title']}\n\n{skill['body']}"
