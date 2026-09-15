from agentkit.spec import ToolContext, ToolError, boolean, string, tool
from agentkit.workspace import RepoBusyError


def _branch(ctx: ToolContext) -> str:
    return f"agent/chat-{ctx.conversation_id[:8]}"


@tool(
    "open_pull_request",
    "Commit your uncommitted changes under src/ and tests/ to a new branch, push it, and open a pull request. "
    "Run the tests first and include the results in the body.",
    {
        "title": string("Pull request title."),
        "body": string("Markdown description: what changed, why, and test results."),
        "draft": boolean("Open as a draft (for example when tests fail)."),
    },
    required=("title", "body"),
    needs_confirmation=True,
)
def open_pull_request(ctx: ToolContext, title: str, body: str, draft: bool = False) -> str:
    agent, workspace = ctx.agent, ctx.agent.workspace
    own = workspace.own
    if workspace.holder and workspace.holder != ctx.conversation_id:
        raise RepoBusyError(f"The uncommitted changes belong to conversation {workspace.holder}.")
    paths = [path for path in own.changed_paths() if path.split("/")[0] in own.ref.editable_dirs or path == "uv.lock"]
    if not paths:
        raise ToolError(f"There are no changes under {', '.join(own.ref.editable_dirs)} to commit.")

    branch = _branch(ctx)
    own.commit_and_push(branch, title, paths)
    try:
        url = agent.github.create_pull_request(
            own.slug,
            head=branch,
            base=own.ref.branch,
            title=title,
            body=f"{body}\n\n_Opened by the {agent.spec.name} chat agent._",
            draft=draft,
        )
    finally:
        # The commit is on the pushed branch; return the checkout to a clean main.
        own.reset_to_origin()
        workspace.release(ctx.conversation_id)
        ctx.wrote_files = True
    agent.store.upsert_pr_outcome(url, state="open", conversation_id=ctx.conversation_id)
    agent.log_event("pr_opened", f"Opened PR: {title}", url=url, conversation_id=ctx.conversation_id)
    return f"Opened pull request: {url}"


def _preview(ctx: ToolContext, title: str, body: str, draft: bool = False) -> str:
    own = ctx.agent.workspace.own
    target = f"{_branch(ctx)} → {own.ref.branch}{' (draft)' if draft else ''}"
    return f"Repo: {own.slug}\nBranch: {target}\n\n{own.diffstat() or 'No changes.'}"


open_pull_request.preview = _preview
