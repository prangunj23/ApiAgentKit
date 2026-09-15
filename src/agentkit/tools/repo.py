from agentkit.spec import ToolContext, ToolError, integer, string, tool
from agentkit.workspace import Repo, RepoBusyError

REPO = string("Repo name or owner/name. Defaults to your own repo.")


def _repo(ctx: ToolContext, name: str = "") -> Repo:
    repo = ctx.agent.workspace.repo(name)
    if not repo.exists():
        raise ToolError(f"{repo.name} isn't cloned yet. Try again in a minute.")
    return repo


@tool("list_files", "List files in a repo, optionally under a directory.", {"repo": REPO, "path": string("Directory relative to the repo root.")})
def list_files(ctx: ToolContext, repo: str = "", path: str = "") -> str:
    return "\n".join(_repo(ctx, repo).list_files(path)) or "No files."


@tool("read_file", "Read a file from a repo.", {"path": string("File path relative to the repo root."), "repo": REPO}, required=("path",), max_result_chars=40_000)
def read_file(ctx: ToolContext, path: str, repo: str = "") -> str:
    return _repo(ctx, repo).read_file(path)


@tool(
    "search_code",
    "Search a repo with git grep (basic regular expressions). Returns file:line:text matches.",
    {"pattern": string("Pattern to search for."), "repo": REPO, "path": string("Limit to this path.")},
    required=("pattern",),
)
def search_code(ctx: ToolContext, pattern: str, repo: str = "", path: str = "") -> str:
    return _repo(ctx, repo).search(pattern, path)


@tool(
    "git_log",
    "Show recent commits.",
    {"repo": REPO, "revision_range": string("Optional range such as abc123..HEAD."), "limit": integer("Max commits (default 20).")},
)
def git_log(ctx: ToolContext, repo: str = "", revision_range: str = "", limit: int = 20) -> str:
    return _repo(ctx, repo).log(revision_range, min(max(limit, 1), 200)) or "No commits."


@tool(
    "git_diff",
    "Show a diff. With no commits, shows your uncommitted changes. With before/after, diffs those commits.",
    {"repo": REPO, "before": string("Base commit."), "after": string("Target commit (default HEAD)."), "path": string("Limit to this path.")},
    max_result_chars=60_000,
)
def git_diff(ctx: ToolContext, repo: str = "", before: str = "", after: str = "", path: str = "") -> str:
    return _repo(ctx, repo).diff(before, after, path)


@tool("git_status", "Show the branch and uncommitted changes.", {"repo": REPO})
def git_status(ctx: ToolContext, repo: str = "") -> str:
    return _repo(ctx, repo).status()


@tool(
    "write_file",
    "Create or overwrite a file in your own repo with its complete new content. Only src/ and tests/ can be edited.",
    {"path": string("File path relative to the repo root."), "content": string("The complete file content.")},
    required=("path", "content"),
)
def write_file(ctx: ToolContext, path: str, content: str) -> str:
    own = _repo(ctx)
    own.resolve(path, for_write=True)
    ctx.agent.workspace.claim(ctx.conversation_id)
    changed = own.write_file(path, content)
    ctx.wrote_files = True
    return f"Wrote {path} ({len(content.splitlines())} lines)." if changed else f"{path} already had that content."


@tool("revert_changes", "Discard every uncommitted change in your own repo.")
def revert_changes(ctx: ToolContext) -> str:
    workspace = ctx.agent.workspace
    own = _repo(ctx)
    if workspace.holder and workspace.holder != ctx.conversation_id:
        raise RepoBusyError(f"The uncommitted changes belong to conversation {workspace.holder}; revert them there.")
    own.discard_changes()
    workspace.release(ctx.conversation_id)
    ctx.wrote_files = True
    return f"Reverted all uncommitted changes in {own.name}."


@tool("run_tests", "Run the test suite of your own repo with `uv run pytest`.")
def run_tests(ctx: ToolContext) -> str:
    own = _repo(ctx)
    passed, output = own.run_tests()
    outcome = "passed" if passed else "failed"
    ctx.agent.log_event("tests_run", f"Tests {outcome} in {own.name}", conversation_id=ctx.conversation_id)
    return f"Tests {outcome.upper()}.\n\n{output}"


@tool("sync_repo", "Fetch the latest commits and fast-forward your checkout when it has no uncommitted changes.")
def sync_repo(ctx: ToolContext) -> str:
    workspace = ctx.agent.workspace
    if not _repo(ctx).is_dirty():
        workspace.release(ctx.conversation_id)
    return "\n".join(ctx.agent.sync_workspace()) or "Already up to date."
