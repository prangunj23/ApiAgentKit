"""Pull request review between developer agents.

A developer's agent asks other developer agents to review a PR (request_reviews). Each reviewer loads the change
(review_pull_request), emails its own developer a draft, and replies. The asking agent then emails its developer a
digest. Nothing reaches GitHub unless a developer approves post_review_comment.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import httpx

from agentkit.messaging import deliver
from agentkit.registry import PeerError
from agentkit.spec import ToolContext, ToolError, string, string_list, tool

if TYPE_CHECKING:
    from agentkit.agent import Agent

REVIEW_TIMEOUT = 900

# Budgets for each section of review_pull_request's result, in characters.
DESCRIPTION_CHARS = 2_000
CHECKS_CHARS = 1_000
FILES_CHARS = 2_000
DIFF_CHARS = 45_000
CONTENT_CHARS = 20_000
COMMENTS_CHARS = 5_000
MAX_CONTENT_LINES = 1_500
MAX_CONTENT_FILES = 30
MAX_FILE_ROWS = 100

SKIPPED_NAMES = {"uv.lock", "poetry.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock", "VENDOR.json"}
SKIPPED_PARTS = {".venv", "node_modules", "__pycache__", "vendor", "dist", "build"}


def skipped(path: str) -> bool:
    """Lock files, vendored copies and build output: noise for a reviewer."""
    parts = path.split("/")
    return parts[-1] in SKIPPED_NAMES or parts[-1].endswith((".lock", ".min.js")) or any(part in SKIPPED_PARTS for part in parts[:-1])


def clip(text: str, budget: int, note: str = "") -> str:
    if len(text) <= budget:
        return text
    return text[:budget] + f"\n…(truncated{': ' + note if note else ''})"


def _section(title: str, body: str) -> str:
    return f"## {title}\n{body.strip() or '(none)'}"


def _where_to_look(agent: "Agent", repo_slug: str) -> str:
    lines = []
    local = next((repo for repo in agent.workspace.repos() if repo.slug.lower() == repo_slug.lower()), None)
    if local:
        lines.append(
            f'- read_file / search_code with repo="{local.name}" for code the PR didn\'t change. '
            f"That checkout is at {local.ref.branch}, not the PR's head."
        )
    else:
        lines.append(f"- {repo_slug} isn't in your read-only repos, so only the files above are available.")
    for peer in agent.peers.others():
        if peer.kind != "service":
            continue
        try:
            slug = (agent.peers.info(peer.id).get("repo") or {}).get("slug", "")
        except PeerError:
            continue
        if slug.lower() == repo_slug.lower():
            lines.append(f'- read_research(agent="{peer.id}", file="codebase-map.md") for how {repo_slug} fits together.')
    return "\n".join(lines)


def build_review_context(agent: "Agent", pr_url: str) -> str:
    """Everything a reviewer is shown about a pull request, in a fixed order with a budget per section."""
    gh = agent.github
    repo, number = gh.parse_pr_url(pr_url)
    try:
        pr = gh.pull_request(pr_url)
        files = gh.pull_request_files(pr_url)
    except httpx.HTTPError as err:
        raise ToolError(f"Couldn't load {pr_url}: {err}") from err
    head_sha = pr["head"]["sha"]

    header = "\n".join(
        [
            f"- Repo: {repo}",
            f"- Number: #{number}",
            f"- Title: {pr.get('title', '')}",
            f"- Author: @{(pr.get('user') or {}).get('login', '')}",
            f"- State: {pr.get('state', '')}{' (draft)' if pr.get('draft') else ''}{' (merged)' if pr.get('merged') else ''}",
            f"- Branches: {pr['head'].get('ref', '')} → {pr['base'].get('ref', '')}",
            f"- Head commit: {head_sha}",
            f"- Created: {pr.get('created_at', '')}, updated: {pr.get('updated_at', '')}",
            f"- URL: {pr.get('html_url', pr_url)}",
        ]
    )

    try:
        runs = gh.check_runs(repo, head_sha)
        checks = "\n".join(f"- {run.get('name')}: {run.get('conclusion') or run.get('status')} ({run.get('html_url', '')})" for run in runs)
        checks = checks or "No checks reported."
    except httpx.HTTPError as err:
        checks = f"Checks unavailable: {err}"

    shown = [item for item in files if not skipped(item["filename"])]
    hidden = [item["filename"] for item in files if skipped(item["filename"])]
    rows = ["| File | Status | Changes |", "|---|---|---|"]
    rows += [f"| {item['filename']} | {item['status']} | +{item['additions']} / -{item['deletions']} |" for item in files[:MAX_FILE_ROWS]]
    if len(files) > MAX_FILE_ROWS:
        rows.append(f"| …and {len(files) - MAX_FILE_ROWS} more | | |")
    rows.append(
        f"\n{len(files)} file(s), +{sum(item['additions'] for item in files)} / -{sum(item['deletions'] for item in files)}."
    )
    if hidden:
        rows.append(f"Not shown below (lock files, vendored or generated): {', '.join(hidden)}.")

    diffs = []
    for item in shown:
        patch = item.get("patch")
        body = f"```diff\n{patch}\n```" if patch else "_No patch: the file is binary or too large._"
        diffs.append(f"### {item['filename']} ({item['status']})\n{body}")
    diff = clip("\n\n".join(diffs), DIFF_CHARS, "read the files for the rest")

    fetched, left_out = {}, []
    for item in [item for item in shown if item["status"] != "removed" and item.get("patch")][:MAX_CONTENT_FILES]:
        try:
            fetched[item["filename"]] = gh.file_at_ref(repo, item["filename"], head_sha)
        except httpx.HTTPError:
            left_out.append(item["filename"])
    contents, used = [], 0
    # Smallest first, so one large file can't use up the whole budget.
    for path, text in sorted(fetched.items(), key=lambda pair: len(pair[1])):
        block = f"### {path}\n```\n{text}\n```"
        if text.count("\n") > MAX_CONTENT_LINES or used + len(block) > CONTENT_CHARS:
            left_out.append(path)
            continue
        contents.append(block)
        used += len(block)
    if left_out:
        contents.append(f"Too large to include: {', '.join(left_out)}.")

    try:
        comments = gh.review_comments(pr_url)
        comment_text = "\n".join(
            f"- @{comment['author']}{' on ' + comment['path'] if comment.get('path') else ''}: {comment['body'].strip()}"
            for comment in comments
        )
    except httpx.HTTPError as err:
        comment_text = f"Comments unavailable: {err}"

    return "\n\n".join(
        [
            f"# Pull request {repo}#{number}",
            _section("Pull request", header),
            _section("Description", clip(pr.get("body") or "", DESCRIPTION_CHARS)),
            _section("Checks", clip(checks, CHECKS_CHARS)),
            _section("Files changed", clip("\n".join(rows), FILES_CHARS)),
            _section("Diff", diff),
            _section(f"Full content after the change (at {head_sha[:7]})", "\n\n".join(contents)),
            _section("Existing comments", clip(comment_text, COMMENTS_CHARS) or "No comments yet."),
            _section("Where to look next", _where_to_look(agent, repo)),
        ]
    )


@tool(
    "review_pull_request",
    "Load a pull request for review: description, checks, changed files, diff, the changed files' new content, "
    "and existing comments.",
    {"pr_url": string("The pull request URL, e.g. https://github.com/owner/repo/pull/42.")},
    required=("pr_url",),
    max_result_chars=80_000,
)
def review_pull_request(ctx: ToolContext, pr_url: str) -> str:
    return build_review_context(ctx.agent, pr_url)


def review_request_message(agent: "Agent", pr: dict[str, Any], pr_url: str, focus: str) -> str:
    person = (agent.peers.developer_of(agent.spec.id) or {}).get("name") or agent.spec.name
    repo, _ = agent.github.parse_pr_url(pr_url)
    lines = [
        f"{person} is asking for a review of a pull request.",
        "",
        f"- Pull request: {pr.get('html_url', pr_url)}",
        f"- Repo: {repo}",
        f"- Title: {pr.get('title', '')}",
        f"- Author: @{(pr.get('user') or {}).get('login', '')}",
    ]
    if focus.strip():
        lines.append(f"- Requested focus: {focus.strip()}")
    lines += [
        "",
        "Load the change with review_pull_request and review it. Send your review to your developer with "
        f"notify_developer, then reply here with the same review so {person}'s agent can summarize it for them.",
    ]
    return "\n".join(lines)


@tool(
    "request_reviews",
    "Ask other developers' agents to review a pull request and wait for their reviews. Each reviewer also emails "
    "its own developer. Afterwards, send your developer a digest with notify_developer.",
    {
        "pr_url": string("The pull request URL."),
        "reviewers": string_list("Developer agent ids to ask, e.g. dev-bob."),
        "focus": string("Optional: what the reviewers should look at most closely."),
    },
    required=("pr_url", "reviewers"),
    max_result_chars=60_000,
    top_level_only=True,
)
def request_reviews(ctx: ToolContext, pr_url: str, reviewers: list[str], focus: str = "") -> str:
    agent = ctx.agent
    if ctx.depth > 0:
        raise ToolError("You can't ask for reviews while answering another agent.")
    reviewers = list(dict.fromkeys(reviewer.strip() for reviewer in reviewers if reviewer.strip()))
    if not reviewers:
        raise ToolError("Name at least one reviewer.")
    peers = {}
    for reviewer in reviewers:
        if reviewer == agent.spec.id:
            raise ToolError("You can't review your own request.")
        try:
            peer = agent.peers.get(reviewer)
        except PeerError as err:
            raise ToolError(str(err)) from err
        if peer.kind != "developer":
            raise ToolError(f"{reviewer} isn't a developer's agent.")
        peers[reviewer] = peer
    try:
        pr = agent.github.pull_request(pr_url)
    except httpx.HTTPError as err:
        raise ToolError(f"Couldn't load {pr_url}: {err}") from err
    message = review_request_message(agent, pr, pr_url, focus)

    def ask(reviewer: str) -> dict[str, Any]:
        try:
            return deliver(agent, reviewer, message, parent_conversation_id=ctx.conversation_id, timeout=REVIEW_TIMEOUT)
        except PeerError as err:
            return {"status": "error", "reply": "", "error": str(err)}

    with ThreadPoolExecutor(max_workers=len(reviewers)) as pool:
        results = dict(zip(reviewers, pool.map(ask, reviewers), strict=True))

    sections = []
    for reviewer, result in results.items():
        peer = peers[reviewer]
        person = (peer.developer or {}).get("name", reviewer)
        heading = f"### From {peer.name} ({person})"
        if result["status"] in {"replied", "duplicate"} and result.get("reply"):
            sections.append(f"{heading}\n{result['reply']}")
        elif result["status"] == "awaiting_approval":
            sections.append(f"{heading}\n_Waiting for {person} to approve an action; the review will arrive in your thread with {reviewer}._")
        else:
            sections.append(f"{heading}\n_No review: {result.get('error') or 'unknown error'}_")
    done = sum(result["status"] in {"replied", "duplicate"} for result in results.values())
    agent.log_event(
        "reviews_requested",
        f"Asked {', '.join(reviewers)} to review {pr.get('title', pr_url)}; {done} replied",
        url=pr.get("html_url", pr_url),
        conversation_id=ctx.conversation_id,
    )
    return (
        "\n\n".join(sections)
        + "\n\nNext: send your developer one digest of these reviews with notify_developer, naming anything that "
        "would block the merge."
    )


@tool(
    "post_review_comment",
    "Post a comment on a pull request, on behalf of your developer. Only do this when they ask. They must approve.",
    {"pr_url": string("The pull request URL."), "body_markdown": string("The comment, in markdown.")},
    required=("pr_url", "body_markdown"),
    needs_confirmation=True,
)
def post_review_comment(ctx: ToolContext, pr_url: str, body_markdown: str) -> str:
    agent = ctx.agent
    try:
        url = agent.github.create_issue_comment(pr_url, _signed(ctx, body_markdown))
    except httpx.HTTPError as err:
        raise ToolError(f"Couldn't comment on {pr_url}: {err}") from err
    agent.log_event("review_posted", f"Commented on {pr_url}", url=url, conversation_id=ctx.conversation_id)
    return f"Posted: {url}"


def _signed(ctx: ToolContext, body_markdown: str) -> str:
    agent = ctx.agent
    github = (agent.peers.developer_of(agent.spec.id) or {}).get("github")
    on_behalf = f" on behalf of @{github}" if github else ""
    return f"{body_markdown.rstrip()}\n\n_Posted by {agent.spec.name}{on_behalf}._"


def _preview(ctx: ToolContext, pr_url: str, body_markdown: str) -> str:
    return f"Comment on {pr_url}\n\n{_signed(ctx, body_markdown)}"


post_review_comment.preview = _preview

REVIEW_TOOLS = [review_pull_request, request_reviews, post_review_comment]
