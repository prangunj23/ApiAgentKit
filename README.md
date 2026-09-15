# agentkit (ApiAgentKit)

A framework for chat agents that each maintain one service repo. An agent is defined by an `AgentSpec`, and agentkit supplies everything else:

- a tool loop on NVIDIA NIM, with user approval for actions that reach outside
- SQLite history covering chats, agent-to-agent threads, emails, events, feedback, and learnings
- the agent's own clones of its repos
- memory, a research folder, and a codebase map that keeps itself current
- self-improvement from feedback and PR outcomes
- one HTTP API that the UI ([ApiAgentUI](../ApiAgentUI)) uses for every agent

## This repo is the source of truth, not a dependency

Each service repo deploys on its own server, so it can't depend on this directory being next to it. Instead every service carries a copy of agentkit at `chat_agent/vendor/agentkit/`, committed to that repo, and installs it from there:

```toml
[tool.uv.sources]
agentkit = { path = "vendor/agentkit", editable = true }
```

The install is editable on purpose. A copied install of a local package keeps its old code after a sync: uv reuses its build while the version number is unchanged, so a redeploy would go on serving the previous agentkit.

**Always change agentkit here**, run the tests here, then push the copies out:

```sh
uv run pytest                          # agentkit's own tests live only in this repo
uv run scripts/sync_vendor.py          # copy src/agentkit -> every service's vendor/agentkit
uv run scripts/sync_vendor.py --check  # report drift instead of copying (exit 1 if any)
```

A sync rewrites `vendor/agentkit/` wholesale, so an edit made inside a service's `vendor/` is lost at the next sync. Each service's CI guards against that: the sync writes a `VENDOR.json` of sha256 digests alongside a standalone `vendor/check_vendor.py`, and the service's `chat-agent` job runs it to catch a copy that was edited in place. That check needs no access to this repo, which is what makes the service repos self-contained.

After syncing, commit the changed `vendor/agentkit/` in each service repo and re-run its `uv lock` if agentkit's dependencies changed.

## Define an agent

```python
from agentkit import AgentSpec, RepoRef
from agentkit.tools.email import send_email

SPEC = AgentSpec(
    id="service2",
    name="Consumer (Service2)",
    repo=RepoRef("prangunj23/ApiAgentService2"),   # cloned read-write; edits limited to src/ and tests/
    reads=[RepoRef("prangunj23/ApiAgentService1")], # cloned read-only, as a sibling directory
    system_prompt="You maintain ApiAgentService2 ...",
    tools=[send_email],                             # added to the generic tools
    features={"emails"},                            # tells the UI to show the Emails tab
)
```

Every agent gets these generic tools:
- **Repo:** `list_files`, `read_file`, `search_code`, `git_log`, `git_diff`, `git_status`, `write_file`, `revert_changes`, `run_tests`, `sync_repo`
- **Pull requests:** `open_pull_request` (needs approval)
- **Other agents:** `message_agent`
- **Research:** `save_research`, `list_research`, `read_research`
- **Memory:** `remember`, `forget`, `update_codebase_notes`
- **Learning:** `propose_lesson`, `propose_skill`, `read_skill`

Service-specific tools use the `@tool` decorator:

```python
from agentkit import ToolContext, string, tool

@tool("notify", "Tell the team.", {"text": string("Message.")}, required=("text",), needs_confirmation=True)
def notify(ctx: ToolContext, text: str) -> str:
    ...
```

## Commands

```sh
uv run agentkit serve service1_agent.spec:SPEC --port 9001        # one agent (reads .env in the current directory)
uv run agentkit dev --registry ../ApiAgentUI/public/registry.json # every agent with a `local` entry
uv run agentkit init --id service3 --repo owner/ApiAgentService3 \
    --reads owner/ApiAgentService1 --dir ../ApiAgentService3/chat_agent \
    --registry ../ApiAgentUI/public/registry.json                # scaffold a new agent and register it
uv run agentkit openapi --out openapi.json                       # API schema for the UI's generated types
```

## Configuration

Each agent reads its own `.env`:

| Variable | Purpose |
|---|---|
| `NVIDIA_API_KEY` | NIM API key (required to chat) |
| `NIM_MODEL` | Default `moonshotai/kimi-k3` |
| `GITHUB_TOKEN` | Fine-grained token: Contents and Pull requests (read and write) on the agent's repo, and Contents (read) on the repos it reads |
| `AGENT_REGISTRY` | Path or URL of `registry.json`. Set by `agentkit dev` |
| `AGENT_SHARED_TOKEN` | Secret the agents use on agent-to-agent routes. Set by `agentkit dev` |
| `AGENT_HOME` | Folder holding one data folder per agent. Default `~/.apiagent` |
| `AGENT_DATA_DIR` | This agent's data folder. Default `$AGENT_HOME/<id>` |
| `UI_ORIGINS` | Browser origins allowed to call the agent. Default `http://localhost:5173` |
| `AUTO_APPROVE_LEARNINGS` | `true` activates proposed lessons and skills without review. Default `false` |
| `EMAIL_TO`, `EMAIL_PROVIDER`, `EMAIL_FROM`, `RESEND_API_KEY`, `SMTP_*` | Needed by `send_email` |

## Data

Everything an agent keeps lives outside the repos:

```
~/.apiagent/<id>/
  agent.db                     conversations, messages, emails, events, feedback, learnings, PR outcomes
  workspace/<repo>/            the agent's clones (never your dev checkout)
  research/codebase-map.md     current structure, important files, public contract, run/test/config
  research/<date>-<slug>.md    research the agent saved
  memory/notes.md              facts the agent chose to remember
  memory/recent.md             one line per recent conversation or event (maintained automatically)
  memory/lessons.md            approved lessons
  skills/<name>.md             approved skills
```

**Automatic updates:**
- **Codebase map:** rebuilt when new commits reach `origin/main` (checked every 5 minutes) and after the agent edits files. The LLM only re-describes files that changed.
- **Recent activity:** one summary line per conversation, written about 60 seconds after each turn.
- **Pull requests:** checked every 15 minutes. Merges, closes, and review comments start a reflection, and its proposals wait in the UI for approval.

## HTTP API

All agents serve the same routes, so the UI needs only each agent's URL. Run `agentkit openapi` for the full schema.

| Area | Routes |
|---|---|
| Info | `GET /health`, `GET /api/info` |
| Conversations | `GET/POST /api/conversations`, `GET /api/conversations/{id}`, `POST …/{id}/messages` (SSE), `POST …/{id}/confirm` (SSE), `GET …/{id}/stream` (SSE: replay and follow a turn in progress), `GET /api/pending` |
| Agent-to-agent (needs `X-Agent-Token`) | `POST /api/agent-messages`, `POST /api/agent-messages/reply`, `POST /api/events/inbound` |
| Data | `GET /api/emails[/{id}]`, `GET /api/events`, `GET /api/research[/{file}]`, `GET /api/memory`, `GET /api/codebase-map`, `POST /api/codebase-map/rebuild` |
| Learning | `POST /api/messages/{id}/feedback`, `GET /api/learnings`, `POST /api/learnings/{id}`, `GET /api/pr-outcomes` |
| Workspace | `GET /api/workspace`, `POST /api/workspace/sync` |

## Security

The agents have no login. Keep them on localhost:
- **Network:** they bind to `127.0.0.1` only.
- **Host header:** requests whose `Host` isn't localhost on the agent's port are rejected, which stops DNS rebinding.
- **Origin:** writes from origins outside `UI_ORIGINS` are rejected, which stops cross-site POSTs.
- **Agent routes:** need the shared token.
- **Secrets:** redacted from tool output.

## Test

```sh
uv sync
uv run pytest
```

The tests use `agentkit.testing`: a scripted `FakeLLM`, local bare git repos in place of GitHub, and in-process HTTP between agents. They need no network access or API keys.
