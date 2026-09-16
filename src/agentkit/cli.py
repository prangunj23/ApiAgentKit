"""agentkit command line: serve an agent, run every local agent, onboard developers, show what a model is given,
connect an assistant over MCP, export the API schema, or scaffold a new agent."""

import argparse
import importlib
import json
import logging
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from string import Template
from urllib.parse import urlparse

from agentkit.config import Settings, load_dotenv
from agentkit.registry import load_registry, validate_registry
from agentkit.spec import AgentSpec, RepoRef

KIT_ROOT = Path(__file__).resolve().parents[2]


def load_spec(target: str) -> AgentSpec:
    module_name, _, attribute = target.partition(":")
    spec = getattr(importlib.import_module(module_name), attribute or "SPEC")
    if not isinstance(spec, AgentSpec):
        raise SystemExit(f"{target} is not an AgentSpec")
    return spec


def check_registry(source: str) -> None:
    """Refuse to start with a registry whose links or owners are inconsistent: messages would be misrouted."""
    if not source or source.startswith(("http://", "https://")):
        return
    problems = validate_registry(load_registry(source))
    if problems:
        raise SystemExit(f"{source} is inconsistent:\n" + "\n".join(f"- {problem}" for problem in problems))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from agentkit.server import create_app

    load_dotenv(Path.cwd() / ".env")
    spec = load_spec(args.spec)
    settings = Settings.from_env(spec.id, args.port)
    check_registry(settings.registry)
    if args.host:
        settings.host = args.host
    if settings.host not in {"127.0.0.1", "localhost", "::1"}:
        print(f"WARNING: {spec.id} has no login and is listening on {settings.host}. Only do this on a private network.", file=sys.stderr)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s {spec.id} %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(spec, settings), host=settings.host, port=settings.port, log_level="info")


def _relay(name: str, process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{name}] {line}", end="", flush=True)


def _spawn(registry: Path, token: str, agent_id: str, url: str, local: dict) -> subprocess.Popen[str]:
    project = (registry.parent / local["path"]).resolve()
    port = str(local.get("port") or urlparse(url).port)
    # The onboarding token is for the onboarding service only; the agents never get it.
    env = {key: value for key, value in os.environ.items() if not key.startswith("ONBOARDING_") and key != "VIRTUAL_ENV"}
    env |= {"AGENT_REGISTRY": str(registry), "AGENT_SHARED_TOKEN": token}
    command = ["uv", "run", "--project", str(project), "agentkit", "serve", local["spec"], "--port", port]
    process = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    threading.Thread(target=_relay, args=(agent_id, process), daemon=True).start()
    print(f"Starting {agent_id} at {url} from {project}", flush=True)
    return process


def _developer_project(entries: list) -> str | None:
    """The `local.path` of the developer agents' project, from an existing developer entry."""
    return next((entry.local["path"] for entry in entries if entry.local and entry.local.get("spec", "").startswith("dev_agent.")), None)


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def cmd_dev(args: argparse.Namespace) -> None:
    # Onboarding settings (and shared email settings) from the .env where `agentkit dev` runs, e.g. ApiAgentUI/.env.
    load_dotenv(Path.cwd() / ".env")
    registry = Path(args.registry).resolve()
    check_registry(str(registry))
    all_entries = load_registry(str(registry))
    entries = [entry for entry in all_entries if entry.local]
    if not entries:
        raise SystemExit(f"No agents with a `local` entry in {registry}")
    token = os.environ.get("AGENT_SHARED_TOKEN") or secrets.token_urlsafe(24)
    # Stop the agents on SIGTERM too (a closed terminal, `kill`), not only on Ctrl+C, so none are left running.
    signal.signal(signal.SIGTERM, _interrupt)

    processes: list[tuple[str, subprocess.Popen[str]]] = []
    # Agents started by onboarding while this runs. One of them failing doesn't stop the rest.
    added: list[tuple[str, subprocess.Popen[str]]] = []
    for entry in entries:
        processes.append((entry.id, _spawn(registry, token, entry.id, entry.url, entry.local or {})))

    if args.admin_port:
        _start_local_admin(args, registry, all_entries, lambda entry: _start_added(registry, token, entry, added))

    try:
        while all(process.poll() is None for _, process in processes):
            time.sleep(0.5)
        for name, process in processes:
            if process.poll() is not None:
                print(f"{name} exited with code {process.returncode}; stopping the others.")
    except KeyboardInterrupt:
        pass
    finally:
        # A second Ctrl+C or SIGTERM (uv run forwards one too) must not cut the cleanup short and orphan agents.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        everything = processes + added
        for _, process in everything:
            if process.poll() is None:
                process.terminate()
        for _, process in everything:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def _start_added(registry: Path, token: str, entry: dict, added: list) -> str:
    if not entry.get("local"):
        return f"{entry['id']} has no local block, so start it yourself."
    added.append((entry["id"], _spawn(registry, token, entry["id"], entry["url"], entry["local"])))
    return f"Started {entry['id']} at {entry['url']}. It takes a few seconds to come up."


def _start_local_admin(args: argparse.Namespace, registry: Path, entries: list, starter) -> None:
    import uvicorn

    from agentkit.onboarding import OnboardingConfig, create_admin_app

    project = args.developers_project or _developer_project(entries)
    if project is None:
        print("Onboarding is off: no developer agent in the registry to copy from; pass --developers-project.")
        return
    config = OnboardingConfig(
        registry_path=registry,
        developers_dir=(registry.parent / project / "developers").resolve(),
        agent_url_base="http://127.0.0.1",
        local_project=project,
        default_reads=_env_list("ONBOARDING_READS"),
        github_admin_token=os.environ.get("ONBOARDING_GITHUB_TOKEN", "").strip(),
        allowed_hosts={"localhost", "127.0.0.1", f"localhost:{args.admin_port}", f"127.0.0.1:{args.admin_port}"},
        starter=starter,
        mcp_hint=f"uv run --project {KIT_ROOT} --extra mcp agentkit mcp --agent {{agent_id}} --registry {{registry}}",
    )
    app = create_admin_app(config)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.admin_port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="onboarding").start()
    print(f"Onboarding at http://127.0.0.1:{args.admin_port} (registry {registry})", flush=True)


def _env_list(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def cmd_admin(args: argparse.Namespace) -> None:
    """The deployed onboarding service. It owns registry.json and serves it to every agent at /registry.json."""
    import uvicorn

    from agentkit.onboarding import OnboardingConfig, create_admin_app

    load_dotenv(Path.cwd() / ".env")
    registry = Path(args.registry).resolve()
    check_registry(str(registry))
    host = args.host or os.environ.get("ADMIN_HOST", "127.0.0.1")
    config = OnboardingConfig(
        registry_path=registry,
        developers_dir=Path(args.developers_dir).resolve(),
        agent_url_base=args.agent_url_base,
        default_reads=_env_list("ONBOARDING_READS"),
        ui_url=os.environ.get("AGENTKIT_UI_URL", "http://localhost:5173"),
        ui_origins=_env_list("UI_ORIGINS") or ["http://localhost:5173", "http://127.0.0.1:5173"],
        allowed_hosts={"localhost", "127.0.0.1", f"localhost:{args.port}", f"127.0.0.1:{args.port}", *_env_list("ALLOWED_HOSTS")},
        github_admin_token=os.environ.get("ONBOARDING_GITHUB_TOKEN", "").strip(),
        mcp_hint=args.mcp_hint,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s onboarding %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_admin_app(config), host=host, port=args.port, log_level="info")


def cmd_inspect(args: argparse.Namespace) -> None:
    """Print exactly what the model would be given, without running a turn."""
    from agentkit.agent import Agent
    from agentkit.tools.review import build_review_context

    load_dotenv(Path.cwd() / ".env")
    spec = load_spec(args.spec)
    settings = Settings.from_env(spec.id, args.port)
    settings.background = False
    agent = Agent(spec, settings)
    if args.review_context:
        print(build_review_context(agent, args.review_context))
        return
    print(agent.system_prompt(args.depth))
    print("\n# Tools the model can call", f"(depth {args.depth})")
    for item in agent.tools_for(args.depth):
        print(f"- {item.name}{' (needs approval)' if item.needs_confirmation else ''}: {item.description}")


def cmd_mcp(args: argparse.Namespace) -> None:
    """Serve an MCP server on stdio for one agent. Stdout carries the protocol, so nothing else may print there."""
    import httpx

    try:
        from agentkit.mcp_server import build_server
    except ImportError as err:
        raise SystemExit(f"The mcp extra isn't installed ({err}). Run with: uv run --extra mcp agentkit mcp ...") from err
    from agentkit.bridge import AgentBridge

    source = args.registry or os.environ.get("AGENT_REGISTRY", "")
    entry = next((item for item in load_registry(source) if item.id == args.agent), None) if source else None
    url = args.url or (entry.url if entry else "")
    if not url:
        raise SystemExit(f"No URL for agent {args.agent!r}: pass --url, or --registry with an entry for it.")
    client = httpx.Client(base_url=url, timeout=30)
    bridge = AgentBridge(client, ui_url=args.ui_url, agent_id=args.agent)
    build_server(bridge, name=entry.name if entry else args.agent, agent_id=args.agent).run("stdio")


def cmd_openapi(args: argparse.Namespace) -> None:
    from agentkit.server import create_app
    from agentkit.tools.email import send_email

    spec = AgentSpec(
        id="example", name="Example", repo=RepoRef("owner/example"), system_prompt="", tools=[send_email], features={"emails"}
    )
    with tempfile.TemporaryDirectory() as data_dir:
        schema = create_app(spec, Settings(data_dir=Path(data_dir), background=False)).openapi()
    text = json.dumps(schema, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)


PYPROJECT = Template('''[project]
name = "$project_name"
version = "0.1.0"
description = "Chat agent for $repo_slug"
requires-python = ">=3.13"
dependencies = [
    "agentkit",
]

[dependency-groups]
dev = [
    "pytest>=8",
]

[tool.uv.sources]
$agentkit_source

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/$package"]

[tool.pytest.ini_options]
testpaths = ["tests"]
''')

SPEC_PY = Template('''from agentkit import AgentSpec, RepoRef

from $package.tools import TOOLS

PROMPT = """You are the maintainer agent for $repo_slug.
Help the user understand and change this repo. Read the code before answering, keep changes small, run the
tests before proposing a pull request, and say which files you changed and why."""

SPEC = AgentSpec(
    id="$agent_id",
    name="$name",
    repo=RepoRef("$repo_slug"),
    reads=[$reads],
    system_prompt=PROMPT,
    tools=TOOLS,
    features=set(),
)
''')

TOOLS_PY = '''"""Tools specific to this service. The agent already has agentkit's generic tools (files, git, tests, PRs, research, memory).

Example:

    from agentkit import ToolContext, string, tool

    @tool("lookup_status", "Describe what the tool does.", {"name": string("What to look up.")}, required=("name",))
    def lookup_status(ctx: ToolContext, name: str) -> str:
        return "..."
"""

from agentkit import Tool

TOOLS: list[Tool] = []
'''

TEST_PY = Template('''from fastapi.testclient import TestClient

from agentkit.server import create_app
from agentkit.testing import FakeLLM, make_settings
from $package.spec import SPEC


def test_info_describes_this_agent(tmp_path):
    client = TestClient(create_app(SPEC, make_settings(tmp_path), llm=FakeLLM()))
    info = client.get("/api/info").json()
    assert info["id"] == "$agent_id"
    assert info["repo"]["slug"] == "$repo_slug"
''')

ENV_EXAMPLE = Template('''# Copy to .env. `agentkit dev` sets AGENT_REGISTRY and AGENT_SHARED_TOKEN for you.
NVIDIA_API_KEY=
# NIM_MODEL=moonshotai/kimi-k3

# Fine-grained token with Contents and Pull requests (read and write) on $repo_slug,
# and Contents (read) on the repos this agent reads.
GITHUB_TOKEN=

# AGENT_DATA_DIR=~/.apiagent/$agent_id
# UI_ORIGINS=http://localhost:5173
# AUTO_APPROVE_LEARNINGS=false
''')

README_MD = Template('''# $name chat agent

The `$agent_id` agent for $repo_slug, built on [agentkit]($agentkit_link).

```sh
cp .env.example .env   # add NVIDIA_API_KEY and GITHUB_TOKEN
uv sync
uv run pytest
```

Run it with every other agent in the registry:

```sh
uv run --project $agentkit_rel agentkit dev --registry <path to ApiAgentUI/public/registry.json>
```

Edit `src/$package/spec.py` to change the prompt and `src/$package/tools.py` to add service-specific tools.
''')


def cmd_init(args: argparse.Namespace) -> None:
    target = Path(args.dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{target} already exists and isn't empty")
    package = re.sub(r"\W", "_", args.id) + "_agent"
    name = args.name or args.id

    if (KIT_ROOT / "pyproject.toml").is_file():
        agentkit_rel = os.path.relpath(KIT_ROOT, target)
        agentkit_source = f'agentkit = {{ path = "{agentkit_rel}", editable = true }}'
    else:
        agentkit_rel = "../ApiAgentKit"
        agentkit_source = '# agentkit = { path = "../../ApiAgentKit", editable = true }'

    values = {
        "agent_id": args.id,
        "name": name,
        "package": package,
        "project_name": f"{args.id.replace('_', '-')}-agent",
        "repo_slug": args.repo,
        "reads": ", ".join(f'RepoRef("{slug}")' for slug in args.reads or []),
        "agentkit_source": agentkit_source,
        "agentkit_rel": agentkit_rel,
        "agentkit_link": agentkit_rel,
    }
    files = {
        "pyproject.toml": PYPROJECT.substitute(values),
        "README.md": README_MD.substitute(values),
        ".env.example": ENV_EXAMPLE.substitute(values),
        ".gitignore": ".venv\n__pycache__\n.pytest_cache\n.env\n",
        f"src/{package}/__init__.py": "",
        f"src/{package}/spec.py": SPEC_PY.substitute(values),
        f"src/{package}/tools.py": TOOLS_PY,
        "tests/test_spec.py": TEST_PY.substitute(values),
    }
    for rel, content in files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(f"Created {target}")

    entry = {
        "id": args.id,
        "name": name,
        "url": f"http://127.0.0.1:{args.port}",
        "local": {"spec": f"{package}.spec:SPEC", "port": args.port},
    }
    if args.registry:
        registry = Path(args.registry).resolve()
        entry["local"] = {"path": os.path.relpath(target, registry.parent), **entry["local"]}
        entries = json.loads(registry.read_text()) if registry.exists() else []
        if any(existing["id"] == args.id for existing in entries):
            print(f"{registry} already lists {args.id}; left it unchanged.")
        else:
            registry.write_text(json.dumps([*entries, entry], indent=2) + "\n")
            print(f"Added {args.id} to {registry}")
    else:
        print("Add this to registry.json (set local.path relative to the registry file):")
        print(json.dumps(entry, indent=2))
    print(f"Next: cd {target} && cp .env.example .env && uv sync && uv run pytest")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agentkit", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Run one agent")
    serve.add_argument("spec", help="module:ATTRIBUTE of the AgentSpec, e.g. service1_agent.spec:SPEC")
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--host", help="Default 127.0.0.1 (or AGENT_HOST)")

    dev = commands.add_parser("dev", help="Run every agent that has a `local` entry in the registry, and onboarding")
    dev.add_argument("--registry", required=True)
    dev.add_argument("--admin-port", type=int, default=9100, help="Port for the onboarding service; 0 turns it off")
    dev.add_argument("--developers-project", help="Developer agents' project, relative to the registry (default: from its entries)")

    admin = commands.add_parser("admin", help="Run the onboarding service, which owns and serves registry.json")
    admin.add_argument("--registry", required=True, help="The registry.json this service owns")
    admin.add_argument("--developers-dir", required=True, help="Where developer files are written")
    admin.add_argument("--agent-url-base", required=True, help="Scheme and host of new developer agents, e.g. http://apiagent-devs")
    admin.add_argument("--port", type=int, default=9100)
    admin.add_argument("--host", help="Default 127.0.0.1 (or ADMIN_HOST)")
    admin.add_argument(
        "--mcp-hint",
        default="uv run --project <path to ApiAgentKit> --extra mcp agentkit mcp --agent {agent_id} --registry http://apiagent-devs:9100/registry.json",
        help="Command shown in the welcome email for connecting Claude Code",
    )

    inspect = commands.add_parser("inspect", help="Print an agent's system prompt and tools, or a PR's review context")
    inspect.add_argument("spec", help="module:ATTRIBUTE of the AgentSpec")
    inspect.add_argument("--depth", type=int, choices=(0, 1), default=0, help="0: chatting with a person; 1: answering an agent")
    inspect.add_argument("--review-context", metavar="PR_URL", help="Print what review_pull_request returns for this PR")
    inspect.add_argument("--port", type=int, default=9000)

    mcp = commands.add_parser("mcp", help="Let an assistant such as Claude Code talk to one agent over MCP (stdio)")
    mcp.add_argument("--agent", required=True, help="Agent id, e.g. dev-pranit")
    mcp.add_argument("--registry", help="registry.json to find the agent's URL in (default: AGENT_REGISTRY)")
    mcp.add_argument("--url", help="The agent's URL, instead of looking it up")
    mcp.add_argument("--ui-url", default=os.environ.get("AGENTKIT_UI_URL", "http://localhost:5173"), help="Where the UI runs, for approval links")

    openapi = commands.add_parser("openapi", help="Print the agent HTTP API schema")
    openapi.add_argument("--out")

    init = commands.add_parser("init", help="Scaffold a chat_agent package for a new service")
    init.add_argument("--id", required=True, help="Agent id, e.g. service3")
    init.add_argument("--repo", required=True, help="GitHub repo the agent maintains, e.g. owner/ApiAgentService3")
    init.add_argument("--name", help="Display name")
    init.add_argument("--reads", action="append", help="Repo the agent may read (repeatable), e.g. owner/ApiAgentService1")
    init.add_argument("--port", type=int, default=9003)
    init.add_argument("--dir", default="chat_agent")
    init.add_argument("--registry", help="registry.json to add the new agent to")

    args = parser.parse_args(argv)
    commands_by_name = {
        "serve": cmd_serve,
        "dev": cmd_dev,
        "admin": cmd_admin,
        "inspect": cmd_inspect,
        "mcp": cmd_mcp,
        "openapi": cmd_openapi,
        "init": cmd_init,
    }
    commands_by_name[args.command](args)


if __name__ == "__main__":
    main()
