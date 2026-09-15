"""agentkit command line: serve an agent, run every local agent, export the API schema, or scaffold a new agent."""

import argparse
import importlib
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from string import Template
from urllib.parse import urlparse

from agentkit.config import Settings, load_dotenv
from agentkit.registry import load_registry
from agentkit.spec import AgentSpec, RepoRef

KIT_ROOT = Path(__file__).resolve().parents[2]


def load_spec(target: str) -> AgentSpec:
    module_name, _, attribute = target.partition(":")
    spec = getattr(importlib.import_module(module_name), attribute or "SPEC")
    if not isinstance(spec, AgentSpec):
        raise SystemExit(f"{target} is not an AgentSpec")
    return spec


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from agentkit.server import create_app

    load_dotenv(Path.cwd() / ".env")
    spec = load_spec(args.spec)
    settings = Settings.from_env(spec.id, args.port)
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


def cmd_dev(args: argparse.Namespace) -> None:
    registry = Path(args.registry).resolve()
    entries = [entry for entry in load_registry(str(registry)) if entry.local]
    if not entries:
        raise SystemExit(f"No agents with a `local` entry in {registry}")
    token = os.environ.get("AGENT_SHARED_TOKEN") or secrets.token_urlsafe(24)

    processes: list[tuple[str, subprocess.Popen[str]]] = []
    for entry in entries:
        local = entry.local or {}
        project = (registry.parent / local["path"]).resolve()
        port = str(local.get("port") or urlparse(entry.url).port)
        env = {**os.environ, "AGENT_REGISTRY": str(registry), "AGENT_SHARED_TOKEN": token}
        env.pop("VIRTUAL_ENV", None)
        command = ["uv", "run", "--project", str(project), "agentkit", "serve", local["spec"], "--port", port]
        process = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=_relay, args=(entry.id, process), daemon=True).start()
        processes.append((entry.id, process))
        print(f"Starting {entry.id} at {entry.url} from {project}")

    try:
        while all(process.poll() is None for _, process in processes):
            time.sleep(0.5)
        for name, process in processes:
            if process.poll() is not None:
                print(f"{name} exited with code {process.returncode}; stopping the others.")
    except KeyboardInterrupt:
        pass
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


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

    dev = commands.add_parser("dev", help="Run every agent that has a `local` entry in the registry")
    dev.add_argument("--registry", required=True)

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
    {"serve": cmd_serve, "dev": cmd_dev, "openapi": cmd_openapi, "init": cmd_init}[args.command](args)


if __name__ == "__main__":
    main()
