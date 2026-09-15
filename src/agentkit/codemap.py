"""Keeps research/codebase-map.md describing the agent's own repo.

Most sections are extracted from the code with `ast` on every update. The LLM only writes file purposes and
the architecture summary, and only for files that changed since the map's last commit.
"""

import ast
import difflib
import json
import logging
import re
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentkit import frontmatter
from agentkit.llm import complete_json
from agentkit.research import MAP_FILE
from agentkit.spec import ToolError
from agentkit.store import now
from agentkit.workspace import Repo

if TYPE_CHECKING:
    from agentkit.agent import Agent

log = logging.getLogger(__name__)

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
SKIP_PARTS = {".venv", "__pycache__", ".pytest_cache", "node_modules"}
TREE_LIMIT = 200
FILE_CHARS = 6_000
DESCRIBE_CHARS = 40_000
PROMPT_SECTIONS = ("1.", "2.", "3.", "4.", "8.")
NO_NOTES = "_No notes yet. The agent adds them with update_codebase_notes._"

DESCRIBE_PROMPT = """You document a codebase for an AI agent that maintains it.
For each file listed under "Describe these files", write its purpose in one sentence of at most 25 words.
If an architecture summary is requested, write 2-4 short markdown paragraphs: how requests flow, how the
pieces fit together, and how this repo relates to the repos it depends on.

Respond with only a JSON object:
{"files": {"path": "purpose"}, "architecture": "markdown, or an empty string if not requested"}"""


@dataclass
class Route:
    method: str
    path: str
    handler: str
    file: str
    response_model: str = ""
    params: list[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    file: str
    bases: list[str]
    methods: list[str]
    fields: list[str]


@dataclass
class FileFacts:
    path: str
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    routes: list[Route] = field(default_factory=list)
    env_vars: set[str] = field(default_factory=set)
    imports: dict[str, set[str]] = field(default_factory=dict)
    attributes: set[str] = field(default_factory=set)
    exports: list[str] = field(default_factory=list)


@dataclass
class RepoFacts:
    slug: str
    files: list[str]
    python: dict[str, FileFacts]
    overview: dict[str, Any]
    packages: list[str]
    readme_commands: list[str]
    workflows: list[dict[str, Any]]

    @property
    def models(self) -> list[ClassInfo]:
        return [cls for facts in self.python.values() for cls in facts.classes if cls.fields]

    @property
    def routes(self) -> list[Route]:
        return [route for facts in self.python.values() for route in facts.routes]


# Extraction


def _const_str(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def parse_python(path: str, text: str) -> FileFacts:
    facts = FileFacts(path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return facts

    prefixes: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func).split(".")[-1] in {"APIRouter", "FastAPI"}:
                prefix = next((_const_str(kw.value) for kw in node.value.keywords if kw.arg == "prefix"), None) or ""
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        prefixes[target.id] = prefix

    model_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [ast.unparse(base) for base in node.bases]
            methods = [
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_")
            ]
            fields: list[str] = []
            if any(base.split(".")[-1] == "BaseModel" or base in model_names for base in bases):
                model_names.add(node.name)
                fields = [
                    f"{item.target.id}: {ast.unparse(item.annotation)}"
                    for item in node.body
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                ]
            facts.classes.append(ClassInfo(node.name, path, bases, methods, fields))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                facts.functions.append(node.name)
            for decorator in node.decorator_list:
                if route := _route(decorator, node, prefixes, path):
                    facts.routes.append(route)
        elif isinstance(node, ast.Assign) and path.endswith("__init__.py"):
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets) and isinstance(node.value, (ast.List, ast.Tuple)):
                facts.exports = [value for elt in node.value.elts if (value := _const_str(elt))]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in {"os.environ.get", "os.getenv", "environ.get", "getenv"}:
            if node.args and (name := _const_str(node.args[0])):
                facts.env_vars.add(name)
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value) in {"os.environ", "environ"}:
            if name := _const_str(node.slice):
                facts.env_vars.add(name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            facts.imports.setdefault(node.module, set()).update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                facts.imports.setdefault(alias.name, set())
        elif isinstance(node, ast.Attribute):
            facts.attributes.add(node.attr)

    if path.endswith("__init__.py") and not facts.exports:
        facts.exports = sorted(
            alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names if alias.name != "*"
        )
    return facts


def _route(decorator: ast.expr, func: ast.FunctionDef | ast.AsyncFunctionDef, prefixes: dict[str, str], path: str) -> Route | None:
    if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)):
        return None
    if decorator.func.attr not in HTTP_METHODS or not isinstance(decorator.func.value, ast.Name):
        return None
    route_path = _const_str(decorator.args[0]) if decorator.args else None
    if route_path is None:
        return None
    response_model = next((ast.unparse(kw.value) for kw in decorator.keywords if kw.arg == "response_model"), "")
    params = [ast.unparse(arg.annotation) for arg in func.args.args if arg.annotation]
    prefix = prefixes.get(decorator.func.value.id, "")
    return Route(decorator.func.attr.upper(), prefix + route_path, func.name, path, response_model, params)


def _read_pyproject(root: Path) -> tuple[dict[str, Any], list[str]]:
    path = root / "pyproject.toml"
    if not path.is_file():
        return {}, []
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError:
        return {}, []
    project = data.get("project", {})
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    overview = {
        "name": project.get("name", ""),
        "version": project.get("version", ""),
        "description": project.get("description", ""),
        "requires_python": project.get("requires-python", ""),
        "dependencies": project.get("dependencies", []),
        "path_dependencies": {
            name: source["path"] for name, source in sources.items() if isinstance(source, dict) and "path" in source
        },
    }
    wheel = data.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {})
    packages = [Path(package).name for package in wheel.get("packages", [])]
    return overview, packages


def _readme_commands(root: Path) -> list[str]:
    path = root / "README.md"
    if not path.is_file():
        return []
    commands: list[str] = []
    for block in re.findall(r"```(?:sh|bash|shell)\n(.*?)```", path.read_text(), flags=re.DOTALL):
        for line in block.splitlines():
            if line.strip() and line.strip() not in commands:
                commands.append(line.strip())
    return commands[:15]


def _workflow(root: Path, rel: str) -> dict[str, Any]:
    text = (root / rel).read_text(errors="replace")
    name = re.search(r"^name:\s*(.+)$", text, flags=re.MULTILINE)
    triggers: list[str] = []
    inline = re.search(r"^on:\s*\[(.+)\]\s*$", text, flags=re.MULTILINE)
    if inline:
        triggers = [item.strip() for item in inline.group(1).split(",")]
    else:
        block = re.search(r"^on:\s*\n((?:[ \t]+.*\n|\s*\n)+)", text, flags=re.MULTILINE)
        if block:
            triggers = re.findall(r"^  ([A-Za-z_][\w-]*):", block.group(1), flags=re.MULTILINE)
    return {"file": rel, "name": name.group(1).strip() if name else rel, "triggers": triggers}


def collect(repo: Repo) -> RepoFacts:
    files = [f for f in repo.list_files() if not set(Path(f).parts) & SKIP_PARTS and (repo.root / f).is_file()]
    python = {f: parse_python(f, (repo.root / f).read_text(errors="replace")) for f in files if f.endswith(".py")}
    overview, packages = _read_pyproject(repo.root)
    if not packages:
        packages = sorted({Path(f).parts[1] for f in files if f.startswith("src/") and f.endswith("/__init__.py") and len(Path(f).parts) == 3})
    workflows = [_workflow(repo.root, f) for f in files if f.startswith(".github/workflows/") and f.endswith((".yml", ".yaml"))]
    return RepoFacts(repo.slug, files, python, overview, packages, _readme_commands(repo.root), workflows)


def important_files(facts: RepoFacts) -> list[str]:
    keep = [f for f in facts.files if f.endswith(".py") or f in {"pyproject.toml", "README.md"} or f.startswith(".github/workflows/")]
    return keep[:60]


def contract_text(facts: RepoFacts) -> str:
    lines = [f"{route.method} {route.path} -> {route.response_model or '?'} ({', '.join(route.params)})" for route in facts.routes]
    lines += [f"model {model.name}: {', '.join(model.fields)}" for model in facts.models]
    for path, file_facts in sorted(facts.python.items()):
        if path.endswith("__init__.py") and file_facts.exports:
            lines.append(f"exports {path}: {', '.join(file_facts.exports)}")
    lines += [
        f"client {cls.name}: {', '.join(cls.methods)}"
        for file_facts in facts.python.values()
        for cls in file_facts.classes
        if cls.name.endswith("Client")
    ]
    return "\n".join(sorted(lines))


def cross_repo_usage(facts: RepoFacts, others: list[RepoFacts]) -> list[str]:
    lines = []
    for other in others:
        classes = {cls.name: cls for file_facts in other.python.values() for cls in file_facts.classes}
        for path, file_facts in sorted(facts.python.items()):
            names: set[str] = set()
            for module, imported in file_facts.imports.items():
                if module.split(".")[0] in other.packages:
                    names |= imported or {module}
            if not names:
                continue
            methods = sorted({m for name in names if name in classes for m in classes[name].methods} & file_facts.attributes)
            line = f"- `{path}` uses `{other.slug}`: {', '.join(sorted(names))}"
            lines.append(line + (f"; methods called: {', '.join(methods)}" if methods else ""))
    return lines


# Rendering


def _tree(files: list[str]) -> str:
    visible = [f for f in files if Path(f).name != "uv.lock"]
    shown = visible[:TREE_LIMIT]
    lines, seen = [], set()
    for f in shown:
        parts = Path(f).parts
        for depth in range(len(parts) - 1):
            directory = "/".join(parts[: depth + 1])
            if directory not in seen:
                seen.add(directory)
                lines.append(f"{'  ' * depth}{parts[depth]}/")
        lines.append(f"{'  ' * (len(parts) - 1)}{parts[-1]}")
    if len(visible) > len(shown):
        lines.append(f"… {len(visible) - len(shown)} more")
    return "\n".join(lines)


def _symbols(file_facts: FileFacts | None) -> str:
    if file_facts is None:
        return ""
    parts = [f"`{route.method} {route.path}` → `{route.handler}`" for route in file_facts.routes]
    parts += [f"model `{cls.name}`" if cls.fields else f"class `{cls.name}`" for cls in file_facts.classes]
    routed = {route.handler for route in file_facts.routes}
    functions = [name for name in file_facts.functions if name not in routed]
    if functions:
        parts.append("functions: " + ", ".join(f"`{name}`" for name in functions[:12]))
    return "; ".join(parts)


def render(facts: RepoFacts, others: list[RepoFacts], state: dict[str, Any], changes: list[str], diffstat: str) -> str:
    overview = facts.overview
    purposes: dict[str, str] = state.get("purposes", {})
    outdated = set(state.get("outdated", []))
    out = [f"# Codebase map: {facts.slug}", ""]

    out += ["## 1. Overview", ""]
    if overview:
        out.append(f"- **Package:** `{overview['name']}` {overview['version']} — {overview['description']}")
        out.append(f"- **Python:** {overview['requires_python'] or 'unspecified'}")
        out.append(f"- **Dependencies:** {', '.join(overview['dependencies']) or 'none'}")
        for name, path in overview["path_dependencies"].items():
            out.append(f"- **Path dependency:** `{name}` from `{path}`")
    out.append(f"- **Import packages provided:** {', '.join(f'`{p}`' for p in facts.packages) or 'none'}")

    out += ["", "## 2. Directory tree", "", "```text", _tree(facts.files), "```"]

    out += ["", "## 3. Important files", "", "| File | Purpose | Key symbols |", "|---|---|---|"]
    for path in important_files(facts):
        purpose = purposes.get(path, "_not described yet_")
        if path in outdated:
            purpose += " _(may be outdated)_"
        out.append(f"| `{path}` | {purpose} | {_symbols(facts.python.get(path))} |")

    out += ["", "## 4. Public contract", ""]
    if facts.routes:
        out += ["### HTTP endpoints", "", "| Method | Path | Handler | Parameters | Response |", "|---|---|---|---|---|"]
        out += [
            f"| {r.method} | `{r.path}` | `{r.file}:{r.handler}` | {', '.join(r.params) or '—'} | {r.response_model or '—'} |"
            for r in facts.routes
        ]
        out.append("")
    if facts.models:
        out += ["### Models", ""] + [f"- `{m.name}` (`{m.file}`): {', '.join(m.fields)}" for m in facts.models] + [""]
    exports = [(path, ff.exports) for path, ff in sorted(facts.python.items()) if path.endswith("__init__.py") and ff.exports]
    if exports:
        out += ["### Package exports", ""] + [f"- `{path}`: {', '.join(names)}" for path, names in exports] + [""]
    clients = [cls for ff in facts.python.values() for cls in ff.classes if cls.name.endswith("Client")]
    if clients:
        out += ["### Client methods", ""] + [f"- `{cls.name}`: {', '.join(cls.methods)}" for cls in clients] + [""]
    if not (facts.routes or facts.models or exports or clients):
        out.append("_No public contract detected._")

    out += ["", "## 5. Cross-repo usage", ""]
    out += cross_repo_usage(facts, others) or ["_No imports from the repos this agent reads._"]

    out += ["", "## 6. Run, test, and config", ""]
    if facts.readme_commands:
        out += ["**Commands from the README:**", "", "```sh", *facts.readme_commands, "```", ""]
    if facts.workflows:
        out += ["**CI workflows:**", ""] + [f"- {w['name']} (`{w['file']}`): {', '.join(w['triggers']) or '?'}" for w in facts.workflows] + [""]
    env_vars = sorted({(name, path) for path, ff in facts.python.items() for name in ff.env_vars})
    if env_vars:
        out += ["**Environment variables:**", ""] + [f"- `{name}` (`{path}`)" for name, path in env_vars]

    out += ["", "## 7. Architecture summary", "", state.get("architecture") or "_Not written yet._"]
    out += ["", "## 8. Notes & gotchas", "", state.get("notes") or NO_NOTES]
    if changes:
        out += ["", "## 9. Uncommitted changes", "", *[f"- `{path}`" for path in changes], "", "```text", diffstat, "```"]
    return "\n".join(out) + "\n"


class CodeMap:
    def __init__(self, agent: "Agent") -> None:
        self.agent = agent
        self.path = agent.research.dir / MAP_FILE
        self.state_path = agent.research.dir / ".codebase-map.json"
        self._lock = threading.Lock()
        self._update_running = threading.Lock()

    @property
    def repo(self) -> Repo:
        return self.agent.workspace.own

    def load_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def _save(self, facts: RepoFacts, state: dict[str, Any]) -> None:
        changes = self.repo.changed_paths()
        state["dirty"] = bool(changes)
        state["updated_at"] = now()
        self.state_path.write_text(json.dumps(state, indent=2))
        others = [collect(repo) for repo in self.agent.workspace.read_only if repo.exists()]
        body = render(facts, others, state, changes, self.repo.diffstat() if changes else "")
        meta = {
            "title": f"Codebase map: {facts.slug}",
            "repo": facts.slug,
            "commit": state.get("commit", ""),
            "updated_at": state["updated_at"],
            "dirty": state["dirty"],
        }
        self.path.write_text(frontmatter.dump(meta, body))

    def is_stale(self) -> bool:
        return self.repo.exists() and self.load_state().get("commit") != self.repo.head()

    def view(self) -> dict[str, Any]:
        state = self.load_state()
        return {
            "exists": self.path.exists(),
            "markdown": frontmatter.parse(self.path.read_text())[1] if self.path.exists() else "",
            "commit": state.get("commit", ""),
            "updated_at": state.get("updated_at", ""),
            "dirty": bool(state.get("dirty")),
            "stale": self.is_stale() if self.path.exists() else False,
            "outdated": state.get("outdated", []),
        }

    def update(self, *, full: bool = False) -> dict[str, Any]:
        """Regenerate the map for the current HEAD, describing only files that changed since the last map."""
        repo = self.repo
        if not repo.exists():
            raise ToolError(f"{repo.name} isn't cloned yet")
        with self._lock:
            state = self.load_state()
            head = repo.head()
            facts = collect(repo)
            important = important_files(facts)
            old_commit = state.get("commit")
            purposes = {path: text for path, text in state.get("purposes", {}).items() if path in important}
            changed: set[str] = set()

            if full or not old_commit or not repo.commit_exists(old_commit):
                to_describe, describe_architecture = important, True
            else:
                if old_commit != head:
                    changed = set(repo.git("diff", "--name-only", f"{old_commit}..{head}").split())
                to_describe = [path for path in important if path in changed or path not in purposes]
                describe_architecture = not state.get("architecture")

            contract = contract_text(facts)
            old_contract = state.get("contract")
            contract_changed = bool(old_contract) and old_contract != contract
            if old_commit and old_commit != head and (len(changed) > 3 or contract_changed):
                describe_architecture = True

            architecture = state.get("architecture", "")
            outdated = set(state.get("outdated", [])) & set(important)
            if to_describe or describe_architecture:
                try:
                    result = self._describe(facts, to_describe, describe_architecture, architecture)
                    for path in to_describe:
                        if text := str(result.get("files", {}).get(path, "")).strip():
                            purposes[path] = text
                            outdated.discard(path)
                        elif path in purposes:
                            outdated.add(path)
                    if describe_architecture and str(result.get("architecture", "")).strip():
                        architecture = str(result["architecture"]).strip()
                except Exception as err:
                    # The extracted sections are still written; only the LLM descriptions are missing.
                    log.warning("Couldn't describe files for the codebase map: %s", err)
                    outdated |= {path for path in to_describe if path in purposes}

            state |= {
                "commit": head,
                "purposes": purposes,
                "architecture": architecture,
                "contract": contract,
                "outdated": sorted(outdated),
            }
            self._save(facts, state)

        if full or old_commit != head:
            self.agent.log_event("map_updated", f"Codebase map updated to {head[:7]} ({len(to_describe)} file(s) described)")
        if contract_changed:
            diff = "\n".join(difflib.unified_diff((old_contract or "").splitlines(), contract.splitlines(), lineterm="", n=0))[:2000]
            summary = f"Public contract of {repo.slug} changed at {head[:7]}"
            self.agent.log_event("contract_changed", summary)
            self.agent.peers.notify_dependents(repo.slug, {"type": "contract_changed", "summary": f"{summary}:\n{diff}"})
        return {"commit": head, "described": to_describe, "contract_changed": contract_changed}

    def schedule_update(self, *, full: bool = False) -> bool:
        """Run update() in the background unless one is already running. Returns False if skipped."""
        if not self._update_running.acquire(blocking=False):
            return False

        def work() -> None:
            try:
                self.update(full=full)
            except Exception:
                log.exception("Codebase map update failed")
            finally:
                self._update_running.release()

        if self.agent.settings.background:
            threading.Thread(target=work, daemon=True, name=f"{self.agent.spec.id}-codemap").start()
        else:
            work()
        return True

    def refresh_working_tree(self) -> None:
        """Re-extract facts from uncommitted edits. No LLM call."""
        if not self.repo.exists() or not self.path.exists():
            return
        with self._lock:
            self._save(collect(self.repo), self.load_state())

    def set_notes(self, notes: str) -> None:
        with self._lock:
            state = self.load_state()
            state["notes"] = notes.strip()
            if self.repo.exists():
                self._save(collect(self.repo), state)
            else:
                self.state_path.write_text(json.dumps(state, indent=2))

    def prompt_excerpt(self, budget: int = 6_000) -> str:
        if not self.path.exists():
            return ""
        _, body = frontmatter.parse(self.path.read_text())
        sections = body.split("\n## ")
        text = "\n## ".join([sections[0], *[s for s in sections[1:] if s.startswith(PROMPT_SECTIONS)]])
        if len(text) > budget:
            text = text[:budget] + "\n…(truncated; read_research codebase-map.md for the full map)"
        return text

    def _describe(self, facts: RepoFacts, paths: list[str], architecture: bool, current_architecture: str) -> dict[str, Any]:
        contents, used = [], 0
        for path in paths:
            text = (self.repo.root / path).read_text(errors="replace")[:FILE_CHARS]
            if used + len(text) > DESCRIBE_CHARS:
                text = text[: max(0, DESCRIBE_CHARS - used)]
            used += len(text)
            contents.append(f"--- {path} ---\n{text}")
        user = "\n".join(
            [
                f"# Repo: {facts.slug}",
                f"Overview: {json.dumps(facts.overview)}",
                "# All files",
                "\n".join(facts.files),
                "# Describe these files",
                "\n".join(paths) or "(none)",
                f"# Architecture summary requested: {'yes' if architecture else 'no'}",
                f"Current summary:\n{current_architecture or '(none)'}",
                "# File contents",
                "\n\n".join(contents),
            ]
        )
        return complete_json(self.agent.llm, DESCRIBE_PROMPT, user)
