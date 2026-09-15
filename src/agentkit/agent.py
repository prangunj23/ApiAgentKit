"""An agent at runtime: its spec plus storage, workspace, memory, and background work."""

import logging
import threading
from typing import Any

from agentkit.codemap import CodeMap
from agentkit.config import Settings
from agentkit.gh import GitHub
from agentkit.learning import Learning
from agentkit.llm import LLM, NimLLM
from agentkit.memory import Memory, strip_markers
from agentkit.registry import HttpFactory, Peers, default_http_factory
from agentkit.research import Research
from agentkit.runs import Runs
from agentkit.spec import AgentSpec, Tool, ToolError
from agentkit.store import Store
from agentkit.tools import GENERIC_TOOLS
from agentkit.workspace import Workspace

log = logging.getLogger(__name__)

API_VERSION = 1


def _clip(text: str, budget: int) -> str:
    return text if len(text) <= budget else text[:budget] + "\n…(truncated)"


class Agent:
    def __init__(
        self, spec: AgentSpec, settings: Settings, *, llm: LLM | None = None, http_factory: HttpFactory = default_http_factory
    ) -> None:
        self.spec = spec
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(settings.data_dir / "agent.db")
        self.workspace = Workspace(settings.data_dir / "workspace", spec, settings)
        self.research = Research(settings.data_dir / "research")
        self.memory = Memory(self)
        self.codemap = CodeMap(self)
        self.learning = Learning(self)
        self.llm: LLM = llm or NimLLM(settings)
        self.http_factory = http_factory
        self.peers = Peers(settings.registry, spec.id, settings.shared_token, http_factory)
        self.github = GitHub(settings.github_token, http_factory(settings.github_api))
        self.tools: dict[str, Tool] = {tool.name: tool for tool in [*GENERIC_TOOLS, *spec.tools]}
        self._conversation_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self.runs = Runs(settings.secrets)
        self._stop = threading.Event()

    def tools_for(self, depth: int) -> list[Tool]:
        return [tool for tool in self.tools.values() if not (depth > 0 and tool.name == "message_agent")]

    def conversation_lock(self, conversation_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._conversation_locks.setdefault(conversation_id, threading.Lock())

    def log_event(self, type: str, summary: str, *, url: str | None = None, conversation_id: str | None = None) -> dict[str, Any]:
        event = self.store.add_event(type, summary, url=url, conversation_id=conversation_id)
        # Memory holds one line per event; the full summary (e.g. a contract diff) stays in the events table.
        headline = summary.strip().splitlines()[0] if summary.strip() else type
        self.memory.record_event(f"{headline} ({url})" if url else headline)
        return event

    def info(self) -> dict[str, Any]:
        return {
            "api_version": API_VERSION,
            "id": self.spec.id,
            "name": self.spec.name,
            "description": self.spec.description,
            "repo": {"slug": self.spec.repo.slug, "name": self.spec.repo.name},
            "reads": [{"slug": ref.slug, "name": ref.name} for ref in self.spec.reads],
            "features": sorted(self.spec.features),
            "tools": [
                {"name": tool.name, "description": tool.description, "needs_confirmation": tool.needs_confirmation}
                for tool in self.tools.values()
            ],
        }

    def system_prompt(self, depth: int) -> str:
        spec, own = self.spec, self.workspace.own
        editable = ", ".join(f"{d}/" for d in own.ref.editable_dirs)
        lines = [
            spec.system_prompt.strip(),
            "",
            "# Operating context",
            f"You are the `{spec.id}` agent ({spec.name}).",
            f"- Your repo: {own.slug}. You may edit only {editable} in it, and you propose changes by opening a pull request.",
        ]
        if self.workspace.read_only:
            names = ", ".join(f"{repo.slug} (`{repo.name}`)" for repo in self.workspace.read_only)
            lines.append(f"- Read-only repos: {names}. Pass `repo` to the read tools to use them.")
        if own.exists():
            changed = own.changed_paths()
            state = f"uncommitted changes in {', '.join(changed[:10])}" if changed else "no uncommitted changes"
            lines.append(f"- Checkout: {own.branch()} at {own.head()[:7]}, {state}.")
        else:
            lines.append("- Your checkout isn't cloned yet; repo tools will fail until it is.")
        confirm = [tool.name for tool in self.tools_for(depth) if tool.needs_confirmation]
        if confirm:
            lines.append(
                f"- These tools need the user's approval: {', '.join(confirm)}. Call them when ready; the user sees an approval card."
            )
        if depth == 0:
            peers = self.peers.others()
            if peers:
                lines.append("- Other agents you can ask with message_agent: " + ", ".join(f"`{p.id}` ({p.name})" for p in peers) + ".")
        else:
            lines.append("- You are answering a message from another agent. Reply directly; you can't message agents from here.")
        lines.append(
            "- Keep what's worth remembering: save_research for findings, remember for durable facts, "
            "and propose_lesson or propose_skill when you learn how to work better."
        )

        lessons = "\n".join(f"- **{item['title']}**: {item['body']}" for item in self.learning.active("lesson")[:30])
        skills = "\n".join(f"- {item['title']}: {item['body'].splitlines()[0][:150]}" for item in self.learning.active("skill"))
        research = "\n".join(f"- {item['file']}: {item['title']}" for item in self.research.list()[:30])
        codemap = self.codemap.prompt_excerpt()
        if codemap and self.codemap.is_stale():
            codemap = "_This map is behind your checkout and is being rebuilt; verify details by reading files._\n\n" + codemap
            if self.settings.background:
                self.codemap.schedule_update()

        sections = [
            ("Lessons you follow", lessons, 2_500),
            ("Skills (load one with read_skill)", skills, 800),
            ("Memory: notes", self.memory.notes(), 2_000),
            ("Memory: recent activity", "\n".join(strip_markers(self.memory.recent())), 3_000),
            ("Research files", research, 600),
            ("Codebase map", codemap, 6_000),
        ]
        for title, body, budget in sections:
            if body.strip():
                lines += ["", f"# {title}", _clip(body.strip(), budget)]
        return "\n".join(lines)

    # Background work

    def start_background(self) -> None:
        if not self.settings.background:
            return
        threading.Thread(target=self._repo_loop, daemon=True, name=f"{self.spec.id}-repos").start()
        threading.Thread(target=self._pr_loop, daemon=True, name=f"{self.spec.id}-prs").start()

    def stop(self) -> None:
        self._stop.set()

    def _repo_loop(self) -> None:
        try:
            for note in self.workspace.ensure_cloned():
                log.info(note)
            self.codemap.update()
        except Exception:
            log.exception("Workspace setup failed")
        while not self._stop.wait(self.settings.repo_poll_seconds):
            self.sync_workspace()

    def sync_workspace(self) -> list[str]:
        try:
            moved, notes = self.workspace.sync()
        except ToolError as err:
            log.warning("Workspace sync failed: %s", err)
            return [str(err)]
        if moved:
            self.log_event("repo_synced", notes[-1])
            try:
                self.codemap.update()
            except Exception:
                log.exception("Codebase map update failed after sync")
        return notes

    def _pr_loop(self) -> None:
        while not self._stop.wait(self.settings.pr_poll_seconds):
            try:
                self.learning.poll_prs()
            except Exception:
                log.exception("PR polling failed")
