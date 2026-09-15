"""The agent's own clones of its repos, kept apart from anyone's dev checkout."""

import os
import subprocess
import threading
from pathlib import Path
from typing import Any

from agentkit.config import Settings
from agentkit.security import redact
from agentkit.spec import AgentSpec, RepoRef, ToolError

EMPTY_TREE = "4b825dc642cb6eb9a060ae81c5c29fbd9e0f06a0"
GIT_AUTHOR = ("ApiAgent chat agent", "agent@apiagent.local")


class PathError(ToolError):
    pass


class RepoBusyError(ToolError):
    pass


def _env_without_virtualenv() -> dict[str, str]:
    # Drop the agent's own virtualenv so `uv` uses the repo's environment.
    return {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}


def run(cmd: list[str], cwd: Path, *, check: bool = True, env: dict[str, str] | None = None, timeout: float = 300) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as err:
        raise ToolError(f"`{redact(' '.join(cmd))}` timed out after {timeout:.0f}s") from err
    if check and result.returncode != 0:
        output = (result.stderr or result.stdout).strip()[-2000:]
        raise ToolError(redact(f"`{' '.join(cmd)}` failed ({result.returncode}): {output}"))
    return result


class Repo:
    def __init__(self, root: Path, ref: RepoRef, *, writable: bool) -> None:
        self.root = root
        self.ref = ref
        self.writable = writable

    @property
    def name(self) -> str:
        return self.ref.name

    @property
    def slug(self) -> str:
        return self.ref.slug

    def exists(self) -> bool:
        return (self.root / ".git").exists()

    def git(self, *args: str, check: bool = True, timeout: float = 120) -> str:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        return run(["git", *args], self.root, check=check, env=env, timeout=timeout).stdout

    def resolve(self, path: str, *, for_write: bool = False) -> Path:
        """Return the absolute path for `path`, refusing anything outside the repo (or outside editable dirs for writes)."""
        rel = str(path or "").strip().removeprefix("./")
        parts = Path(rel).parts
        if Path(rel).is_absolute() or ".." in parts:
            raise PathError(f"Paths must be relative and stay inside {self.name}: {rel!r}")
        if parts and parts[0] == ".git":
            raise PathError(".git is off limits")
        target = (self.root / rel).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise PathError(f"Paths must stay inside {self.name}: {rel!r}")
        if for_write:
            if not self.writable:
                raise PathError(f"{self.name} is read-only for this agent")
            if not parts or parts[0] not in self.ref.editable_dirs:
                allowed = ", ".join(f"{d}/" for d in self.ref.editable_dirs)
                raise PathError(f"Edits in {self.name} are limited to {allowed}: {rel!r}")
        return target

    def list_files(self, path: str = "") -> list[str]:
        self.resolve(path)
        args = ["ls-files", "--cached", "--others", "--exclude-standard"]
        if path:
            args += ["--", path]
        return sorted(set(self.git(*args).splitlines()))

    def read_file(self, path: str) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"No file at {path!r} in {self.name}")
        return target.read_text(errors="replace")

    def write_file(self, path: str, content: str) -> bool:
        target = self.resolve(path, for_write=True)
        original = target.read_text() if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return original != content

    def search(self, pattern: str, path: str = "") -> str:
        self.resolve(path)
        args = ["grep", "-n", "-I", "--untracked", "-e", pattern]
        if path:
            args += ["--", path]
        result = run(["git", *args], self.root, check=False)
        if result.returncode == 1:
            return "No matches."
        if result.returncode != 0:
            raise ToolError(result.stderr.strip())
        return result.stdout

    def head(self) -> str:
        return self.git("rev-parse", "HEAD").strip()

    def branch(self) -> str:
        return self.git("rev-parse", "--abbrev-ref", "HEAD").strip()

    def commit_exists(self, ref: str) -> bool:
        return run(["git", "cat-file", "-e", f"{ref}^{{commit}}"], self.root, check=False).returncode == 0

    def changed_paths(self) -> list[str]:
        paths = []
        for line in self.git("status", "--porcelain", "--untracked-files=all").splitlines():
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            paths.append(path.strip('"'))
        return paths

    def is_dirty(self) -> bool:
        return bool(self.changed_paths())

    def untracked(self, path: str = "") -> list[str]:
        return self.git("ls-files", "--others", "--exclude-standard", *(["--", path] if path else [])).split()

    def diff(self, before: str = "", after: str = "", path: str = "") -> str:
        if path:
            self.resolve(path)
        pathspec = ["--", path or ".", ":(exclude)uv.lock"]
        if before or after:
            return self.git("diff", before or EMPTY_TREE, after or "HEAD", *pathspec) or "No differences."
        out = self.git("diff", "HEAD", *pathspec)
        if untracked := self.untracked(path):
            out += "\nUntracked files:\n" + "\n".join(untracked)
        return out or "No uncommitted changes."

    def diffstat(self) -> str:
        out = self.git("diff", "HEAD", "--stat")
        if untracked := self.untracked():
            out += "\nNew files: " + ", ".join(untracked)
        return out.strip()

    def log(self, revision_range: str = "", limit: int = 20) -> str:
        args = ["log", f"-n{limit}", "--format=%h %ad %an %s", "--date=short"]
        if revision_range:
            args.append(revision_range)
        return self.git(*args)

    def status(self) -> str:
        return self.git("status", "--short", "--branch")

    def run_tests(self, timeout: float = 900) -> tuple[bool, str]:
        cmd = ["uv", "run", "pytest", "-q", "-p", "no:warnings"]
        try:
            result = subprocess.run(cmd, cwd=self.root, env=_env_without_virtualenv(), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"Tests timed out after {timeout:.0f}s"
        return result.returncode == 0, (result.stdout + result.stderr)[-3000:]

    def uv_sync(self) -> None:
        run(["uv", "sync"], self.root, env=_env_without_virtualenv(), timeout=600)

    def fetch(self) -> None:
        self.git("fetch", "--quiet", "origin", timeout=300)

    def behind(self) -> int:
        try:
            return int(self.git("rev-list", "--count", f"HEAD..origin/{self.ref.branch}").strip())
        except ToolError:
            return 0

    def fast_forward(self) -> None:
        self.git("merge", "--ff-only", f"origin/{self.ref.branch}")

    def discard_changes(self) -> None:
        self.git("reset", "--hard", "HEAD")
        self.git("clean", "-fd", "--", *self.ref.editable_dirs)

    def reset_to_origin(self) -> None:
        self.git("checkout", "-B", self.ref.branch, f"origin/{self.ref.branch}")
        self.git("reset", "--hard", f"origin/{self.ref.branch}")
        self.git("clean", "-fd", "--", *self.ref.editable_dirs)

    def commit_and_push(self, branch: str, title: str, paths: list[str]) -> None:
        name, email = GIT_AUTHOR
        start = self.branch()
        self.git("checkout", "-B", branch)
        self.git("add", "--all", "--", *paths)
        self.git("-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-m", title)
        try:
            self.git("push", "--force", "origin", branch, timeout=300)
        except ToolError:
            # Put the changes back as uncommitted edits on the starting branch so nothing is lost.
            self.git("reset", "--soft", "HEAD~1")
            self.git("checkout", start)
            self.git("branch", "-D", branch)
            raise


class Workspace:
    def __init__(self, root: Path, spec: AgentSpec, settings: Settings) -> None:
        self.root = root
        self.settings = settings
        self.own = Repo(root / spec.repo.name, spec.repo, writable=True)
        self.read_only = [Repo(root / ref.name, ref, writable=False) for ref in spec.reads]
        self.holder: str | None = None
        self._claim_lock = threading.Lock()
        self.sync_lock = threading.Lock()

    def repos(self) -> list[Repo]:
        return [self.own, *self.read_only]

    def repo(self, name: str | None = None) -> Repo:
        if not name:
            return self.own
        key = name.lower()
        for repo in self.repos():
            if key in {repo.name.lower(), repo.slug.lower()}:
                return repo
        raise ToolError(f"Unknown repo {name!r}. Available: {', '.join(repo.name for repo in self.repos())}")

    def claim(self, conversation_id: str) -> None:
        """Only one conversation at a time may have uncommitted changes in the agent's own repo."""
        with self._claim_lock:
            if self.holder and self.holder != conversation_id and self.own.is_dirty():
                raise RepoBusyError(
                    f"{self.own.name} has uncommitted changes from conversation {self.holder}. "
                    "Open a PR or revert those changes first."
                )
            self.holder = conversation_id

    def release(self, conversation_id: str | None = None) -> None:
        with self._claim_lock:
            if conversation_id is None or self.holder == conversation_id:
                self.holder = None

    def clone_url(self, slug: str) -> str:
        base = self.settings.git_base_url.rstrip("/")
        if self.settings.github_token and base.startswith("https://github.com"):
            base = base.replace("https://", f"https://x-access-token:{self.settings.github_token}@", 1)
        return f"{base}/{slug}.git"

    def ensure_cloned(self) -> list[str]:
        notes = []
        self.root.mkdir(parents=True, exist_ok=True)
        for repo in self.repos():
            if not repo.exists():
                run(["git", "clone", "--quiet", self.clone_url(repo.slug), str(repo.root)], self.root, timeout=600)
                notes.append(f"Cloned {repo.slug}")
        if not (self.own.root / ".venv").exists():
            try:
                self.own.uv_sync()
                notes.append(f"Installed dependencies in {self.own.name}")
            except ToolError as err:
                notes.append(f"uv sync failed in {self.own.name}: {err}")
        return notes

    def sync(self) -> tuple[bool, list[str]]:
        """Fetch every repo and fast-forward the ones that are safe to move. Returns (own repo moved, notes)."""
        with self.sync_lock:
            notes: list[str] = []
            for repo in self.read_only:
                repo.fetch()
                before = repo.head()
                repo.reset_to_origin()
                if repo.head() != before:
                    notes.append(f"Updated read-only {repo.name} to {repo.head()[:7]}")

            self.own.fetch()
            behind = self.own.behind()
            if not behind:
                return False, notes
            if self.own.is_dirty() or self.holder:
                notes.append(
                    f"{self.own.name} is {behind} commit(s) behind origin/{self.own.ref.branch} "
                    "but has uncommitted changes, so it wasn't updated."
                )
                return False, notes
            self.own.fast_forward()
            try:
                self.own.uv_sync()
            except ToolError as err:
                notes.append(f"uv sync failed: {err}")
            notes.append(f"Fast-forwarded {self.own.name} by {behind} commit(s) to {self.own.head()[:7]}")
            return True, notes

    def status(self) -> list[dict[str, Any]]:
        rows = []
        for repo in self.repos():
            row: dict[str, Any] = {"name": repo.name, "slug": repo.slug, "writable": repo.writable, "cloned": repo.exists()}
            if repo.exists():
                changed = repo.changed_paths()
                row |= {
                    "head": repo.head()[:7],
                    "branch": repo.branch(),
                    "behind": repo.behind(),
                    "dirty": bool(changed),
                    "changed": changed[:50],
                    "lock_holder": self.holder if repo is self.own else None,
                }
            rows.append(row)
        return rows
