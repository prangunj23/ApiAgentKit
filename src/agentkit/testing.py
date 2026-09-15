"""Helpers for testing agents without NIM, GitHub, or network access."""

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from agentkit.config import Settings
from agentkit.registry import HttpFactory

GIT_IDENTITY = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]


class FakeLLM:
    """Scripted model.

    `replies` feed stream(): each is text, or {"content": ..., "tool_calls": [(name, arguments), ...]}.
    `completions` feed complete() in order; after they run out, `responder(system, user)` answers.
    """

    def __init__(
        self,
        replies: list[Any] | None = None,
        completions: list[str] | None = None,
        responder: Callable[[str, str], str] | None = None,
    ) -> None:
        self.replies = list(replies or [])
        self.completions = list(completions or [])
        self.responder = responder
        self.calls: list[dict[str, Any]] = []
        self.complete_calls: list[tuple[str, str]] = []

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        self.calls.append({"messages": messages, "tools": tools})
        reply = self.replies.pop(0) if self.replies else "OK"
        if isinstance(reply, str):
            reply = {"content": reply}
        content = reply.get("content", "")
        if content:
            yield {"type": "token", "text": content}
        tool_calls = [
            {"id": f"call_{len(self.calls)}_{index}", "name": name, "arguments": json.dumps(arguments)}
            for index, (name, arguments) in enumerate(reply.get("tool_calls", []))
        ]
        yield {"type": "message", "content": content, "tool_calls": tool_calls, "reasoning": reply.get("reasoning", "")}

    def complete(self, system: str, user: str) -> str:
        self.complete_calls.append((system, user))
        if self.completions:
            return self.completions.pop(0)
        if self.responder:
            return self.responder(system, user)
        return "Summary of the conversation."


def make_settings(data_dir: Path, *, port: int = 9000, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "data_dir": data_dir,
        "port": port,
        "background": False,
        "summary_delay_seconds": 0,
        "shared_token": "test-shared-token",
        "extra_hosts": ["testserver"],
    }
    values.update(overrides)
    return Settings(**values)


def parse_sse(text: str) -> list[dict[str, Any]]:
    events = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *GIT_IDENTITY, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def make_remote(origin_root: Path, slug: str, files: dict[str, str]) -> Path:
    """Create a bare 'GitHub' repo at origin_root/<slug>.git with one commit. Returns a working clone for more commits."""
    bare = origin_root / f"{slug}.git"
    bare.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--quiet", "--bare", "-b", "main", str(bare)], check=True)
    seed = origin_root / "_seeds" / slug
    seed.mkdir(parents=True)
    git(seed, "init", "--quiet", "-b", "main")
    git(seed, "remote", "add", "origin", str(bare))
    commit_files(seed, files, "Initial commit")
    return seed


def commit_files(worktree: Path, files: dict[str, str | None], message: str) -> str:
    """Write (or delete, for None) files, commit, and push to origin/main. Returns the new commit SHA."""
    for rel, content in files.items():
        path = worktree / rel
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    git(worktree, "add", "--all")
    git(worktree, "commit", "--quiet", "-m", message)
    git(worktree, "push", "--quiet", "origin", "main")
    return git(worktree, "rev-parse", "HEAD")


def http_factory_for(apps: dict[str, Any], fallback: Callable[[str], httpx.Client] | None = None) -> HttpFactory:
    """Route HTTP calls for each base URL to an in-process ASGI app. Other URLs use `fallback` or get 503s."""

    def factory(base_url: str) -> httpx.Client:
        if base_url in apps:
            return TestClient(apps[base_url], base_url=base_url)
        if fallback:
            return fallback(base_url)
        return httpx.Client(base_url=base_url, transport=httpx.MockTransport(lambda request: httpx.Response(503)))

    return factory
