"""What an agent is: the repos it works on, its prompt, its tools, and the UI features it offers."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentkit.agent import Agent


@dataclass(frozen=True)
class RepoRef:
    """A GitHub repo cloned into the agent's workspace."""

    slug: str
    dirname: str = ""
    branch: str = "main"
    editable_dirs: tuple[str, ...] = ("src", "tests")

    @property
    def name(self) -> str:
        return self.dirname or self.slug.rsplit("/", 1)[-1]


class ToolError(Exception):
    """A tool failed in a way the model should see and can recover from."""


@dataclass
class ToolContext:
    """Passed to every tool call."""

    agent: "Agent"
    conversation_id: str
    # 0 when chatting with the user; 1 when answering another agent.
    depth: int = 0
    wrote_files: bool = False


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]
    needs_confirmation: bool = False
    # Text shown on the approval card, e.g. a diffstat. Called with the same arguments as fn.
    preview: Callable[..., str] | None = None
    # Called when the user denies a confirmation, with (ctx, args, reason).
    on_deny: Callable[..., None] | None = None
    max_result_chars: int = 20_000

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def tool(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: tuple[str, ...] | list[str] = (),
    *,
    needs_confirmation: bool = False,
    max_result_chars: int = 20_000,
) -> Callable[[Callable[..., str]], Tool]:
    """Turn `fn(ctx: ToolContext, **arguments) -> str` into a Tool."""

    def wrap(fn: Callable[..., str]) -> Tool:
        return Tool(
            name=name,
            description=description,
            parameters={"type": "object", "properties": properties or {}, "required": list(required)},
            fn=fn,
            needs_confirmation=needs_confirmation,
            max_result_chars=max_result_chars,
        )

    return wrap


def string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def integer(description: str) -> dict[str, Any]:
    return {"type": "integer", "description": description}


def boolean(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def string_list(description: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": description}


@dataclass
class AgentSpec:
    id: str
    name: str
    repo: RepoRef
    system_prompt: str
    reads: list[RepoRef] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    # Optional UI features, e.g. {"emails"}.
    features: set[str] = field(default_factory=set)
    description: str = ""
