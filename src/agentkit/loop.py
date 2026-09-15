"""The tool loop: stream the model, run tools, and pause for the user's approval on tools that need it.

Every step is an event dict. The HTTP layer streams them as SSE; agent-to-agent calls collect them.
"""

import json
import logging
from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING, Any

from agentkit.security import redact
from agentkit.spec import Tool, ToolContext, ToolError
from agentkit.store import now

if TYPE_CHECKING:
    from agentkit.agent import Agent

log = logging.getLogger(__name__)

Event = dict[str, Any]
MAX_ROUNDS = {0: 25, 1: 10}


def depth_of(conversation: dict[str, Any]) -> int:
    return 1 if conversation["kind"] == "agent" else 0


def _title(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else "Conversation"
    return first[:60] + ("…" if len(first) > 60 else "")


def history(agent: "Agent", conversation_id: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in agent.store.list_messages(conversation_id):
        if message["role"] == "tool":
            messages.append({"role": "tool", "tool_call_id": message["tool_call_id"], "content": message["content"]})
        elif message["role"] == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message["content"] or ""}
            if message.get("reasoning"):
                # Reasoning models such as Kimi K3 require their reasoning to be sent back in later turns.
                entry["reasoning_content"] = message["reasoning"]
            if message["tool_calls"]:
                entry["tool_calls"] = [
                    {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
                    for call in message["tool_calls"]
                ]
            messages.append(entry)
        else:
            content = message["content"]
            if message["sender"] not in {"", "user"}:
                content = f"[Message from the {message['sender']} agent]\n{content}"
            messages.append({"role": "user", "content": content})
    return messages


def run_turn(agent: "Agent", conversation_id: str, text: str | None, *, sender: str = "user") -> Iterator[Event]:
    lock = agent.conversation_lock(conversation_id)
    if not lock.acquire(blocking=False):
        yield {"type": "error", "message": "This conversation is already running."}
        return
    ctx: ToolContext | None = None
    try:
        conversation = agent.store.get_conversation(conversation_id)
        if conversation is None:
            yield {"type": "error", "message": f"No conversation {conversation_id}"}
            return
        if conversation["pending_action"]:
            yield {"type": "error", "message": "Approve or deny the pending action first."}
            return
        if text is not None:
            message = agent.store.add_message(conversation_id, "user", text, sender=sender)
            if not conversation["title"]:
                agent.store.set_title(conversation_id, _title(text))
            yield {"type": "message", "message": message}
        ctx = ToolContext(agent, conversation_id, depth=depth_of(conversation))
        yield from _rounds(ctx)
    finally:
        if ctx:
            _after_turn(ctx)
        lock.release()


def resume(agent: "Agent", conversation_id: str, approve: bool, reason: str = "") -> Iterator[Event]:
    lock = agent.conversation_lock(conversation_id)
    if not lock.acquire(blocking=False):
        yield {"type": "error", "message": "This conversation is already running."}
        return
    ctx: ToolContext | None = None
    try:
        conversation = agent.store.get_conversation(conversation_id)
        action = conversation and conversation["pending_action"]
        if not conversation or not action:
            yield {"type": "error", "message": "Nothing is waiting for approval."}
            return
        ctx = ToolContext(agent, conversation_id, depth=depth_of(conversation))
        agent.store.set_pending_action(conversation_id, None)
        tool = agent.tools.get(action["name"])
        if approve and tool:
            result = _execute(tool, ctx, action["arguments"])
        else:
            result = "The user denied this action." + (f" Reason: {reason}" if reason else "")
            if tool and tool.on_deny:
                try:
                    tool.on_deny(ctx, action["arguments"], reason)
                except Exception:
                    log.exception("on_deny failed for %s", action["name"])
            agent.learning.record_denial(conversation_id, action, reason)
        yield {"type": "decision", "approved": approve, "name": action["name"]}
        yield from _store_result(ctx, action["tool_call_id"], action["name"], result)
        if not (yield from _run_calls(ctx, action.get("remaining", []))):
            yield from _rounds(ctx)
    finally:
        if ctx:
            _after_turn(ctx)
        lock.release()


def collect(events: Iterator[Event]) -> dict[str, Any]:
    """Run a turn to the end and report the outcome, for replying to another agent."""
    reply, status, error = "", "replied", None
    for event in events:
        if event["type"] == "message" and event["message"]["role"] == "assistant" and event["message"]["content"]:
            reply = event["message"]["content"]
        elif event["type"] == "confirm_required":
            status = "awaiting_approval"
        elif event["type"] == "error":
            error = event["message"]
    if status == "replied" and error and not reply:
        status = "error"
    return {"status": status, "reply": reply, "error": error}


def _rounds(ctx: ToolContext) -> Iterator[Event]:
    agent = ctx.agent
    schemas = [tool.schema() for tool in agent.tools_for(ctx.depth)]
    system = agent.system_prompt(ctx.depth)
    rounds = MAX_ROUNDS[min(ctx.depth, 1)]
    for _ in range(rounds):
        messages = [{"role": "system", "content": system}, *history(agent, ctx.conversation_id)]
        content, calls, reasoning = "", [], ""
        try:
            for event in agent.llm.stream(messages, schemas):
                if event["type"] == "token":
                    yield event
                elif event["type"] == "message":
                    content, calls, reasoning = event["content"], event["tool_calls"], event.get("reasoning", "")
        except Exception as err:
            log.exception("Model call failed")
            yield {"type": "error", "message": redact(str(err), agent.settings.secrets)}
            return
        message = agent.store.add_message(
            ctx.conversation_id, "assistant", content, sender=agent.spec.id, tool_calls=calls or None, reasoning=reasoning
        )
        yield {"type": "message", "message": message}
        if not calls:
            yield {"type": "done", "message_id": message["id"]}
            return
        if (yield from _run_calls(ctx, calls)):
            return
    yield {"type": "error", "message": f"Stopped after {rounds} tool rounds. Send another message to continue."}


def _run_calls(ctx: ToolContext, calls: list[dict[str, Any]]) -> Generator[Event, None, bool]:
    """Run tool calls in order. Returns True if one of them is now waiting for approval."""
    agent = ctx.agent
    available = {tool.name: tool for tool in agent.tools_for(ctx.depth)}
    for index, call in enumerate(calls):
        try:
            arguments = json.loads(call["arguments"] or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as err:
            yield from _store_result(ctx, call["id"], call["name"], f"Error: invalid arguments: {err}")
            continue
        yield {"type": "tool_call", "id": call["id"], "name": call["name"], "arguments": arguments}
        tool = available.get(call["name"])
        if tool is None:
            yield from _store_result(ctx, call["id"], call["name"], f"Error: unknown tool {call['name']!r}")
            continue
        if tool.needs_confirmation:
            action = {
                "tool_call_id": call["id"],
                "name": tool.name,
                "arguments": arguments,
                "preview": _preview(tool, ctx, arguments),
                "remaining": calls[index + 1 :],
                "requested_at": now(),
            }
            agent.store.set_pending_action(ctx.conversation_id, action)
            yield {"type": "confirm_required", "conversation_id": ctx.conversation_id, "action": action}
            return True
        yield from _store_result(ctx, call["id"], tool.name, _execute(tool, ctx, arguments))
    return False


def _execute(tool: Tool, ctx: ToolContext, arguments: dict[str, Any]) -> str:
    try:
        result = str(tool.fn(ctx, **arguments))
    except ToolError as err:
        result = f"Error: {err}"
    except TypeError as err:
        result = f"Error: {err}"
    except Exception as err:
        log.exception("Tool %s failed", tool.name)
        result = f"Error: {type(err).__name__}: {err}"
    result = redact(result, ctx.agent.settings.secrets)
    if len(result) > tool.max_result_chars:
        result = result[: tool.max_result_chars] + f"\n…(truncated {len(result) - tool.max_result_chars} characters)"
    return result


def _preview(tool: Tool, ctx: ToolContext, arguments: dict[str, Any]) -> str | None:
    if tool.preview is None:
        return None
    try:
        return redact(str(tool.preview(ctx, **arguments)), ctx.agent.settings.secrets)
    except Exception as err:
        return f"(Preview unavailable: {err})"


def _store_result(ctx: ToolContext, call_id: str, name: str, result: str) -> Iterator[Event]:
    message = ctx.agent.store.add_message(ctx.conversation_id, "tool", result, sender=name, tool_call_id=call_id)
    yield {"type": "tool_result", "id": call_id, "name": name, "content": result, "message_id": message["id"]}


def _after_turn(ctx: ToolContext) -> None:
    if ctx.wrote_files:
        try:
            ctx.agent.codemap.refresh_working_tree()
        except Exception:
            log.exception("Couldn't refresh the codebase map")
    ctx.agent.memory.schedule_summary(ctx.conversation_id)
