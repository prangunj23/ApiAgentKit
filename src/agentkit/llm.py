"""NVIDIA NIM through the OpenAI SDK: streaming chat with tools, plus plain completions."""

import json
import re
from collections.abc import Iterator
from typing import Any, Protocol

import openai
from openai import OpenAI

from agentkit.config import Settings


class LLM(Protocol):
    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Yield {"type": "token", "text"} events, then one {"type": "message", "content", "tool_calls"}."""
        ...

    def complete(self, system: str, user: str) -> str: ...


def strip_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a model reply, ignoring reasoning blocks and code fences."""
    text = strip_reasoning(text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in model response")
    return json.loads(text[start : end + 1])


def complete_json(llm: LLM, system: str, user: str) -> dict[str, Any]:
    reply = llm.complete(system, user)
    try:
        return extract_json(reply)
    except ValueError:
        return extract_json(llm.complete(system, f"{user}\n\nYour last reply was not valid JSON. Reply with only the JSON object."))


class ThinkFilter:
    """Drops <think>…</think> blocks from streamed text, even when a tag is split across chunks."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buffer = ""
        self.thinking = False

    def feed(self, text: str) -> str:
        self.buffer += text
        out: list[str] = []
        while True:
            tag = self.CLOSE if self.thinking else self.OPEN
            index = self.buffer.find(tag)
            if index == -1:
                keep = _partial_tag_length(self.buffer, tag)
                emit = self.buffer[: len(self.buffer) - keep]
                self.buffer = self.buffer[len(self.buffer) - keep :]
                if not self.thinking:
                    out.append(emit)
                return "".join(out)
            if not self.thinking:
                out.append(self.buffer[:index])
            self.buffer = self.buffer[index + len(tag) :]
            self.thinking = not self.thinking

    def flush(self) -> str:
        rest, self.buffer = self.buffer, ""
        return "" if self.thinking else rest


def _partial_tag_length(text: str, tag: str) -> int:
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class NimLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.settings.nvidia_api_key:
                raise RuntimeError("NVIDIA_API_KEY is not set. Add it to this agent's .env file.")
            self._client = OpenAI(
                base_url=self.settings.nim_base_url, api_key=self.settings.nvidia_api_key, timeout=600, max_retries=5
            )
        return self._client

    def _connection_error(self, err: openai.APIConnectionError) -> RuntimeError:
        # The SDK only says "Connection error."; include the underlying network error.
        return RuntimeError(f"Couldn't reach NVIDIA NIM at {self.settings.nim_base_url}: {err.__cause__!r}")

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        request: dict[str, Any] = {
            "model": self.settings.nim_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 16384,
            "stream": True,
        }
        if tools:
            request["tools"] = tools

        text = ThinkFilter()
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        try:
            for chunk in self.client.chat.completions.create(**request):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if piece := getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None):
                    reasoning.append(piece)
                if delta.content:
                    if piece := text.feed(delta.content):
                        content.append(piece)
                        yield {"type": "token", "text": piece}
                for call in delta.tool_calls or []:
                    entry = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                    entry["id"] = call.id or entry["id"]
                    if call.function:
                        entry["name"] += call.function.name or ""
                        entry["arguments"] += call.function.arguments or ""
        except openai.APIConnectionError as err:
            raise self._connection_error(err) from err

        if rest := text.flush():
            content.append(rest)
            yield {"type": "token", "text": rest}
        tool_calls = [{**call, "id": call["id"] or f"call_{index}"} for index, call in sorted(calls.items())]
        yield {"type": "message", "content": "".join(content).strip(), "tool_calls": tool_calls, "reasoning": "".join(reasoning)}

    def complete(self, system: str, user: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.settings.nim_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=8192,
            )
        except openai.APIConnectionError as err:
            raise self._connection_error(err) from err
        return strip_reasoning(completion.choices[0].message.content or "")
