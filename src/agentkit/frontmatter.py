"""Minimal `---` frontmatter for markdown files. Values that aren't plain strings are stored as JSON."""

import json
from typing import Any


def _encode(value: Any) -> str:
    if isinstance(value, str) and "\n" not in value:
        try:
            json.loads(value)
        except ValueError:
            return value
    return json.dumps(value)


def dump(meta: dict[str, Any], body: str) -> str:
    header = "\n".join(f"{key}: {_encode(value)}" for key, value in meta.items())
    return f"---\n{header}\n---\n\n{body.lstrip(chr(10))}"


def parse(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta: dict[str, Any] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            meta[key.strip()] = json.loads(value)
        except ValueError:
            meta[key.strip()] = value
    return meta, text[end + 4 :].lstrip("\n")
