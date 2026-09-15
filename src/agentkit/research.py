"""The agent's research folder: markdown notes with frontmatter, plus the codebase map."""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentkit import frontmatter
from agentkit.spec import ToolError

MAP_FILE = "codebase-map.md"


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "note"


def today() -> str:
    return datetime.now(UTC).date().isoformat()


class Research:
    def __init__(self, directory: Path) -> None:
        self.dir = directory
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, file: str) -> Path:
        name = str(file).strip()
        if not name or "/" in name or "\\" in name or name.startswith(".") or not name.endswith(".md"):
            raise ToolError(f"Not a research file name: {file!r}")
        return self.dir / name

    def save(self, title: str, content: str, *, conversation_id: str | None = None, sources: list[str] | None = None) -> str:
        base = f"{today()}-{slugify(title)}"
        name, n = f"{base}.md", 2
        while (self.dir / name).exists():
            name, n = f"{base}-{n}.md", n + 1
        meta = {"title": title, "date": today(), "conversation_id": conversation_id or "", "sources": sources or []}
        (self.dir / name).write_text(frontmatter.dump(meta, content))
        return name

    def list(self) -> list[dict[str, Any]]:
        items = []
        for path in self.dir.glob("*.md"):
            meta, _ = frontmatter.parse(path.read_text())
            stat = path.stat()
            items.append(
                {
                    "file": path.name,
                    "title": str(meta.get("title") or path.stem),
                    "date": str(meta.get("date") or ""),
                    "updated_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec="seconds"),
                    "size": stat.st_size,
                }
            )
        items.sort(key=lambda item: item["updated_at"], reverse=True)
        items.sort(key=lambda item: item["file"] != MAP_FILE)
        return items

    def read(self, file: str) -> str:
        path = self.path(file)
        if not path.is_file():
            raise ToolError(f"No research file named {file!r}")
        return path.read_text()
