"""The registry lists every agent. The UI and the agents read the same file to find each other."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

HttpFactory = Callable[[str], httpx.Client]


def default_http_factory(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=30)


@dataclass
class RegistryEntry:
    id: str
    name: str
    url: str
    local: dict[str, Any] | None = None


def load_registry(source: str) -> list[RegistryEntry]:
    if not source:
        return []
    if source.startswith(("http://", "https://")):
        data = httpx.get(source, timeout=10).json()
    else:
        data = json.loads(Path(source).expanduser().read_text())
    return [
        RegistryEntry(id=item["id"], name=item.get("name", item["id"]), url=item["url"].rstrip("/"), local=item.get("local"))
        for item in data
    ]


class PeerError(Exception):
    pass


class Peers:
    INFO_TTL_SECONDS = 60

    def __init__(self, source: str, self_id: str, token: str, http_factory: HttpFactory = default_http_factory) -> None:
        self.source = source
        self.self_id = self_id
        self.token = token
        self.http_factory = http_factory
        self._info: dict[str, tuple[float, dict[str, Any]]] = {}

    def entries(self) -> list[RegistryEntry]:
        try:
            return load_registry(self.source)
        except (OSError, ValueError, KeyError, httpx.HTTPError):
            return []

    def others(self) -> list[RegistryEntry]:
        return [entry for entry in self.entries() if entry.id != self.self_id]

    def get(self, peer_id: str) -> RegistryEntry:
        for entry in self.others():
            if entry.id == peer_id:
                return entry
        known = ", ".join(entry.id for entry in self.others()) or "none"
        raise PeerError(f"No agent named {peer_id!r} in the registry. Other agents: {known}.")

    def request(
        self, peer_id: str, method: str, path: str, *, json: Any = None, params: dict[str, Any] | None = None, timeout: float = 30
    ) -> Any:
        entry = self.get(peer_id)
        headers = {"X-Agent-Token": self.token} if self.token else {}
        client = self.http_factory(entry.url)
        try:
            response = client.request(method, path, json=json, params=params, headers=headers, timeout=timeout)
        except httpx.TransportError as err:
            raise PeerError(f"Couldn't reach agent {peer_id} at {entry.url}: {err}") from err
        finally:
            client.close()
        if response.is_error:
            raise PeerError(f"Agent {peer_id} returned {response.status_code}: {response.text[:500]}")
        return response.json()

    def info(self, peer_id: str) -> dict[str, Any]:
        cached = self._info.get(peer_id)
        if cached and time.monotonic() - cached[0] < self.INFO_TTL_SECONDS:
            return cached[1]
        info = self.request(peer_id, "GET", "/api/info")
        self._info[peer_id] = (time.monotonic(), info)
        return info

    def notify_dependents(self, repo_slug: str, event: dict[str, Any]) -> list[str]:
        """Send an event to every agent that reads `repo_slug`. Returns the ids that received it."""
        notified = []
        for entry in self.others():
            try:
                reads = [repo["slug"] for repo in self.info(entry.id).get("reads", [])]
                if repo_slug in reads:
                    self.request(entry.id, "POST", "/api/events/inbound", json={"from_agent": self.self_id, **event})
                    notified.append(entry.id)
            except PeerError:
                continue
        return notified
