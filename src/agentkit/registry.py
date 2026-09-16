"""The registry lists every agent. The UI and the agents read the same file to find each other.

Besides where each agent lives, an entry can say:
- `kind`: "service" (the default; maintains a repo) or "developer" (acts for one person, named in `developer`).
- `owners` / `oncall`: on a service, the developer agents that own it and the one on call.
- `links`: the agents this one may talk to. Links must be symmetric. An entry without `links` may talk to everyone.

The registry is a file, or a URL (the onboarding service serves the deployed one). A URL is cached briefly, and the
last copy that loaded is kept on disk, so an agent keeps working while the registry's host is down.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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
    kind: str = "service"
    # For developer agents: {"name", "email", "github"}.
    developer: dict[str, Any] | None = None
    owners: list[str] = field(default_factory=list)
    oncall: str | None = None
    links: list[str] | None = None

    def linked_to(self, other_id: str) -> bool:
        return self.links is None or other_id in self.links


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def read_registry_data(source: str) -> list[dict[str, Any]]:
    if is_url(source):
        response = httpx.get(source, timeout=10)
        response.raise_for_status()
        data = response.json()
    else:
        data = json.loads(Path(source).expanduser().read_text())
    if not isinstance(data, list):
        raise ValueError("registry.json must be a JSON array")
    return data


def load_registry(source: str) -> list[RegistryEntry]:
    return parse_registry(read_registry_data(source)) if source else []


def parse_registry(data: list[dict[str, Any]]) -> list[RegistryEntry]:
    return [
        RegistryEntry(
            id=item["id"],
            name=item.get("name", item["id"]),
            url=item["url"].rstrip("/"),
            local=item.get("local"),
            kind=item.get("kind", "service"),
            developer=item.get("developer"),
            owners=list(item.get("owners", [])),
            oncall=item.get("oncall"),
            links=list(item["links"]) if "links" in item else None,
        )
        for item in data
    ]


def validate_registry(entries: list[RegistryEntry]) -> list[str]:
    """Problems that would make routing wrong. Empty when the registry is consistent."""
    problems = []
    by_id = {entry.id: entry for entry in entries}
    for entry in entries:
        if entry.kind not in {"service", "developer"}:
            problems.append(f"{entry.id}: unknown kind {entry.kind!r}")
        if entry.kind == "developer" and not (entry.developer or {}).get("email"):
            problems.append(f"{entry.id}: a developer agent needs developer.email")
        for other_id in entry.links or []:
            other = by_id.get(other_id)
            if other is None:
                problems.append(f"{entry.id}: links to unknown agent {other_id!r}")
            elif other.links is not None and entry.id not in other.links:
                problems.append(f"{entry.id} links to {other_id}, but {other_id} doesn't link back")
        for owner_id in entry.owners:
            owner = by_id.get(owner_id)
            if owner is None or owner.kind != "developer":
                problems.append(f"{entry.id}: owner {owner_id!r} isn't a developer agent")
            elif not entry.linked_to(owner_id):
                problems.append(f"{entry.id}: owner {owner_id} isn't in its links")
        if entry.oncall and entry.oncall not in entry.owners:
            problems.append(f"{entry.id}: on-call {entry.oncall!r} isn't one of its owners")
    return problems


class PeerError(Exception):
    pass


class Peers:
    INFO_TTL_SECONDS = 60
    URL_TTL_SECONDS = 15

    def __init__(
        self,
        source: str,
        self_id: str,
        token: str,
        http_factory: HttpFactory = default_http_factory,
        *,
        cache_path: Path | None = None,
    ) -> None:
        self.source = source
        self.self_id = self_id
        self.token = token
        self.http_factory = http_factory
        self.cache_path = cache_path
        self._info: dict[str, tuple[float, dict[str, Any]]] = {}
        self._url_cache: tuple[float, list[dict[str, Any]]] | None = None

    def entries(self) -> list[RegistryEntry]:
        try:
            return parse_registry(self._data())
        except (OSError, ValueError, KeyError, TypeError, httpx.HTTPError):
            return []

    def _data(self) -> list[dict[str, Any]]:
        if not is_url(self.source):
            return read_registry_data(self.source) if self.source else []
        if self._url_cache and time.monotonic() - self._url_cache[0] < self.URL_TTL_SECONDS:
            return self._url_cache[1]
        try:
            data = read_registry_data(self.source)
        except (OSError, ValueError, httpx.HTTPError):
            # Keep working from the last registry that loaded, in memory or on disk.
            if self._url_cache:
                return self._url_cache[1]
            if self.cache_path and self.cache_path.is_file():
                return json.loads(self.cache_path.read_text())
            raise
        self._url_cache = (time.monotonic(), data)
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(data))
        return data

    def me(self) -> RegistryEntry | None:
        return next((entry for entry in self.entries() if entry.id == self.self_id), None)

    def others(self) -> list[RegistryEntry]:
        """The agents this one may talk to: every other entry, narrowed by `links` when set."""
        me = self.me()
        return [entry for entry in self.entries() if entry.id != self.self_id and (me is None or me.linked_to(entry.id))]

    def accepts(self, sender_id: str) -> bool:
        """Whether a message from `sender_id` is allowed in. Both ends check links, so one bad registry copy can't open a path."""
        me = self.me()
        return me is None or me.linked_to(sender_id)

    def get(self, peer_id: str) -> RegistryEntry:
        for entry in self.others():
            if entry.id == peer_id:
                return entry
        known = ", ".join(entry.id for entry in self.others()) or "none"
        if any(entry.id == peer_id for entry in self.entries()):
            raise PeerError(f"You aren't linked to agent {peer_id!r}. Agents you can reach: {known}.")
        raise PeerError(f"No agent named {peer_id!r} in the registry. Other agents: {known}.")

    # Ownership

    def owners(self) -> list[RegistryEntry]:
        """The developer agents that own this agent's service."""
        me = self.me()
        return [peer for peer in self.others() if me and peer.id in me.owners and peer.kind == "developer"]

    def oncall(self) -> RegistryEntry | None:
        me = self.me()
        return next((peer for peer in self.owners() if me and peer.id == me.oncall), None)

    def developer_of(self, agent_id: str) -> dict[str, Any] | None:
        entry = next((entry for entry in self.entries() if entry.id == agent_id), None)
        return entry.developer if entry and entry.kind == "developer" else None

    def owned_by(self, agent_id: str) -> list[RegistryEntry]:
        """Services that list `agent_id` as an owner."""
        return [entry for entry in self.entries() if entry.kind == "service" and agent_id in entry.owners]

    # HTTP

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
        """Send an event to every service agent that reads `repo_slug`. Returns the ids that received it.

        Developer agents are skipped: they hear about their services through the owners route, not as dependents.
        """
        notified = []
        for entry in self.others():
            if entry.kind != "service":
                continue
            try:
                reads = [repo["slug"] for repo in self.info(entry.id).get("reads", [])]
                if repo_slug in reads:
                    self.request(entry.id, "POST", "/api/events/inbound", json={"from_agent": self.self_id, **event})
                    notified.append(entry.id)
            except PeerError:
                continue
        return notified
