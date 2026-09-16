"""Onboarding a developer: their agent, their place in the registry, repo access, and a welcome email.

The onboarding service (`agentkit admin`, or inside `agentkit dev --admin-port`) owns registry.json. It serves the file
at /registry.json, so deployed agents read it by URL, and it is the only thing that writes it.

Onboarding someone:
1. checks the request and that the GitHub login exists, before anything is written;
2. writes developers/<id>.toml and adds the agent to registry.json: linked to every other developer agent and to the
   services it owns, and added to those services' owners (and on-call, if asked). Only services that already list
   owners can be owned, so a service kept away from developers (service2) stays that way;
3. starts the agent (locally), or leaves that to the devs VM, which starts agents for new developer files;
4. invites the GitHub login to the repos of the services it owns;
5. emails the developer how to use their agent.
"""

import json
import logging
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import markdown
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agentkit.config import DEFAULT_UI_ORIGINS
from agentkit.registry import HttpFactory, default_http_factory, parse_registry, validate_registry
from agentkit.security import LocalOnlyMiddleware
from agentkit.tools import email

log = logging.getLogger(__name__)

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
GITHUB_LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
TOML_STRING = re.compile(r'["\\\n]')


class OnboardingError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class OnboardingConfig:
    registry_path: Path
    developers_dir: Path
    # Scheme and host new developer agents are reached at, e.g. http://127.0.0.1 or http://apiagent-devs.
    agent_url_base: str = "http://127.0.0.1"
    first_port: int = 9101
    # Set when running locally: the developer agents' project, relative to the registry file, for its `local` block.
    local_project: str | None = None
    # Repos a new developer's agent reads, on top of every service's repo it can find.
    default_reads: list[str] = field(default_factory=list)
    ui_url: str = "http://localhost:5173"
    ui_origins: list[str] = field(default_factory=lambda: list(DEFAULT_UI_ORIGINS))
    allowed_hosts: set[str] = field(default_factory=lambda: {"localhost", "127.0.0.1"})
    # Needs admin rights on the service repos, to send collaborator invites. Invites are skipped without it.
    github_admin_token: str = ""
    github_api: str = "https://api.github.com"
    http_factory: HttpFactory = default_http_factory
    # Starts a new agent from its registry entry and says what happened. None: something else starts it.
    starter: Callable[[dict[str, Any]], str] | None = None
    mcp_hint: str = "uv run --project <ApiAgentKit> --extra mcp agentkit mcp --agent {agent_id} --registry {registry}"


class OnboardRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    email: str
    github: str
    owns: list[str] = []
    oncall: list[str] = []
    invite: bool = True
    welcome_email: bool = True


class Step(BaseModel):
    step: str
    status: Literal["done", "skipped", "failed"]
    detail: str = ""


class OnboardResult(BaseModel):
    agent_id: str
    url: str
    steps: list[Step]


class ServiceOption(BaseModel):
    id: str
    name: str
    owners: list[str]
    oncall: str | None


class DeveloperSummary(BaseModel):
    id: str
    name: str
    github: str | None


class OnboardingOptions(BaseModel):
    services: list[ServiceOption]
    developers: list[DeveloperSummary]
    github_invites: bool
    welcome_email: bool


def agent_id_for(github: str) -> str:
    return "dev-" + github.lower()


def _port(url: str) -> int | None:
    return urlparse(url).port


def _toml_string(value: str) -> str:
    return '"' + TOML_STRING.sub(lambda m: {'"': '\\"', "\\": "\\\\", "\n": "\\n"}[m.group()], value) + '"'


def developer_toml(name: str, reads: list[str]) -> str:
    return (
        f"# {name}'s agent, created by onboarding. Name, email and GitHub login are in registry.json.\n"
        "# Everything in review_preferences is added to the agent's system prompt as written.\n"
        f"name = {_toml_string(f"{name}'s agent")}\n\n"
        "# Repos the agent can read. It can't change any of them.\n"
        f"reads = [{', '.join(_toml_string(slug) for slug in reads)}]\n\n"
        'review_preferences = """\n"""\n'
    )


class Onboarding:
    def __init__(self, config: OnboardingConfig) -> None:
        self.config = config
        self._lock = threading.Lock()

    # Registry

    def registry_data(self) -> list[dict[str, Any]]:
        return json.loads(self.config.registry_path.read_text())

    def _write_registry(self, data: list[dict[str, Any]]) -> None:
        path = self.config.registry_path
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(temporary, path)

    def options(self) -> dict[str, Any]:
        data = self.registry_data()
        return {
            "services": [
                {"id": e["id"], "name": e.get("name", e["id"]), "owners": e["owners"], "oncall": e.get("oncall")}
                for e in data
                if e.get("kind", "service") == "service" and "owners" in e
            ],
            "developers": [
                {"id": e["id"], "name": (e.get("developer") or {}).get("name", e["id"]), "github": (e.get("developer") or {}).get("github")}
                for e in data
                if e.get("kind") == "developer"
            ],
            "github_invites": bool(self.config.github_admin_token),
            "welcome_email": not email.missing_settings(),
        }

    def plan(self, data: list[dict[str, Any]], request: OnboardRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """The registry with the new developer added, and their entry. Raises OnboardingError; changes nothing."""
        name, address, login = request.name.strip(), request.email.strip(), request.github.strip()
        if not name:
            raise OnboardingError("Enter a name.")
        if not EMAIL.match(address):
            raise OnboardingError(f"{address!r} isn't an email address.")
        if not GITHUB_LOGIN.match(login):
            raise OnboardingError(f"{login!r} isn't a valid GitHub login.")
        agent_id = agent_id_for(login)
        by_id = {entry["id"]: entry for entry in data}
        if agent_id in by_id:
            raise OnboardingError(f"{login} is already onboarded as {agent_id}.", status=409)
        for entry in data:
            if (entry.get("developer") or {}).get("email", "").lower() == address.lower():
                raise OnboardingError(f"{address} already belongs to {entry['id']}.", status=409)
        ownable = {entry["id"] for entry in data if entry.get("kind", "service") == "service" and "owners" in entry}
        if unknown := set(request.owns) - ownable:
            raise OnboardingError(f"Can't own {', '.join(sorted(unknown))}. Services that take owners: {', '.join(sorted(ownable)) or 'none'}.")
        if stray := set(request.oncall) - set(request.owns):
            raise OnboardingError(f"On call for {', '.join(sorted(stray))} requires owning it.")

        base = self.config.agent_url_base.rstrip("/")
        host = urlparse(base).hostname
        used = {_port(entry["url"]) for entry in data if urlparse(entry["url"]).hostname == host}
        port = self.config.first_port
        while port in used:
            port += 1

        data = json.loads(json.dumps(data))
        by_id = {entry["id"]: entry for entry in data}
        developers = [entry["id"] for entry in data if entry.get("kind") == "developer"]
        entry: dict[str, Any] = {
            "id": agent_id,
            "name": f"{name}'s agent",
            "url": f"{base}:{port}",
            "kind": "developer",
            "developer": {"name": name, "email": address, "github": login},
            "links": [*request.owns, *developers],
        }
        if self.config.local_project is not None:
            entry["local"] = {"path": self.config.local_project, "spec": f"dev_agent.spec:{agent_id.replace('-', '_')}", "port": port}
        for other_id in [*request.owns, *developers]:
            other = by_id[other_id]
            if other.get("links") is not None and agent_id not in other["links"]:
                other["links"].append(agent_id)
        for service_id in request.owns:
            by_id[service_id]["owners"].append(agent_id)
        for service_id in request.oncall:
            by_id[service_id]["oncall"] = agent_id
        data.append(entry)
        if problems := validate_registry(parse_registry(data)):
            raise OnboardingError("The registry would be inconsistent: " + "; ".join(problems))
        return data, entry

    # Steps

    def _github(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.config.github_admin_token:
            headers["Authorization"] = f"Bearer {self.config.github_admin_token}"
        client = self.config.http_factory(self.config.github_api)
        try:
            return client.request(method, path, headers=headers, timeout=30, **kwargs)
        finally:
            client.close()

    def check_github_login(self, login: str) -> str:
        try:
            response = self._github("GET", f"/users/{login}")
        except httpx.HTTPError as err:
            raise OnboardingError(f"Couldn't check GitHub login {login}: {err}", status=502) from err
        if response.status_code == 404:
            raise OnboardingError(f"There is no GitHub user {login}.")
        if response.is_error:
            raise OnboardingError(f"Couldn't check GitHub login {login}: HTTP {response.status_code}", status=502)
        return response.json().get("login", login)

    def service_repos(self, data: list[dict[str, Any]]) -> dict[str, str]:
        """Each service agent's repo slug, from its /api/info. Services that don't answer are left out."""
        repos = {}
        for entry in data:
            if entry.get("kind", "service") != "service":
                continue
            client = self.config.http_factory(entry["url"])
            try:
                slug = ((client.get("/api/info", timeout=5).json() or {}).get("repo") or {}).get("slug")
            except (httpx.HTTPError, ValueError):
                continue
            finally:
                client.close()
            if slug:
                repos[entry["id"]] = slug
        return repos

    def invite(self, login: str, slugs: list[str]) -> Step:
        if not slugs:
            return Step(step="GitHub access", status="skipped", detail="They don't own a service with a known repo.")
        if not self.config.github_admin_token:
            return Step(step="GitHub access", status="skipped", detail="Set ONBOARDING_GITHUB_TOKEN to send collaborator invites.")
        done, failed = [], []
        for slug in slugs:
            try:
                response = self._github("PUT", f"/repos/{slug}/collaborators/{login}", json={"permission": "push"})
            except httpx.HTTPError as err:
                failed.append(f"{slug} ({err})")
                continue
            if response.status_code == 201:
                done.append(f"invited to {slug}")
            elif response.status_code == 204:
                done.append(f"already a collaborator on {slug}")
            else:
                failed.append(f"{slug} (HTTP {response.status_code}: {response.text[:120]})")
        detail = "; ".join(done + [f"failed: {item}" for item in failed])
        return Step(step="GitHub access", status="failed" if failed else "done", detail=detail)

    def welcome(self, entry: dict[str, Any], owns: list[str], oncall: list[str]) -> Step:
        if missing := email.missing_settings():
            return Step(step="Welcome email", status="skipped", detail=f"Email isn't configured: {', '.join(missing)}.")
        developer = entry["developer"]
        agent_id = entry["id"]
        ui = f"{self.config.ui_url.rstrip('/')}/#/agents/{agent_id}"
        lines = [
            f"Hi {developer['name']},",
            "",
            f"You now have your own agent, **{entry['name']}** (`{agent_id}`). It works only for you:",
            "",
            "- It tells you about the services you own"
            + (f" ({', '.join(owns)})" if owns else "")
            + ", and decides what is worth an email.",
            "- Other developers' agents can ask it to review their pull requests. It emails you a draft review first;"
            " nothing is posted on GitHub unless you ask it to and approve.",
            "- You can ask it to get reviews on your own pull requests.",
        ]
        if oncall:
            lines.append(f"- You are on call for {', '.join(oncall)}, so urgent news about it comes to you.")
        lines += [
            "",
            f"Chat with it in the agent UI: {ui}",
            "",
            "To use it from Claude Code, add it as an MCP server:",
            "",
            "```",
            "claude mcp add apiagent -- " + self.config.mcp_hint.format(agent_id=agent_id, registry=self.config.registry_path),
            "```",
            "",
            "To tell it how you like reviews done, add review_preferences to its developer file, "
            f"developers/{agent_id}.toml.",
        ]
        body = "\n".join(lines)
        subject = f"Your agent {agent_id} is ready"
        try:
            email.send([developer["email"]], subject, body, markdown.markdown(body, extensions=["fenced_code"]))
        except Exception as err:
            return Step(step="Welcome email", status="failed", detail=str(err))
        return Step(step="Welcome email", status="done", detail=f"Sent to {developer['email']}.")

    def onboard(self, request: OnboardRequest) -> OnboardResult:
        with self._lock:
            data, entry = self.plan(self.registry_data(), request)
            login = self.check_github_login(request.github.strip())
            entry["developer"]["github"] = login
            repos = self.service_repos(data)
            reads = sorted({*self.config.default_reads, *repos.values()})

            self.config.developers_dir.mkdir(parents=True, exist_ok=True)
            toml_path = self.config.developers_dir / f"{entry['id']}.toml"
            toml_path.write_text(developer_toml(request.name.strip(), reads))
            self._write_registry(data)
            steps = [
                Step(step="GitHub login", status="done", detail=f"@{login} exists."),
                Step(step="Developer file", status="done", detail=f"{toml_path} (reads {', '.join(reads) or 'nothing'})"),
                Step(
                    step="Registry",
                    status="done",
                    detail=f"{entry['id']} at {entry['url']}, linked to {', '.join(entry['links']) or 'nobody'}"
                    + (f"; owns {', '.join(request.owns)}" if request.owns else "")
                    + (f"; on call for {', '.join(request.oncall)}" if request.oncall else ""),
                ),
            ]

        if self.config.starter:
            try:
                steps.append(Step(step="Agent", status="done", detail=self.config.starter(entry)))
            except Exception as err:
                log.exception("Couldn't start %s", entry["id"])
                steps.append(Step(step="Agent", status="failed", detail=str(err)))
        else:
            steps.append(Step(step="Agent", status="done", detail="The devs VM starts agents for new developer files within a minute."))

        owned_repos = [repos[service] for service in request.owns if service in repos]
        steps.append(self.invite(login, owned_repos) if request.invite else Step(step="GitHub access", status="skipped", detail="Not requested."))
        steps.append(
            self.welcome(entry, request.owns, request.oncall)
            if request.welcome_email
            else Step(step="Welcome email", status="skipped", detail="Not requested.")
        )
        return OnboardResult(agent_id=entry["id"], url=entry["url"], steps=steps)


def create_admin_app(config: OnboardingConfig) -> FastAPI:
    onboarding = Onboarding(config)
    app = FastAPI(title="Onboarding", version="1")
    app.state.onboarding = onboarding
    app.add_middleware(LocalOnlyMiddleware, allowed_hosts=config.allowed_hosts, allowed_origins=config.ui_origins)
    app.add_middleware(CORSMiddleware, allow_origins=config.ui_origins, allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "onboarding"}

    @app.get("/registry.json")
    def registry() -> list[dict[str, Any]]:
        return onboarding.registry_data()

    @app.get("/api/onboarding/options", response_model=OnboardingOptions)
    def options() -> dict[str, Any]:
        return onboarding.options()

    @app.post("/api/onboard", response_model=OnboardResult)
    def onboard(body: OnboardRequest) -> OnboardResult:
        try:
            return onboarding.onboard(body)
        except OnboardingError as err:
            raise HTTPException(err.status, str(err)) from err

    return app
