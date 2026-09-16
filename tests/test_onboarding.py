"""Onboarding a developer from the UI, and agents reading the registry the onboarding service serves."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agentkit.onboarding import Onboarding, OnboardingConfig, OnboardingError, OnboardRequest, create_admin_app, developer_toml
from agentkit.registry import Peers, load_registry, validate_registry

REGISTRY = [
    {"id": "service1", "name": "Operation", "url": "http://127.0.0.1:9001", "kind": "service",
     "owners": ["dev-pranit"], "oncall": "dev-pranit", "links": ["service2", "dev-pranit"]},
    {"id": "service2", "name": "Consumer", "url": "http://127.0.0.1:9002", "kind": "service", "links": ["service1"]},
    {"id": "dev-pranit", "name": "Pranit's agent", "url": "http://127.0.0.1:9101", "kind": "developer",
     "developer": {"name": "Pranit", "email": "pranit@example.com", "github": "prangunj23"}, "links": ["service1"],
     "local": {"path": "../../ApiAgentDevs", "spec": "dev_agent.spec:dev_pranit", "port": 9101}},
]  # fmt: skip


def fake_http(invites: list, *, github_users=("sam-dev",), collaborator_status=201):
    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if host == "api.github.com":
            if path.startswith("/users/"):
                login = path.rsplit("/", 1)[1]
                return httpx.Response(200, json={"login": "Sam-Dev"}) if login.lower() in github_users else httpx.Response(404)
            if request.method == "PUT" and "/collaborators/" in path:
                invites.append((path, json.loads(request.content), request.headers.get("authorization")))
                return httpx.Response(collaborator_status)
        if path == "/api/info":
            repos = {9001: "acme/ApiAgentService1", 9002: "acme/ApiAgentService2"}
            return httpx.Response(200, json={"repo": {"slug": repos[request.url.port]}})
        return httpx.Response(404)

    return lambda base_url: httpx.Client(base_url=base_url, transport=httpx.MockTransport(handler))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(REGISTRY))
    sent, invites, started = [], [], []
    monkeypatch.setattr("agentkit.tools.email.send", lambda to, subject, text, html: sent.append({"to": to, "subject": subject, "text": text}))
    monkeypatch.setenv("EMAIL_PROVIDER", "resend")
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")

    def build(**overrides):
        values = dict(
            registry_path=registry,
            developers_dir=tmp_path / "developers",
            local_project="../../ApiAgentDevs",
            github_admin_token="ghp_admin",
            http_factory=fake_http(invites),
            starter=lambda entry: started.append(entry) or f"Started {entry['id']}",
        )
        return OnboardingConfig(**(values | overrides))

    return {"registry": registry, "sent": sent, "invites": invites, "started": started, "build": build, "tmp": tmp_path}


def request(**overrides) -> OnboardRequest:
    return OnboardRequest(**({"name": "Sam", "email": "sam@example.com", "github": "sam-dev", "owns": ["service1"]} | overrides))


def test_onboarding_from_the_ui_sets_up_everything(setup):
    client = TestClient(create_admin_app(setup["build"]()), base_url="http://localhost")

    options = client.get("/api/onboarding/options").json()
    assert [service["id"] for service in options["services"]] == ["service1"]
    assert options["developers"] == [{"id": "dev-pranit", "name": "Pranit", "github": "prangunj23"}]
    assert options["github_invites"] is True and options["welcome_email"] is True

    response = client.post("/api/onboard", json={"name": "Sam", "email": "sam@example.com", "github": "sam-dev", "owns": ["service1"]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert (result["agent_id"], result["url"]) == ("dev-sam-dev", "http://127.0.0.1:9102")
    assert [(step["step"], step["status"]) for step in result["steps"]] == [
        ("GitHub login", "done"),
        ("Developer file", "done"),
        ("Registry", "done"),
        ("Agent", "done"),
        ("GitHub access", "done"),
        ("Welcome email", "done"),
    ]

    data = json.loads(setup["registry"].read_text())
    assert validate_registry(load_registry(str(setup["registry"]))) == []
    sam = data[-1]
    assert sam == {
        "id": "dev-sam-dev",
        "name": "Sam's agent",
        "url": "http://127.0.0.1:9102",
        "kind": "developer",
        "developer": {"name": "Sam", "email": "sam@example.com", "github": "Sam-Dev"},
        "links": ["service1", "dev-pranit"],
        "local": {"path": "../../ApiAgentDevs", "spec": "dev_agent.spec:dev_sam_dev", "port": 9102},
    }
    assert data[0]["owners"] == ["dev-pranit", "dev-sam-dev"] and data[0]["oncall"] == "dev-pranit"
    assert "dev-sam-dev" in data[0]["links"] and "dev-sam-dev" in data[2]["links"]
    assert "dev-sam-dev" not in data[1]["links"]
    assert client.get("/registry.json").json() == data

    toml = (setup["tmp"] / "developers" / "dev-sam-dev.toml").read_text()
    assert 'name = "Sam\'s agent"' in toml
    assert 'reads = ["acme/ApiAgentService1", "acme/ApiAgentService2"]' in toml
    assert setup["started"] == [sam]

    [(path, body, auth)] = setup["invites"]
    assert (path, body, auth) == ("/repos/acme/ApiAgentService1/collaborators/Sam-Dev", {"permission": "push"}, "Bearer ghp_admin")

    [email] = setup["sent"]
    assert email["to"] == ["sam@example.com"] and email["subject"] == "Your agent dev-sam-dev is ready"
    assert "http://localhost:5173/#/agents/dev-sam-dev" in email["text"]
    assert "agentkit mcp --agent dev-sam-dev" in email["text"]
    assert "(service1)" in email["text"] and "on call" not in email["text"]

    again = client.post("/api/onboard", json={"name": "Sam", "email": "other@example.com", "github": "sam-dev"})
    assert again.status_code == 409 and "already onboarded" in again.text


def test_on_call_moves_to_the_new_owner(setup):
    result = Onboarding(setup["build"]()).onboard(request(oncall=["service1"]))
    data = json.loads(setup["registry"].read_text())
    assert data[0]["oncall"] == "dev-sam-dev"
    assert "You are on call for service1" in setup["sent"][0]["text"]
    assert result.steps[2].detail.endswith("on call for service1")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"email": "not-an-email"}, "isn't an email address"),
        ({"github": "-bad-"}, "isn't a valid GitHub login"),
        ({"owns": ["service2"]}, "Can't own service2"),
        ({"owns": [], "oncall": ["service1"]}, "requires owning it"),
        ({"email": "PRANIT@example.com"}, "already belongs to dev-pranit"),
        ({"github": "nobody-here"}, "There is no GitHub user nobody-here"),
    ],
)
def test_bad_requests_change_nothing(setup, overrides, message):
    before = setup["registry"].read_text()
    with pytest.raises(OnboardingError, match=message):
        Onboarding(setup["build"]()).onboard(request(**overrides))
    assert setup["registry"].read_text() == before
    assert not (setup["tmp"] / "developers").exists()
    assert setup["started"] == [] and setup["sent"] == [] and setup["invites"] == []


def test_ports_skip_ones_in_use_and_vm_entries_have_no_local_block(setup):
    config = setup["build"](agent_url_base="http://apiagent-devs", local_project=None, starter=None)
    data = [dict(entry, url=entry["url"].replace("127.0.0.1", "apiagent-devs")) for entry in REGISTRY]
    data[2].pop("local")
    _, entry = Onboarding(config).plan(data, request())
    assert entry["url"] == "http://apiagent-devs:9102" and "local" not in entry

    setup["registry"].write_text(json.dumps(data))
    result = Onboarding(config).onboard(request())
    assert result.steps[3].detail == "The devs VM starts agents for new developer files within a minute."


def test_optional_steps_are_skipped_when_not_configured(setup, monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY")
    result = Onboarding(setup["build"](github_admin_token="")).onboard(request())
    steps = {step.step: step for step in result.steps}
    assert steps["GitHub access"].status == "skipped" and "ONBOARDING_GITHUB_TOKEN" in steps["GitHub access"].detail
    assert steps["Welcome email"].status == "skipped" and "RESEND_API_KEY" in steps["Welcome email"].detail
    assert setup["invites"] == [] and setup["sent"] == []


def test_a_failed_invite_is_reported_but_onboarding_finishes(setup):
    config = setup["build"](http_factory=fake_http(setup["invites"], collaborator_status=403))
    steps = {step.step: step for step in Onboarding(config).onboard(request()).steps}
    assert steps["GitHub access"].status == "failed" and "HTTP 403" in steps["GitHub access"].detail
    assert steps["Welcome email"].status == "done"


def test_the_admin_service_only_answers_allowed_hosts(setup):
    client = TestClient(create_admin_app(setup["build"]()), base_url="http://evil.example")
    assert client.get("/registry.json").status_code == 403


def test_developer_files_escape_names():
    assert 'name = "Jo \\"JJ\\" Smith\'s agent"' in developer_toml('Jo "JJ" Smith', [])


def test_agents_keep_the_last_registry_when_its_url_is_down(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_get(url, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=REGISTRY, request=httpx.Request("GET", url))
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", fake_get)
    cache = tmp_path / "registry-cache.json"
    peers = Peers("http://apiagent-devs:9100/registry.json", "service1", "", cache_path=cache)
    peers.URL_TTL_SECONDS = 0
    assert [peer.id for peer in peers.others()] == ["service2", "dev-pranit"]
    assert json.loads(cache.read_text()) == REGISTRY
    assert [peer.id for peer in peers.others()] == ["service2", "dev-pranit"]

    restarted = Peers("http://apiagent-devs:9100/registry.json", "service1", "", cache_path=cache)
    assert [peer.id for peer in restarted.others()] == ["service2", "dev-pranit"]
    assert Peers("http://apiagent-devs:9100/registry.json", "service1", "").others() == []
