"""Developer agents: agents without a repo, owner routing through the registry, and PR reviews between developers."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agentkit import AgentSpec, ToolContext
from agentkit.registry import Peers, load_registry, validate_registry
from agentkit.testing import FakeLLM, commit_files, http_factory_for
from agentkit.tools import REPO_WRITE_TOOLS
from agentkit.tools.owner import notify_developer, notify_owners
from agentkit.tools.review import REVIEW_TOOLS, build_review_context
from helpers import OPERATION_APP, SERVICE1, SERVICE2, build_agent, build_app, decide, describer, send, service1_spec, service2_spec, start

URLS = {
    "service1": "http://127.0.0.1:9001",
    "service2": "http://127.0.0.1:9002",
    "dev-alice": "http://127.0.0.1:9101",
    "dev-bob": "http://127.0.0.1:9102",
    "dev-carol": "http://127.0.0.1:9103",
}
PR_URL = "https://github.com/acme/ApiAgentService1/pull/42"
EMAILS = {"dev-alice": "alice@example.com", "dev-bob": "bob@example.com", "dev-carol": "carol@example.com"}
NAMES = {"dev-alice": "Alice Chen", "dev-bob": "Bob Ruiz", "dev-carol": "Carol Diaz"}


def registry_entries() -> list[dict]:
    """The user's topology: devs ↔ service1, alice ↔ bob, service1 ↔ service2. Carol only reviews for Alice."""
    developer = lambda agent_id, links: {  # noqa: E731
        "id": agent_id,
        "name": f"{NAMES[agent_id].split()[0]}'s agent",
        "url": URLS[agent_id],
        "kind": "developer",
        "developer": {"name": NAMES[agent_id], "email": EMAILS[agent_id], "github": agent_id.removeprefix("dev-")},
        "links": links,
    }
    return [
        {
            "id": "service1",
            "name": "Operation",
            "url": URLS["service1"],
            "owners": ["dev-alice", "dev-bob"],
            "oncall": "dev-alice",
            "links": ["service2", "dev-alice", "dev-bob"],
        },
        {"id": "service2", "name": "Consumer", "url": URLS["service2"], "links": ["service1"]},
        developer("dev-alice", ["service1", "dev-bob", "dev-carol"]),
        developer("dev-bob", ["service1", "dev-alice"]),
        developer("dev-carol", ["dev-alice"]),
    ]


def dev_spec(agent_id: str) -> AgentSpec:
    return AgentSpec(
        id=agent_id,
        name=f"{NAMES[agent_id].split()[0]}'s agent",
        system_prompt="You act for one developer.",
        reads=[SERVICE1, SERVICE2],
        tools=[notify_developer, *REVIEW_TOOLS],
        excluded_tools=set(REPO_WRITE_TOOLS),
        features={"emails"},
    )


@pytest.fixture
def emails(monkeypatch):
    sent = []
    monkeypatch.setattr("agentkit.tools.email.send", lambda to, subject, text, html: sent.append({"to": to, "subject": subject, "text": text}))
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    return sent


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry_entries()))
    return path


def network(tmp_path, remotes, registry, llms: dict[str, FakeLLM], github=None) -> dict:
    """In-process agents keyed by id, talking to each other over routed HTTP."""
    apps: dict = {}
    fallback = (lambda url: httpx.Client(base_url=url, transport=httpx.MockTransport(github))) if github else None
    factory = http_factory_for(apps, fallback=fallback)
    specs = {"service1": service1_spec(tools=[notify_owners]), "service2": service2_spec()}
    for agent_id, llm in llms.items():
        spec = specs.get(agent_id) or dev_spec(agent_id)
        port = int(URLS[agent_id].rsplit(":", 1)[1])
        apps[URLS[agent_id]] = build_app(
            tmp_path, remotes, spec, llm=llm, port=port, http_factory=factory, registry=str(registry), github_token="ghp_testtoken123"
        )
    return {agent_id: apps[URLS[agent_id]] for agent_id in llms}


def client(apps: dict, agent_id: str) -> TestClient:
    return TestClient(apps[agent_id], base_url=URLS[agent_id])


def settle(app) -> None:
    """Wait for every turn an agent is running, e.g. the triage a notification started in the background."""
    agent = app.state.agent
    for conversation in agent.store.list_conversations():
        if run := agent.runs.get(conversation["id"]):
            for _ in run.follow(heartbeat_seconds=0.05):
                pass


def triage_reply(subject: str = "service1 changed") -> FakeLLM:
    """A developer agent that emails its developer and then replies."""
    return FakeLLM([{"tool_calls": [("notify_developer", {"subject": subject, "body_markdown": "Heads up."})]}, "Told my developer."])


# An agent without a repo


def test_an_agent_can_own_no_repo(tmp_path, remotes, registry):
    app = build_app(tmp_path, remotes, dev_spec("dev-alice"), registry=str(registry))
    api = TestClient(app)
    info = api.get("/api/info").json()
    assert info["repo"] is None and info["kind"] == "developer"
    tools = {tool["name"] for tool in info["tools"]}
    assert tools.isdisjoint(REPO_WRITE_TOOLS) and {"read_file", "notify_developer", "request_reviews"} <= tools

    repos = api.get("/api/workspace").json()["repos"]
    assert [(repo["name"], repo["writable"], repo["cloned"]) for repo in repos] == [
        ("ApiAgentService1", False, True),
        ("ApiAgentService2", False, True),
    ]
    assert api.post("/api/codebase-map/rebuild").status_code == 409
    assert api.post("/api/workspace/sync").status_code == 200

    agent = app.state.agent
    read = agent.tools["read_file"].fn
    assert read(ToolContext(agent, "c"), path="README.md", repo="ApiAgentService1").startswith("# operation")
    with pytest.raises(Exception, match="owns no repo"):
        read(ToolContext(agent, "c"), path="README.md")
    assert agent.codemap.prompt_excerpt() == "" and not agent.codemap.is_stale()


def test_developer_prompt_names_the_person_and_only_linked_agents(tmp_path, remotes, registry):
    alice = build_agent(tmp_path, remotes, dev_spec("dev-alice"), registry=str(registry))
    prompt = alice.system_prompt(0)
    assert "You work for Alice Chen <alice@example.com> (GitHub @alice)." in prompt
    assert "- They own: `service1` (Operation)." in prompt
    assert "- They are on call for: `service1`." in prompt
    assert "You own no repo. You can read: acme/ApiAgentService1 (`ApiAgentService1`), acme/ApiAgentService2" in prompt
    assert "`service1` (Operation), `dev-bob` (Bob's agent), `dev-carol` (Carol's agent)." in prompt
    assert "service2" not in prompt.split("Other agents you can ask", 1)[1].splitlines()[0]
    assert "- These tools need the user's approval: post_review_comment." in prompt
    assert len(prompt) < 20_000

    bob = build_agent(tmp_path, remotes, dev_spec("dev-bob"), registry=str(registry))
    assert "- They own: `service1`" in bob.system_prompt(0) and "on call" not in bob.system_prompt(0)

    service1 = build_agent(tmp_path, remotes, service1_spec(tools=[notify_owners]), registry=str(registry))
    assert (
        "- This service is owned by `dev-alice` (Alice Chen), `dev-bob` (Bob Ruiz); `dev-alice` is on call." in service1.system_prompt(0)
    )
    service2 = build_agent(tmp_path, remotes, service2_spec(), registry=str(registry))
    assert "Other agents you can ask with message_agent: `service1` (Operation)." in service2.system_prompt(0)


# The registry


def test_links_limit_who_an_agent_can_reach(registry):
    service2 = Peers(str(registry), "service2", "token")
    assert [peer.id for peer in service2.others()] == ["service1"]
    with pytest.raises(Exception, match="aren't linked to agent 'dev-alice'"):
        service2.get("dev-alice")
    assert service2.owners() == [] and service2.oncall() is None

    service1 = Peers(str(registry), "service1", "token")
    assert [peer.id for peer in service1.owners()] == ["dev-alice", "dev-bob"]
    assert service1.oncall().id == "dev-alice"
    assert service1.developer_of("dev-bob")["email"] == "bob@example.com"
    assert service1.developer_of("service2") is None
    assert [entry.id for entry in service1.owned_by("dev-bob")] == ["service1"]
    assert validate_registry(load_registry(str(registry))) == []


def test_agents_refuse_messages_from_agents_they_arent_linked_to(tmp_path, remotes, registry):
    alice = TestClient(build_app(tmp_path, remotes, dev_spec("dev-alice"), clone=False, registry=str(registry)))
    headers = {"X-Agent-Token": "test-shared-token"}
    body = {"from_agent": "service2", "thread_id": "t1", "message": "Hi Alice"}
    response = alice.post("/api/agent-messages", json=body, headers=headers)
    assert response.status_code == 403 and "isn't linked to service2" in response.text
    assert alice.post("/api/events/inbound", json={"from_agent": "service2", "summary": "x"}, headers=headers).status_code == 403
    assert alice.post("/api/agent-messages/reply", json={"from_agent": "service2", "thread_id": "t1", "reply": "x"}, headers=headers).status_code == 403
    assert alice.post("/api/events/inbound", json={"from_agent": "service1", "summary": "x"}, headers=headers).status_code == 200


def test_entries_without_links_keep_reaching_everyone(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps([{"id": "a", "url": "http://a"}, {"id": "b", "url": "http://b"}, {"id": "c", "url": "http://c"}]))
    assert [peer.id for peer in Peers(str(path), "a", "").others()] == ["b", "c"]
    assert [entry.kind for entry in load_registry(str(path))] == ["service"] * 3


def test_inconsistent_registries_are_reported(tmp_path):
    entries = registry_entries()
    entries[1]["links"] = []  # service1 links to service2, but service2 no longer links back
    entries[0]["oncall"] = "dev-carol"  # not an owner
    entries[3]["developer"] = {"name": "Bob"}  # no email
    problems = validate_registry(load_registry(_write(tmp_path, entries)))
    assert "service1 links to service2, but service2 doesn't link back" in problems
    assert "service1: on-call 'dev-carol' isn't one of its owners" in problems
    assert "dev-bob: a developer agent needs developer.email" in problems


def _write(tmp_path, entries) -> str:
    path = tmp_path / "bad-registry.json"
    path.write_text(json.dumps(entries))
    return str(path)


# Service1 → owners


def test_urgent_news_goes_to_the_oncall_owner_only(tmp_path, remotes, registry, emails):
    call = {"headline": "numeric_op now returns `value`", "detail_markdown": "Service2 still reads `result`.", "urgency": "urgent", "url": "https://x/1"}
    apps = network(
        tmp_path,
        remotes,
        registry,
        {"service1": FakeLLM([{"tool_calls": [("notify_owners", call)]}, "Done."]), "dev-alice": triage_reply(), "dev-bob": FakeLLM()},
    )
    events = send(client(apps, "service1"), start(client(apps, "service1")), "Tell whoever is on call")
    result = next(event for event in events if event["type"] == "tool_result")
    assert result["content"] == "- dev-alice received it and is deciding whether to email its developer."
    settle(apps["dev-alice"])

    alice_llm = apps["dev-alice"].state.agent.llm
    received = alice_llm.calls[0]["messages"][-1]["content"]
    assert received.startswith("[Message from the service1 agent]\n[service1] News about acme/ApiAgentService1.")
    assert "You are receiving this as: on-call owner." in received
    assert "Urgency: urgent\nHeadline: numeric_op now returns `value`" in received
    assert "Link: https://x/1" in received and "Alice Chen owns this service." in received
    depth1_tools = [tool["function"]["name"] for tool in alice_llm.calls[0]["tools"]]
    assert "notify_developer" in depth1_tools and "request_reviews" not in depth1_tools and "message_agent" not in depth1_tools

    [email] = emails
    assert email["to"] == ["alice@example.com"] and email["subject"] == "service1 changed"
    assert "Sent by dev-alice (Alice's agent)" in email["text"] and "/#/agents/dev-alice" in email["text"]
    assert apps["dev-bob"].state.agent.llm.calls == []
    assert any(event["type"] == "owner_notified" for event in client(apps, "service1").get("/api/events").json())


def test_fyi_news_goes_to_every_owner(tmp_path, remotes, registry, emails):
    call = {"headline": "Docs updated", "detail_markdown": "README now says subtract.", "urgency": "fyi"}
    apps = network(
        tmp_path,
        remotes,
        registry,
        {"service1": FakeLLM([{"tool_calls": [("notify_owners", call)]}, "Done."]), "dev-alice": FakeLLM(["Not worth an email."]), "dev-bob": triage_reply("Docs")},
    )
    send(client(apps, "service1"), start(client(apps, "service1")), "Let the owners know")
    settle(apps["dev-alice"])
    settle(apps["dev-bob"])
    assert "You are receiving this as: owner." in apps["dev-bob"].state.agent.llm.calls[0]["messages"][-1]["content"]
    assert [email["to"] for email in emails] == [["bob@example.com"]]


def test_service2_news_is_relayed_by_service1_after_its_reply(tmp_path, remotes, registry, emails):
    relay = {"headline": "service2 will break", "detail_markdown": "operation_client.py reads `result`.", "urgency": "needs_attention"}
    apps = network(
        tmp_path,
        remotes,
        registry,
        {
            "service2": FakeLLM([{"tool_calls": [("message_agent", {"to": "service1", "message": "Your rename breaks me."})]}, "Told service1."]),
            "service1": FakeLLM([{"tool_calls": [("notify_owners", relay)]}, "Noted; I've told the on-call owner."]),
            "dev-alice": triage_reply("service2 will break"),
        },
    )
    service1 = apps["service1"].state.agent
    events = send(client(apps, "service2"), start(client(apps, "service2")), "Warn service1")
    assert "Noted; I've told the on-call owner." in next(e for e in events if e["type"] == "tool_result")["content"]
    assert service1.llm.calls[1]["messages"][-1]["content"] == "Queued for dev-alice; it will be delivered after this reply."

    [queued] = client(apps, "service1").get("/api/outbox").json()
    assert (queued["recipient"], queued["status"]) == ("dev-alice", "queued")
    assert apps["dev-alice"].state.agent.llm.calls == [] and emails == []

    assert service1.deliver_outbox() == 1
    settle(apps["dev-alice"])
    received = apps["dev-alice"].state.agent.llm.calls[0]["messages"][-1]["content"]
    assert "Reported by: service2." in received and "Headline: service2 will break" in received
    assert [email["to"] for email in emails] == [["alice@example.com"]]
    assert client(apps, "service1").get("/api/outbox").json()[0]["status"] == "sent"
    assert service1.deliver_outbox() == 0

    # Service1 crashed before recording the delivery: the redelivery is recognized, and nothing runs twice.
    service1.store.update_outbox(queued["id"], status="queued", attempts=1)
    assert service1.deliver_outbox() == 1
    settle(apps["dev-alice"])
    assert len(apps["dev-alice"].state.agent.llm.calls) == 2
    assert [email["to"] for email in emails] == [["alice@example.com"]]
    [thread] = client(apps, "dev-alice").get("/api/conversations", params={"kind": "agent"}).json()
    assert thread["thread_id"] == queued["id"]


def test_notifications_dont_wait_for_the_owners_turn(tmp_path, remotes, registry):
    apps = network(tmp_path, remotes, registry, {"dev-alice": FakeLLM(["Noted."])})
    api = client(apps, "dev-alice")
    body = {"from_agent": "service1", "thread_id": "t-1", "message": "News", "wait": False}
    headers = {"X-Agent-Token": "test-shared-token"}
    first = api.post("/api/agent-messages", json=body, headers=headers).json()
    assert first["status"] == "accepted" and first["reply"] == ""
    settle(apps["dev-alice"])
    again = api.post("/api/agent-messages", json=body, headers=headers).json()
    assert (again["status"], again["reply"]) == ("duplicate", "Noted.")
    assert len(apps["dev-alice"].state.agent.llm.calls) == 1


def test_undeliverable_notifications_fail_after_three_attempts(tmp_path, remotes, registry):
    # No dev-alice app: every delivery gets a 503.
    service1 = network(tmp_path, remotes, registry, {"service1": FakeLLM()})["service1"].state.agent
    notify_owners.fn(ToolContext(service1, "c", depth=1), headline="Down", detail_markdown="It broke.", urgency="urgent")
    for _ in range(3):
        assert service1.deliver_outbox() == 0
    [row] = service1.store.list_outbox()
    assert (row["status"], row["attempts"]) == ("failed", 3) and "503" in row["error"]
    assert any(event["type"] == "owner_notify_failed" for event in service1.store.list_events())


def test_a_contract_change_is_queued_for_the_oncall_owner(tmp_path, remotes, registry):
    apps = network(tmp_path, remotes, registry, {"service1": FakeLLM(responder=describer)})
    service1 = apps["service1"].state.agent
    service1.codemap.update()
    commit_files(remotes["service1"], {"src/operation/app.py": OPERATION_APP.replace('"/numeric_op"', '"/subtract"')}, "Rename endpoint")
    service1.sync_workspace()

    [row] = service1.store.list_outbox()
    assert (row["recipient"], row["urgency"]) == ("dev-alice", "needs_attention")
    assert "/v1/operation/subtract" in row["message"]


def test_notify_developer_needs_an_address(tmp_path, remotes, registry):
    service1 = build_agent(tmp_path, remotes, service1_spec(tools=[notify_developer]), registry=str(registry))
    with pytest.raises(Exception, match="no developer.email for service1"):
        notify_developer.fn(ToolContext(service1, "c"), subject="Hi", body_markdown="Hi")


# Reviews


def fake_github(comments_posted: list | None = None):
    long_file = "x = 1\n" * 2_000
    files = [
        {"filename": "src/operation/models.py", "status": "modified", "additions": 1, "deletions": 1, "changes": 2,
         "patch": "@@ -8 +8 @@\n-    result: float\n+    value: float"},
        {"filename": "src/operation/big.py", "status": "added", "additions": 2000, "deletions": 0, "changes": 2000, "patch": "@@ +1 @@\n+x = 1"},
        {"filename": "uv.lock", "status": "modified", "additions": 3, "deletions": 3, "changes": 6, "patch": "@@ lock noise @@"},
        {"filename": "docs/diagram.png", "status": "added", "additions": 0, "deletions": 0, "changes": 0},
        {"filename": "src/operation/old.py", "status": "removed", "additions": 0, "deletions": 4, "changes": 4, "patch": "@@ -1,4 +0,0 @@"},
    ]  # fmt: skip
    contents = {"src/operation/models.py": "class NumericOpResponse(BaseModel):\n    value: float\n", "src/operation/big.py": long_file}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/repos/acme/ApiAgentService1/issues/42/comments":
            comments_posted.append(json.loads(request.content)["body"])
            return httpx.Response(201, json={"html_url": f"{PR_URL}#issuecomment-1"})
        if path == "/repos/acme/ApiAgentService1/pulls/42":
            return httpx.Response(200, json={
                "title": "Rename result -> value", "state": "open", "draft": False, "merged": False, "body": "Renames the field.",
                "user": {"login": "alice"}, "html_url": PR_URL, "created_at": "2026-09-16T10:00:00Z", "updated_at": "2026-09-16T11:00:00Z",
                "head": {"sha": "abc1234def5678", "ref": "rename-result"}, "base": {"ref": "main"},
            })  # fmt: skip
        if path == "/repos/acme/ApiAgentService1/pulls/42/files":
            return httpx.Response(200, json=files if request.url.params["page"] == "1" else [])
        if path.startswith("/repos/acme/ApiAgentService1/contents/"):
            assert request.url.params["ref"] == "abc1234def5678"
            return httpx.Response(200, text=contents[path.split("/contents/", 1)[1]])
        if path == "/repos/acme/ApiAgentService1/commits/abc1234def5678/check-runs":
            return httpx.Response(200, json={"check_runs": [{"name": "tests", "conclusion": "failure", "html_url": "https://ci/1"}]})
        if path == "/repos/acme/ApiAgentService1/pulls/42/comments":
            return httpx.Response(200, json=[{"id": 1, "user": {"login": "carol"}, "body": "Why rename?", "path": "src/operation/models.py"}])
        if path == "/repos/acme/ApiAgentService1/issues/42/comments":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    return handler


def test_review_context_has_every_section_in_order(tmp_path, remotes, registry):
    apps = network(tmp_path, remotes, registry, {"dev-bob": FakeLLM(), "service1": FakeLLM()}, github=fake_github())
    context = build_review_context(apps["dev-bob"].state.agent, PR_URL)

    headings = [line for line in context.splitlines() if line.startswith(("# ", "## "))]
    assert headings == [
        "# Pull request acme/ApiAgentService1#42",
        "## Pull request",
        "## Description",
        "## Checks",
        "## Files changed",
        "## Diff",
        "## Full content after the change (at abc1234)",
        "## Existing comments",
        "## Where to look next",
    ]
    assert "- Author: @alice" in context and "- Branches: rename-result → main" in context
    assert "- tests: failure (https://ci/1)" in context
    assert "| uv.lock | modified | +3 / -3 |" in context
    assert "Not shown below (lock files, vendored or generated): uv.lock." in context
    assert "lock noise" not in context
    assert "+    value: float" in context
    assert "### docs/diagram.png (added)\n_No patch: the file is binary or too large._" in context
    assert "### src/operation/models.py\n```\nclass NumericOpResponse" in context
    assert "Too large to include: src/operation/big.py." in context
    assert "- @carol on src/operation/models.py: Why rename?" in context
    assert 'repo="ApiAgentService1"' in context
    assert 'read_research(agent="service1", file="codebase-map.md")' in context
    assert "alice@example.com" not in context


def test_reviews_reach_both_inboxes(tmp_path, remotes, registry, emails):
    def reviewer(verdict: str, subject: str) -> FakeLLM:
        return FakeLLM(
            [
                {"tool_calls": [("review_pull_request", {"pr_url": PR_URL})]},
                {"tool_calls": [("notify_developer", {"subject": subject, "body_markdown": verdict, "url": PR_URL})]},
                verdict,
            ]
        )

    alice = FakeLLM(
        [
            {"tool_calls": [("request_reviews", {"pr_url": PR_URL, "reviewers": ["dev-bob", "dev-carol"], "focus": "Does it break service2?"})]},
            {"tool_calls": [("notify_developer", {"subject": "2 reviews on your PR", "body_markdown": "Bob blocks it.", "url": PR_URL})]},
            "Sent you the digest.",
        ]
    )
    apps = network(
        tmp_path,
        remotes,
        registry,
        {
            "dev-alice": alice,
            "dev-bob": reviewer("**Verdict:** request changes — service2 reads `result`.", "Review draft: #42"),
            "dev-carol": reviewer("**Verdict:** approve", "Review draft: #42"),
        },
        github=fake_github(),
    )
    api = client(apps, "dev-alice")
    events = send(api, start(api), "Get Bob and Carol to review PR 42")

    result = next(e for e in events if e["type"] == "tool_result" and e["name"] == "request_reviews")["content"]
    assert "### From Bob's agent (Bob Ruiz)\n**Verdict:** request changes" in result
    assert "### From Carol's agent (Carol Diaz)\n**Verdict:** approve" in result
    assert "Next: send your developer one digest" in result

    request = apps["dev-bob"].state.agent.llm.calls[0]["messages"][-1]["content"]
    assert request.startswith("[Message from the dev-alice agent]\nAlice Chen is asking for a review of a pull request.")
    assert f"- Pull request: {PR_URL}" in request and "- Requested focus: Does it break service2?" in request
    bob_tool_result = apps["dev-bob"].state.agent.llm.calls[1]["messages"][-1]["content"]
    assert bob_tool_result.startswith("# Pull request acme/ApiAgentService1#42")

    by_recipient = {email["to"][0]: email for email in emails}
    assert set(by_recipient) == {"alice@example.com", "bob@example.com", "carol@example.com"}
    assert by_recipient["bob@example.com"]["subject"] == "Review draft: #42"
    assert f"Link: {PR_URL}" in by_recipient["alice@example.com"]["text"]

    threads = api.get("/api/conversations", params={"kind": "agent"}).json()
    assert sorted(thread["peer_agent"] for thread in threads) == ["dev-bob", "dev-carol"]
    assert any(event["type"] == "reviews_requested" for event in api.get("/api/events").json())


def test_reviews_can_only_be_requested_from_linked_developers(tmp_path, remotes, registry):
    bob = network(tmp_path, remotes, registry, {"dev-bob": FakeLLM()}, github=fake_github())["dev-bob"].state.agent
    request = REVIEW_TOOLS[1].fn
    with pytest.raises(Exception, match="aren't linked to agent 'dev-carol'"):
        request(ToolContext(bob, "c"), pr_url=PR_URL, reviewers=["dev-carol"])
    with pytest.raises(Exception, match="service1 isn't a developer's agent"):
        request(ToolContext(bob, "c"), pr_url=PR_URL, reviewers=["service1"])
    with pytest.raises(Exception, match="while answering another agent"):
        request(ToolContext(bob, "c", depth=1), pr_url=PR_URL, reviewers=["dev-alice"])


def test_posting_a_review_comment_needs_approval(tmp_path, remotes, registry):
    posted: list[str] = []
    llm = FakeLLM([{"tool_calls": [("post_review_comment", {"pr_url": PR_URL, "body_markdown": "Please keep `result`."})]}, "Posted."])
    apps = network(tmp_path, remotes, registry, {"dev-bob": llm}, github=fake_github(posted))
    api = client(apps, "dev-bob")
    conversation_id = start(api)

    events = send(api, conversation_id, "Post that on the PR")
    assert events[-1]["type"] == "confirm_required" and posted == []
    assert "Please keep `result`.\n\n_Posted by Bob's agent on behalf of @bob._" in events[-1]["action"]["preview"]

    events = decide(api, conversation_id, True)
    assert posted == ["Please keep `result`.\n\n_Posted by Bob's agent on behalf of @bob._"]
    assert events[1]["content"] == f"Posted: {PR_URL}#issuecomment-1"
