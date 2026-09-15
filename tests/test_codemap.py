from agentkit import frontmatter
from agentkit.codemap import DESCRIBE_PROMPT, parse_python
from agentkit.testing import FakeLLM, commit_files
from helpers import (
    CONSUMER_CONFIG,
    OPERATION_APP,
    OPERATION_INIT,
    OPERATION_MODELS,
    build_agent,
    describer,
    service1_spec,
    service2_spec,
)


def requested_files(llm: FakeLLM) -> list[str]:
    requests = [user for system, user in llm.complete_calls if system == DESCRIBE_PROMPT]
    return [request.split("# Describe these files\n")[1].split("\n# Architecture")[0] for request in requests]


def test_python_facts_are_extracted_with_ast():
    app = parse_python("src/operation/app.py", OPERATION_APP)
    assert [(r.method, r.path, r.handler, r.response_model, r.params) for r in app.routes] == [
        ("POST", "/v1/operation/numeric_op", "numeric_op", "NumericOpResponse", ["NumericOpRequest"]),
        ("GET", "/health", "health", "", []),
    ]
    models = parse_python("src/operation/models.py", OPERATION_MODELS)
    assert [(c.name, c.fields) for c in models.classes] == [
        ("NumericOpRequest", ["a: float", "b: float"]),
        ("NumericOpResponse", ["result: float"]),
    ]
    assert parse_python("src/operation/__init__.py", OPERATION_INIT).exports == ["NumericOpRequest", "NumericOpResponse", "OperationClient"]
    assert parse_python("src/consumer/config.py", CONSUMER_CONFIG).env_vars == {"OPERATION_BASE_URL"}
    assert parse_python("broken.py", "def (").routes == []


def test_first_build_writes_every_section(tmp_path, remotes):
    agent = build_agent(tmp_path, remotes, service2_spec(), llm=FakeLLM(responder=describer))
    agent.codemap.update()

    meta, body = frontmatter.parse(agent.codemap.path.read_text())
    assert meta["commit"] == agent.workspace.own.head() and meta["dirty"] is False
    headings = ["Overview", "Directory tree", "Important files", "Public contract", "Cross-repo usage", "Run, test, and config", "Architecture summary", "Notes & gotchas"]
    for number, heading in enumerate(headings, start=1):
        assert f"## {number}. {heading}" in body
    assert "| `src/consumer/app.py` | Purpose of src/consumer/app.py |" in body
    assert "- `src/consumer/app.py` uses `acme/ApiAgentService1`: OperationClient; methods called: health, numeric_op" in body
    assert "- `OPERATION_BASE_URL` (`src/consumer/config.py`)" in body
    assert "- **Path dependency:** `operation` from `../ApiAgentService1`" in body
    assert "Requests flow through FastAPI." in body
    tree = body.split("## 2. Directory tree")[1].split("## 3.")[0]
    assert "src/\n  consumer/\n    __init__.py" in tree and "more" not in tree
    assert agent.research.list()[0]["file"] == "codebase-map.md"
    assert any(event["type"] == "map_updated" for event in agent.store.list_events())


def test_contract_changes_redescribe_only_changed_files(tmp_path, remotes):
    llm = FakeLLM(responder=describer)
    agent = build_agent(tmp_path, remotes, service1_spec(), llm=llm)
    agent.codemap.update()
    llm.complete_calls.clear()

    commit_files(remotes["service1"], {"src/operation/app.py": OPERATION_APP.replace('"/numeric_op"', '"/subtract"')}, "Rename endpoint")
    agent.sync_workspace()

    assert requested_files(llm) == ["src/operation/app.py"]
    assert "# Architecture summary requested: yes" in llm.complete_calls[-1][1]
    body = agent.codemap.path.read_text()
    assert "`/v1/operation/subtract`" in body and "/v1/operation/numeric_op" not in body
    assert {"repo_synced", "map_updated", "contract_changed"} <= {event["type"] for event in agent.store.list_events()}


def test_small_commits_skip_the_architecture_summary(tmp_path, remotes):
    llm = FakeLLM(responder=describer)
    agent = build_agent(tmp_path, remotes, service1_spec(), llm=llm)
    agent.codemap.update()
    llm.complete_calls.clear()

    commit_files(remotes["service1"], {"README.md": "# operation\n\nMore docs.\n"}, "Docs")
    agent.sync_workspace()

    assert requested_files(llm) == ["README.md"]
    assert "# Architecture summary requested: no" in llm.complete_calls[-1][1]
    assert "contract_changed" not in {event["type"] for event in agent.store.list_events()}


def test_uncommitted_edits_mark_the_map_dirty_without_llm_calls(tmp_path, remotes):
    llm = FakeLLM(responder=describer)
    agent = build_agent(tmp_path, remotes, service1_spec(), llm=llm)
    agent.codemap.update()
    llm.complete_calls.clear()

    agent.workspace.own.write_file("src/operation/extra.py", "def added():\n    return 1\n")
    agent.codemap.refresh_working_tree()
    view = agent.codemap.view()
    assert view["dirty"] is True
    assert "## 9. Uncommitted changes" in view["markdown"] and "- `src/operation/extra.py`" in view["markdown"]
    assert llm.complete_calls == []

    agent.workspace.own.discard_changes()
    agent.codemap.refresh_working_tree()
    view = agent.codemap.view()
    assert view["dirty"] is False and "## 9." not in view["markdown"]


def test_notes_survive_regeneration_and_prompt_excerpt_keeps_core_sections(tmp_path, remotes):
    agent = build_agent(tmp_path, remotes, service1_spec(), llm=FakeLLM(responder=describer))
    agent.codemap.update()
    agent.codemap.set_notes("Result is a - b, so the order of inputs matters.")
    agent.codemap.update(full=True)
    assert "Result is a - b, so the order of inputs matters." in agent.codemap.path.read_text()

    excerpt = agent.codemap.prompt_excerpt()
    assert "## 4. Public contract" in excerpt and "## 8. Notes & gotchas" in excerpt
    assert "## 6." not in excerpt and "## 7." not in excerpt


def test_llm_failure_still_writes_extracted_sections(tmp_path, remotes):
    def broken(system: str, user: str) -> str:
        raise RuntimeError("NIM is down")

    agent = build_agent(tmp_path, remotes, service1_spec(), llm=FakeLLM(responder=broken))
    agent.codemap.update()
    body = agent.codemap.path.read_text()
    assert "_not described yet_" in body and "`/v1/operation/numeric_op`" in body
