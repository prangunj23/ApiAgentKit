import importlib
import importlib.util
import json

from agentkit.cli import main


def test_init_scaffolds_a_working_agent_and_registers_it(tmp_path, monkeypatch):
    registry = tmp_path / "ApiAgentUI" / "public" / "registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text("[]")
    target = tmp_path / "ApiAgentService3" / "chat_agent"

    main(
        [
            "init",
            "--id", "service3",
            "--repo", "acme/ApiAgentService3",
            "--reads", "acme/ApiAgentService1",
            "--dir", str(target),
            "--port", "9003",
            "--registry", str(registry),
        ]
    )

    assert json.loads(registry.read_text()) == [
        {
            "id": "service3",
            "name": "service3",
            "url": "http://127.0.0.1:9003",
            "local": {"path": "../../ApiAgentService3/chat_agent", "spec": "service3_agent.spec:SPEC", "port": 9003},
        }
    ]
    assert "editable = true" in (target / "pyproject.toml").read_text()

    monkeypatch.syspath_prepend(str(target / "src"))
    spec = importlib.import_module("service3_agent.spec").SPEC
    assert spec.reads[0].slug == "acme/ApiAgentService1"

    module_spec = importlib.util.spec_from_file_location("generated_test_spec", target / "tests" / "test_spec.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.test_info_describes_this_agent(tmp_path / "data")


def test_openapi_describes_the_agent_api(tmp_path):
    out = tmp_path / "openapi.json"
    main(["openapi", "--out", str(out)])
    schema = json.loads(out.read_text())
    assert {
        "/api/info",
        "/api/conversations/{conversation_id}/messages",
        "/api/emails",
        "/api/learnings/{learning_id}",
        "/api/codebase-map",
    } <= set(schema["paths"])
    assert {"AgentInfo", "Conversation", "Message", "Email", "Learning"} <= set(schema["components"]["schemas"])


def test_inspect_prints_the_prompt_and_tools(tmp_path, monkeypatch, capsys):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps([{"id": "dev-ann", "url": "http://x", "kind": "developer", "developer": {"name": "Ann", "email": "ann@example.com"}}]))
    (tmp_path / "inspect_spec.py").write_text(
        "from agentkit import AgentSpec\n"
        "from agentkit.tools import REPO_WRITE_TOOLS\n"
        "from agentkit.tools.owner import notify_developer\n"
        "SPEC = AgentSpec(id='dev-ann', name=\"Ann's agent\", system_prompt='You act for Ann.', tools=[notify_developer],"
        " excluded_tools=set(REPO_WRITE_TOOLS))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENT_REGISTRY", str(registry))

    main(["inspect", "inspect_spec:SPEC", "--depth", "1"])
    out = capsys.readouterr().out
    assert out.startswith("You act for Ann.\n\n# Operating context")
    assert "- You work for Ann <ann@example.com>." in out
    assert "- You are answering a message from another agent." in out
    assert "# Tools the model can call (depth 1)" in out
    assert "- notify_developer: Email the developer you work for." in out
    assert "write_file" not in out and "message_agent" not in out


def test_dev_refuses_an_inconsistent_registry(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps([{"id": "a", "url": "http://a", "links": ["b"]}, {"id": "b", "url": "http://b", "links": []}]))
    try:
        main(["dev", "--registry", str(registry)])
    except SystemExit as exit:
        assert "a links to b, but b doesn't link back" in str(exit)
    else:
        raise AssertionError("dev started with an inconsistent registry")


def test_dev_keeps_the_onboarding_token_from_the_agents(tmp_path, monkeypatch):
    import subprocess

    from agentkit import cli

    captured = {}

    class FakePopen:
        stdout = iter(())

        def __init__(self, command, cwd, env, **kwargs):
            captured.update(env)

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setenv("ONBOARDING_GITHUB_TOKEN", "gho_secret")
    monkeypatch.setenv("RESEND_API_KEY", "re_shared")
    cli._spawn(tmp_path / "registry.json", "shared", "dev-x", "http://127.0.0.1:9109", {"path": ".", "spec": "x:SPEC"})
    assert "ONBOARDING_GITHUB_TOKEN" not in captured
    assert captured["RESEND_API_KEY"] == "re_shared" and captured["AGENT_SHARED_TOKEN"] == "shared"
