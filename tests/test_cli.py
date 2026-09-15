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
