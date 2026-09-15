import pytest

from agentkit import ToolContext, ToolError
from agentkit.testing import commit_files
from agentkit.tools.repo import revert_changes, write_file
from agentkit.workspace import PathError, RepoBusyError
from helpers import build_agent, service1_spec


def test_paths_are_confined_and_other_repos_are_read_only(tmp_path, remotes):
    agent = build_agent(tmp_path, remotes, service1_spec())
    own, other = agent.workspace.own, agent.workspace.repo("ApiAgentService2")

    for bad in ["../escape.py", "/etc/passwd", ".git/config"]:
        with pytest.raises(PathError):
            own.resolve(bad)
    with pytest.raises(PathError, match="limited to src/, tests/"):
        own.resolve("pyproject.toml", for_write=True)
    with pytest.raises(PathError, match="read-only"):
        other.resolve("src/consumer/app.py", for_write=True)

    (own.root / "src" / "link").symlink_to(tmp_path)
    with pytest.raises(PathError):
        own.resolve("src/link/secret.txt")

    assert "src/consumer/app.py" in other.list_files("src")
    assert agent.workspace.repo("acme/ApiAgentService2") is other


def test_one_conversation_owns_uncommitted_changes(tmp_path, remotes):
    agent = build_agent(tmp_path, remotes, service1_spec())
    ctx_a, ctx_b = ToolContext(agent, "conversation-a"), ToolContext(agent, "conversation-b")

    assert write_file.fn(ctx_a, path="src/operation/new.py", content="x = 1\n").startswith("Wrote")
    with pytest.raises(RepoBusyError):
        write_file.fn(ctx_b, path="src/operation/other.py", content="y = 2\n")
    with pytest.raises(RepoBusyError):
        revert_changes.fn(ctx_b)

    revert_changes.fn(ctx_a)
    assert not agent.workspace.own.is_dirty()
    assert write_file.fn(ctx_b, path="src/operation/other.py", content="y = 2\n").startswith("Wrote")


def test_sync_fast_forwards_only_clean_checkouts(tmp_path, remotes):
    agent = build_agent(tmp_path, remotes, service1_spec())
    own = agent.workspace.own
    new_head = commit_files(remotes["service1"], {"README.md": "# changed\n"}, "Docs")

    own.write_file("src/operation/wip.py", "pass\n")
    moved, notes = agent.workspace.sync()
    assert not moved and "wasn't updated" in notes[-1] and own.head() != new_head

    own.discard_changes()
    moved, _ = agent.workspace.sync()
    assert moved and own.head() == new_head

    read_only_head = commit_files(remotes["service2"], {"README.md": "# consumer\n"}, "Docs")
    agent.workspace.sync()
    assert agent.workspace.repo("ApiAgentService2").head() == read_only_head


def test_failed_push_keeps_the_changes(tmp_path, remotes):
    agent = build_agent(tmp_path, remotes, service1_spec())
    own = agent.workspace.own
    own.write_file("src/operation/wip.py", "pass\n")
    own.git("remote", "set-url", "origin", str(tmp_path / "missing.git"))

    with pytest.raises(ToolError):
        own.commit_and_push("agent/chat-x", "WIP", ["src/operation/wip.py"])
    assert own.branch() == "main"
    assert "src/operation/wip.py" in own.changed_paths()
    assert "agent/chat-x" not in own.git("branch")
