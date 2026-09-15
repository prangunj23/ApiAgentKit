import pytest

from agentkit.testing import make_remote
from agentkit.workspace import Repo
from helpers import CONSUMER_FILES, OPERATION_FILES, SERVICE1, SERVICE2


@pytest.fixture(autouse=True)
def no_dependency_installs(monkeypatch):
    """Tests never install a repo's dependencies or run its real test suite."""
    monkeypatch.setattr(Repo, "uv_sync", lambda self: None)
    monkeypatch.setattr(Repo, "run_tests", lambda self, timeout=900: (True, "2 passed"))


@pytest.fixture
def remotes(tmp_path):
    root = tmp_path / "origin"
    return {
        "root": root,
        "service1": make_remote(root, SERVICE1.slug, OPERATION_FILES),
        "service2": make_remote(root, SERVICE2.slug, CONSUMER_FILES),
    }
