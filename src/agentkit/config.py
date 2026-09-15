import os
from dataclasses import dataclass, field
from pathlib import Path

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "moonshotai/kimi-k3"
DEFAULT_UI_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def env(name: str, default: str = "") -> str:
    # Pasted secrets often end in a newline, which isn't allowed in an HTTP header.
    return os.environ.get(name, "").strip() or default


def env_list(name: str) -> list[str]:
    return [item.strip() for item in env(name).split(",") if item.strip()]


def load_dotenv(path: Path) -> None:
    """Set variables from a .env file without overriding ones already in the environment."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.removeprefix("export ").strip(), value.strip().strip("'\""))


@dataclass
class Settings:
    data_dir: Path
    port: int = 9000
    host: str = "127.0.0.1"
    nvidia_api_key: str = ""
    nim_model: str = DEFAULT_MODEL
    nim_base_url: str = NIM_BASE_URL
    github_token: str = ""
    github_api: str = "https://api.github.com"
    git_base_url: str = "https://github.com"
    # Path or URL of the registry.json that lists every agent.
    registry: str = ""
    shared_token: str = ""
    ui_origins: list[str] = field(default_factory=lambda: list(DEFAULT_UI_ORIGINS))
    extra_hosts: list[str] = field(default_factory=list)
    auto_approve_learnings: bool = False
    # Background threads: clone/poll repos, poll PRs, and debounce summaries. Tests turn this off.
    background: bool = True
    repo_poll_seconds: int = 300
    pr_poll_seconds: int = 900
    summary_delay_seconds: float = 60

    @property
    def allowed_hosts(self) -> set[str]:
        return {"localhost", "127.0.0.1", f"localhost:{self.port}", f"127.0.0.1:{self.port}", *self.extra_hosts}

    @property
    def secrets(self) -> list[str]:
        return [self.nvidia_api_key, self.github_token, self.shared_token]

    @classmethod
    def from_env(cls, agent_id: str, port: int) -> "Settings":
        return cls(
            # AGENT_HOME holds one folder per agent, so several agents can share it; AGENT_DATA_DIR overrides one agent's folder.
            data_dir=Path(env("AGENT_DATA_DIR") or Path(env("AGENT_HOME") or Path.home() / ".apiagent") / agent_id).expanduser(),
            port=port,
            host=env("AGENT_HOST", "127.0.0.1"),
            nvidia_api_key=env("NVIDIA_API_KEY"),
            nim_model=env("NIM_MODEL", DEFAULT_MODEL),
            github_token=env("GITHUB_TOKEN"),
            registry=env("AGENT_REGISTRY"),
            shared_token=env("AGENT_SHARED_TOKEN"),
            ui_origins=env_list("UI_ORIGINS") or list(DEFAULT_UI_ORIGINS),
            extra_hosts=env_list("ALLOWED_HOSTS"),
            auto_approve_learnings=env("AUTO_APPROVE_LEARNINGS").lower() in {"1", "true", "yes"},
        )
