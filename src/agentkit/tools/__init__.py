"""Tools every agent gets. Service-specific tools (and send_email) are added by each AgentSpec."""

from agentkit.tools.agents import message_agent
from agentkit.tools.github import open_pull_request
from agentkit.tools.learning import propose_lesson, propose_skill, read_skill
from agentkit.tools.memory import forget, remember, update_codebase_notes
from agentkit.tools.repo import (
    git_diff,
    git_log,
    git_status,
    list_files,
    read_file,
    revert_changes,
    run_tests,
    search_code,
    sync_repo,
    write_file,
)
from agentkit.tools.research import list_research, read_research, save_research

GENERIC_TOOLS = [
    list_files,
    read_file,
    search_code,
    git_log,
    git_diff,
    git_status,
    write_file,
    revert_changes,
    run_tests,
    sync_repo,
    open_pull_request,
    message_agent,
    save_research,
    list_research,
    read_research,
    remember,
    forget,
    update_codebase_notes,
    propose_lesson,
    propose_skill,
    read_skill,
]
