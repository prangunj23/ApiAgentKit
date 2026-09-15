"""Self-improvement: feedback and outcomes → reflection → proposed lessons and skills → user approval."""

import json
import logging
import threading
from typing import TYPE_CHECKING, Any

import httpx

from agentkit import frontmatter
from agentkit.llm import complete_json
from agentkit.memory import format_transcript
from agentkit.research import slugify
from agentkit.spec import ToolError

if TYPE_CHECKING:
    from agentkit.agent import Agent

log = logging.getLogger(__name__)

DECISIONS = {"approve": "active", "reject": "rejected", "retire": "retired"}

REFLECT_PROMPT = """You help an AI agent get better at its work. You get an excerpt of one of its conversations,
a signal about how it went (user feedback or a real-world outcome), and the lessons and skills it already follows.

Propose at most 3 improvements:
- "lesson": a short, general rule for future behavior. Not a fact about one task.
- "skill": a reusable step-by-step procedure for a kind of task that went well.
Only propose what the evidence supports. Don't repeat an existing lesson; to change one, propose the corrected
version with the same title. Propose nothing if there is nothing general to learn.

Respond with only a JSON object:
{"proposals": [{"kind": "lesson" or "skill", "title": "short title", "body": "the rule, or markdown steps for a skill", "why": "the evidence from this conversation"}]}"""


class Learning:
    def __init__(self, agent: "Agent") -> None:
        self.agent = agent
        self.skills_dir = agent.settings.data_dir / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    # Signals

    def record_feedback(self, message_id: str, rating: int, comment: str = "") -> dict[str, Any]:
        store = self.agent.store
        message = store.get_message(message_id)
        if message is None:
            raise ToolError(f"No message {message_id}")
        row = store.add_feedback(
            conversation_id=message["conversation_id"], message_id=message_id, source="thumb", rating=rating, comment=comment
        )
        if rating < 0:
            signal = f"The user gave a thumbs down to this reply:\n{message['content'][:1500]}"
            if comment:
                signal += f"\n\nTheir correction: {comment}"
            self.schedule_reflection(message["conversation_id"], f"thumbs_down:{message_id}", signal)
        return row

    def record_denial(self, conversation_id: str, action: dict[str, Any], reason: str) -> None:
        self.agent.store.add_feedback(conversation_id=conversation_id, source="deny_reason", rating=-1, comment=reason)
        arguments = json.dumps(action.get("arguments", {}))[:1500]
        self.schedule_reflection(
            conversation_id,
            f"denied:{action.get('tool_call_id', '')}",
            f"The user denied `{action['name']}` with arguments {arguments}. Reason: {reason or '(none given)'}",
        )

    # Proposals and decisions

    def propose(self, kind: str, title: str, body: str, why: str = "", evidence_conversation_id: str | None = None) -> dict[str, Any]:
        status = "active" if self.agent.settings.auto_approve_learnings else "proposed"
        row = self.agent.store.add_learning(
            kind=kind, title=title, body=body, why=why, evidence_conversation_id=evidence_conversation_id, status=status
        )
        if status == "active":
            self.mirror()
        return row

    def decide(self, learning_id: str, action: str, *, title: str | None = None, body: str | None = None) -> dict[str, Any]:
        if action not in DECISIONS:
            raise ValueError(f"action must be one of {', '.join(DECISIONS)}")
        row = self.agent.store.update_learning(learning_id, status=DECISIONS[action], title=title, body=body)
        if row is None:
            raise ToolError(f"No learning {learning_id}")
        self.mirror()
        return row

    def active(self, kind: str) -> list[dict[str, Any]]:
        return self.agent.store.list_learnings(status="active", kind=kind)

    def find_skill(self, name: str) -> dict[str, Any] | None:
        key = slugify(name)
        return next((skill for skill in self.active("skill") if slugify(skill["title"]) == key), None)

    def mirror(self) -> None:
        """Write active lessons and skills to markdown so they're easy to read outside the UI."""
        lessons = self.active("lesson")
        lines = [f"- **{lesson['title']}**: {lesson['body']}" for lesson in lessons] or ["_No active lessons._"]
        self.agent.memory.lessons_path.write_text("# Lessons\n\n" + "\n".join(lines) + "\n")

        keep = set()
        for skill in self.active("skill"):
            name = f"{slugify(skill['title'])}.md"
            keep.add(name)
            meta = {"title": skill["title"], "why": skill["why"], "evidence_conversation_id": skill["evidence_conversation_id"] or ""}
            (self.skills_dir / name).write_text(frontmatter.dump(meta, skill["body"]))
        for path in self.skills_dir.glob("*.md"):
            if path.name not in keep:
                path.unlink()

    # Reflection

    def schedule_reflection(self, conversation_id: str, trigger: str, signal: str) -> None:
        if not self.agent.store.claim_reflection(conversation_id, trigger):
            return
        if self.agent.settings.background:
            threading.Thread(target=self.reflect, args=(conversation_id, signal), daemon=True).start()
        else:
            self.reflect(conversation_id, signal)

    def reflect(self, conversation_id: str, signal: str) -> list[dict[str, Any]]:
        store = self.agent.store
        transcript = format_transcript(store.list_messages(conversation_id), 10_000)
        existing = "\n".join(
            f"- [{item['kind']}] {item['title']}: {item['body'][:200]}" for item in store.list_learnings(status="active")
        )
        user = f"# Signal\n{signal}\n\n# Existing lessons and skills\n{existing or 'none'}\n\n# Conversation excerpt\n{transcript}"
        try:
            data = complete_json(self.agent.llm, REFLECT_PROMPT, user)
        except Exception:
            log.exception("Reflection failed for conversation %s", conversation_id)
            return []
        proposals = []
        for item in (data.get("proposals") or [])[:3]:
            kind, title, body = item.get("kind"), str(item.get("title", "")).strip(), str(item.get("body", "")).strip()
            if kind in {"lesson", "skill"} and title and body:
                proposals.append(self.propose(kind, title[:120], body, str(item.get("why", "")), conversation_id))
        return proposals

    # PR outcomes

    def poll_prs(self) -> list[str]:
        agent = self.agent
        notes = []
        for outcome in agent.store.list_pr_outcomes(state="open"):
            url, conversation_id = outcome["pr_url"], outcome["conversation_id"]
            try:
                pr = agent.github.pull_request(url)
                comments = agent.github.review_comments(url)
            except (ToolError, httpx.HTTPError):
                log.exception("Couldn't check %s", url)
                continue
            merged = bool(pr.get("merged"))
            state = "merged" if merged else pr.get("state", "open")
            seen = {comment["id"] for comment in outcome["review_comments"] or []}
            new_comments = [comment for comment in comments if comment["id"] not in seen]
            agent.store.upsert_pr_outcome(url, state=state, merged=merged, review_comments=comments)

            if conversation_id:
                if new_comments:
                    self.schedule_reflection(
                        conversation_id,
                        f"review:{max(comment['id'] for comment in new_comments)}",
                        f"Reviewers commented on the PR this conversation opened ({url}):\n"
                        + "\n".join(f"- {c['author']}: {c['body'][:500]}" for c in new_comments),
                    )
                if state == "closed":
                    self.schedule_reflection(
                        conversation_id, "pr_closed", f"The PR this conversation opened was closed without being merged: {url}"
                    )
                elif merged and agent.store.count_tool_rounds(conversation_id) >= 3:
                    self.schedule_reflection(
                        conversation_id,
                        "pr_merged",
                        f"The PR this conversation opened was merged: {url}. If the approach generalizes, capture it as a skill.",
                    )
            if state != "open":
                agent.log_event(f"pr_{state}", f"PR {state}", url=url, conversation_id=conversation_id)
            notes.append(f"{url}: {state}")
        return notes
