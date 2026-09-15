"""HTTP API for one agent. Every agent serves the same contract, so the UI works with any of them."""

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agentkit import loop
from agentkit.agent import API_VERSION, Agent
from agentkit.config import Settings
from agentkit.llm import LLM
from agentkit.registry import HttpFactory, PeerError, default_http_factory
from agentkit.security import LocalOnlyMiddleware, check_agent_token
from agentkit.spec import AgentSpec, ToolError

SSE_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Server-sent events: message, token, tool_call, tool_result, confirm_required, decision, error, done",
        "content": {"text/event-stream": {}},
    }
}


# Response models


class RepoInfo(BaseModel):
    slug: str
    name: str


class ToolInfo(BaseModel):
    name: str
    description: str
    needs_confirmation: bool


class AgentInfo(BaseModel):
    api_version: int
    id: str
    name: str
    description: str
    repo: RepoInfo
    reads: list[RepoInfo]
    features: list[str]
    tools: list[ToolInfo]


class PendingAction(BaseModel):
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    preview: str | None = None
    requested_at: str = ""


class Conversation(BaseModel):
    id: str
    thread_id: str | None
    kind: Literal["user", "agent"]
    peer_agent: str | None
    channel: str
    title: str
    parent_conversation_id: str | None
    pending_action: PendingAction | None
    created_at: str
    updated_at: str
    preview: str | None = None
    # True while a turn is in progress on the server, whether or not anyone is streaming it.
    running: bool = False


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


class Message(BaseModel):
    id: str
    conversation_id: str
    role: Literal["user", "assistant", "tool"]
    sender: str
    content: str
    tool_calls: list[ToolCall] | None
    tool_call_id: str | None
    created_at: str


class ConversationDetail(BaseModel):
    conversation: Conversation
    messages: list[Message]


class EmailSummary(BaseModel):
    id: str
    conversation_id: str | None
    recipients: list[str]
    subject: str
    provider: str
    status: Literal["sent", "failed", "denied"]
    error: str | None
    created_at: str


class Email(EmailSummary):
    body_markdown: str
    body_html: str


class Event(BaseModel):
    id: str
    type: str
    summary: str
    url: str | None
    conversation_id: str | None
    created_at: str


class ResearchFile(BaseModel):
    file: str
    title: str
    date: str
    updated_at: str
    size: int


class ResearchDoc(BaseModel):
    file: str
    content: str


class MemoryView(BaseModel):
    notes: str
    recent: str
    lessons: str
    events: list[Event]


class CodebaseMapView(BaseModel):
    exists: bool
    markdown: str
    commit: str
    updated_at: str
    dirty: bool
    stale: bool
    outdated: list[str]


class Learning(BaseModel):
    id: str
    kind: Literal["lesson", "skill"]
    title: str
    body: str
    why: str
    evidence_conversation_id: str | None
    status: Literal["proposed", "active", "rejected", "retired"]
    created_at: str
    decided_at: str | None


class Feedback(BaseModel):
    id: str
    conversation_id: str
    message_id: str | None
    source: str
    rating: int
    comment: str
    created_at: str


class PrOutcome(BaseModel):
    pr_url: str
    conversation_id: str | None
    state: str
    merged: bool
    review_comments: list[dict[str, Any]]
    checked_at: str


class RepoStatus(BaseModel):
    name: str
    slug: str
    writable: bool
    cloned: bool
    head: str | None = None
    branch: str | None = None
    behind: int = 0
    dirty: bool = False
    changed: list[str] = []
    lock_holder: str | None = None


class WorkspaceView(BaseModel):
    repos: list[RepoStatus]
    notes: list[str] = []


class AgentMessageResult(BaseModel):
    status: Literal["replied", "awaiting_approval", "error"]
    reply: str = ""
    error: str | None = None
    conversation_id: str


class Accepted(BaseModel):
    ok: bool = True
    detail: str = ""


# Request bodies


class NewConversation(BaseModel):
    title: str = ""


class SendMessage(BaseModel):
    message: str = Field(min_length=1)


class Decision(BaseModel):
    approve: bool
    reason: str = ""


class AgentMessageIn(BaseModel):
    from_agent: str
    thread_id: str
    message: str = Field(min_length=1)


class AgentReplyIn(BaseModel):
    from_agent: str
    thread_id: str
    reply: str


class InboundEvent(BaseModel):
    from_agent: str
    type: str = "peer_event"
    summary: str
    url: str | None = None


class FeedbackIn(BaseModel):
    rating: Literal[-1, 1]
    comment: str = ""


class LearningDecision(BaseModel):
    action: Literal["approve", "reject", "retire"]
    title: str | None = None
    body: str | None = None


def sse(events: Iterator[dict[str, Any] | None]) -> StreamingResponse:
    def body() -> Iterator[str]:
        for event in events:
            # None is a keep-alive: an SSE comment the client ignores. Writing it also lets the server
            # notice a listener that has gone away, so the thread serving it stops waiting.
            yield ": keep-alive\n\n" if event is None else f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def create_app(spec: AgentSpec, settings: Settings, *, llm: LLM | None = None, http_factory: HttpFactory = default_http_factory) -> FastAPI:
    agent = Agent(spec, settings, llm=llm, http_factory=http_factory)
    store = agent.store
    runs = agent.runs

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        agent.start_background()
        yield
        agent.stop()

    app = FastAPI(title=f"{spec.name} agent", version=f"{API_VERSION}", lifespan=lifespan)
    app.state.agent = agent
    app.add_middleware(LocalOnlyMiddleware, allowed_hosts=settings.allowed_hosts, allowed_origins=settings.ui_origins)
    app.add_middleware(CORSMiddleware, allow_origins=settings.ui_origins, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(ToolError)
    async def tool_error(request: Request, exc: ToolError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    def conversation_or_404(conversation_id: str) -> dict[str, Any]:
        conversation = store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        return conversation

    def with_running(conversation: dict[str, Any]) -> dict[str, Any]:
        return {**conversation, "running": runs.is_running(conversation["id"])}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "agent": spec.id}

    @app.get("/api/info", response_model=AgentInfo)
    def info() -> dict[str, Any]:
        return agent.info()

    # Conversations

    @app.get("/api/conversations", response_model=list[Conversation])
    def list_conversations(kind: Literal["user", "agent"] | None = None) -> list[dict[str, Any]]:
        return [with_running(conversation) for conversation in store.list_conversations(kind)]

    @app.post("/api/conversations", response_model=Conversation)
    def create_conversation(body: NewConversation) -> dict[str, Any]:
        return with_running(store.create_conversation(kind="user", title=body.title))

    @app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
    def get_conversation(conversation_id: str) -> dict[str, Any]:
        return {"conversation": with_running(conversation_or_404(conversation_id)), "messages": store.list_messages(conversation_id)}

    # Each turn runs on its own thread (agentkit.runs) and a response only follows it. Reloading or closing
    # the page ends the stream, not the turn; GET …/stream picks the turn up again.

    @app.post("/api/conversations/{conversation_id}/messages", responses=SSE_RESPONSE)
    def send_message(conversation_id: str, body: SendMessage) -> StreamingResponse:
        if conversation_or_404(conversation_id)["kind"] != "user":
            raise HTTPException(409, "Conversations between agents are read-only")
        return sse(runs.start(conversation_id, loop.run_turn(agent, conversation_id, body.message)).follow())

    @app.get("/api/conversations/{conversation_id}/stream", responses=SSE_RESPONSE)
    def follow_turn(conversation_id: str) -> StreamingResponse:
        """Replay the turn in progress from its first event, then follow it. Ends at once if none is running."""
        conversation_or_404(conversation_id)
        run = runs.get(conversation_id)
        return sse(run.follow() if run else iter(()))

    @app.post("/api/conversations/{conversation_id}/confirm", responses=SSE_RESPONSE)
    def confirm(conversation_id: str, body: Decision) -> StreamingResponse:
        conversation = conversation_or_404(conversation_id)
        if not conversation["pending_action"]:
            raise HTTPException(409, "Nothing is waiting for approval")
        events = loop.resume(agent, conversation_id, body.approve, body.reason)
        if conversation["kind"] == "agent" and conversation["channel"] == "agent_http":
            events = forward_reply(conversation, events)
        return sse(runs.start(conversation_id, events).follow())

    def forward_reply(conversation: dict[str, Any], events: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """After the user decides for a thread another agent started, send the final reply back to that agent."""
        reply, paused = "", False
        for event in events:
            if event["type"] == "message" and event["message"]["role"] == "assistant" and event["message"]["content"]:
                reply = event["message"]["content"]
            elif event["type"] == "confirm_required":
                paused = True
            yield event
        if reply and not paused:
            try:
                agent.peers.request(
                    conversation["peer_agent"],
                    "POST",
                    "/api/agent-messages/reply",
                    json={"from_agent": spec.id, "thread_id": conversation["thread_id"], "reply": reply},
                )
            except PeerError as err:
                yield {"type": "error", "message": f"Couldn't deliver the reply to {conversation['peer_agent']}: {err}"}

    @app.get("/api/pending", response_model=list[Conversation])
    def pending() -> list[dict[str, Any]]:
        return [with_running(conversation) for conversation in store.pending_conversations()]

    # Agent-to-agent

    @app.post("/api/agent-messages", response_model=AgentMessageResult)
    def receive_agent_message(body: AgentMessageIn, x_agent_token: str | None = Header(default=None)) -> dict[str, Any]:
        check_agent_token(settings.shared_token, x_agent_token)
        conversation = store.find_thread(body.thread_id)
        if conversation is None:
            conversation = store.create_conversation(
                kind="agent", channel="agent_http", peer_agent=body.from_agent, thread_id=body.thread_id
            )
        elif conversation["peer_agent"] != body.from_agent:
            raise HTTPException(409, "That thread belongs to a different agent")
        if conversation["pending_action"]:
            return {"status": "awaiting_approval", "conversation_id": conversation["id"]}
        run = runs.start(conversation["id"], loop.run_turn(agent, conversation["id"], body.message, sender=body.from_agent))
        result = loop.collect(event for event in run.follow() if event is not None)
        return {**result, "conversation_id": conversation["id"]}

    @app.post("/api/agent-messages/reply", response_model=Accepted)
    def receive_agent_reply(body: AgentReplyIn, x_agent_token: str | None = Header(default=None)) -> dict[str, Any]:
        check_agent_token(settings.shared_token, x_agent_token)
        conversation = store.find_thread(body.thread_id)
        if conversation is None or conversation["peer_agent"] != body.from_agent:
            raise HTTPException(404, "Thread not found")
        store.add_message(conversation["id"], "user", body.reply, sender=body.from_agent)
        agent.memory.schedule_summary(conversation["id"])
        return {"ok": True}

    @app.post("/api/events/inbound", response_model=Accepted)
    def inbound_event(body: InboundEvent, x_agent_token: str | None = Header(default=None)) -> dict[str, Any]:
        check_agent_token(settings.shared_token, x_agent_token)
        agent.log_event(body.type, f"[from {body.from_agent}] {body.summary}", url=body.url)
        return {"ok": True}

    # Emails, events, research, memory, codebase map

    @app.get("/api/emails", response_model=list[EmailSummary])
    def list_emails() -> list[dict[str, Any]]:
        return store.list_emails()

    @app.get("/api/emails/{email_id}", response_model=Email)
    def get_email(email_id: str) -> dict[str, Any]:
        if (email := store.get_email(email_id)) is None:
            raise HTTPException(404, "Email not found")
        return email

    @app.get("/api/events", response_model=list[Event])
    def list_events(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
        return store.list_events(limit)

    @app.get("/api/research", response_model=list[ResearchFile])
    def list_research() -> list[dict[str, Any]]:
        return agent.research.list()

    @app.get("/api/research/{file}", response_model=ResearchDoc)
    def read_research(file: str) -> dict[str, Any]:
        try:
            return {"file": file, "content": agent.research.read(file)}
        except ToolError as err:
            raise HTTPException(404, str(err)) from err

    @app.get("/api/memory", response_model=MemoryView)
    def memory() -> dict[str, Any]:
        return {**agent.memory.view(), "events": store.list_events(10)}

    @app.get("/api/codebase-map", response_model=CodebaseMapView)
    def codebase_map() -> dict[str, Any]:
        return agent.codemap.view()

    @app.post("/api/codebase-map/rebuild", response_model=Accepted, status_code=202)
    def rebuild_codebase_map() -> dict[str, Any]:
        if not agent.workspace.own.exists():
            raise HTTPException(409, "The repo isn't cloned yet")
        started = agent.codemap.schedule_update(full=True)
        return {"ok": started, "detail": "Rebuilding the codebase map" if started else "A rebuild is already running"}

    # Learning

    @app.post("/api/messages/{message_id}/feedback", response_model=Feedback)
    def feedback(message_id: str, body: FeedbackIn) -> dict[str, Any]:
        try:
            return agent.learning.record_feedback(message_id, body.rating, body.comment)
        except ToolError as err:
            raise HTTPException(404, str(err)) from err

    @app.get("/api/learnings", response_model=list[Learning])
    def list_learnings(
        status: Literal["proposed", "active", "rejected", "retired"] | None = None, kind: Literal["lesson", "skill"] | None = None
    ) -> list[dict[str, Any]]:
        return store.list_learnings(status=status, kind=kind)

    @app.post("/api/learnings/{learning_id}", response_model=Learning)
    def decide_learning(learning_id: str, body: LearningDecision) -> dict[str, Any]:
        try:
            return agent.learning.decide(learning_id, body.action, title=body.title, body=body.body)
        except ToolError as err:
            raise HTTPException(404, str(err)) from err

    @app.get("/api/pr-outcomes", response_model=list[PrOutcome])
    def pr_outcomes() -> list[dict[str, Any]]:
        return store.list_pr_outcomes()

    # Workspace

    @app.get("/api/workspace", response_model=WorkspaceView)
    def workspace() -> dict[str, Any]:
        return {"repos": agent.workspace.status()}

    @app.post("/api/workspace/sync", response_model=WorkspaceView)
    def sync_workspace() -> dict[str, Any]:
        notes = agent.workspace.ensure_cloned()
        notes += agent.sync_workspace()
        if not agent.codemap.path.exists():
            agent.codemap.schedule_update()
        return {"repos": agent.workspace.status(), "notes": notes}

    return app
