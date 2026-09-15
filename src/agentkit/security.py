"""Localhost protection. The agents have no login, so only this machine's UI and other agents may call them."""

import hmac
import re
from collections.abc import Iterable

from fastapi import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
CREDENTIALS_IN_URL = re.compile(r"(https?://)[^/@\s]+@")


class LocalOnlyMiddleware:
    """Rejects requests with an unexpected Host (DNS rebinding) or a foreign Origin on writes (cross-site POSTs)."""

    def __init__(self, app: ASGIApp, allowed_hosts: Iterable[str], allowed_origins: Iterable[str]) -> None:
        self.app = app
        self.allowed_hosts = set(allowed_hosts)
        self.allowed_origins = set(allowed_origins)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope["headers"]}
        if headers.get("host", "") not in self.allowed_hosts:
            await JSONResponse({"detail": "Host not allowed"}, status_code=403)(scope, receive, send)
            return
        origin = headers.get("origin")
        if origin and scope["method"] not in SAFE_METHODS and origin not in self.allowed_origins:
            await JSONResponse({"detail": "Origin not allowed"}, status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def check_agent_token(expected: str, provided: str | None) -> None:
    if not expected:
        raise HTTPException(503, "AGENT_SHARED_TOKEN is not configured on this agent")
    if not provided or not hmac.compare_digest(expected, provided):
        raise HTTPException(401, "Missing or wrong X-Agent-Token")


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    text = CREDENTIALS_IN_URL.sub(r"\1***@", text)
    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "***")
    return text
