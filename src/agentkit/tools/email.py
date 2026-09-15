"""send_email: provider-agnostic email with a saved record of every attempt.

Ported from ApiAgentService2/agent/email_sender.py. Pick the provider with EMAIL_PROVIDER (resend or smtp).
"""

import smtplib
from email.message import EmailMessage

import httpx
import markdown

from agentkit.config import env
from agentkit.spec import ToolContext, ToolError, string, string_list, tool

PROVIDER_SETTINGS = {
    "resend": ["RESEND_API_KEY"],
    "smtp": ["SMTP_HOST"],
}


def email_provider() -> str:
    return env("EMAIL_PROVIDER", "resend").lower()


def missing_settings() -> list[str]:
    provider = email_provider()
    if provider not in PROVIDER_SETTINGS:
        return [f"EMAIL_PROVIDER (unknown value {provider!r}; use resend or smtp)"]
    return [name for name in PROVIDER_SETTINGS[provider] if not env(name)]


def send(to: list[str], subject: str, text: str, html: str) -> None:
    if email_provider() == "smtp":
        _send_smtp(to, subject, text, html)
    else:
        _send_resend(to, subject, text, html)


def _send_resend(to: list[str], subject: str, text: str, html: str) -> None:
    response = httpx.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {env('RESEND_API_KEY')}"},
        json={
            "from": env("EMAIL_FROM") or "ApiAgent <onboarding@resend.dev>",
            "to": to,
            "subject": subject,
            "text": text,
            "html": html,
        },
        timeout=30,
    )
    response.raise_for_status()


def _send_smtp(to: list[str], subject: str, text: str, html: str) -> None:
    host = env("SMTP_HOST")
    port = int(env("SMTP_PORT") or 587)
    username = env("SMTP_USERNAME")
    password = env("SMTP_PASSWORD")

    message = EmailMessage()
    message["From"] = env("EMAIL_FROM") or username
    message["To"] = ", ".join(to)
    message["Subject"] = subject
    message.set_content(text)
    message.add_alternative(html, subtype="html")

    # Port 465 expects TLS from the start; other ports (usually 587) upgrade with STARTTLS.
    if port == 465:
        server = smtplib.SMTP_SSL(host, port, timeout=30)
    else:
        server = smtplib.SMTP(host, port, timeout=30)
        server.starttls()
    with server:
        if username and password:
            server.login(username, password)
        server.send_message(message)


def _recipients(to: list[str] | None) -> list[str]:
    return [address.strip() for address in (to or env("EMAIL_TO").split(",")) if address.strip()]


def _record(ctx: ToolContext, recipients: list[str], subject: str, body_markdown: str, status: str, error: str | None = None) -> None:
    ctx.agent.store.add_email(
        conversation_id=ctx.conversation_id,
        recipients=recipients,
        subject=subject,
        body_markdown=body_markdown,
        body_html=markdown.markdown(body_markdown, extensions=["fenced_code", "tables"]),
        provider=email_provider(),
        status=status,
        error=error,
    )


@tool(
    "send_email",
    "Send an email update. The user must approve. Recipients default to EMAIL_TO.",
    {
        "subject": string("Subject line."),
        "body_markdown": string("Email body in markdown."),
        "to": string_list("Recipients. Leave empty to use EMAIL_TO."),
    },
    required=("subject", "body_markdown"),
    needs_confirmation=True,
)
def send_email(ctx: ToolContext, subject: str, body_markdown: str, to: list[str] | None = None) -> str:
    recipients = _recipients(to)
    if not recipients:
        _record(ctx, recipients, subject, body_markdown, "failed", "No recipients")
        raise ToolError("No recipients. Pass `to` or set EMAIL_TO in the agent's .env.")
    if missing := missing_settings():
        error = f"Missing email settings: {', '.join(missing)}"
        _record(ctx, recipients, subject, body_markdown, "failed", error)
        raise ToolError(error)
    html = markdown.markdown(body_markdown, extensions=["fenced_code", "tables"])
    try:
        send(recipients, subject, body_markdown, html)
    except (httpx.HTTPError, smtplib.SMTPException, OSError) as err:
        _record(ctx, recipients, subject, body_markdown, "failed", str(err))
        raise ToolError(f"Email failed: {err}") from err
    _record(ctx, recipients, subject, body_markdown, "sent")
    ctx.agent.log_event("email_sent", f"Emailed {', '.join(recipients)}: {subject}", conversation_id=ctx.conversation_id)
    return f"Emailed {', '.join(recipients)}: {subject}"


def _preview(ctx: ToolContext, subject: str, body_markdown: str, to: list[str] | None = None) -> str:
    return f"To: {', '.join(_recipients(to)) or '(none: set EMAIL_TO)'}\nSubject: {subject}\n\n{body_markdown}"


def _deny(ctx: ToolContext, arguments: dict, reason: str) -> None:
    _record(ctx, _recipients(arguments.get("to")), arguments.get("subject", ""), arguments.get("body_markdown", ""), "denied", reason or None)


send_email.preview = _preview
send_email.on_deny = _deny
