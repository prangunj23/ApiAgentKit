import httpx
from fastapi.testclient import TestClient

from agentkit.testing import FakeLLM
from helpers import build_app, decide, send, service2_spec, start


def email_call(subject: str) -> dict:
    return {"tool_calls": [("send_email", {"subject": subject, "body_markdown": "Hello **team**"})]}


def test_sent_failed_and_denied_emails_are_all_saved(tmp_path, remotes, monkeypatch):
    monkeypatch.setenv("EMAIL_TO", "dev@example.com")
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    failures = iter([None, httpx.ConnectError("resend down")])

    def fake_send(to, subject, text, html):
        if error := next(failures):
            raise error

    monkeypatch.setattr("agentkit.tools.email.send", fake_send)
    llm = FakeLLM([email_call("First"), "Sent.", email_call("Second"), "It failed.", email_call("Third"), "Not sent."])
    client = TestClient(build_app(tmp_path, remotes, service2_spec(), llm=llm, clone=False))
    conversation_id = start(client)

    for approve in (True, True, False):
        events = send(client, conversation_id, "Email the team")
        assert events[-1]["type"] == "confirm_required"
        assert events[-1]["action"]["preview"].startswith("To: dev@example.com\nSubject: ")
        decide(client, conversation_id, approve, "" if approve else "Not now")

    emails = client.get("/api/emails").json()
    assert [(email["subject"], email["status"]) for email in emails] == [("Third", "denied"), ("Second", "failed"), ("First", "sent")]
    assert emails[0]["error"] == "Not now" and emails[1]["error"] == "resend down"

    detail = client.get(f"/api/emails/{emails[2]['id']}").json()
    assert detail["body_html"] == "<p>Hello <strong>team</strong></p>"
    assert detail["recipients"] == ["dev@example.com"]
    assert client.get("/api/emails/missing").status_code == 404
    assert client.get("/api/info").json()["features"] == ["emails"]


def test_missing_email_settings_fail_clearly(tmp_path, remotes, monkeypatch):
    monkeypatch.setenv("EMAIL_TO", "dev@example.com")
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    llm = FakeLLM([email_call("Hi"), "Couldn't send."])
    client = TestClient(build_app(tmp_path, remotes, service2_spec(), llm=llm, clone=False))
    conversation_id = start(client)
    send(client, conversation_id, "Email the team")

    events = decide(client, conversation_id, True)
    assert events[1]["content"] == "Error: Missing email settings: RESEND_API_KEY"
    assert client.get("/api/emails").json()[0]["status"] == "failed"
