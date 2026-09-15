"""Small GitHub REST client: pull requests, review comments, and repository_dispatch."""

import re
from typing import Any

import httpx

from agentkit.spec import ToolError

PR_URL = re.compile(r"https://github\.com/([^/]+/[^/]+)/pull/(\d+)")


class GitHub:
    def __init__(self, token: str, client: httpx.Client) -> None:
        self.token = token
        self.client = client

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        if not self.token:
            raise ToolError("GITHUB_TOKEN is not set. Add a token with Contents and Pull requests access to this agent's .env.")
        return self.client.request(
            method,
            path,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30,
            **kwargs,
        )

    def create_pull_request(self, repo: str, *, head: str, base: str, title: str, body: str, draft: bool) -> str:
        response = self._request(
            "POST", f"/repos/{repo}/pulls", json={"title": title, "head": head, "base": base, "body": body, "draft": draft}
        )
        if response.status_code == 422:
            # A PR for this branch is already open; the force push updated it.
            owner = repo.split("/")[0]
            existing = self._request("GET", f"/repos/{repo}/pulls", params={"head": f"{owner}:{head}", "state": "open"})
            existing.raise_for_status()
            if existing.json():
                return existing.json()[0]["html_url"]
        response.raise_for_status()
        return response.json()["html_url"]

    def pull_request(self, pr_url: str) -> dict[str, Any]:
        repo, number = self._parse(pr_url)
        response = self._request("GET", f"/repos/{repo}/pulls/{number}")
        response.raise_for_status()
        return response.json()

    def review_comments(self, pr_url: str) -> list[dict[str, Any]]:
        repo, number = self._parse(pr_url)
        comments = []
        for path in (f"/repos/{repo}/pulls/{number}/comments", f"/repos/{repo}/issues/{number}/comments"):
            response = self._request("GET", path)
            response.raise_for_status()
            comments += [
                {"id": item["id"], "author": item.get("user", {}).get("login", ""), "body": item.get("body", ""), "path": item.get("path")}
                for item in response.json()
            ]
        return comments

    def dispatch(self, repo: str, event_type: str, payload: dict[str, Any]) -> None:
        response = self._request("POST", f"/repos/{repo}/dispatches", json={"event_type": event_type, "client_payload": payload})
        response.raise_for_status()

    @staticmethod
    def _parse(pr_url: str) -> tuple[str, str]:
        match = PR_URL.match(pr_url)
        if not match:
            raise ToolError(f"Not a GitHub pull request URL: {pr_url}")
        return match.group(1), match.group(2)
