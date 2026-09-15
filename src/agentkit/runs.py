"""Turns that outlive the HTTP request that started them.

A turn can take minutes, since every model call is a round trip to NIM. It used to run inside the SSE
response, so reloading or closing the page stopped it at its next event and lost the reply. Now each
turn runs on its own thread and records its events. Any number of listeners follow along, including one
that attaches after a reload, and none of them keeps the turn alive or can stop it.
"""

import logging
import threading
from collections.abc import Iterator
from typing import Any

from agentkit.security import redact

log = logging.getLogger(__name__)

Event = dict[str, Any]


class Run:
    """One turn's events, recorded as they happen so a listener can replay and follow them."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self.events: list[Event] = []
        self.done = False
        self._changed = threading.Condition()

    def _add(self, event: Event) -> None:
        with self._changed:
            self.events.append(event)
            self._changed.notify_all()

    def _finish(self) -> None:
        with self._changed:
            self.done = True
            self._changed.notify_all()

    def follow(self, heartbeat_seconds: float = 15) -> Iterator[Event | None]:
        """Yield every event from the start of the turn, then new ones until it ends.

        Yields None after `heartbeat_seconds` without an event, so the HTTP layer can send a keep-alive
        and notice a listener that has gone away. Abandoning this iterator never affects the turn.
        """
        index = 0
        while True:
            with self._changed:
                if index >= len(self.events) and not self.done:
                    self._changed.wait(heartbeat_seconds)
                pending = self.events[index:]
                finished = self.done
            if pending:
                index += len(pending)
                yield from pending
            elif finished:
                return
            else:
                yield None


class Runs:
    """The turns in progress for one agent: at most one per conversation."""

    def __init__(self, secrets: list[str] | None = None) -> None:
        self._secrets = secrets or []
        self._active: dict[str, Run] = {}
        self._guard = threading.Lock()

    def start(self, conversation_id: str, events: Iterator[Event]) -> Run:
        """Run `events` to the end on a background thread.

        If the conversation already has a turn running, `events` is discarded unstarted and the returned
        run holds a single error, the same one the loop reports for a conversation that's busy.
        """
        with self._guard:
            if conversation_id in self._active:
                refused = Run(conversation_id)
                refused._add({"type": "error", "message": "This conversation is already running."})
                refused._finish()
                close = getattr(events, "close", None)
                if close:
                    close()
                return refused
            run = Run(conversation_id)
            self._active[conversation_id] = run
        threading.Thread(target=self._drive, args=(run, events), daemon=True, name=f"turn-{conversation_id[:8]}").start()
        return run

    def _drive(self, run: Run, events: Iterator[Event]) -> None:
        try:
            for event in events:
                run._add(event)
        except Exception as err:
            log.exception("Turn in conversation %s failed", run.conversation_id)
            run._add({"type": "error", "message": redact(f"{type(err).__name__}: {err}", self._secrets)})
        finally:
            # Stop reporting the conversation as running before listeners see the end, so a page that
            # reloads when its stream closes finds the turn finished.
            with self._guard:
                if self._active.get(run.conversation_id) is run:
                    del self._active[run.conversation_id]
            run._finish()

    def get(self, conversation_id: str) -> Run | None:
        with self._guard:
            return self._active.get(conversation_id)

    def is_running(self, conversation_id: str) -> bool:
        with self._guard:
            return conversation_id in self._active

    def wait(self, conversation_id: str, timeout: float | None = None) -> bool:
        """Block until the conversation's turn ends. Returns False on timeout."""
        run = self.get(conversation_id)
        if run is None:
            return True
        with run._changed:
            return run._changed.wait_for(lambda: run.done, timeout)
