"""Pause-tolerant end-of-thought controller.

Fuses three signals into one `thought_end` decision:

  * Parakeet-EOU `<EOU>` token  -> a *candidate* end-of-thought (fast, semantic).
  * Smart Turn (raw-audio CPU)  -> *confirms* or *vetoes* the candidate.
  * Silero pause / max-wait     -> when to ask Smart Turn, and a hard ceiling so a
                                   stubborn "incomplete" can never hang the turn.

`<EOB>` backchannels never create a candidate. Pure logic, no model I/O, so the
policy is unit-testable; `now` is injectable for deterministic tests.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from gateway.config import Settings


class EndpointController:
    def __init__(self, settings: Settings) -> None:
        self.require_smart_turn = bool(settings.endpoint_require_smart_turn)
        self.max_wait = float(settings.endpoint_max_wait_secs)
        self._segments: list[str] = []
        self._candidate = False
        self._candidate_at = 0.0
        self._smart_turn_complete = False
        self.reset()

    def reset(self) -> None:
        """Start a fresh thought: drop accumulated finals and candidate state."""
        self._segments = []
        self._candidate = False
        self._candidate_at = 0.0
        self._smart_turn_complete = False

    def ingest_stream(self, result: dict[str, Any], now: Optional[float] = None) -> None:
        """Fold one STT streaming step into the controller.

        `result` is the dict from `STTStreamSession.feed/step`: it carries any
        newly-closed `finals` and the `eou`/`eob` flags. Each `<EOU>` (re)arms the
        candidate and refreshes the max-wait timer; `<EOB>` is ignored.
        """
        now = time.time() if now is None else now
        if result.get("finals"):
            self._segments.extend(s for s in result["finals"] if s)
        if result.get("eou"):
            self._candidate = True
            self._candidate_at = now
            self._smart_turn_complete = False

    def on_smart_turn(self, complete: bool) -> None:
        """Feed a Smart Turn verdict (True = user finished). A veto (False) is not
        sticky-fatal: the candidate persists so the max-wait ceiling still applies."""
        self._smart_turn_complete = bool(complete)

    def should_fire(self, now: Optional[float] = None) -> bool:
        if not self._candidate:
            return False
        if not self.require_smart_turn:
            return True
        if self._smart_turn_complete:
            return True
        now = time.time() if now is None else now
        return (now - self._candidate_at) >= self.max_wait

    def take_thought_end(self, reason: str) -> dict[str, Any]:
        """Build the thought_end payload (segments accumulated since the last one)
        and reset for the next thought. Call only when `should_fire()` / on commit."""
        segments = [s for s in self._segments if s]
        payload = {
            "type": "thought_end",
            "text": " ".join(segments).strip(),
            "segments": segments,
            "reason": reason,
        }
        self.reset()
        return payload

    @property
    def has_candidate(self) -> bool:
        return self._candidate

    @property
    def has_content(self) -> bool:
        return bool(self._segments)
