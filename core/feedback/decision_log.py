"""Append-only decision log for audit trail."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class DecisionLogEntry:
    """One row in the decision log."""
    id: int
    timestamp: str
    action: str
    target: str
    outcome: str
    detail: str = ""


class DecisionLog:
    """Append-only log of every decision the session makes."""

    def __init__(self) -> None:
        self.entries: list[DecisionLogEntry] = []
        self._counter = 0

    def add(self, action: str, target: str, outcome: str = "",
            detail: str = "") -> DecisionLogEntry:
        self._counter += 1
        entry = DecisionLogEntry(
            id=self._counter,
            timestamp=datetime.utcnow().isoformat() + "Z",
            action=action, target=target, outcome=outcome, detail=detail,
        )
        self.entries.append(entry)
        return entry

    def get(self, entry_id: int) -> DecisionLogEntry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    def to_list(self) -> list[dict[str, Any]]:
        return [
            {"id": e.id, "timestamp": e.timestamp, "action": e.action,
             "target": e.target, "outcome": e.outcome, "detail": e.detail}
            for e in self.entries
        ]

    def __len__(self) -> int:
        return len(self.entries)
