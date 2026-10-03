from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        self.ref = self.root / "reference.json"

    def append(self, event: dict[str, Any]) -> None:
        event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        with self.path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def write_reference(self, payload: dict[str, Any]) -> None:
        self.ref.write_text(json.dumps(payload))

    def read_reference(self) -> dict[str, Any]:
        return json.loads(self.ref.read_text())
