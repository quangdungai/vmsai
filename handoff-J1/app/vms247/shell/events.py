"""
L5 — Event store + evidence snapshot.

Lưu mọi Event ra bộ nhớ + file JSONL; chụp ảnh bằng chứng (evidence) cho
event nếu plugin chưa tự gắn evidence_path. Phục vụ review/label về sau.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from vms247.core.schemas import Event


class EventStore:
    def __init__(self, root: str | Path = "data/evidence", jsonl: str | Path = "data/events.jsonl") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jsonl = Path(jsonl)
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[Event] = []

    def add(self, event: Event, frame: np.ndarray | None = None) -> Event:
        if event.evidence_path is None and frame is not None:
            event.evidence_path = self._snapshot(event, frame)
        self._events.append(event)
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_row(), ensure_ascii=False) + "\n")
        return event

    def _snapshot(self, event: Event, frame: np.ndarray) -> str:
        ts = int(event.time or time.time())
        fname = self.root / f"{event.module.value}_{event.type.value}_{ts}_{len(self._events)}.jpg"
        cv2.imwrite(str(fname), frame)
        return str(fname)

    def recent(self, n: int = 50) -> list[dict]:
        return [e.to_row() for e in self._events[-n:]]

    def all(self) -> list[Event]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()
