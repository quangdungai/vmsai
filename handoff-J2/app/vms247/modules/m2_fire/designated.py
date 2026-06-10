"""
M2 Cháy/khói — DESIGNATED (Track 1, Phase 1b).  CHỦ: Lane B.  CHẠY CLOUD.

LocateAnything-3B (VLM) — 2 vai trong 1 plugin:
  1. Cold-start  : locate("smoke","flame") zero-shot khi chưa đủ nhãn nhà máy.
  2. VERIFY      : với mỗi candidate, hỏi VLM "có thật là cháy/khói không?" để
                   CẮT báo giả hàn/hơi nước/nắng (nhiễu ngữ nghĩa debounce không cắt được).
Chạy TRIGGERED (trigger_every) — VLM nặng, không mọi frame.

Chưa cấu hình vlm_endpoint -> log + trả [] (không crash demo).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vms247.core.interfaces import ModulePlugin
from vms247.core.registry import register
from vms247.core.rules import Cooldown, Debouncer
from vms247.core.schemas import (
    Detection,
    Engine,
    Event,
    EventType,
    FrameMeta,
    ModuleId,
    Severity,
)


def _crop(frame: np.ndarray, box) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(max(0, v)) for v in box)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame


@register(ModuleId.M2, Engine.DESIGNATED)
class M2Designated(ModulePlugin):
    def __init__(self) -> None:
        self.vlm_endpoint: str | None = None
        self.vlm_model = "locate-anything-3b"
        self.classes = ["fire", "smoke"]
        self.trigger_every = 3
        self.do_verify = True
        self.debounce_s = 2.5
        self.cooldown_s = 30.0
        self._vlm = None

    def configure(self, cfg: dict[str, Any]) -> None:
        self.vlm_endpoint = cfg.get("vlm_endpoint", self.vlm_endpoint)
        self.vlm_model = cfg.get("vlm_model", self.vlm_model)
        self.classes = cfg.get("classes", self.classes)
        self.trigger_every = cfg.get("trigger_every", self.trigger_every)
        self.do_verify = cfg.get("verify", self.do_verify)

    def setup(self) -> None:
        from vms247.integrations.vlm_client import VLMClient  # noqa: PLC0415

        self._vlm = VLMClient(self.vlm_endpoint, self.vlm_model) if self.vlm_endpoint else None
        if self._vlm is None:
            print("[M2/designated] chưa cấu hình vlm_endpoint — phần Phase 1b, môi trường Lead cấp.")

    @staticmethod
    def _norm(label: str) -> str:
        n = label.lower()
        if "smoke" in n:
            return "smoke"
        if "fire" in n or "flame" in n:
            return "fire"
        return n

    def process(self, frame: np.ndarray, meta: FrameMeta) -> list[Detection]:
        if self._vlm is None or meta.frame_index % max(1, self.trigger_every) != 0:
            return []
        out: list[Detection] = []
        try:
            candidates = self._vlm.locate(frame, self.classes)
        except Exception as e:  # pragma: no cover
            print(f"[M2/designated] VLM locate lỗi: {e}")
            return []
        for o in candidates:
            label = self._norm(o["label"])
            box = o["box"]
            if self.do_verify:
                try:
                    v = self._vlm.verify(
                        _crop(frame, box),
                        f"Is there real {label} here (NOT welding sparks, steam, or sunlight glare)?",
                    )
                    if not v.get("is_true"):
                        continue
                except Exception:  # pragma: no cover
                    pass
            out.append(Detection(label=label, confidence=float(o.get("score", 1.0)), box=tuple(box)))
        return out

    def events(self, detections: list[Detection], state: dict[str, Any]) -> list[Event]:
        now = time.time()
        deb: dict[str, Debouncer] = state.setdefault("deb", {})
        cd: dict[str, Cooldown] = state.setdefault("cd", {})
        events: list[Event] = []
        for lab, etype in (("fire", EventType.FIRE), ("smoke", EventType.SMOKE)):
            cur = [d for d in detections if d.label == lab]
            db = deb.setdefault(lab, Debouncer(min_seconds=self.debounce_s))
            c = cd.setdefault(lab, Cooldown(seconds=self.cooldown_s))
            if db.update(bool(cur), now) and c.ready(now):
                c.fire(now)
                best = max(cur, key=lambda x: x.confidence)
                events.append(
                    Event(
                        type=etype,
                        module=self.module_id,
                        time=now,
                        camera="",
                        box=best.box,
                        confidence=best.confidence,
                        severity=Severity.HIGH,
                        label=f"{lab} (VLM verified)",
                    )
                )
        return events
