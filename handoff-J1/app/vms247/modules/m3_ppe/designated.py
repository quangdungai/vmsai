"""
M3 PPE — DESIGNATED (Track 1, Phase 1b).  CHỦ: Lane A.  CHẠY CLOUD.

DEIMv2-Wholebody49 làm MỎ NEO head/body chính xác + classifier mũ nhẹ phía sau.
Ý tưởng: kiểm "mũ trên đầu" dựa anchor head thay vì detect mũ rời.

ĐIỂM CẮM (Lane A): (1) Wholebody49.detect (xem integrations/wholebody.py),
(2) classifier mũ trên crop head -> helmet/no_helmet.
Chưa cấu hình -> log + trả [] (không crash demo).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vms247.core.interfaces import ModulePlugin
from vms247.core.registry import register
from vms247.core.rules import Cooldown, Debouncer, box_center, point_in_polygon
from vms247.core.schemas import (
    Detection,
    Engine,
    Event,
    EventType,
    FrameMeta,
    ModuleId,
    Severity,
)


@register(ModuleId.M3, Engine.DESIGNATED)
class M3Designated(ModulePlugin):
    def __init__(self) -> None:
        self.wholebody_onnx: str | None = None
        self.zones: list[list[tuple[float, float]]] = []
        self.debounce_s = 1.5
        self.cooldown_s = 20.0
        self._wb = None
        self._warned = False

    def configure(self, cfg: dict[str, Any]) -> None:
        self.wholebody_onnx = cfg.get("wholebody_onnx", self.wholebody_onnx)
        self.zones = [[tuple(p) for p in z] for z in cfg.get("zones", [])]

    def setup(self) -> None:
        from vms247.integrations.wholebody import Wholebody49  # noqa: PLC0415

        wb = Wholebody49(self.wholebody_onnx)
        if wb.available():
            try:
                wb.setup()
                self._wb = wb
            except Exception as e:  # pragma: no cover
                print(f"[M3/designated] Wholebody load lỗi: {e}")
        if self._wb is None:
            print("[M3/designated] chưa cấu hình wholebody_onnx — phần Phase 1b, môi trường Lead cấp.")

    def process(self, frame: np.ndarray, meta: FrameMeta) -> list[Detection]:
        if self._wb is None:
            return []
        try:
            parts = self._wb.detect(frame)
        except NotImplementedError:
            if not self._warned:
                print("[M3/designated] Wholebody.detect + classifier mũ chưa implement (điểm cắm).")
                self._warned = True
            return []
        except Exception as e:  # pragma: no cover
            print(f"[M3/designated] detect lỗi: {e}")
            return []

        out: list[Detection] = []
        for d in parts:
            if d.get("part") != "head":
                continue
            # TODO Lane A: classifier mũ trên crop head -> d["helmet"] True/False.
            has_helmet = d.get("helmet")
            if has_helmet is None:
                continue  # chưa có classifier -> chưa kết luận
            out.append(
                Detection(
                    label="helmet" if has_helmet else "no_helmet",
                    confidence=float(d.get("score", 0.0)),
                    box=tuple(d["box"]),
                )
            )
        return out

    def events(self, detections: list[Detection], state: dict[str, Any]) -> list[Event]:
        now = time.time()
        deb: Debouncer = state.setdefault("deb", Debouncer(min_seconds=self.debounce_s))
        cd: Cooldown = state.setdefault("cd", Cooldown(seconds=self.cooldown_s))
        violators = [
            d for d in detections
            if d.label == "no_helmet"
            and ((not self.zones) or any(point_in_polygon(box_center(d.box), z) for z in self.zones))
        ]
        if deb.update(bool(violators), now) and cd.ready(now):
            cd.fire(now)
            d = max(violators, key=lambda x: x.confidence)
            return [
                Event(
                    type=EventType.PPE_VIOLATION,
                    module=self.module_id,
                    time=now,
                    camera="",
                    box=d.box,
                    confidence=d.confidence,
                    severity=Severity.MEDIUM,
                    label="không đội mũ (Wholebody anchor)",
                )
            ]
        return []
