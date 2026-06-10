"""
M1 Xâm nhập — DESIGNATED (Track 1, Phase 1b).  CHỦ: Lane B.  CHẠY CLOUD.

  - Người : DEIMv2-Wholebody49 (engine Lane A publish) — lớp body.
  - Xe/forklift/vật lạ : LocateAnything-3B locate zero-shot (KHÔNG cần nhãn xe).
VLM chạy TRIGGERED (trigger_every) để giữ low-FPS.

Chưa cấu hình (vlm_endpoint / wholebody_onnx) -> log + trả [] (không crash demo).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vms247.core.interfaces import ModulePlugin
from vms247.core.registry import register
from vms247.core.rules import Cooldown, Debouncer, box_bottom_center, point_in_polygon
from vms247.core.schemas import (
    Detection,
    Engine,
    Event,
    EventType,
    FrameMeta,
    ModuleId,
    Severity,
)


@register(ModuleId.M1, Engine.DESIGNATED)
class M1Designated(ModulePlugin):
    def __init__(self) -> None:
        self.roi: list[tuple[float, float]] = []
        self.vlm_endpoint: str | None = None
        self.vlm_model = "locate-anything-3b"
        self.wholebody_onnx: str | None = None
        self.vehicle_classes = ["forklift", "truck", "car", "motorbike"]
        self.trigger_every = 5
        self.debounce_s = 1.0
        self.cooldown_s = 15.0
        self._vlm = None
        self._wb = None
        self._warned = False

    def configure(self, cfg: dict[str, Any]) -> None:
        self.roi = [tuple(p) for p in cfg.get("roi", [])]
        self.vlm_endpoint = cfg.get("vlm_endpoint", self.vlm_endpoint)
        self.vlm_model = cfg.get("vlm_model", self.vlm_model)
        self.wholebody_onnx = cfg.get("wholebody_onnx", self.wholebody_onnx)
        self.vehicle_classes = cfg.get("vehicle_classes", self.vehicle_classes)
        self.trigger_every = cfg.get("trigger_every", self.trigger_every)

    def setup(self) -> None:
        from vms247.integrations.vlm_client import VLMClient  # noqa: PLC0415
        from vms247.integrations.wholebody import Wholebody49  # noqa: PLC0415

        self._vlm = VLMClient(self.vlm_endpoint, self.vlm_model) if self.vlm_endpoint else None
        wb = Wholebody49(self.wholebody_onnx)
        if wb.available():
            try:
                wb.setup()
                self._wb = wb
            except Exception as e:  # pragma: no cover
                print(f"[M1/designated] Wholebody load lỗi: {e}")
        if self._vlm is None and self._wb is None:
            print("[M1/designated] chưa cấu hình (vlm_endpoint/wholebody_onnx) — phần Phase 1b, môi trường Lead cấp.")

    def process(self, frame: np.ndarray, meta: FrameMeta) -> list[Detection]:
        dets: list[Detection] = []
        # person — Wholebody49 (Lane A)
        if self._wb is not None:
            try:
                for d in self._wb.detect(frame):
                    if d.get("part") == "body":
                        dets.append(Detection("person", float(d.get("score", 0.0)), tuple(d["box"])))
            except NotImplementedError:
                if not self._warned:
                    print("[M1/designated] Wholebody.detect chưa implement (điểm cắm cloud).")
                    self._warned = True
            except Exception as e:  # pragma: no cover
                print(f"[M1/designated] Wholebody.detect lỗi: {e}")
        # vehicle — VLM zero-shot, TRIGGERED
        if self._vlm is not None and meta.frame_index % max(1, self.trigger_every) == 0:
            try:
                for o in self._vlm.locate(frame, self.vehicle_classes):
                    dets.append(Detection(o["label"], o["score"], tuple(o["box"])))
            except Exception as e:  # pragma: no cover
                print(f"[M1/designated] VLM locate lỗi: {e}")
        return dets

    def events(self, detections: list[Detection], state: dict[str, Any]) -> list[Event]:
        now = time.time()
        deb: Debouncer = state.setdefault("deb", Debouncer(min_seconds=self.debounce_s))
        cd: Cooldown = state.setdefault("cd", Cooldown(seconds=self.cooldown_s))
        in_roi = [
            d for d in detections
            if (not self.roi) or point_in_polygon(box_bottom_center(d.box), self.roi)
        ]
        if deb.update(bool(in_roi), now) and cd.ready(now):
            cd.fire(now)
            d = max(in_roi, key=lambda x: x.confidence)
            return [
                Event(
                    type=EventType.INTRUSION,
                    module=self.module_id,
                    time=now,
                    camera="",
                    box=d.box,
                    confidence=d.confidence,
                    severity=Severity.MEDIUM,
                    label=f"{d.label} trong ROI",
                )
            ]
        return []
