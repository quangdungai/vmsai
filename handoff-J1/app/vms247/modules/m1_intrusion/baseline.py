"""
M1 Xâm nhập — BASELINE (Track 0).  CHỦ: Lane B.

Model: YOLO11m (person, COCO) + supervision ByteTrack + ROI/line-cross.
Phạm vi: CHỈ "người trong ROI / vượt line". KHÔNG nhận hành vi trộm.

Đây là IMPLEMENTATION THẬT (bài mẫu chạy được). Lane B nối tiếp:
tinh chỉnh ngưỡng, ROI nhiều vùng, gắn HRM/notify... theo cùng pattern.

NGHIỆM THU:
  ✅ dương: người đi vào ROI / vượt line -> phát Event INTRUSION.
  ✅ âm   : người đi NGOÀI ROI            -> KHÔNG phát.

Cấu hình (configs/default.yaml > M1):
  roi:   [[x1,y1],...]   polygon vùng cấm (rỗng = cả khung là vùng cấm)
  line:  [[x1,y1],[x2,y2]]  đoạn đếm-vượt (null = tắt)
  model_path, conf_thr, debounce_s, cooldown_s
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vms247.core.interfaces import ModulePlugin
from vms247.core.registry import register
from vms247.core.rules import (
    Cooldown,
    Debouncer,
    LineCrosser,
    box_bottom_center,
    point_in_polygon,
)
from vms247.core.schemas import (
    Detection,
    Engine,
    Event,
    EventType,
    FrameMeta,
    ModuleId,
    Severity,
)

PERSON_CLASS_ID = 0  # 'person' trong COCO


@register(ModuleId.M1, Engine.BASELINE)
class M1Baseline(ModulePlugin):
    def __init__(self) -> None:
        self.roi: list[tuple[float, float]] = []
        self.line: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.model_path = "yolo11m.pt"
        self.conf_thr = 0.35
        self.debounce_s = 1.0
        self.cooldown_s = 15.0
        self._model = None
        self._tracker = None
        self._sv = None

    def configure(self, cfg: dict[str, Any]) -> None:
        self.roi = [tuple(p) for p in cfg.get("roi", [])]
        line = cfg.get("line")
        self.line = (tuple(line[0]), tuple(line[1])) if line else None
        self.model_path = cfg.get("model_path", self.model_path)
        self.conf_thr = cfg.get("conf_thr", self.conf_thr)
        self.debounce_s = cfg.get("debounce_s", self.debounce_s)
        self.cooldown_s = cfg.get("cooldown_s", self.cooldown_s)

    # 1) LOAD --------------------------------------------------------------- #
    def setup(self) -> None:
        # QUY TẮC VÀNG: import nặng Ở ĐÂY (không ở đầu file) -> import shell
        # không kéo theo CUDA/torch.
        from ultralytics import YOLO  # noqa: PLC0415
        import supervision as sv  # noqa: PLC0415

        self._sv = sv
        self._model = YOLO(self.model_path)          # tự tải weight lần đầu
        self._tracker = sv.ByteTrack()               # ID ổn định khi bị che khuất

    # 2) INFER -------------------------------------------------------------- #
    def process(self, frame: np.ndarray, meta: FrameMeta) -> list[Detection]:
        if self._model is None:
            return []
        sv = self._sv
        results = self._model(
            frame, classes=[PERSON_CLASS_ID], conf=self.conf_thr, verbose=False
        )[0]
        det = sv.Detections.from_ultralytics(results)
        det = self._tracker.update_with_detections(det)

        out: list[Detection] = []
        for i in range(len(det)):
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[i])
            tid = (
                int(det.tracker_id[i])
                if det.tracker_id is not None and det.tracker_id[i] is not None
                else None
            )
            conf = float(det.confidence[i]) if det.confidence is not None else 0.0
            out.append(
                Detection(label="person", confidence=conf, box=(x1, y1, x2, y2), track_id=tid)
            )
        return out

    # 3) RULE --------------------------------------------------------------- #
    def events(self, detections: list[Detection], state: dict[str, Any]) -> list[Event]:
        now = time.time()
        events: list[Event] = []

        # state bền vững theo camera
        deb: Debouncer = state.setdefault("deb", Debouncer(min_seconds=self.debounce_s))
        cd: Cooldown = state.setdefault("cd", Cooldown(seconds=self.cooldown_s))
        crosser: LineCrosser | None = None
        if self.line is not None:
            crosser = state.get("crosser")
            if crosser is None:
                crosser = LineCrosser(a=self.line[0], b=self.line[1])
                state["crosser"] = crosser

        in_roi: list[Detection] = []
        for d in detections:
            foot = box_bottom_center(d.box)  # điểm chân — ổn định cho người đứng
            if (not self.roi) or point_in_polygon(foot, self.roi):
                in_roi.append(d)
            # vượt line: theo từng track_id, báo ngay (mỗi lần đổi phía)
            if crosser is not None and d.track_id is not None and crosser.update(d.track_id, foot):
                events.append(
                    Event(
                        type=EventType.INTRUSION,
                        module=self.module_id,
                        time=now,
                        camera="",  # router điền
                        box=d.box,
                        confidence=d.confidence,
                        severity=Severity.HIGH,
                        label=f"vượt line (track #{d.track_id})",
                    )
                )

        # có người trong ROI liên tục đủ debounce + qua cooldown -> 1 event
        if deb.update(bool(in_roi), now) and cd.ready(now):
            cd.fire(now)
            d = max(in_roi, key=lambda x: x.confidence)
            events.append(
                Event(
                    type=EventType.INTRUSION,
                    module=self.module_id,
                    time=now,
                    camera="",
                    box=d.box,
                    confidence=d.confidence,
                    severity=Severity.MEDIUM,
                    label=f"người trong ROI ({len(in_roi)})",
                )
            )
        return events
