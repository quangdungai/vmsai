"""
M3 PPE mũ bảo hộ — BASELINE (Track 0).  CHỦ: Lane A.

Model: hardhat YOLO train trên Hard Hat Workers (models/hardhat_yolo.pt) —
NẾU CHƯA train, fallback **YOLO-World zero-shot** (prompt "helmet"/"head")
để demo chạy ngay. Quy ước: class "head" (đầu không mũ) -> nhãn "no_helmet".

IMPLEMENTATION THẬT (bài mẫu cho Lane A). Lane A nối tiếp: train hardhat_yolo,
gắn zones bắt buộc đội mũ, chỉnh ngưỡng.

NGHIỆM THU:
  ✅ dương: người CỞI mũ trong zone -> Event PPE_VIOLATION.
  ✅ âm   : người ĐỘI mũ             -> im.
"""

from __future__ import annotations

import time
from pathlib import Path
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


@register(ModuleId.M3, Engine.BASELINE)
class M3Baseline(ModulePlugin):
    def __init__(self) -> None:
        self.model_path = "models/hardhat_yolo.pt"
        self.prompts = ["helmet", "head"]      # fallback YOLO-World
        self.conf_thr = 0.4
        self.debounce_s = 1.5
        self.cooldown_s = 20.0
        self.zones: list[list[tuple[float, float]]] = []  # vùng bắt buộc đội mũ
        self._model = None

    def configure(self, cfg: dict[str, Any]) -> None:
        self.model_path = cfg.get("model_path", self.model_path)
        self.prompts = cfg.get("prompts", self.prompts)
        self.conf_thr = cfg.get("conf_thr", self.conf_thr)
        self.debounce_s = cfg.get("debounce_s", self.debounce_s)
        self.cooldown_s = cfg.get("cooldown_s", self.cooldown_s)
        self.zones = [[tuple(p) for p in z] for z in cfg.get("zones", [])]

    def setup(self) -> None:
        if self.model_path and Path(self.model_path).exists():
            from ultralytics import YOLO  # noqa: PLC0415

            self._model = YOLO(self.model_path)
        else:
            from ultralytics import YOLOWorld  # noqa: PLC0415

            m = YOLOWorld("yolov8s-world.pt")
            m.set_classes(self.prompts)
            self._model = m
            print("[M3] Chưa có hardhat_yolo.pt -> dùng YOLO-World zero-shot (helmet/head).")

    @staticmethod
    def _norm(name: str) -> str:
        n = name.lower()
        if "helmet" in n or "hardhat" in n or "hard hat" in n:
            return "helmet"
        if "head" in n or "no_helmet" in n or "no-helmet" in n:
            return "no_helmet"
        return n

    def process(self, frame: np.ndarray, meta: FrameMeta) -> list[Detection]:
        if self._model is None:
            return []
        res = self._model(frame, conf=self.conf_thr, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        names = res.names
        out: list[Detection] = []
        for i in range(len(cls)):
            label = self._norm(names[cls[i]])
            if label not in ("helmet", "no_helmet"):
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            out.append(Detection(label=label, confidence=float(conf[i]), box=(x1, y1, x2, y2)))
        return out

    def events(self, detections: list[Detection], state: dict[str, Any]) -> list[Event]:
        now = time.time()
        deb: Debouncer = state.setdefault("deb", Debouncer(min_seconds=self.debounce_s))
        cd: Cooldown = state.setdefault("cd", Cooldown(seconds=self.cooldown_s))

        violators = []
        for d in detections:
            if d.label != "no_helmet":
                continue
            c = box_center(d.box)
            if (not self.zones) or any(point_in_polygon(c, z) for z in self.zones):
                violators.append(d)

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
                    label=f"không đội mũ ({len(violators)})",
                )
            ]
        return []
