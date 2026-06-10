"""
M2 Cháy/khói — BASELINE (Track 0).  CHỦ: Lane B.

Model: YOLO11 train trên D-Fire (models/fire_yolo.pt) — NẾU CHƯA train, tự
fallback **YOLO-World zero-shot** (prompt "fire"/"smoke") để demo chạy ngay.
Module RỦI RO NO-GO CAO NHẤT — đuôi tuning negative reel dài nhất.

IMPLEMENTATION THẬT (bài mẫu). Lane B nối tiếp: train fire_yolo, chỉnh
ngưỡng/debounce trên negative reel (hàn/hơi nước/nắng).

NGHIỆM THU:
  ✅ dương: clip lửa/khói        -> Event FIRE/SMOKE.
  ✅ âm   : hàn / hơi nước / nắng -> KHÔNG phát (phần khó — chỉnh debounce + Phase 1b VLM verify).
"""

from __future__ import annotations

import time
from pathlib import Path
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


@register(ModuleId.M2, Engine.BASELINE)
class M2Baseline(ModulePlugin):
    def __init__(self) -> None:
        self.model_path = "models/fire_yolo.pt"
        self.prompts = ["fire", "smoke"]   # dùng khi fallback YOLO-World
        self.conf_thr = 0.35
        self.debounce_s = 2.5              # cháy thật kéo dài > nhấp nháy
        self.cooldown_s = 30.0
        self._model = None

    def configure(self, cfg: dict[str, Any]) -> None:
        self.model_path = cfg.get("model_path", self.model_path)
        self.prompts = cfg.get("prompts", self.prompts)
        self.conf_thr = cfg.get("conf_thr", self.conf_thr)
        self.debounce_s = cfg.get("debounce_s", self.debounce_s)
        self.cooldown_s = cfg.get("cooldown_s", self.cooldown_s)

    def setup(self) -> None:
        # QUY TẮC VÀNG: import nặng ở đây.
        if self.model_path and Path(self.model_path).exists():
            from ultralytics import YOLO  # noqa: PLC0415

            self._model = YOLO(self.model_path)
        else:
            # Fallback zero-shot: chạy được ngay khi chưa train fire_yolo.
            from ultralytics import YOLOWorld  # noqa: PLC0415

            m = YOLOWorld("yolov8s-world.pt")
            m.set_classes(self.prompts)
            self._model = m
            print("[M2] Chưa có fire_yolo.pt -> dùng YOLO-World zero-shot (fire/smoke).")

    @staticmethod
    def _norm(name: str) -> str:
        n = name.lower()
        if "smoke" in n:
            return "smoke"
        if "fire" in n or "flame" in n:
            return "fire"
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
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            out.append(Detection(label=label, confidence=float(conf[i]), box=(x1, y1, x2, y2)))
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
                        label=lab,
                    )
                )
        return events
