"""
M4 Chấm công — DESIGNATED (Track 1, Phase 1b).  CHỦ: Lane A.  CHẠY CLOUD.

DEIMv2-Wholebody49 (face/head detect + align, THAY SCRFD) + ArcFace embed.
Mục tiêu consolidation: cùng backbone Wholebody49 cho M1/M3/M4. Đối chứng AdaFace
(ảnh chất lượng thấp) là tuỳ chọn benchmark.

Tái dùng store cosine của baseline (`_FaceStore`). ĐIỂM CẮM (Lane A):
(1) Wholebody49.detect lấy face + landmark, (2) align + embed (ArcFace/AdaFace).
Chưa cấu hình -> log + trả [] (không crash demo).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vms247.core.interfaces import M4Plugin
from vms247.core.registry import register
from vms247.core.schemas import (
    Detection,
    Engine,
    Event,
    EventType,
    FrameMeta,
    ModuleId,
    Severity,
)


@register(ModuleId.M4, Engine.DESIGNATED)
class M4Designated(M4Plugin):
    def __init__(self) -> None:
        self.wholebody_onnx: str | None = None
        self.threshold = 0.45
        self.db_path = "data/enroll/faces_designated.npz"
        self.log_cooldown_s = 300.0
        self._wb = None
        self._store = None
        self._warned = False

    def configure(self, cfg: dict[str, Any]) -> None:
        self.wholebody_onnx = cfg.get("wholebody_onnx", self.wholebody_onnx)
        self.threshold = cfg.get("threshold", self.threshold)
        self.db_path = cfg.get("db_path_designated", self.db_path)

    def setup(self) -> None:
        from vms247.integrations.wholebody import Wholebody49  # noqa: PLC0415

        from .baseline import _FaceStore  # tái dùng store cosine  # noqa: PLC0415

        self._store = _FaceStore(self.db_path)
        wb = Wholebody49(self.wholebody_onnx)
        if wb.available():
            try:
                wb.setup()
                self._wb = wb
            except Exception as e:  # pragma: no cover
                print(f"[M4/designated] Wholebody load lỗi: {e}")
        if self._wb is None:
            print("[M4/designated] chưa cấu hình wholebody_onnx — phần Phase 1b, môi trường Lead cấp.")

    def _embed(self, face_crop: np.ndarray) -> np.ndarray | None:
        """ĐIỂM CẮM: align + ArcFace/AdaFace embed (512-d, normed). Lane A implement."""
        raise NotImplementedError("M4/designated embed: align + ArcFace — điểm cắm Phase 1b")

    def _faces(self, frame: np.ndarray) -> list[dict]:
        if self._wb is None:
            return []
        try:
            return [d for d in self._wb.detect(frame) if d.get("part") == "face"]
        except NotImplementedError:
            if not self._warned:
                print("[M4/designated] Wholebody.detect + embed chưa implement (điểm cắm).")
                self._warned = True
            return []
        except Exception as e:  # pragma: no cover
            print(f"[M4/designated] detect lỗi: {e}")
            return []

    def enroll(self, name: str, faces: list[np.ndarray]) -> bool:
        if self._store is None:
            self.setup()
        embs = []
        for img in faces:
            for f in self._faces(img):
                try:
                    e = self._embed(_crop(img, f["box"]))
                    if e is not None:
                        embs.append(e)
                except NotImplementedError:
                    return False
        if not embs:
            return False
        t = np.mean(embs, axis=0)
        self._store.add(name, t / (np.linalg.norm(t) + 1e-9))
        return True

    def process(self, frame: np.ndarray, meta: FrameMeta) -> list[Detection]:
        if self._store is None:
            return []
        out: list[Detection] = []
        for f in self._faces(frame):
            try:
                emb = self._embed(_crop(frame, f["box"]))
            except NotImplementedError:
                return []
            if emb is None:
                continue
            name, sim = self._store.match(emb)
            if sim < self.threshold:
                name = "Unknown"
            out.append(
                Detection(
                    label="face",
                    confidence=float(f.get("score", 0.0)),
                    box=tuple(f["box"]),
                    attributes={"name": name, "sim": round(sim, 3)},
                )
            )
        return out

    def events(self, detections: list[Detection], state: dict[str, Any]) -> list[Event]:
        now = time.time()
        logged: dict[str, float] = state.setdefault("logged", {})
        events: list[Event] = []
        for d in detections:
            name = d.attributes.get("name")
            if not name or name == "Unknown":
                continue
            if now - logged.get(name, 0.0) < self.log_cooldown_s:
                continue
            logged[name] = now
            events.append(
                Event(
                    type=EventType.ATTENDANCE,
                    module=self.module_id,
                    time=now,
                    camera="",
                    box=d.box,
                    confidence=float(d.attributes.get("sim", 0.0)),
                    severity=Severity.INFO,
                    label=name,
                    meta={"engine": "designated", "liveness": "NOT CHECKED (MVP2)"},
                )
            )
        return events


def _crop(frame: np.ndarray, box) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(max(0, v)) for v in box)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame
