"""
M4 Chấm công khuôn mặt — BASELINE (Track 0).  CHỦ: Lane A.

Model: InsightFace `buffalo_l` (SCRFD detect + ArcFace embed, 512-d, normed).
Store: numpy .npz (cosine match) — đủ cho MVP1; nâng pgvector khi scale.
M4 = superset: có thêm enroll().

⚠️ CHƯA có liveness ở MVP1 → ảnh điện thoại VẪN nhận. Event ghi rõ
   meta["liveness"]="NOT CHECKED (MVP2)" để báo cáo lỗ này.

IMPLEMENTATION THẬT (bài mẫu superset). Lane A nối tiếp: nối HRM, thêm
liveness (MiniFASNet) ở MVP2, đổi store sang pgvector khi cần.

NGHIỆM THU:
  ✅ dương: người ĐÃ enroll  -> đúng tên + giờ (Event ATTENDANCE).
  ✅ âm   : người lạ          -> "Unknown", không log.
"""

from __future__ import annotations

import time
from pathlib import Path
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


class _FaceStore:
    """Kho embedding đơn giản (numpy + npz). Cosine = dot vì embedding đã normed."""

    def __init__(self, path: str) -> None:
        self.path = Path(path).with_suffix(".npz")
        self.names: list[str] = []
        self.embs: np.ndarray = np.zeros((0, 512), dtype=np.float32)
        if self.path.exists():
            d = np.load(self.path, allow_pickle=True)
            self.names = list(d["names"])
            self.embs = d["embs"].astype(np.float32)

    def add(self, name: str, emb: np.ndarray) -> None:
        self.names.append(name)
        self.embs = np.vstack([self.embs, emb.reshape(1, -1).astype(np.float32)])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, names=np.array(self.names, dtype=object), embs=self.embs)

    def match(self, emb: np.ndarray) -> tuple[str, float]:
        if not self.names:
            return "Unknown", 0.0
        sims = self.embs @ emb.astype(np.float32)
        j = int(np.argmax(sims))
        return self.names[j], float(sims[j])


@register(ModuleId.M4, Engine.BASELINE)
class M4Baseline(M4Plugin):
    def __init__(self) -> None:
        self.threshold = 0.45            # cosine ngưỡng match (tune theo dữ liệu)
        self.db_path = "data/enroll/faces.npz"
        self.log_cooldown_s = 300.0      # 1 người log lại sau 5 phút (production: theo ca/ngày)
        self.det_size = (640, 640)
        self._app = None
        self._store: _FaceStore | None = None

    def configure(self, cfg: dict[str, Any]) -> None:
        self.threshold = cfg.get("threshold", self.threshold)
        self.db_path = cfg.get("db_path", self.db_path)
        self.log_cooldown_s = cfg.get("log_cooldown_s", self.log_cooldown_s)

    def setup(self) -> None:
        # QUY TẮC VÀNG: import nặng ở đây.
        from insightface.app import FaceAnalysis  # noqa: PLC0415

        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=self.det_size)
        self._store = _FaceStore(self.db_path)

    @staticmethod
    def _largest(faces):
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def enroll(self, name: str, faces: list[np.ndarray]) -> bool:
        if self._app is None:
            self.setup()
        embs = []
        for img in faces:
            found = self._app.get(img)
            if not found:
                continue
            embs.append(self._largest(found).normed_embedding)
        if not embs:
            print(f"[M4] enroll('{name}') thất bại — không bắt được mặt trong ảnh.")
            return False
        template = np.mean(embs, axis=0)
        template = template / (np.linalg.norm(template) + 1e-9)
        self._store.add(name, template)
        print(f"[M4] enroll OK: {name} ({len(embs)} ảnh hợp lệ)")
        return True

    def process(self, frame: np.ndarray, meta: FrameMeta) -> list[Detection]:
        if self._app is None or self._store is None:
            return []
        out: list[Detection] = []
        for f in self._app.get(frame):
            name, sim = self._store.match(f.normed_embedding)
            if sim < self.threshold:
                name = "Unknown"
            x1, y1, x2, y2 = (float(v) for v in f.bbox)
            out.append(
                Detection(
                    label="face",
                    confidence=float(getattr(f, "det_score", 0.0)),
                    box=(x1, y1, x2, y2),
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
                    meta={"sim": d.attributes.get("sim"), "liveness": "NOT CHECKED (MVP2)"},
                )
            )
        return events
