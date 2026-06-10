"""
VMS247 MVP1 — Data contracts (FROZEN interface, L2).

Đây là "ngôn ngữ chung" giữa shell (Lead) và plugin (Junior).
KHÔNG đổi các field cốt lõi nếu chưa thống nhất với Lead — mọi plugin
và mọi phần shell phụ thuộc vào shape này.

Cốt lõi (theo kế hoạch MVP1):
    Detection : 1 vật thể trên 1 frame  (box · label · score)
    Event     : {type, time, camera, box, confidence, evidence_path}
    Metric    : {fps, vram_mb, latency_ms, engine}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Enums — định danh module & track
# --------------------------------------------------------------------------- #
class ModuleId(str, Enum):
    M1 = "M1"  # Xâm nhập (person + ROI/line-cross)
    M2 = "M2"  # Cháy/khói
    M3 = "M3"  # PPE (mũ bảo hộ)
    M4 = "M4"  # Chấm công khuôn mặt


class Engine(str, Enum):
    """2-track: baseline (lưới an toàn) ↔ designated (đang đánh giá)."""
    BASELINE = "baseline"      # YOLO11 / hardhat / fire-YOLO / InsightFace
    DESIGNATED = "designated"  # DEIMv2-Wholebody49 / LocateAnything-3B


class EventType(str, Enum):
    INTRUSION = "intrusion"          # M1
    FIRE = "fire"                    # M2
    SMOKE = "smoke"                  # M2
    PPE_VIOLATION = "ppe_violation"  # M3 (không đội mũ)
    ATTENDANCE = "attendance"        # M4 (chấm công thành công)


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Bounding box: (x1, y1, x2, y2) theo pixel, gốc trên-trái.
BBox = tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# FrameMeta — ngữ cảnh 1 frame (router cấp cho plugin)
# --------------------------------------------------------------------------- #
@dataclass
class FrameMeta:
    camera: str          # id/tên nguồn (vd "cam01", "file:fire.mp4", "webcam0")
    frame_index: int     # số thứ tự frame đã được sample
    timestamp: float     # epoch seconds (thời điểm decode)
    width: int
    height: int


# --------------------------------------------------------------------------- #
# Detection — output của plugin.process()
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """Một vật thể phát hiện trên 1 frame."""
    label: str                 # vd "person", "helmet", "no_helmet", "fire", "smoke", "face"
    confidence: float          # [0..1]
    box: BBox                  # (x1, y1, x2, y2) pixel
    track_id: Optional[int] = None        # id bám vết (ByteTrack/BoT-SORT) nếu có
    attributes: dict[str, Any] = field(default_factory=dict)
    # attributes ví dụ:
    #   M3: {"helmet": False}
    #   M4: {"name": "Nguyen Van A", "sim": 0.62, "embedding_id": 12}


# --------------------------------------------------------------------------- #
# Event — output của plugin.events() (sau Rule Engine)
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """Sự kiện nghiệp vụ sau khi áp luật (ROI/ngưỡng/đổi-trạng-thái)."""
    type: EventType
    module: ModuleId
    time: float                          # epoch seconds
    camera: str
    box: Optional[BBox] = None
    confidence: float = 0.0
    severity: Severity = Severity.MEDIUM
    label: str = ""                      # mô tả ngắn (vd tên người, "không đội mũ")
    evidence_path: Optional[str] = None  # đường dẫn snapshot/clip (shell tự điền nếu trống)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Phẳng hoá để log/jsonl/UI table."""
        return {
            "time": self.time,
            "module": self.module.value,
            "type": self.type.value,
            "camera": self.camera,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "severity": self.severity.value,
            "box": self.box,
            "evidence_path": self.evidence_path,
            **self.meta,
        }


# --------------------------------------------------------------------------- #
# Metric — output đo lường (L7 metric logger)
# --------------------------------------------------------------------------- #
@dataclass
class Metric:
    module: ModuleId
    engine: Engine
    fps: float = 0.0
    latency_ms: float = 0.0   # thời gian process() 1 frame
    vram_mb: float = 0.0
    time: float = 0.0

    def to_row(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "module": self.module.value,
            "engine": self.engine.value,
            "fps": round(self.fps, 2),
            "latency_ms": round(self.latency_ms, 2),
            "vram_mb": round(self.vram_mb, 1),
        }
