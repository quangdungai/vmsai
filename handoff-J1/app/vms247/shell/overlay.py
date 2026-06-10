"""
L5 — Overlay. Vẽ box/label + banner sự kiện lên frame.
Dùng `supervision` nếu có, fallback OpenCV thuần.
"""

from __future__ import annotations

import cv2
import numpy as np

from vms247.core.schemas import Detection, Event

_COLORS = {
    "person": (0, 200, 0),
    "no_helmet": (0, 0, 255),
    "helmet": (0, 200, 0),
    "fire": (0, 0, 255),
    "smoke": (0, 165, 255),
    "face": (255, 200, 0),
}
_DEFAULT_COLOR = (200, 200, 200)


def draw_detections(frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        color = _COLORS.get(d.label, _DEFAULT_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"{d.label} {d.confidence:.2f}"
        if d.track_id is not None:
            tag = f"#{d.track_id} {tag}"
        name = d.attributes.get("name")
        if name:
            tag = f"{name} {tag}"
        cv2.putText(
            out, tag, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return out


def draw_roi(frame: np.ndarray, polygon: list[tuple[float, float]]) -> np.ndarray:
    if not polygon:
        return frame
    out = frame.copy()
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
    return out


def draw_event_banner(frame: np.ndarray, events: list[Event]) -> np.ndarray:
    if not events:
        return frame
    out = frame.copy()
    txt = " | ".join(f"{e.type.value.upper()}" for e in events[:3])
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 180), -1)
    cv2.putText(
        out, f"ALERT: {txt}", (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out
