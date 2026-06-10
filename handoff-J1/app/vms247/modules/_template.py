"""
VÍ DỤ MẪU HOÀN CHỈNH cho Junior — KHÔNG dùng model nặng (chỉ OpenCV).

Đây là cách 1 plugin "đúng chuẩn" trông như thế nào: implement đủ
setup / process / events, dùng helper rule (ROI + debounce + cooldown),
lưu state per-camera. Junior COPY pattern này cho M1..M4.

Chạy thử standalone (không cần shell):
    python -m vms247.modules._template data/videos/any.mp4
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from vms247.core.interfaces import ModulePlugin
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


class TemplateMotionPlugin(ModulePlugin):
    """Phát hiện chuyển động (frame-diff) làm mẫu — KHÔNG đăng ký vào registry."""

    module_id = ModuleId.M1
    engine = Engine.BASELINE

    def __init__(self) -> None:
        self.roi: list[tuple[float, float]] = []
        self.min_area = 1500

    def configure(self, cfg: dict[str, Any]) -> None:
        """(Tuỳ chọn) nhận config từ shell: ROI, ngưỡng..."""
        self.roi = [tuple(p) for p in cfg.get("roi", [])]
        self.min_area = cfg.get("min_area", self.min_area)

    # 1) LOAD --------------------------------------------------------------- #
    def setup(self) -> None:
        self._prev_gray: np.ndarray | None = None

    # 2) INFER -------------------------------------------------------------- #
    def process(self, frame: np.ndarray, meta: FrameMeta) -> list[Detection]:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        dets: list[Detection] = []
        if self._prev_gray is not None:
            delta = cv2.absdiff(self._prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                if cv2.contourArea(c) < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                dets.append(
                    Detection(label="motion", confidence=0.5, box=(x, y, x + w, y + h))
                )
        self._prev_gray = gray
        return dets

    # 3) RULE --------------------------------------------------------------- #
    def events(self, detections: list[Detection], state: dict[str, Any]) -> list[Event]:
        # state bền vững theo camera — khởi tạo timer 1 lần
        deb: Debouncer = state.setdefault("deb", Debouncer(min_seconds=1.0))
        cd: Cooldown = state.setdefault("cd", Cooldown(seconds=10.0))
        now = time.time()

        in_roi = any(
            (not self.roi) or point_in_polygon(box_bottom_center(d.box), self.roi)
            for d in detections
        )
        fired = deb.update(in_roi, now)
        if fired and cd.ready(now):
            cd.fire(now)
            d = detections[0]
            return [
                Event(
                    type=EventType.INTRUSION,
                    module=self.module_id,
                    time=now,
                    camera="",  # router sẽ điền
                    box=d.box,
                    confidence=d.confidence,
                    severity=Severity.MEDIUM,
                    label="motion in ROI (template)",
                )
            ]
        return []


def _demo(path: str) -> None:  # pragma: no cover
    from vms247.shell.input import VideoSource
    from vms247.shell.overlay import draw_detections, draw_event_banner

    p = TemplateMotionPlugin()
    p.setup()
    st: dict = {}
    with VideoSource(path, target_fps=5.0) as src:
        for frame, meta in src.frames():
            dets = p.process(frame, meta)
            evs = p.events(dets, st)
            vis = draw_event_banner(draw_detections(frame, dets), evs)
            cv2.imshow("template", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover
    import sys

    _demo(sys.argv[1] if len(sys.argv) > 1 else 0)
