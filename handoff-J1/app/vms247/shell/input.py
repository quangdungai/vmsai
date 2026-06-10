"""
L4 — Input layer.

Đọc video từ file / RTSP / webcam bằng OpenCV, sample về target FPS (3-5),
yield (frame_BGR, FrameMeta). Thống nhất 1 nguồn frame cho mọi module.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import cv2
import numpy as np

from vms247.core.schemas import FrameMeta


class VideoSource:
    """
    source:
        - int hoặc "webcam0"  -> webcam index
        - "rtsp://..."        -> camera IP
        - đường dẫn file       -> video file
    target_fps: tần suất sample (None = giữ nguyên FPS nguồn).
    """

    def __init__(
        self,
        source: str | int,
        target_fps: float | None = 5.0,
        camera_id: str | None = None,
    ) -> None:
        self.source = self._normalize(source)
        self.camera_id = camera_id or self._auto_camera_id(source)
        self.target_fps = target_fps
        self._cap: cv2.VideoCapture | None = None

    @staticmethod
    def _normalize(source: str | int) -> str | int:
        if isinstance(source, int):
            return source
        if source.lower().startswith("webcam"):
            tail = source[6:] or "0"
            return int(tail)
        return source

    @staticmethod
    def _auto_camera_id(source: str | int) -> str:
        if isinstance(source, int) or str(source).lower().startswith("webcam"):
            return f"webcam{source}"
        if str(source).lower().startswith("rtsp"):
            return "rtsp"
        return f"file:{str(source).split('/')[-1].split(chr(92))[-1]}"

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn video: {self.source!r}")

    @property
    def src_fps(self) -> float:
        if self._cap is None:
            return 0.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    def frames(self) -> Iterator[tuple[np.ndarray, FrameMeta]]:
        """Generator yield (frame, meta) đã sample theo target_fps."""
        if self._cap is None:
            self.open()
        assert self._cap is not None

        src_fps = self.src_fps
        stride = 1
        if self.target_fps and self.target_fps < src_fps:
            stride = max(1, round(src_fps / self.target_fps))

        raw_idx = 0
        out_idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if raw_idx % stride == 0:
                h, w = frame.shape[:2]
                meta = FrameMeta(
                    camera=self.camera_id,
                    frame_index=out_idx,
                    timestamp=time.time(),
                    width=w,
                    height=h,
                )
                yield frame, meta
                out_idx += 1
            raw_idx += 1

        self.release()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
