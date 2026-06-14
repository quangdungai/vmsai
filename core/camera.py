import asyncio
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

from config.settings import settings


class CameraStream:
    """Camera stream với hỗ trợ USB và RTSP IP camera, auto-reconnect."""

    def __init__(self, camera_id: str, source: str):
        self.camera_id = camera_id
        self.source = source
        self._capture = None
        self._connected = False
        self._last_frame = None
        self._last_annotated = None
        self._last_preview = None
        self._running = False
        self._lock = asyncio.Lock()
        self._is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp")

        self._open_camera()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _parse_source(self):
        source = self.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        return source

    def _open_camera(self):
        source = self._parse_source()
        # Windows: CAP_DSHOW ổn định hơn cho webcam USB
        if isinstance(source, int):
            self._capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self._capture = cv2.VideoCapture(source)

        if self._is_rtsp:
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, settings.RTSP_BUFFER_SIZE)
            self._capture.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)

        if settings.CAMERA_WIDTH:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
        if settings.CAMERA_HEIGHT:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)

        self._connected = self._capture.isOpened()
        if not self._connected:
            logger.error(f"Cannot open camera source: {self.source}")
        else:
            logger.info(f"Camera {self.camera_id} opened: {self.source}")

    def _reconnect(self):
        if self._capture is not None:
            self._capture.release()
        logger.warning(f"Reconnecting camera {self.camera_id}...")
        self._open_camera()

    def get_frame(self):
        if not self._connected or self._capture is None:
            return None

        ret, frame = self._capture.read()
        if not ret or frame is None:
            if self._is_rtsp:
                self._connected = False
                self._reconnect()
                ret, frame = self._capture.read() if self._capture else (False, None)
            if not ret:
                return self._last_frame

        self._last_frame = frame
        return frame

    def update_preview(self, frame: np.ndarray):
        """Cập nhật preview nhẹ cho stream — gọi mỗi frame."""
        if frame is None:
            return
        h, w = frame.shape[:2]
        sw = settings.STREAM_WIDTH
        if w > sw:
            sh = int(h * sw / w)
            self._last_preview = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        else:
            self._last_preview = frame

    def get_stream_frame(self):
        """Frame cho MJPEG — ưu tiên annotated, fallback preview/raw."""
        frame = self._last_annotated
        if frame is None:
            frame = self._last_preview
        if frame is None:
            frame = self._last_frame
        if frame is None:
            frame = self.get_frame()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        sw = settings.STREAM_WIDTH
        if w > sw:
            sh = int(h * sw / w)
            return cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        return frame

    async def stream_frames(self):
        fail_count = 0
        while True:
            frame = self.get_frame()
            if frame is not None:
                fail_count = 0
                yield frame
            else:
                fail_count += 1
                if fail_count >= 10 and self._is_rtsp:
                    await asyncio.sleep(settings.RTSP_RECONNECT_DELAY)
                    self._reconnect()
                    fail_count = 0
            await asyncio.sleep(1.0 / settings.CAMERA_FPS)

    def stop(self):
        self._running = False
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._connected = False
        logger.info(f"Camera {self.camera_id} stopped")


class CameraManager:
    def __init__(self):
        self._cameras: Dict[str, CameraStream] = {}

    def add_camera(self, camera_id: str, source: str):
        if camera_id in self._cameras:
            return self._cameras[camera_id]
        camera = CameraStream(camera_id, source)
        self._cameras[camera_id] = camera
        return camera

    def get_camera(self, camera_id: str) -> Optional[CameraStream]:
        return self._cameras.get(camera_id)

    def stop_all(self):
        for cam in self._cameras.values():
            cam.stop()
        self._cameras.clear()

    def get_status(self):
        return {
            camera_id: {
                "source": cam.source,
                "connected": cam.is_connected,
                "is_rtsp": cam._is_rtsp,
            }
            for camera_id, cam in self._cameras.items()
        }


camera_manager = CameraManager()
