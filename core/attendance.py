import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, Deque
from collections import deque
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from .face_engine import face_engine, RecognitionResult
from .anti_spoofing import anti_spoofing, FASResult
from .database import AttendanceLog, FaceAuditLog, Employee
from config.settings import settings


class AttendanceConfirmationBuffer:
    """
    Xác minh đa khung hình — yêu cầu N frame liên tiếp pass FAS + recognition.
    Chống spoof tấn công 1-frame và giảm FAR.
    """

    def __init__(self, required_frames: int, timeout_sec: float):
        self.required_frames = required_frames
        self.timeout_sec = timeout_sec
        self._buffer: Deque[str] = deque(maxlen=required_frames)
        self._started_at: Optional[datetime] = None

    def reset(self):
        self._buffer.clear()
        self._started_at = None

    def pending_count(self) -> int:
        return len(self._buffer)

    def add(self, employee_id: Optional[str], passed: bool) -> bool:
        """Thêm kết quả frame. Trả về True nếu đủ điều kiện chấm công."""
        now = datetime.now()

        if not passed or employee_id is None:
            self.reset()
            return False

        if self._started_at is None:
            self._started_at = now
        elif (now - self._started_at).total_seconds() > self.timeout_sec:
            self.reset()
            self._started_at = now

        self._buffer.append(employee_id)

        if len(self._buffer) < self.required_frames:
            return False

        if all(eid == employee_id for eid in self._buffer):
            self.reset()
            return True

        self.reset()
        return False


class AttendanceManager:
    """Quản lý luồng chấm công: detect → FAS → recognize → xác minh → ghi DB."""

    def __init__(self):
        self._cooldown_map: Dict[str, datetime] = {}
        self._confirm_buffer = AttendanceConfirmationBuffer(
            required_frames=settings.ATTENDANCE_CONFIRM_FRAMES,
            timeout_sec=settings.ATTENDANCE_CONFIRM_TIMEOUT_SEC,
        )
        self._audit_counter = 0
        self._processing = False

    def process_frame_sync(
        self,
        frame: np.ndarray,
        camera_id: str = "cam_01",
    ) -> Tuple[np.ndarray, Optional[dict], Optional[dict]]:
        """CPU-heavy pipeline — chạy trong thread pool, không block event loop."""
        annotated = frame
        attendance_result = None
        audit_payload = None

        faces = face_engine.detect_faces(frame)
        if not faces:
            self._confirm_buffer.reset()
            annotated = frame.copy()
            self._draw_status(annotated, "No face detected", (100, 100, 255))
            return annotated, None, None

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        track_id = f"cam_{camera_id}"

        fas_result = anti_spoofing.analyze(
            frame=frame,
            bbox=face.bbox,
            landmarks_5=face.landmarks,
            track_id=track_id
        )
        recog_result = face_engine.recognize(face)

        self._audit_counter += 1
        if self._audit_counter % settings.AUDIT_LOG_INTERVAL == 0:
            audit_payload = self._build_audit_payload(face, recog_result, fas_result, camera_id)

        passed = fas_result.is_live and recog_result.matched
        emp_id = recog_result.employee_id if passed else None

        annotated = frame.copy()
        if passed and self._confirm_buffer.add(emp_id, True):
            if not self._check_cooldown(emp_id):
                msg = f"Da cham — cooldown {settings.ATTENDANCE_COOLDOWN_MINUTES} phut"
                self._draw_status(annotated, msg, (200, 200, 0))
            else:
                attendance_result = self._build_attendance_payload(
                    recog_result, fas_result, frame, camera_id
                )
                self._update_cooldown(emp_id)
                anti_spoofing.reset_tracker(track_id)
        elif passed:
            done = self._confirm_buffer.pending_count
            total = settings.ATTENDANCE_CONFIRM_FRAMES
            self._draw_status(annotated, f"Xac minh... ({done}/{total})", (0, 200, 255))
        elif not fas_result.is_live:
            self._confirm_buffer.reset()
            reason = fas_result.reject_reason or "Spoof detected"
            self._draw_status(annotated, reason[:60], (0, 0, 255))
        elif recog_result.reject_reason:
            self._confirm_buffer.reset()
            self._draw_status(annotated, recog_result.reject_reason[:60], (0, 165, 255))

        annotated = face_engine.draw_result(
            annotated, recog_result, fas_result.is_live, fas_result.overall_score
        )
        self._draw_fas_overlay(annotated, fas_result)
        return annotated, attendance_result, audit_payload

    async def persist_results(
        self,
        db: AsyncSession,
        attendance_payload: Optional[dict],
        audit_payload: Optional[dict],
    ) -> Optional[dict]:
        if audit_payload:
            db.add(FaceAuditLog(**audit_payload))
        if attendance_payload:
            return await self._save_attendance(db, attendance_payload)
        return None

    async def process_frame(
        self,
        frame: np.ndarray,
        db: AsyncSession,
        camera_id: str = "cam_01"
    ) -> Tuple[np.ndarray, Optional[dict]]:
        annotated, attendance_payload, audit_payload = self.process_frame_sync(frame, camera_id)
        result = await self.persist_results(db, attendance_payload, audit_payload)
        return annotated, result

    def _check_cooldown(self, employee_id: str) -> bool:
        if employee_id not in self._cooldown_map:
            return True
        elapsed = datetime.now() - self._cooldown_map[employee_id]
        return elapsed.total_seconds() > settings.ATTENDANCE_COOLDOWN_MINUTES * 60

    def _update_cooldown(self, employee_id: str):
        self._cooldown_map[employee_id] = datetime.now()

    def _build_attendance_payload(
        self,
        recog_result: RecognitionResult,
        fas_result: FASResult,
        frame: np.ndarray,
        camera_id: str,
    ) -> dict:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        return {
            "recog_result": recog_result,
            "fas_result": fas_result,
            "frame": frame,
            "camera_id": camera_id,
            "now": now,
            "date_str": date_str,
            "time_str": time_str,
        }

    def _build_audit_payload(self, face, recog_result, fas_result, camera_id: str) -> dict:
        return dict(
            face_detected=True,
            face_count=1,
            detection_score=float(face.score),
            matched_employee_id=recog_result.employee_id,
            recognition_score=float(recog_result.similarity),
            recognition_passed=recog_result.matched,
            fas_texture_score=fas_result.texture_score,
            fas_liveness_score=fas_result.liveness_score,
            fas_freq_score=fas_result.freq_score,
            fas_overall_score=fas_result.overall_score,
            fas_passed=fas_result.is_live,
            fas_reject_reason=fas_result.reject_reason or recog_result.reject_reason,
            attendance_recorded=recog_result.matched and fas_result.is_live,
            camera_id=camera_id,
        )

    async def _save_attendance(self, db: AsyncSession, payload: dict):
        recog_result = payload["recog_result"]
        fas_result = payload["fas_result"]
        frame = payload["frame"]
        camera_id = payload["camera_id"]
        now = payload["now"]
        date_str = payload["date_str"]
        time_str = payload["time_str"]

        existing = await db.execute(
            select(AttendanceLog).where(
                and_(
                    AttendanceLog.employee_id == recog_result.employee_id,
                    AttendanceLog.date == date_str,
                    AttendanceLog.type == "checkin"
                )
            )
        )
        has_checkin = existing.scalars().first() is not None
        att_type = "checkout" if has_checkin else "checkin"

        is_late = False
        is_early_leave = False
        if att_type == "checkin":
            checkin_late = datetime.strptime(settings.CHECKIN_LATE_THRESHOLD, "%H:%M").time()
            is_late = now.time() > checkin_late
        elif att_type == "checkout":
            checkout_early = datetime.strptime(settings.CHECKOUT_EARLY_THRESHOLD, "%H:%M").time()
            is_early_leave = now.time() < checkout_early

        image_path = None
        if settings.LOG_ATTENDANCE_IMAGES:
            img_dir = Path(settings.IMAGES_PATH) / date_str
            img_dir.mkdir(parents=True, exist_ok=True)
            img_filename = f"{recog_result.employee_id}_{att_type}_{now.strftime('%H%M%S')}.jpg"
            image_path = str(img_dir / img_filename)
            cv2.imwrite(image_path, frame)

        log = AttendanceLog(
            employee_id=recog_result.employee_id,
            employee_name=recog_result.employee_name,
            timestamp=now,
            date=date_str,
            time=time_str,
            type=att_type,
            status="late" if is_late else ("early_leave" if is_early_leave else "valid"),
            recognition_score=float(recog_result.similarity),
            liveness_score=float(fas_result.liveness_score),
            fas_passed=fas_result.is_live,
            camera_id=camera_id,
            image_path=image_path,
            is_late=is_late,
            is_early_leave=is_early_leave,
        )
        db.add(log)
        await db.flush()

        type_str = "Check-in" if att_type == "checkin" else "Check-out"
        logger.success(
            f"{type_str}: {recog_result.employee_name} ({recog_result.employee_id}) "
            f"luc {time_str} | sim={recog_result.similarity:.3f} fas={fas_result.overall_score:.3f}"
        )
        return {
            "success": True,
            "employee_id": recog_result.employee_id,
            "employee_name": recog_result.employee_name,
            "type": att_type,
            "time": time_str,
            "date": date_str,
            "is_late": is_late,
            "is_early_leave": is_early_leave,
            "recognition_score": float(recog_result.similarity),
            "fas_score": float(fas_result.overall_score),
        }

    def _draw_fas_overlay(self, frame: np.ndarray, fas: FASResult):
        h, w = frame.shape[:2]
        panel_h = 145
        cv2.rectangle(frame, (10, 10), (300, panel_h), (0, 0, 0), -1)

        color_ok = (0, 255, 0)
        color_fail = (0, 0, 255)
        y = 30
        cv2.putText(frame, "ANTI-SPOOFING", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += 18

        items = [
            ("Texture", fas.texture_score, settings.FAS_TEXTURE_THRESHOLD),
            ("Liveness", fas.liveness_score, 0.5),
            ("Frequency", fas.freq_score, settings.FAS_FREQ_THRESHOLD),
            ("Temporal", fas.overall_score, 0.5),
            ("DL/Heur", fas.dl_score, 0.6),
        ]
        for name, score, threshold in items:
            c = color_ok if score >= threshold else color_fail
            bar_len = int(min(score, 1.0) * 80)
            cv2.putText(frame, f"{name}: {score:.2f}", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1)
            cv2.rectangle(frame, (130, y-9), (130 + bar_len, y-2), c, -1)
            y += 15

        blink_txt = "Blink:OK" if fas.blink_detected else "Blink:--"
        motion_txt = "Motion:OK" if fas.motion_detected else "Motion:--"
        cv2.putText(frame, f"{blink_txt} | {motion_txt}", (15, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    def _draw_status(self, frame: np.ndarray, message: str, color: tuple):
        h, w = frame.shape[:2]
        cv2.putText(frame, message, (20, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


attendance_manager = AttendanceManager()
