"""
FastAPI Application - Chấm công nhận diện khuôn mặt
API Endpoints:
- /stream         : Video stream từ camera (MJPEG)
- /ws/attendance  : WebSocket real-time attendance events
- /api/employees  : CRUD nhân viên
- /api/attendance : Xem báo cáo chấm công
- /api/register   : Đăng ký khuôn mặt
- /               : Web UI dashboard
"""
import asyncio
import base64
import io
import json
from datetime import datetime, date
from typing import Optional, List
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from loguru import logger
from pydantic import BaseModel

from config.settings import settings
from core.database import init_db, get_db, db_session, Employee, AttendanceLog, FaceAuditLog
from core.face_engine import face_engine
from core.anti_spoofing import anti_spoofing
from core.camera import camera_manager
from core.attendance import attendance_manager

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="Face Attendance System",
    description="Hệ thống chấm công nhận diện khuôn mặt với Anti-Spoofing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_path = Path(__file__).parent / "web" / "static"
static_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# WebSocket connections
ws_clients: List[WebSocket] = []


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================
@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting Face Attendance System...")
    
    # Init DB
    await init_db()
    logger.success("✅ Database initialized")
    
    # Init face engine
    await face_engine.initialize()
    
    # Start main camera
    camera_manager.add_camera("cam_01", settings.CAMERA_SOURCE)
    logger.success(f"✅ Camera started: {settings.CAMERA_SOURCE}")
    
    # Start processing loop
    asyncio.create_task(attendance_processing_loop())
    logger.success("✅ Processing loop started")


@app.on_event("shutdown")
async def shutdown():
    camera_manager.stop_all()
    logger.info("👋 System shutdown")


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================
async def attendance_processing_loop():
    """AI loop — chạy ML trong thread pool, không block stream."""
    cam = camera_manager.get_camera("cam_01")
    if not cam:
        return

    frame_count = 0
    logger.info("Attendance processing loop started")

    async for frame in cam.stream_frames():
        try:
            frame_count += 1
            cam.update_preview(frame)

            if frame_count % settings.FRAME_SKIP != 0:
                continue
            if attendance_manager._processing:
                continue

            attendance_manager._processing = True
            annotated, attendance_payload, audit_payload = await asyncio.to_thread(
                attendance_manager.process_frame_sync, frame, "cam_01"
            )
            attendance_manager._processing = False

            cam._last_annotated = annotated

            if attendance_payload or audit_payload:
                async with db_session() as db:
                    result = await attendance_manager.persist_results(
                        db, attendance_payload, audit_payload
                    )
            else:
                result = None

            if result and ws_clients:
                msg = json.dumps({"type": "attendance", "data": result})
                dead_clients = []
                for ws in ws_clients:
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead_clients.append(ws)
                for dc in dead_clients:
                    ws_clients.remove(dc)

        except Exception as e:
            attendance_manager._processing = False
            logger.error(f"Processing loop error: {e}")
            await asyncio.sleep(0.05)


# ============================================================
# VIDEO STREAM
# ============================================================
@app.get("/stream")
async def video_stream(camera_id: str = "cam_01"):
    """MJPEG video stream endpoint"""
    
    async def generate():
        cam = camera_manager.get_camera(camera_id)
        if not cam:
            return

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, settings.STREAM_JPEG_QUALITY]
        interval = 1.0 / settings.STREAM_FPS

        while True:
            frame = cam.get_stream_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    buffer.tobytes() +
                    b'\r\n'
                )
            await asyncio.sleep(interval)
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )


# ============================================================
# WEBSOCKET
# ============================================================
@app.websocket("/ws/attendance")
async def ws_attendance(websocket: WebSocket):
    """WebSocket for real-time attendance events"""
    await websocket.accept()
    ws_clients.append(websocket)
    logger.info(f"WebSocket connected. Total: {len(ws_clients)}")
    
    try:
        # Gửi ping để giữ kết nối
        while True:
            await websocket.send_text(json.dumps({"type": "ping"}))
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        ws_clients.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining: {len(ws_clients)}")


# ============================================================
# EMPLOYEE APIs
# ============================================================
class EmployeeCreate(BaseModel):
    employee_id: str
    full_name: str
    department: Optional[str] = None
    position: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


@app.post("/api/employees")
async def create_employee(data: EmployeeCreate, db: AsyncSession = Depends(get_db)):
    """Tạo nhân viên mới"""
    existing = await db.execute(
        select(Employee).where(Employee.employee_id == data.employee_id)
    )
    if existing.scalars().first():
        raise HTTPException(400, f"Employee {data.employee_id} already exists")
    
    emp = Employee(**data.model_dump())
    db.add(emp)
    await db.flush()
    return {"success": True, "message": f"Created employee {data.employee_id}"}


@app.get("/api/employees")
async def list_employees(db: AsyncSession = Depends(get_db)):
    """Danh sách nhân viên"""
    result = await db.execute(select(Employee).where(Employee.is_active == True))
    employees = result.scalars().all()
    return [
        {
            "employee_id": e.employee_id,
            "full_name": e.full_name,
            "department": e.department,
            "position": e.position,
            "face_samples": e.face_samples,
            "is_active": e.is_active,
        }
        for e in employees
    ]


@app.post("/api/employees/register")
async def register_employee_with_face(
    data: EmployeeCreate,
    n_frames: int = 15,
    db: AsyncSession = Depends(get_db)
):
    """Tạo nhân viên + đăng ký khuôn mặt từ camera đang chạy."""
    result = await db.execute(
        select(Employee).where(Employee.employee_id == data.employee_id)
    )
    emp = result.scalars().first()
    if not emp:
        emp = Employee(**data.model_dump())
        db.add(emp)
        await db.flush()

    cam = camera_manager.get_camera("cam_01")
    if not cam or not cam.is_connected:
        raise HTTPException(503, "Camera không khả dụng. Kiểm tra CAMERA_SOURCE trong .env")

    images = []
    for _ in range(150):
        frame = cam.get_frame()
        if frame is not None:
            faces = face_engine.detect_faces(frame)
            if len(faces) == 1 and faces[0].score >= settings.RECOGNITION_MIN_QUALITY:
                images.append(frame.copy())
                if len(images) >= n_frames:
                    break
        await asyncio.sleep(0.08)

    if len(images) < 5:
        raise HTTPException(
            400,
            f"Không đủ ảnh khuôn mặt ({len(images)}/5). "
            "Nhìn thẳng camera, đủ ánh sáng, chỉ 1 người trong khung hình."
        )

    reg_result = await face_engine.register_employee(
        employee_id=data.employee_id,
        name=data.full_name,
        images=images,
        replace=True
    )

    if reg_result["success"]:
        emp.face_samples = reg_result["count"]
        emp.embedding_path = f"{settings.EMBEDDINGS_PATH}/face_db.pkl"
        await db.flush()

    return {
        **reg_result,
        "employee_id": data.employee_id,
        "full_name": data.full_name,
    }


@app.post("/api/employees/{employee_id}/register-face")
async def register_face(
    employee_id: str,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Đăng ký khuôn mặt cho nhân viên
    Upload 10-20 ảnh với các góc khác nhau: thẳng, trái, phải, lên, xuống
    """
    # Kiểm tra nhân viên tồn tại
    result = await db.execute(
        select(Employee).where(Employee.employee_id == employee_id)
    )
    emp = result.scalars().first()
    if not emp:
        raise HTTPException(404, f"Employee {employee_id} not found")
    
    # Đọc ảnh
    images = []
    for f in files:
        content = await f.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    
    if not images:
        raise HTTPException(400, "No valid images provided")
    
    # Đăng ký
    reg_result = await face_engine.register_employee(
        employee_id=employee_id,
        name=emp.full_name,
        images=images,
        replace=True
    )
    
    if reg_result["success"]:
        # Cập nhật DB
        emp.face_samples = reg_result["count"]
        emp.embedding_path = f"{settings.EMBEDDINGS_PATH}/face_db.pkl"
        await db.flush()
    
    return reg_result


@app.post("/api/employees/{employee_id}/register-face-camera")
async def register_face_from_camera(
    employee_id: str,
    n_frames: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Đăng ký khuôn mặt trực tiếp từ camera"""
    result = await db.execute(
        select(Employee).where(Employee.employee_id == employee_id)
    )
    emp = result.scalars().first()
    if not emp:
        raise HTTPException(404, f"Employee {employee_id} not found")
    
    cam = camera_manager.get_camera("cam_01")
    if not cam or not cam.is_connected:
        raise HTTPException(503, "Camera not available")
    
    # Capture n_frames
    images = []
    collected = 0
    max_wait = 100  # frames
    
    for _ in range(max_wait):
        frame = cam.get_frame()
        if frame is not None:
            faces = face_engine.detect_faces(frame)
            if len(faces) == 1:  # Chỉ lấy nếu có đúng 1 khuôn mặt
                images.append(frame)
                collected += 1
                if collected >= n_frames:
                    break
        await asyncio.sleep(0.1)
    
    if len(images) < 5:
        raise HTTPException(400, f"Not enough frames captured ({len(images)}). Make sure face is visible.")
    
    reg_result = await face_engine.register_employee(
        employee_id=employee_id,
        name=emp.full_name,
        images=images,
        replace=True
    )
    
    if reg_result["success"]:
        emp.face_samples = reg_result["count"]
        await db.flush()
    
    return reg_result


# ============================================================
# ATTENDANCE APIs
# ============================================================
@app.get("/api/attendance")
async def get_attendance(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    employee_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Lấy danh sách chấm công"""
    query = select(AttendanceLog).order_by(AttendanceLog.timestamp.desc())
    
    filters = []
    if date_from:
        filters.append(AttendanceLog.date >= date_from)
    if date_to:
        filters.append(AttendanceLog.date <= date_to)
    if employee_id:
        filters.append(AttendanceLog.employee_id == employee_id)
    
    if filters:
        query = query.where(and_(*filters))
    
    result = await db.execute(query.limit(500))
    logs = result.scalars().all()
    
    return [
        {
            "id": l.id,
            "employee_id": l.employee_id,
            "employee_name": l.employee_name,
            "date": l.date,
            "time": l.time,
            "type": l.type,
            "status": l.status,
            "is_late": l.is_late,
            "recognition_score": round(l.recognition_score or 0, 3),
            "fas_passed": l.fas_passed,
        }
        for l in logs
    ]


@app.get("/api/attendance/today")
async def get_today_attendance(db: AsyncSession = Depends(get_db)):
    """Báo cáo chấm công hôm nay"""
    today = date.today().isoformat()
    result = await db.execute(
        select(AttendanceLog)
        .where(AttendanceLog.date == today)
        .order_by(AttendanceLog.timestamp)
    )
    logs = result.scalars().all()
    
    # Summary
    checkins = [l for l in logs if l.type == "checkin"]
    checkouts = [l for l in logs if l.type == "checkout"]
    late = [l for l in checkins if l.is_late]
    
    return {
        "date": today,
        "total_checkin": len(checkins),
        "total_checkout": len(checkouts),
        "late_count": len(late),
        "logs": [
            {
                "employee_id": l.employee_id,
                "employee_name": l.employee_name,
                "time": l.time,
                "type": l.type,
                "is_late": l.is_late,
                "status": l.status,
            }
            for l in logs
        ]
    }


@app.get("/api/system/status")
async def system_status():
    """Status hệ thống"""
    return {
        "status": "running",
        "cameras": camera_manager.get_status(),
        "employees_loaded": len(face_engine.embeddings_db),
        "fas_config": {
            "texture_enabled": settings.FAS_TEXTURE_ENABLED,
            "liveness_enabled": settings.FAS_LIVENESS_ENABLED,
            "freq_enabled": settings.FAS_FREQ_ENABLED,
            "temporal_enabled": settings.FAS_TEMPORAL_ENABLED,
            "dl_enabled": settings.FAS_DL_ENABLED,
            "final_threshold": settings.FAS_FINAL_THRESHOLD,
            "min_layers_pass": settings.FAS_MIN_LAYERS_PASS,
            "confirm_frames": settings.ATTENDANCE_CONFIRM_FRAMES,
        },
        "recognition_threshold": settings.RECOGNITION_THRESHOLD,
        "recognition_margin": settings.RECOGNITION_MARGIN,
        "strict_mode": settings.STRICT_MODE,
        "version": "1.0.0"
    }


# ============================================================
# WEB UI
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Dashboard chấm công"""
    html_path = Path(__file__).parent / "web" / "templates" / "dashboard.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Face Attendance System</h1><p>UI not found. See /docs for API.</p>"
