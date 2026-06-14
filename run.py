#!/usr/bin/env python3
"""
Entry point - Khởi động hệ thống chấm công
Usage:
  python run.py               # Start server
  python run.py register      # Register employee (interactive)
  python run.py test-camera   # Test camera connection
  python run.py report        # Xuất báo cáo hôm nay
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import uvicorn
import asyncio
import argparse
import cv2
import numpy as np
from pathlib import Path
from loguru import logger


def setup_logging():
    from config.settings import settings
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>: {message}"
    )
    logger.add(
        f"{settings.LOGS_PATH}/attendance.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )


def start_server():
    """Khởi động FastAPI server"""
    from config.settings import settings
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("  FACE ATTENDANCE SYSTEM v1.0")
    logger.info("=" * 60)
    logger.info(f"  Server: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"  Camera: {settings.CAMERA_SOURCE}")
    logger.info(f"  FAS Layers: Texture + Liveness + Frequency")
    logger.info(f"  Recognition Threshold: {settings.RECOGNITION_THRESHOLD}")
    logger.info("=" * 60)
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=False
    )


async def register_employee_cli():
    """CLI để đăng ký nhân viên"""
    from config.settings import settings
    from core.face_engine import face_engine
    from core.database import init_db, AsyncSessionLocal, Employee
    from sqlalchemy import select
    
    await init_db()
    await face_engine.initialize()
    
    print("\n" + "="*50)
    print("  ĐĂNG KÝ NHÂN VIÊN MỚI")
    print("="*50)
    
    emp_id = input("Mã nhân viên: ").strip()
    name = input("Họ tên: ").strip()
    dept = input("Phòng ban: ").strip()
    
    # Lưu vào DB
    async with AsyncSessionLocal() as db:
        existing = await db.execute(select(Employee).where(Employee.employee_id == emp_id))
        if not existing.scalars().first():
            emp = Employee(employee_id=emp_id, full_name=name, department=dept)
            db.add(emp)
            await db.commit()
            print(f"✅ Đã tạo nhân viên {name} ({emp_id})")
    
    print(f"\nBắt đầu chụp ảnh cho {name}...")
    print("Hướng dẫn:")
    print("  - Nhìn thẳng vào camera")
    print("  - Xoay đầu sang trái, phải, lên, xuống nhẹ")
    print("  - Nhấn SPACE để chụp, Q để kết thúc")
    
    cap = cv2.VideoCapture(
        int(settings.CAMERA_SOURCE) if settings.CAMERA_SOURCE.isdigit() else settings.CAMERA_SOURCE,
        cv2.CAP_DSHOW if settings.CAMERA_SOURCE.isdigit() else 0
    )
    if not cap.isOpened():
        print("❌ Không mở được camera!")
        print("   → Nếu server đang chạy, hãy đăng ký qua web: http://localhost:8000")
        print("   → Hoặc dừng server rồi chạy lại: python run.py register")
        return
    images = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display = frame.copy()
        faces = face_engine.detect_faces(frame)
        
        for face in faces:
            bbox = face.bbox
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        cv2.putText(display, f"Anh da chup: {len(images)}/20", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=chup | Q=ket thuc", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.imshow(f"Dang ky: {name}", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if faces:
                images.append(frame)
                print(f"  📷 Chụp ảnh {len(images)}/20")
            else:
                print("  ⚠️  Không phát hiện khuôn mặt!")
        elif key == ord('q') or len(images) >= 20:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if images:
        print(f"\nĐang xử lý {len(images)} ảnh...")
        result = await face_engine.register_employee(emp_id, name, images, replace=True)
        
        async with AsyncSessionLocal() as db:
            emp_rec = await db.execute(select(Employee).where(Employee.employee_id == emp_id))
            emp_obj = emp_rec.scalars().first()
            if emp_obj:
                emp_obj.face_samples = result.get("count", 0)
                await db.commit()
        
        if result["success"]:
            print(f"✅ Đăng ký thành công! {result['count']} mẫu khuôn mặt.")
        else:
            print(f"❌ Thất bại: {result['message']}")


def test_camera():
    """Test kết nối camera"""
    from config.settings import settings
    
    source = settings.CAMERA_SOURCE
    if source.isdigit():
        source = int(source)
    
    print(f"Testing camera: {source}")
    backend = cv2.CAP_DSHOW if isinstance(source, int) else 0
    cap = cv2.VideoCapture(source, backend)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        return
    
    print("✅ Camera connected!")
    print("Nhấn Q để thoát")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không đọc được frame!")
            break
            
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Resolution: {w}x{h}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Camera Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Attendance System")
    parser.add_argument("command", nargs="?", default="start",
                       choices=["start", "register", "test-camera", "report"],
                       help="Command to run")
    
    args = parser.parse_args()
    
    # Ensure data dirs exist
    from config.settings import settings as settings_module
    for d in [settings_module.EMBEDDINGS_PATH, settings_module.LOGS_PATH, 
              settings_module.IMAGES_PATH, settings_module.MODELS_PATH]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    if args.command == "start":
        start_server()
    elif args.command == "register":
        asyncio.run(register_employee_cli())
    elif args.command == "test-camera":
        test_camera()
    elif args.command == "report":
        print("Xem báo cáo tại: http://localhost:8000/api/attendance/today")
