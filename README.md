# 👤 Face Attendance System — Nhận Diện Khuôn Mặt & Chấm Công

Hệ thống chấm công tự động dựa trên nhận diện khuôn mặt với anti-spoofing 4 tầng, hỗ trợ camera USB và IP camera (RTSP).

## 🎯 Tính Năng

- **🔍 Nhận Diện Khuôn Mặt**
  - RetinaFace (SOTA detection)
  - ArcFace (512-dim embedding, >99% LFW accuracy)
  - Cosine similarity matching

- **🛡️ Anti-Spoofing 4 Tầng**
  - Texture Analysis (phát hiện ảnh in/màn hình)
  - Liveness Detection (blink, head motion)
  - Frequency Domain Analysis (JPEG artifacts, 60Hz screen detection)
  - Weighted voting system

- **📱 Real-time Dashboard**
  - Live camera feed (MJPEG stream)
  - WebSocket attendance events
  - Stats & history
  - Dark mode UI

- **⚙️ Chấm Công Thông Minh**
  - Auto check-in/check-out detection
  - Late arrival tracking
  - Early leave detection
  - Cooldown management (30 phút default)

- **🗂️ Quản Lý Nhân Viên**
  - CRUD API cho nhân viên
  - Face registration (10-20 ảnh, multi-angle)
  - Embedding database (pickle format)

- **📊 Audit Log**
  - Mọi lần nhận diện được ghi lại
  - Detection/recognition/FAS scores
  - Image snapshots (tuỳ chọn)

## 📋 Requirements

- Python 3.10+
- OpenCV 4.7+
- SQLite (or PostgreSQL)
- GPU optional (CUDA/TensorRT support)

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url> face-attendance
cd face-attendance

# Tạo virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Cài dependencies
pip install -r requirements.txt
```

### 2. Cấu Hình

```bash
# Copy environment template
cp .env.example .env

# Sửa .env để cấu hình:
# - CAMERA_SOURCE=0 (USB camera)
# - CAMERA_SOURCE=rtsp://admin:pass@192.168.1.100:554/stream1 (IP camera)
# - RECOGNITION_THRESHOLD=0.45 (thấp hơn = chặt hơn)
# - FAS_FINAL_THRESHOLD=0.80 (anti-spoofing ngưỡng)
```

### 3. Khởi Động Server

```bash
python run.py start
```

Server chạy tại `http://localhost:8000`

- **Dashboard**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8000/ws/attendance

### 4. Đăng Ký Nhân Viên

```bash
python run.py register
```

Hướng dẫn:
1. Nhập mã nhân viên & họ tên
2. Nhìn thẳng vào camera
3. Xoay đầu nhẹ (trái, phải, lên, xuống)
4. Nhấn SPACE để chụp (20 ảnh khuyến nghị)
5. Nhấn Q để kết thúc

### 5. Test Camera (tuỳ chọn)

```bash
python run.py test-camera
```

## 🐳 Docker Deployment

```bash
# Build & run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Environment variables trong `docker-compose.yml`:
- USB camera: `CAMERA_SOURCE=0`
- IP camera: `CAMERA_SOURCE=rtsp://...`
- GPU support: Uncomment `runtime: nvidia` section

## 📁 Project Structure

```
face-attendance/
├── core/
│   ├── face_engine.py      # RetinaFace + ArcFace
│   ├── anti_spoofing.py    # FAS 4-layer engine
│   ├── attendance.py       # Chấm công logic
│   ├── camera.py           # Camera manager
│   ├── database.py         # SQLAlchemy models
│   └── __init__.py
├── config/
│   ├── settings.py         # Configuration (Pydantic)
│   └── __init__.py
├── web/
│   ├── templates/
│   │   └── dashboard.html  # Real-time UI
│   └── static/             # CSS, JS assets
├── main.py                 # FastAPI app
├── run.py                  # Entry point + CLI
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker config
├── Dockerfile
├── .env.example            # Environment template
└── README.md
```

## 🔌 API Endpoints

### Employees

- `POST /api/employees` - Tạo nhân viên
- `GET /api/employees` - Danh sách nhân viên
- `POST /api/employees/{id}/register-face` - Đăng ký khuôn mặt (upload ảnh)
- `POST /api/employees/{id}/register-face-camera` - Đăng ký từ camera

### Attendance

- `GET /api/attendance` - Danh sách chấm công (với filter)
- `GET /api/attendance/today` - Báo cáo hôm nay

### Streaming

- `GET /stream` - MJPEG video stream
- `WS /ws/attendance` - Real-time events

### System

- `GET /api/system/status` - System status & config

## ⚙️ Cấu Hình Nâng Cao

### Recognition Threshold

Thấp hơn = chặt hơn (ít false positive, nhiều false negative)
- `0.45` (default) - balanced
- `0.38` - strict mode
- `0.50` - relaxed

### Anti-Spoofing Layers

Có thể tắt layer bất kỳ trong `.env`:
- `FAS_TEXTURE_ENABLED=True` - Phát hiện ảnh in/màn hình
- `FAS_LIVENESS_ENABLED=True` - Yêu cầu blink & head motion
- `FAS_FREQ_ENABLED=True` - Phát hiện artifacts JPEG

### Chấm Công Rules

```env
ATTENDANCE_COOLDOWN_MINUTES=30      # Không chấm lại trong 30 phút
CHECKIN_LATE_THRESHOLD=08:30        # Đi muộn sau giờ này
CHECKOUT_EARLY_THRESHOLD=17:30      # Về sớm trước giờ này
```

### GPU Acceleration

Để dùng GPU:
1. Cài CUDA & cuDNN
2. Trong `.env`: `DETECTION_BACKEND=cuda`
3. Docker: Uncomment `runtime: nvidia`

## 📊 Database Schema

### employees
- `employee_id` (PK)
- `full_name`, `department`, `position`
- `face_samples`, `embedding_path`
- `is_active`, `created_at`

### attendance_logs
- `id` (PK)
- `employee_id`, `employee_name`
- `date`, `time`, `type` (checkin/checkout)
- `recognition_score`, `liveness_score`, `fas_passed`
- `status` (valid/late/early_leave)
- `image_path` (snapshot)

### face_audit_logs
- Mọi lần phát hiện khuôn mặt
- Detection, recognition, FAS scores
- Reject reasons nếu fail

## 🐛 Troubleshooting

### Camera not detected
```bash
python run.py test-camera
# Kiểm tra CAMERA_SOURCE trong .env
```

### No face detected
- Đảm bảo lighting tốt
- Khuôn mặt phải rõ ràng (>80x80 pixel)
- Không có occlusion (mask, kính)

### Too many false positives
- Tăng `RECOGNITION_THRESHOLD` (0.45 → 0.50)
- Tăng `FAS_FINAL_THRESHOLD` (0.80 → 0.85)

### Memory issues
- Giảm `CAMERA_WIDTH/HEIGHT` (1280 → 640)
- Tăng `FRAME_SKIP` (2 → 4)

### InsightFace not available
- `pip install insightface`
- Fallback dùng OpenCV Haar Cascade (accuracy thấp hơn)

## 📝 License

MIT License - See LICENSE file

## 👨‍💻 Support

Có vấn đề? Check `/data/logs/attendance.log` để xem chi tiết error.

---

**Built with ❤️ using FastAPI, InsightFace, SQLAlchemy**
