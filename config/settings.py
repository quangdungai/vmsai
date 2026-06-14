"""
Cấu hình hệ thống chấm công nhận diện khuôn mặt
"""
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # pydantic v1 fallback

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    BASE_DIR: Path = BASE_DIR
    # ============================================================
    # SERVER
    # ============================================================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # ============================================================
    # CAMERA
    # ============================================================
    CAMERA_SOURCE: str = "0"
    CAMERA_FPS: int = 25
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    FRAME_SKIP: int = 3
    PROCESS_WIDTH: int = 640          # Resize trước khi chạy AI
    STREAM_WIDTH: int = 640
    STREAM_FPS: int = 12
    STREAM_JPEG_QUALITY: int = 60
    RTSP_RECONNECT_DELAY: float = 3.0
    RTSP_BUFFER_SIZE: int = 1

    # ============================================================
    # FACE DETECTION (InsightFace RetinaFace)
    # ============================================================
    DETECTION_MODEL: str = "buffalo_l"
    DETECTION_THRESHOLD: float = 0.7
    MIN_FACE_SIZE: int = 80
    MAX_FACE_SIZE: int = 600
    DETECTION_BACKEND: str = "cpu"
    DETECTION_SIZE: int = 320         # 320 nhanh hơn 640 trên CPU (~3-4x)

    # ============================================================
    # FACE RECOGNITION (ArcFace)
    # ============================================================
    RECOGNITION_THRESHOLD: float = 0.45
    RECOGNITION_THRESHOLD_STRICT: float = 0.38
    RECOGNITION_MARGIN: float = 0.08       # Khoảng cách tối thiểu best vs 2nd-best (giảm FAR)
    RECOGNITION_MIN_QUALITY: float = 0.85    # Ngưỡng chất lượng detection khi đăng ký
    EMBEDDING_DIM: int = 512
    TOP_K_MATCH: int = 1
    STRICT_MODE: bool = True               # Bật margin check + ngưỡng chặt

    # ============================================================
    # FACE ANTI-SPOOFING (FAS) — ĐA TẦNG
    # ============================================================
    FAS_TEXTURE_ENABLED: bool = True
    FAS_TEXTURE_THRESHOLD: float = 0.75
    FAS_LIVENESS_ENABLED: bool = True
    FAS_LIVENESS_FRAMES: int = 15
    FAS_BLINK_REQUIRED: bool = True
    FAS_HEAD_MOTION_REQUIRED: bool = True
    FAS_FREQ_ENABLED: bool = True
    FAS_FREQ_THRESHOLD: float = 0.7
    FAS_IR_ENABLED: bool = False
    FAS_FINAL_THRESHOLD: float = 0.80
    FAS_MIN_LAYERS_PASS: int = 4
    FAS_DL_ENABLED: bool = True             # ONNX MiniFASNet (nếu có model)
    FAS_TEMPORAL_ENABLED: bool = True       # Phát hiện ảnh/video tĩnh
    FAS_TEMPORAL_FRAMES: int = 20

    # ============================================================
    # XÁC MINH ĐA KHUNG HÌNH (chống spoof nhanh)
    # ============================================================
    ATTENDANCE_CONFIRM_FRAMES: int = 5      # Số frame liên tiếp phải pass
    ATTENDANCE_CONFIRM_TIMEOUT_SEC: float = 3.0

    # ============================================================
    # CHẤM CÔNG
    # ============================================================
    ATTENDANCE_COOLDOWN_MINUTES: int = 30
    ATTENDANCE_START_HOUR: int = 6
    ATTENDANCE_END_HOUR: int = 22
    CHECKIN_LATE_THRESHOLD: str = "08:30"
    CHECKOUT_EARLY_THRESHOLD: str = "17:30"

    # ============================================================
    # DATABASE
    # ============================================================
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/data/attendance.db"

    # ============================================================
    # PATHS
    # ============================================================
    EMBEDDINGS_PATH: str = str(BASE_DIR / "data" / "embeddings")
    LOGS_PATH: str = str(BASE_DIR / "data" / "logs")
    IMAGES_PATH: str = str(BASE_DIR / "data" / "images")
    MODELS_PATH: str = str(BASE_DIR / "models")

    # ============================================================
    # SECURITY
    # ============================================================
    SECRET_KEY: str = "change-this-in-production-secret-key"
    API_KEY: str = "your-api-key-here"

    # ============================================================
    # LOGGING
    # ============================================================
    LOG_LEVEL: str = "INFO"
    LOG_ATTENDANCE_IMAGES: bool = True
    AUDIT_LOG_INTERVAL: int = 30      # Ghi audit mỗi N frame (giảm lag DB)

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
