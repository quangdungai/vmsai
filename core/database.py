import os
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Float,
    DateTime,
    Text,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base

from config.settings import settings

Base = declarative_base()

engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(64), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    department = Column(String(128), nullable=True)
    position = Column(String(128), nullable=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(64), nullable=True)
    face_samples = Column(Integer, default=0)
    embedding_path = Column(String(1024), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(64), nullable=False, index=True)
    employee_name = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    date = Column(String(20), nullable=False)
    time = Column(String(20), nullable=False)
    type = Column(String(32), nullable=False)
    status = Column(String(64), nullable=False)
    recognition_score = Column(Float, nullable=True)
    liveness_score = Column(Float, nullable=True)
    fas_passed = Column(Boolean, default=False)
    camera_id = Column(String(64), nullable=True)
    image_path = Column(String(1024), nullable=True)
    is_late = Column(Boolean, default=False)
    is_early_leave = Column(Boolean, default=False)


class FaceAuditLog(Base):
    __tablename__ = "face_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    face_detected = Column(Boolean, default=False)
    face_count = Column(Integer, default=0)
    detection_score = Column(Float, nullable=True)
    matched_employee_id = Column(String(64), nullable=True)
    recognition_score = Column(Float, nullable=True)
    recognition_passed = Column(Boolean, default=False)
    fas_texture_score = Column(Float, nullable=True)
    fas_liveness_score = Column(Float, nullable=True)
    fas_freq_score = Column(Float, nullable=True)
    fas_overall_score = Column(Float, nullable=True)
    fas_passed = Column(Boolean, default=False)
    fas_reject_reason = Column(Text, nullable=True)
    attendance_recorded = Column(Boolean, default=False)
    camera_id = Column(String(64), nullable=True)


async def init_db() -> None:
    data_dir = Path(settings.BASE_DIR) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    Path(settings.EMBEDDINGS_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.LOGS_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.IMAGES_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.MODELS_PATH).mkdir(parents=True, exist_ok=True)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — async generator."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager cho processing loop."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
