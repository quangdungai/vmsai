"""
Face Detection & Recognition Engine
Sử dụng InsightFace (RetinaFace + ArcFace)
- Detection: RetinaFace (SOTA accuracy)
- Recognition: ArcFace buffalo_l (512-dim embedding, >99% LFW accuracy)
"""
import os
import numpy as np
import pickle
import cv2
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from loguru import logger
from dataclasses import dataclass

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not installed. Using mock mode.")

from config.settings import settings


@dataclass
class FaceDetection:
    """Kết quả detect một khuôn mặt"""
    bbox: np.ndarray        # [x1, y1, x2, y2]
    score: float            # Detection confidence
    embedding: np.ndarray   # 512-dim ArcFace embedding
    landmarks: np.ndarray   # 5 facial landmarks
    age: int = 0
    gender: str = "unknown"


@dataclass
class RecognitionResult:
    """Kết quả nhận diện"""
    employee_id: Optional[str]
    employee_name: Optional[str]
    similarity: float       # Cosine similarity (0-1, cao hơn = giống hơn)
    matched: bool
    face: FaceDetection
    second_best_similarity: float = 0.0
    margin: float = 0.0
    reject_reason: Optional[str] = None


class FaceRecognitionEngine:
    """
    Engine nhận diện khuôn mặt chính
    
    Pipeline:
    1. Detect khuôn mặt bằng RetinaFace
    2. Extract 512-dim ArcFace embedding  
    3. So sánh với database bằng Cosine Similarity
    4. Trả về kết quả với score
    """
    
    def __init__(self):
        self.app = None
        self.embeddings_db: Dict[str, dict] = {}  # {employee_id: {name, embeddings: []}}
        self._initialized = False
        
    async def initialize(self):
        """Khởi tạo model (gọi một lần khi startup)"""
        if self._initialized:
            return
            
        logger.info("🚀 Khởi tạo Face Recognition Engine...")
        
        if INSIGHTFACE_AVAILABLE:
            try:
                self.app = FaceAnalysis(
                    name=settings.DETECTION_MODEL,
                    root=settings.MODELS_PATH,
                    providers=['CPUExecutionProvider']
                )
                if settings.DETECTION_BACKEND == "cuda":
                    try:
                        self.app = FaceAnalysis(
                            name=settings.DETECTION_MODEL,
                            root=settings.MODELS_PATH,
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                        )
                    except Exception:
                        pass
                self.app.prepare(
                    ctx_id=0 if settings.DETECTION_BACKEND == "cuda" else -1,
                    det_size=(settings.DETECTION_SIZE, settings.DETECTION_SIZE),
                    det_thresh=settings.DETECTION_THRESHOLD
                )
                logger.success("✅ InsightFace initialized (RetinaFace + ArcFace)")
            except Exception as e:
                logger.error(f"InsightFace init failed: {e}")
                self.app = None
        
        # Load embedding database
        await self.load_embeddings()
        self._initialized = True
        logger.success(f"✅ Face Engine ready. {len(self.embeddings_db)} employees loaded.")

    async def load_embeddings(self):
        """Load tất cả embeddings từ disk vào RAM"""
        embeddings_path = Path(settings.EMBEDDINGS_PATH)
        embeddings_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_db = {}
        db_file = embeddings_path / "face_db.pkl"
        
        if db_file.exists():
            with open(db_file, "rb") as f:
                self.embeddings_db = pickle.load(f)
            logger.info(f"📊 Loaded {len(self.embeddings_db)} employee embeddings")
        else:
            logger.warning("⚠️ No embedding database found. Please register employees first.")

    async def save_embeddings(self):
        """Lưu embedding database"""
        embeddings_path = Path(settings.EMBEDDINGS_PATH)
        db_file = embeddings_path / "face_db.pkl"
        with open(db_file, "wb") as f:
            pickle.dump(self.embeddings_db, f)
        logger.info(f"💾 Saved {len(self.embeddings_db)} embeddings to disk")

    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect và extract embeddings từ frame
        
        Returns: List[FaceDetection] - mỗi khuôn mặt phát hiện được
        """
        if frame is None or frame.size == 0:
            return []

        # Resize xuống PROCESS_WIDTH để AI nhanh hơn
        h, w = frame.shape[:2]
        scale = 1.0
        if w > settings.PROCESS_WIDTH:
            scale = settings.PROCESS_WIDTH / w
            proc_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            proc_frame = frame

        faces = []
        
        if self.app is not None:
            try:
                detected = self.app.get(proc_frame)
                for face in detected:
                    bbox = face.bbox.astype(int)
                    
                    face_w = bbox[2] - bbox[0]
                    face_h = bbox[3] - bbox[1]
                    min_sz = int(settings.MIN_FACE_SIZE * scale) if scale != 1.0 else settings.MIN_FACE_SIZE
                    max_sz = int(settings.MAX_FACE_SIZE * scale) if scale != 1.0 else settings.MAX_FACE_SIZE
                    if face_w < min_sz or face_h < min_sz:
                        continue
                    if face_w > max_sz or face_h > max_sz:
                        continue

                    if scale != 1.0:
                        bbox = (bbox / scale).astype(int)
                        landmarks = (face.kps / scale).astype(np.float32)
                    else:
                        landmarks = face.kps

                    fd = FaceDetection(
                        bbox=bbox,
                        score=float(face.det_score),
                        embedding=face.normed_embedding,
                        landmarks=landmarks,
                        age=int(face.age) if hasattr(face, 'age') else 0,
                        gender="M" if hasattr(face, 'gender') and face.gender == 1 else "F"
                    )
                    faces.append(fd)
            except Exception as e:
                logger.error(f"Face detection error: {e}")
        else:
            # Mock mode khi không có InsightFace
            faces = self._mock_detect(frame)
            
        return faces

    def recognize(self, face: FaceDetection) -> RecognitionResult:
        """
        Nhận diện khuôn mặt — Cosine similarity + margin check (giảm FAR)
        """
        if len(self.embeddings_db) == 0:
            return RecognitionResult(
                employee_id=None, employee_name=None,
                similarity=0.0, matched=False, face=face,
                reject_reason="No employees registered"
            )

        query_embedding = face.embedding
        scores: List[Tuple[str, str, float]] = []

        for emp_id, emp_data in self.embeddings_db.items():
            if not emp_data.get("is_active", True):
                continue
            embeddings_list = emp_data.get("embeddings", [])
            if not embeddings_list:
                continue

            max_sim = max(
                float(np.dot(query_embedding, db_emb))
                for db_emb in embeddings_list
            )
            scores.append((emp_id, emp_data.get("name", "Unknown"), max_sim))

        if not scores:
            return RecognitionResult(
                employee_id=None, employee_name=None,
                similarity=0.0, matched=False, face=face,
                reject_reason="No active embeddings"
            )

        scores.sort(key=lambda x: x[2], reverse=True)
        best_id, best_name, best_sim = scores[0]
        second_sim = scores[1][2] if len(scores) > 1 else 0.0
        margin = best_sim - second_sim

        threshold = settings.RECOGNITION_THRESHOLD
        if settings.STRICT_MODE:
            threshold = min(threshold, settings.RECOGNITION_THRESHOLD_STRICT)

        sim_threshold = 1.0 - threshold
        matched = best_sim >= sim_threshold
        reject_reason = None

        if matched and settings.STRICT_MODE and len(scores) > 1:
            if margin < settings.RECOGNITION_MARGIN:
                matched = False
                reject_reason = f"Ambiguous match (margin={margin:.3f} < {settings.RECOGNITION_MARGIN})"

        if matched and face.score < settings.RECOGNITION_MIN_QUALITY:
            matched = False
            reject_reason = f"Low face quality ({face.score:.2f})"

        if not matched and reject_reason is None and best_sim < sim_threshold:
            reject_reason = f"Below threshold ({best_sim:.3f} < {sim_threshold:.3f})"

        return RecognitionResult(
            employee_id=best_id if matched else None,
            employee_name=best_name if matched else None,
            similarity=best_sim,
            matched=matched,
            face=face,
            second_best_similarity=second_sim,
            margin=margin,
            reject_reason=reject_reason
        )

    async def register_employee(
        self,
        employee_id: str,
        name: str,
        images: List[np.ndarray],
        replace: bool = False
    ) -> dict:
        """
        Đăng ký nhân viên mới với nhiều ảnh (tối thiểu 5, khuyến nghị 10-20)
        
        Args:
            employee_id: Mã nhân viên
            name: Họ tên
            images: List ảnh (BGR numpy arrays)
            replace: Ghi đè nếu đã tồn tại
            
        Returns: {success, count, message}
        """
        if employee_id in self.embeddings_db and not replace:
            return {"success": False, "message": f"Employee {employee_id} already exists. Use replace=True to overwrite."}

        valid_embeddings = []
        failed = 0

        for i, img in enumerate(images):
            faces = self.detect_faces(img)
            
            if len(faces) == 0:
                logger.warning(f"No face detected in image {i+1}")
                failed += 1
                continue
            if len(faces) > 1:
                logger.warning(f"Multiple faces in image {i+1}, skipping")
                failed += 1
                continue

            # Quality check: face phải đủ lớn và rõ
            face = faces[0]
            if face.score < settings.RECOGNITION_MIN_QUALITY:
                logger.warning(f"Low quality face in image {i+1}: score={face.score:.3f}")
                failed += 1
                continue

            valid_embeddings.append(face.embedding)

        if len(valid_embeddings) < 3:
            return {
                "success": False,
                "message": f"Not enough valid faces. Got {len(valid_embeddings)}, need at least 3. {failed} images failed."
            }

        # Deduplication: loại bỏ embeddings quá giống nhau
        unique_embeddings = self._deduplicate_embeddings(valid_embeddings)

        self.embeddings_db[employee_id] = {
            "name": name,
            "employee_id": employee_id,
            "embeddings": unique_embeddings,
            "is_active": True,
            "registered_count": len(unique_embeddings)
        }

        await self.save_embeddings()
        
        logger.success(f"✅ Registered {name} ({employee_id}) with {len(unique_embeddings)} unique embeddings")
        return {
            "success": True,
            "count": len(unique_embeddings),
            "failed": failed,
            "message": f"Successfully registered {name} with {len(unique_embeddings)} face samples"
        }

    def _deduplicate_embeddings(self, embeddings: List[np.ndarray], threshold: float = 0.98) -> List[np.ndarray]:
        """Loại bỏ embeddings quá giống nhau để tăng diversity"""
        if len(embeddings) <= 1:
            return embeddings
            
        unique = [embeddings[0]]
        for emb in embeddings[1:]:
            sims = [np.dot(emb, u) for u in unique]
            if max(sims) < threshold:  # Chỉ thêm nếu đủ khác biệt
                unique.append(emb)
        return unique

    def _mock_detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Mock detection khi không có InsightFace"""
        # Dùng OpenCV Haar Cascade làm fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detected = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
        
        faces = []
        for (x, y, w, h) in detected:
            # Generate random embedding cho mock mode
            emb = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            faces.append(FaceDetection(
                bbox=np.array([x, y, x+w, y+h]),
                score=0.9,
                embedding=emb,
                landmarks=np.zeros((5, 2))
            ))
        return faces

    def draw_result(
        self,
        frame: np.ndarray,
        result: RecognitionResult,
        fas_passed: bool,
        fas_score: float
    ) -> np.ndarray:
        """Vẽ kết quả lên frame"""
        frame = frame.copy()
        face = result.face
        bbox = face.bbox
        
        # Màu sắc
        if not fas_passed:
            color = (0, 0, 255)    # Đỏ = Fake/Spoof
            label = "⚠ SPOOF DETECTED"
        elif result.matched:
            color = (0, 255, 0)    # Xanh = OK
            label = f"✓ {result.employee_name}"
        else:
            color = (0, 165, 255)  # Cam = Unknown
            label = "UNKNOWN"

        # Vẽ bbox
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Background cho text
        text_y = bbox[1] - 10 if bbox[1] > 30 else bbox[3] + 25
        cv2.rectangle(frame, (bbox[0], text_y - 20), (bbox[2], text_y + 5), color, -1)
        cv2.putText(frame, label, (bbox[0] + 5, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Scores
        score_text = f"Sim:{result.similarity:.2f} FAS:{fas_score:.2f}"
        cv2.putText(frame, score_text, (bbox[0], bbox[3] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Landmarks
        if face.landmarks is not None and len(face.landmarks) == 5:
            for pt in face.landmarks:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

        return frame


# Singleton
face_engine = FaceRecognitionEngine()
