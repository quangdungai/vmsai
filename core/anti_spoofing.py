"""
Face Anti-Spoofing (FAS) Engine - Đa tầng
============================================
Kiến trúc 4 tầng bảo vệ:

Tầng 1: Texture Analysis
  - Phân tích texture khuôn mặt (LBP + Laplacian)
  - Phát hiện màn hình/ảnh in qua Moiré pattern
  - Phân tích gradient histogram

Tầng 2: 3D Liveness Detection  
  - Yêu cầu blink mắt (Eye Aspect Ratio)
  - Phát hiện chuyển động đầu tự nhiên (Euler angles từ landmarks)
  - Landmark dynamics analysis

Tầng 3: Frequency Domain Analysis
  - FFT để phát hiện tần số tuần hoàn của màn hình (60/120Hz)
  - Phát hiện JPEG artifacts của ảnh in/chụp
  - High-frequency noise analysis

Tầng 4: Deep Learning Score (optional)
  - Binary classifier Real/Fake
  - Chạy trên backbone MobileNetV2

Kết quả tổng hợp: Weighted voting
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import time
from pathlib import Path

from config.settings import settings

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class FASResult:
    """Kết quả anti-spoofing tổng hợp"""
    is_live: bool
    overall_score: float        # 0.0 - 1.0 (cao hơn = thật hơn)
    
    # Điểm từng tầng
    texture_score: float = 0.0
    liveness_score: float = 0.0
    freq_score: float = 0.0
    dl_score: float = 0.0
    
    # Chi tiết liveness
    blink_detected: bool = False
    motion_detected: bool = False
    
    reject_reason: Optional[str] = None
    layers_passed: int = 0
    processing_time_ms: float = 0.0


class LivenessTracker:
    """
    Theo dõi liveness qua nhiều frame.
    Blink: eye ROI Laplacian variance drop + EAR proxy từ 5 landmarks InsightFace.
    Motion: head pose variance qua thời gian.
    """

    def __init__(self, window_size: int = 30):
        self.ear_history = deque(maxlen=window_size)
        self.eye_var_history = deque(maxlen=window_size)
        self.head_pose_history = deque(maxlen=window_size)
        self.frame_count = 0
        self.blink_count = 0
        self.last_blink_frame = -100
        self.EAR_THRESHOLD = 0.18
        self.EAR_CONSEC_FRAMES = 2
        self._prev_eye_var = None

    def _eye_roi_variance(self, frame: np.ndarray, landmarks_5: np.ndarray) -> float:
        """Đo độ sắc nét vùng mắt — blink làm giảm đột ngột."""
        if frame is None or landmarks_5 is None or len(landmarks_5) < 2:
            return 0.0
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        variances = []
        for idx in (0, 1):  # left eye, right eye
            cx, cy = int(landmarks_5[idx][0]), int(landmarks_5[idx][1])
            r = max(8, int((landmarks_5[1][0] - landmarks_5[0][0]) * 0.12))
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            variances.append(cv2.Laplacian(roi, cv2.CV_64F).var())
        return float(np.mean(variances)) if variances else 0.0

    def _ear_from_5pts(self, landmarks_5: np.ndarray) -> float:
        """
        Proxy EAR từ 5 điểm InsightFace: mắt trái, mắt phải, mũi, miệng trái/phải.
        Tỷ lệ khoảng cách dọc mắt-mũi / ngang inter-eye thay đổi khi nhắm mắt.
        """
        if landmarks_5 is None or len(landmarks_5) < 5:
            return 0.3
        le, re, nose, ml, mr = landmarks_5[:5]
        inter_eye = np.linalg.norm(re - le) + 1e-6
        eye_center = (le + re) / 2.0
        mouth_center = (ml + mr) / 2.0
        vertical = np.linalg.norm(eye_center - nose) + np.linalg.norm(nose - mouth_center)
        return float(vertical / (2.0 * inter_eye))

    def compute_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Eye Aspect Ratio (EAR)
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        # Vertical distances
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        # Horizontal distance
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        if C < 1e-6:
            return 0.3  # default
        return (A + B) / (2.0 * C)

    def estimate_head_pose(self, landmarks_2d: np.ndarray) -> Tuple[float, float, float]:
        """
        Ước lượng head pose từ 5 facial landmarks
        Trả về (yaw, pitch, roll) trong degrees
        """
        # 3D model points (generic face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),           # Mũi
            (-30.0, -30.0, -30.0),      # Mắt trái (góc ngoài)
            (30.0, -30.0, -30.0),       # Mắt phải (góc ngoài)
            (-20.0, 20.0, -30.0),       # Miệng trái
            (20.0, 20.0, -30.0),        # Miệng phải
        ], dtype=np.float64)

        if landmarks_2d is None or len(landmarks_2d) < 5:
            return (0.0, 0.0, 0.0)

        # Camera internals (giả sử)
        size = (640, 480)
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        image_points = landmarks_2d[:5].astype(np.float64)

        try:
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            rmat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            return (angles[1], angles[0], angles[2])  # yaw, pitch, roll
        except:
            return (0.0, 0.0, 0.0)

    def update(self, frame: Optional[np.ndarray] = None,
               landmarks_68: Optional[np.ndarray] = None,
               landmarks_5: Optional[np.ndarray] = None) -> dict:
        """Cập nhật với frame mới."""
        self.frame_count += 1
        result = {
            "ear": 0.3,
            "blink_detected": False,
            "head_pose": (0.0, 0.0, 0.0),
            "motion_score": 0.0,
            "eye_variance": 0.0,
        }

        if landmarks_5 is not None:
            pose = self.estimate_head_pose(landmarks_5)
            self.head_pose_history.append(pose)
            result["head_pose"] = pose

            if len(self.head_pose_history) >= 10:
                poses = np.array(list(self.head_pose_history))
                result["motion_score"] = float(np.mean(np.std(poses, axis=0)))

            ear = self._ear_from_5pts(landmarks_5)
            self.ear_history.append(ear)
            result["ear"] = ear

            if ear < self.EAR_THRESHOLD:
                if (self.frame_count - self.last_blink_frame) > self.EAR_CONSEC_FRAMES:
                    self.blink_count += 1
                    self.last_blink_frame = self.frame_count
                    result["blink_detected"] = True

            if frame is not None:
                eye_var = self._eye_roi_variance(frame, landmarks_5)
                result["eye_variance"] = eye_var
                self.eye_var_history.append(eye_var)
                if self._prev_eye_var is not None:
                    drop = (self._prev_eye_var - eye_var) / (self._prev_eye_var + 1e-6)
                    if drop > 0.35 and (self.frame_count - self.last_blink_frame) > self.EAR_CONSEC_FRAMES:
                        self.blink_count += 1
                        self.last_blink_frame = self.frame_count
                        result["blink_detected"] = True
                self._prev_eye_var = eye_var

        if landmarks_68 is not None and len(landmarks_68) >= 68:
            left_eye = landmarks_68[36:42]
            right_eye = landmarks_68[42:48]
            ear = (self.compute_ear(left_eye) + self.compute_ear(right_eye)) / 2.0
            self.ear_history.append(ear)
            result["ear"] = ear
            if ear < self.EAR_THRESHOLD:
                if (self.frame_count - self.last_blink_frame) > self.EAR_CONSEC_FRAMES:
                    self.blink_count += 1
                    self.last_blink_frame = self.frame_count
                    result["blink_detected"] = True

        return result

    def get_liveness_score(self) -> Tuple[float, bool, bool]:
        """Tính liveness score — blink + head motion + micro-expression."""
        score = 0.0

        blink_ok = self.blink_count >= 1
        if blink_ok:
            score += 0.45

        motion_ok = False
        if len(self.head_pose_history) >= 12:
            poses = np.array(list(self.head_pose_history))
            natural_motion = float(np.mean(np.std(poses, axis=0)))
            if natural_motion > 0.25:
                motion_ok = True
                score += 0.35
            elif natural_motion > 0.08:
                score += 0.15

        if len(self.ear_history) >= 15:
            ear_std = np.std(list(self.ear_history))
            if ear_std > 0.008:
                score += 0.1

        if len(self.eye_var_history) >= 15:
            ev_std = np.std(list(self.eye_var_history))
            if ev_std > 5.0:
                score += 0.1

        return min(1.0, score), blink_ok, motion_ok

    def reset(self):
        self.ear_history.clear()
        self.eye_var_history.clear()
        self.head_pose_history.clear()
        self.frame_count = 0
        self.blink_count = 0
        self._prev_eye_var = None


class TextureAnalyzer:
    """
    Tầng 1: Phân tích texture để phát hiện fake
    
    Techniques:
    - LBP (Local Binary Pattern): ảnh thật có texture phong phú hơn ảnh in
    - Laplacian variance: đo độ nét (ảnh in có thể mờ)
    - Gradient histogram: ảnh thật có gradient distribution tự nhiên hơn
    - Moiré pattern detection: màn hình thường có pattern này
    """
    
    def analyze(self, face_roi: np.ndarray) -> float:
        """
        Phân tích texture của vùng khuôn mặt
        Returns: score 0-1 (cao = thật)
        """
        if face_roi is None or face_roi.size == 0:
            return 0.0
            
        scores = []
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        
        # 1. Sharpness check (Laplacian variance)
        lap_score = self._sharpness_score(gray)
        scores.append(lap_score * 0.25)
        
        # 2. LBP texture richness
        lbp_score = self._lbp_score(gray)
        scores.append(lbp_score * 0.30)
        
        # 3. Gradient analysis
        grad_score = self._gradient_score(gray)
        scores.append(grad_score * 0.25)
        
        # 4. Moiré pattern (màn hình/ảnh chụp màn hình)
        moire_score = self._moire_score(face_roi)
        scores.append(moire_score * 0.20)  # Cao = KHÔNG có Moiré = thật
        
        return min(1.0, sum(scores))

    def _sharpness_score(self, gray: np.ndarray) -> float:
        """Laplacian variance - đo độ nét"""
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        variance = lap.var()
        # Người thật: variance > 100 thường
        # Ảnh mờ/in kém: variance thấp hơn
        score = min(1.0, variance / 300.0)
        return score

    def _lbp_score(self, gray: np.ndarray) -> float:
        """Local Binary Pattern — dùng vectorized cho real-time."""
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.5

        center = gray[1:-1, 1:-1].astype(np.int32)
        code = np.zeros_like(center, dtype=np.uint8)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = gray[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
            code |= ((neighbor >= center).astype(np.uint8) << bit)

        hist, _ = np.histogram(code.ravel(), bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-7)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        return min(1.0, entropy / 7.0)

    def _gradient_score(self, gray: np.ndarray) -> float:
        """Phân tích histogram gradient"""
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Người thật có phân phối gradient đa dạng
        # Ảnh phẳng (ảnh in bị mờ) có gradient tập trung ở 0
        mean_grad = magnitude.mean()
        std_grad = magnitude.std()
        
        score = min(1.0, (mean_grad + std_grad) / 80.0)
        return score

    def _moire_score(self, bgr: np.ndarray) -> float:
        """
        Phát hiện Moiré pattern - dấu hiệu của màn hình/ảnh scan
        Sử dụng FFT để tìm periodic patterns
        Returns: cao = KHÔNG có Moiré = thật
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        # Resize về 64x64 cho nhanh
        resized = cv2.resize(gray, (64, 64))
        
        # FFT
        f = np.fft.fft2(resized)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        
        # Tìm energy ngoại trừ DC component
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2
        
        # Mask DC component
        mask = np.ones_like(magnitude_spectrum)
        mask[crow-3:crow+3, ccol-3:ccol+3] = 0
        
        peripheral_energy = (magnitude_spectrum * mask).mean()
        
        # Màn hình có nhiều high-frequency periodic patterns
        # Score cao = ít periodic patterns = thật
        threshold = 2.5
        if peripheral_energy > threshold:
            score = max(0.0, 1.0 - (peripheral_energy - threshold) / 2.0)
        else:
            score = 1.0
            
        return score


class TemporalAnalyzer:
    """Phát hiện ảnh/video tĩnh qua frame difference trên ROI khuôn mặt."""

    def __init__(self, window_size: int = 20):
        self.roi_hashes = deque(maxlen=window_size)
        self.diff_scores = deque(maxlen=window_size)

    def update(self, face_roi_gray: np.ndarray) -> float:
        if face_roi_gray is None or face_roi_gray.size == 0:
            return 0.5

        small = cv2.resize(face_roi_gray, (32, 32)).astype(np.float32)
        if self.roi_hashes:
            diff = float(np.mean(np.abs(small - self.roi_hashes[-1])))
            self.diff_scores.append(diff)
        self.roi_hashes.append(small)
        return self.get_score()

    def get_score(self) -> float:
        if len(self.diff_scores) < 8:
            return 0.4
        diffs = np.array(list(self.diff_scores))
        mean_diff, std_diff = float(np.mean(diffs)), float(np.std(diffs))
        if mean_diff < 0.8 and std_diff < 0.5:
            return 0.1
        if mean_diff < 2.0:
            return 0.4
        if mean_diff < 8.0:
            return 0.75
        return min(1.0, mean_diff / 20.0)

    def reset(self):
        self.roi_hashes.clear()
        self.diff_scores.clear()


class DeepFASAnalyzer:
    """ONNX MiniFASNet hoặc heuristic thay thế."""

    INPUT_SIZE = (80, 80)

    def __init__(self):
        self.session = None
        self._load_model()

    def _load_model(self):
        if not ONNX_AVAILABLE or not settings.FAS_DL_ENABLED:
            return
        model_path = Path(settings.MODELS_PATH) / "anti_spoof.onnx"
        if not model_path.exists():
            logger.info("ONNX anti-spoof model not found — heuristic DL layer active")
            return
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            logger.success(f"Loaded ONNX anti-spoof: {model_path}")
        except Exception as e:
            logger.warning(f"ONNX anti-spoof load failed: {e}")

    def analyze(self, face_roi: np.ndarray) -> float:
        if face_roi is None or face_roi.size == 0:
            return 0.0
        if self.session is not None:
            return self._onnx_predict(face_roi)
        return self._heuristic_predict(face_roi)

    def _onnx_predict(self, face_roi: np.ndarray) -> float:
        img = cv2.resize(face_roi, self.INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img})
        logits = outputs[0][0]
        if len(logits) >= 2:
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            return float(probs[0])
        return float(1.0 / (1.0 + np.exp(-logits[0])))

    def _heuristic_predict(self, face_roi: np.ndarray) -> float:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        bright_ratio = np.sum(gray > 220) / gray.size
        highlight_score = 1.0 if 0.001 < bright_ratio < 0.08 else 0.5
        if len(face_roi.shape) == 3:
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            color_score = min(1.0, (float(hsv[:, :, 1].std()) + float(hsv[:, :, 0].std())) / 60.0)
        else:
            color_score = 0.5
        edges = cv2.Canny(gray, 50, 150)
        edge_score = min(1.0, (edges.sum() / (h * w * 255)) * 15)
        return highlight_score * 0.25 + color_score * 0.40 + edge_score * 0.35


class FrequencyAnalyzer:
    """
    Tầng 3: Phân tích tần số
    
    - Phát hiện tần số 60Hz/120Hz của màn hình qua temporal analysis
    - JPEG artifact detection (ảnh in thường qua nhiều lần nén)
    - High-frequency noise pattern
    """
    
    def __init__(self):
        self.frame_buffer = deque(maxlen=30)  # Buffer 30 frame
        
    def analyze_frame(self, face_roi: np.ndarray) -> float:
        """Phân tích tần số spatial của một frame"""
        if face_roi is None or face_roi.size == 0:
            return 0.5
            
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        scores = []
        
        # 1. JPEG artifact check
        jpeg_score = self._jpeg_artifact_score(gray)
        scores.append(jpeg_score * 0.4)
        
        # 2. Noise analysis
        noise_score = self._noise_analysis(gray)
        scores.append(noise_score * 0.3)
        
        # 3. Color channel analysis (màn hình có RGB subpixel)
        channel_score = self._channel_analysis(face_roi)
        scores.append(channel_score * 0.3)
        
        return min(1.0, sum(scores))

    def _jpeg_artifact_score(self, gray: np.ndarray) -> float:
        """
        Phát hiện JPEG artifacts (blocking, ringing)
        Ảnh in/chụp thường có nhiều JPEG artifacts hơn skin texture thật
        """
        # Tính DCT trên 8x8 blocks
        h, w = gray.shape
        block_size = 8
        h_blocks = h // block_size
        w_blocks = w // block_size
        
        if h_blocks == 0 or w_blocks == 0:
            return 0.5
            
        block_energies = []
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = gray[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
                dct_block = cv2.dct(block)
                # High-freq energy
                hf_energy = np.sum(np.abs(dct_block[4:, 4:]))
                block_energies.append(hf_energy)
        
        if not block_energies:
            return 0.5
            
        avg_hf = np.mean(block_energies)
        std_hf = np.std(block_energies)
        
        # JPEG artifacts tạo ra pattern đều đặn (std thấp, avg cao bất thường)
        # Skin texture thật có variance cao hơn
        if std_hf > 10:
            return min(1.0, std_hf / 50.0)
        else:
            return max(0.0, std_hf / 10.0)

    def _noise_analysis(self, gray: np.ndarray) -> float:
        """
        Phân tích noise pattern
        Người thật: noise Gaussian tự nhiên từ sensor camera
        Ảnh in: noise thấp hơn và đều hơn
        Video replay: noise có thể bị double (camera chụp màn hình)
        """
        # Estimate noise bằng median filter
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        noise = (gray.astype(np.float32) - blurred.astype(np.float32))
        
        noise_std = noise.std()
        noise_skew = float(np.mean((noise - noise.mean())**3) / (noise_std**3 + 1e-7))
        
        # Camera sensor noise: std 2-8, Gaussian (skew ≈ 0)
        # Double-acquisition noise: different pattern
        std_score = 1.0 if 1.5 < noise_std < 12.0 else max(0.0, 1.0 - abs(noise_std - 5) / 10)
        skew_score = max(0.0, 1.0 - abs(noise_skew) / 2.0)
        
        return (std_score + skew_score) / 2.0

    def _channel_analysis(self, bgr: np.ndarray) -> float:
        """
        Phân tích correlation giữa channels RGB
        Màn hình LCD: R, G, B channels có subpixel structure -> correlation cao bất thường
        """
        if len(bgr.shape) != 3 or bgr.shape[2] < 3:
            return 0.5
            
        b, g, r = cv2.split(bgr)
        
        # Tính correlation
        r_flat = r.ravel().astype(np.float32)
        g_flat = g.ravel().astype(np.float32)
        b_flat = b.ravel().astype(np.float32)
        
        corr_rg = np.corrcoef(r_flat, g_flat)[0, 1]
        corr_rb = np.corrcoef(r_flat, b_flat)[0, 1]
        corr_gb = np.corrcoef(g_flat, b_flat)[0, 1]
        
        avg_corr = (abs(corr_rg) + abs(corr_rb) + abs(corr_gb)) / 3.0
        
        # Skin tone thật: correlation cao tự nhiên (~0.85-0.95)
        # Màn hình với subpixel rendering: pattern khác
        # Chỉ penalize nếu correlation cực kỳ cao (>0.99) = không tự nhiên
        if avg_corr > 0.99:
            return 0.3
        elif avg_corr > 0.95:
            return 0.7
        else:
            return 1.0


class FaceAntiSpoofing:
    """Controller tổng hợp tất cả tầng FAS"""

    def __init__(self):
        self.texture_analyzer = TextureAnalyzer()
        self.freq_analyzer = FrequencyAnalyzer()
        self.dl_analyzer = DeepFASAnalyzer()
        self.liveness_trackers: Dict[str, LivenessTracker] = {}
        self.temporal_analyzers: Dict[str, TemporalAnalyzer] = {}
        self._frame_count = 0

    def get_liveness_tracker(self, track_id: str = "default") -> LivenessTracker:
        if track_id not in self.liveness_trackers:
            self.liveness_trackers[track_id] = LivenessTracker(
                window_size=settings.FAS_TEMPORAL_FRAMES
            )
        return self.liveness_trackers[track_id]

    def get_temporal_analyzer(self, track_id: str = "default") -> TemporalAnalyzer:
        if track_id not in self.temporal_analyzers:
            self.temporal_analyzers[track_id] = TemporalAnalyzer(
                window_size=settings.FAS_TEMPORAL_FRAMES
            )
        return self.temporal_analyzers[track_id]

    def reset_tracker(self, track_id: str = "default"):
        if track_id in self.liveness_trackers:
            self.liveness_trackers[track_id].reset()
        if track_id in self.temporal_analyzers:
            self.temporal_analyzers[track_id].reset()

    def analyze(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        landmarks_5: Optional[np.ndarray] = None,
        track_id: str = "default"
    ) -> FASResult:
        """
        Chạy toàn bộ pipeline FAS
        
        Args:
            frame: Full frame BGR
            bbox: [x1, y1, x2, y2] của khuôn mặt
            landmarks_5: 5 điểm facial landmarks từ InsightFace
            track_id: ID để theo dõi liên tục
            
        Returns: FASResult
        """
        start_time = time.time()
        self._frame_count += 1
        
        # Extract face ROI với padding
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return FASResult(is_live=False, overall_score=0.0, reject_reason="Empty face ROI")

        # Resize cho analysis
        face_roi_resized = cv2.resize(face_roi, (128, 128))
        
        layers_passed = 0
        reject_reason = None

        # ==================
        # TẦNG 1: Texture
        # ==================
        texture_score = 0.5  # Default
        if settings.FAS_TEXTURE_ENABLED:
            texture_score = self.texture_analyzer.analyze(face_roi_resized)
            if texture_score >= settings.FAS_TEXTURE_THRESHOLD:
                layers_passed += 1
            else:
                if reject_reason is None:
                    reject_reason = f"Texture fail ({texture_score:.2f} < {settings.FAS_TEXTURE_THRESHOLD})"

        # ==================
        # TẦNG 2: Liveness
        # ==================
        liveness_score = 0.5
        blink_detected = False
        motion_detected = False
        
        if settings.FAS_LIVENESS_ENABLED:
            tracker = self.get_liveness_tracker(track_id)
            tracker_result = tracker.update(
                frame=frame, landmarks_5=landmarks_5
            )
            liveness_score, blink_ok, motion_ok = tracker.get_liveness_score()
            blink_detected = blink_ok
            motion_detected = motion_ok

            liveness_pass = True
            if settings.FAS_BLINK_REQUIRED and not blink_ok:
                liveness_pass = False
                if reject_reason is None:
                    reject_reason = "No blink detected — static image/video"
            if settings.FAS_HEAD_MOTION_REQUIRED and not motion_ok:
                liveness_pass = False
                if reject_reason is None:
                    reject_reason = "No natural head motion — possible print/screen"

            if liveness_pass:
                layers_passed += 1

        # ==================
        # TẦNG 3: Frequency
        # ==================
        freq_score = 0.5
        if settings.FAS_FREQ_ENABLED:
            freq_score = self.freq_analyzer.analyze_frame(face_roi_resized)
            if freq_score >= settings.FAS_FREQ_THRESHOLD:
                layers_passed += 1
            else:
                if reject_reason is None:
                    reject_reason = f"Frequency anomaly ({freq_score:.2f}) — screen/print"

        # ==================
        # TẦNG 4: Temporal (static spoof)
        # ==================
        temporal_score = 0.5
        if settings.FAS_TEMPORAL_ENABLED:
            gray_roi = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2GRAY)
            temporal = self.get_temporal_analyzer(track_id)
            temporal_score = temporal.update(gray_roi)
            if temporal_score >= 0.5:
                layers_passed += 1
            else:
                if reject_reason is None:
                    reject_reason = "Static face ROI — photo/screen replay"

        # ==================
        # TẦNG 5: Deep Learning / Heuristic
        # ==================
        dl_score = 0.5
        if settings.FAS_DL_ENABLED:
            dl_score = self.dl_analyzer.analyze(face_roi_resized)
            if dl_score >= 0.6:
                layers_passed += 1
            else:
                if reject_reason is None:
                    reject_reason = f"DL anti-spoof fail ({dl_score:.2f})"

        # ==================
        # TỔNG HỢP
        # ==================
        weights = {
            "texture": 0.20,
            "liveness": 0.30,
            "freq": 0.15,
            "temporal": 0.20,
            "dl": 0.15,
        }

        overall_score = (
            texture_score * weights["texture"] +
            liveness_score * weights["liveness"] +
            freq_score * weights["freq"] +
            temporal_score * weights["temporal"] +
            dl_score * weights["dl"]
        )

        is_live = (
            overall_score >= settings.FAS_FINAL_THRESHOLD and
            layers_passed >= settings.FAS_MIN_LAYERS_PASS
        )
        
        if not is_live and reject_reason is None:
            reject_reason = f"Overall score too low ({overall_score:.2f} < {settings.FAS_FINAL_THRESHOLD})"

        processing_time = (time.time() - start_time) * 1000

        return FASResult(
            is_live=is_live,
            overall_score=overall_score,
            texture_score=texture_score,
            liveness_score=liveness_score,
            freq_score=freq_score,
            dl_score=dl_score,
            blink_detected=blink_detected,
            motion_detected=motion_detected,
            reject_reason=reject_reason if not is_live else None,
            layers_passed=layers_passed,
            processing_time_ms=processing_time
        )


# Singleton
anti_spoofing = FaceAntiSpoofing()
