"""
Wholebody49 — adapter DEIMv2-Wholebody49 (detector human-centric đa lớp:
body / head / face / hands / parts). CHẠY CLOUD.

⚠️ MODEL NICHE. `setup()` nạp ONNX session là CODE THẬT, nhưng `detect()` cần
PARSE OUTPUT theo đúng export upstream (tên/layout tensor) — đánh dấu ĐIỂM CẮM.
Lane A hoàn thiện trên cloud ở Phase 1b (môi trường Lead dựng sẵn).

Mục tiêu consolidation: 1 backbone phục vụ M1(person) + M3(anchor head/body) +
M4(face detect thay SCRFD). Lane A publish engine này cho Lane B/M1.
"""

from __future__ import annotations

from pathlib import Path


class Wholebody49:
    #: nhóm lớp quan tâm (ánh xạ từ 49 lớp gốc — chỉnh theo export thật)
    PARTS = ("body", "head", "face", "hand")

    def __init__(
        self,
        onnx_path: str | None,
        conf_thr: float = 0.4,
        providers: list[str] | None = None,
    ) -> None:
        self.onnx_path = onnx_path
        self.conf_thr = conf_thr
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = None
        self._input_name: str | None = None

    def available(self) -> bool:
        return bool(self.onnx_path) and Path(self.onnx_path).exists()

    def setup(self) -> None:
        """Nạp ONNX session (THẬT). Cần file ONNX export của Wholebody49."""
        if not self.available():
            raise FileNotFoundError(
                f"Không thấy ONNX Wholebody49: {self.onnx_path!r}. "
                "Export theo repo upstream rồi trỏ M*.wholebody_onnx (việc Phase 1b)."
            )
        import onnxruntime as ort  # noqa: PLC0415

        self._sess = ort.InferenceSession(self.onnx_path, providers=self.providers)
        self._input_name = self._sess.get_inputs()[0].name

    def detect(self, frame) -> list[dict]:
        """
        Chạy detector -> list[{part, label, box:[x1,y1,x2,y2], score}].

        ĐIỂM CẮM TÍCH HỢP (Lane A): tiền xử lý (resize/normalize/letterbox),
        chạy self._sess.run(...), rồi hậu xử lý đúng layout output của bản export
        (decode box + map class id -> PARTS). Khác nhau tuỳ cách export Wholebody49.
        """
        raise NotImplementedError(
            "Wholebody49.detect: parse output theo layout của bản export thật (Phase 1b)"
        )
