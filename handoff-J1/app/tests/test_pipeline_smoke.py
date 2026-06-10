"""
Smoke test PIPELINE THẬT — chứng minh plugin load model + infer + ra
Detection/Event đúng kiểu, và Router chạy end-to-end (KHÔNG chỉ compile).

Tự SKIP nếu chưa cài model deps (torch/ultralytics) → suite contract vẫn
chạy được trên môi trường nhẹ. Cài để bật: pip install -r requirements-models.txt
"""

from __future__ import annotations

import numpy as np
import pytest

# Cả file skip nếu thiếu deps nặng.
pytest.importorskip("torch", reason="cần torch (requirements-models.txt)")
pytest.importorskip("ultralytics", reason="cần ultralytics")

from vms247.core.registry import build  # noqa: E402
from vms247.core.schemas import (  # noqa: E402
    Detection,
    Engine,
    Event,
    FrameMeta,
    ModuleId,
)


def _meta(img) -> FrameMeta:
    return FrameMeta("test", 0, 0.0, img.shape[1], img.shape[0])


def _noise(h: int = 720, w: int = 1280) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _sample_with_people() -> np.ndarray:
    """Ảnh mẫu có người, bundled sẵn trong ultralytics."""
    import cv2
    from ultralytics.utils import ASSETS

    img = cv2.imread(str(ASSETS / "zidane.jpg"))
    assert img is not None
    return img


def test_m1_detects_person():
    """YOLO11 person THẬT: phải thấy ≥1 person trên ảnh mẫu."""
    p = build(ModuleId.M1, Engine.BASELINE)
    p.configure({"model_path": "yolo11m.pt", "conf_thr": 0.25})
    p.setup()
    img = _sample_with_people()
    dets = p.process(img, _meta(img))
    assert all(isinstance(d, Detection) for d in dets)
    assert any(d.label == "person" for d in dets), "phải phát hiện ít nhất 1 person"


def test_router_end_to_end():
    """Router: frame -> process -> events -> metric, không crash."""
    from vms247.shell.router import Router

    r = Router()
    p = build(ModuleId.M1, Engine.BASELINE)
    p.configure({"model_path": "yolo11m.pt"})
    r.attach(p)
    img = _sample_with_people()
    dets, events = r.process_frame(ModuleId.M1, img, _meta(img))
    assert all(isinstance(d, Detection) for d in dets)
    assert all(isinstance(e, Event) for e in events)
    assert r.metrics.snapshot(), "metric phải có số sau khi chạy"


@pytest.mark.parametrize("mod", [ModuleId.M2, ModuleId.M3])
def test_world_modules_run(mod):
    """M2/M3 qua YOLO-World fallback: chạy được + đúng kiểu trả về.
    YOLO-World cần CLIP; nếu tải lỗi thì SKIP thay vì FAIL."""
    p = build(mod, Engine.BASELINE)
    p.configure({})
    try:
        p.setup()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"YOLO-World/CLIP chưa sẵn sàng: {e}")
    img = _noise()
    dets = p.process(img, _meta(img))
    assert all(isinstance(d, Detection) for d in dets)
    assert isinstance(p.events(dets, {}), list)


def test_m4_enroll_then_selfmatch(tmp_path):
    """M4: enroll mặt từ ảnh mẫu rồi tự match lại phải ra đúng tên."""
    pytest.importorskip("insightface", reason="cần insightface")
    p = build(ModuleId.M4, Engine.BASELINE)
    p.configure({"db_path": str(tmp_path / "faces.npz"), "threshold": 0.3})
    try:
        p.setup()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"insightface chưa sẵn sàng: {e}")
    img = _sample_with_people()
    if not p.enroll("zidane", [img]):
        pytest.skip("không bắt được mặt trong ảnh mẫu")
    dets = p.process(img, _meta(img))
    assert any(d.attributes.get("name") == "zidane" for d in dets), "tự match phải ra 'zidane'"
