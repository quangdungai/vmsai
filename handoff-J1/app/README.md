# VMS247 MVP1 — App demo + Benchmark harness

Demo 4 module AI (M1 Xâm nhập · M2 Cháy/khói · M3 PPE · M4 Chấm công) trên video file/RTSP/webcam,
kèm khung đo (FPS/VRAM/latency) để **benchmark sớm 2 model designated**.

Kiến trúc **2-track** sau cùng 1 interface:
- **Track 0 — baseline** (lưới an toàn): YOLO11 · hardhat YOLO · fire-YOLO/YOLO-World · InsightFace
- **Track 1 — designated** (đang đánh giá): DEIMv2-Wholebody49 · LocateAnything-3B

> Việc của từng người: xem **file giao việc đi kèm** gói này. Thắc mắc bối cảnh/quyết định → hỏi Lead.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1                           # Windows PowerShell (cmd: .venv\Scripts\activate.bat)
pip install -r requirements.txt                      # shell (chạy được ngay)
pip install -r requirements-models.txt               # khi làm plugin thật (cần CUDA)
pip install -e .                                      # cài package vms247
```

## Chạy

```bash
python -m vms247.app --list                          # liệt kê plugin đã đăng ký (8)
python -m vms247.app --ui                            # mở Gradio UI
# Headless 1 module (cửa sổ cv2 — tiện dev plugin):
python -m vms247.app --source data/videos/x.mp4 --module M2 --engine baseline
pytest -q                                             # smoke test interface contract
```

> Khi chưa có plugin thật, app vẫn chạy: video stream + overlay ROI + metric tick,
> detection rỗng (stub trả `[]`). Cắm model vào là có ngay.

## Cấu trúc

```
vms247/
  core/        # L2 — interface contract (FROZEN): schemas · interfaces · registry · rules
  shell/       # L4-L7 — input · router · overlay · events · metrics · ui   (Lead)
  modules/     # plugin 4 module × 2 track                                   (Junior)
    _template.py          # ví dụ MẪU hoàn chỉnh (OpenCV motion) — COPY pattern này
    m1_intrusion/  baseline.py  designated.py
    m2_fire/       baseline.py  designated.py
    m3_ppe/        baseline.py  designated.py
    m4_attendance/ baseline.py  designated.py
configs/default.yaml       # ROI/zone/ngưỡng/endpoint
tests/test_interfaces.py   # contract smoke test
```

## Viết 1 plugin (cho Junior)

1. Mở file `modules/<module>/baseline.py` đã có sẵn (đã `@register`).
2. Đọc `modules/_template.py` để thấy 1 plugin hoàn chỉnh.
3. Implement đúng 3 hàm: `setup()` (load model), `process()` (→ `Detection`), `events()` (→ `Event`).
4. Dùng helper trong `core/rules.py` (ROI, LineCrosser, Debouncer, Cooldown).
5. Test standalone: `python -m vms247.app --source <video> --module Mx`.
6. Nghiệm thu: **demo dương + demo âm** (xem docstring từng file).

**Quy tắc vàng:** import model nặng (torch/ultralytics/insightface) **bên trong `setup()`**,
không ở đầu file — để import shell không kéo theo CUDA.

## Hợp đồng dữ liệu (tóm tắt)

```python
Detection(label, confidence, box=(x1,y1,x2,y2), track_id=None, attributes={})
Event(type, module, time, camera, box, confidence, severity, label, evidence_path)
Metric(module, engine, fps, latency_ms, vram_mb)
```
