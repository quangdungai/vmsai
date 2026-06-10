"""
VMS247 MVP1 — Demo script: tạo video test tổng hợp + chạy pipeline (overlay + tóm tắt).

    # End-to-end nhanh (tạo video intrusion + chạy M1 baseline, xuất overlay):
    python scripts/demo.py

    # Tạo video test:
    python scripts/demo.py make --scenario all
    python scripts/demo.py make --scenario fire --seconds 12

    # Chạy pipeline trên 1 video (overlay + event + metric -> file + console):
    python scripts/demo.py run --source data/videos/demo_intrusion.mp4 --module M1
    python scripts/demo.py run --source clip.mp4 --module M2 --engine baseline --verify

Lưu ý trung thực:
- 'intrusion' dán SPRITE NGƯỜI THẬT (ảnh mẫu ultralytics) -> YOLO11 detect thật + rule ROI/line chạy thật.
- 'fire'/'ppe' là hình tổng hợp -> chủ yếu demo PIPELINE/overlay; model thật cần footage thật.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# cho phép chạy standalone: thêm app root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vms247.app import _build_verifier, load_config  # noqa: E402

DEF_SIZE = (1280, 720)
VIDEOS = Path("data/videos")


# --------------------------------------------------------------------------- #
# Helpers vẽ
# --------------------------------------------------------------------------- #
def _writer(out: Path, fps: float, size) -> cv2.VideoWriter:
    out.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        raise RuntimeError(f"Không mở được VideoWriter: {out}")
    return w


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _person_sprite(target_h: int) -> np.ndarray:
    """Sprite người: ưu tiên ảnh mẫu ultralytics (YOLO detect thật); fallback silhouette."""
    img = None
    try:
        from ultralytics.utils import ASSETS

        img = cv2.imread(str(ASSETS / "zidane.jpg"))
    except Exception:
        img = None
    if img is None:
        img = np.full((400, 200, 3), 210, np.uint8)
        cv2.circle(img, (100, 70), 48, (110, 110, 110), -1)
        cv2.rectangle(img, (55, 120), (145, 370), (110, 110, 110), -1)
    scale = target_h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), target_h))


def _blit(frame: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    h, w = frame.shape[:2]
    sh, sw = sprite.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + sw), min(h, y + sh)
    if x2 <= x1 or y2 <= y1:
        return
    frame[y1:y2, x1:x2] = sprite[y1 - y : y1 - y + (y2 - y1), x1 - x : x1 - x + (x2 - x1)]


def _draw_ref(frame, roi, line) -> None:
    if roi:
        cv2.polylines(frame, [np.array(roi, np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 1)
    if line:
        cv2.line(frame, line[0], line[1], (255, 255, 0), 1)


def _meta(out: Path, data: dict) -> None:
    out.with_suffix(".meta.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Sinh video theo kịch bản
# --------------------------------------------------------------------------- #
def make_intrusion(out: Path, seconds=12, fps=15, size=DEF_SIZE) -> Path:
    w, h = size
    sprite = _person_sprite(int(h * 0.5))
    sw = sprite.shape[1]
    line_x = int(w * 0.55)
    roi = [(line_x, 0), (w, 0), (w, h), (line_x, h)]  # nửa phải = vùng cấm
    n, y = seconds * fps, int(h * 0.45)
    wr = _writer(out, fps, size)
    inside = []
    for i in range(n):
        frame = np.full((h, w, 3), 70, np.uint8)
        _draw_ref(frame, roi, ((line_x, 0), (line_x, h)))
        x = int(_lerp(-sw, w * 0.78, i / max(1, n - 1)))  # đi trái -> phải, vào ROI
        _blit(frame, sprite, x, y)
        if x + sw // 2 > line_x:
            inside.append(i)
        wr.write(frame)
    wr.release()
    _meta(out, {"scenario": "intrusion", "fps": fps, "size": size, "roi": roi,
                "line": [[line_x, 0], [line_x, h]],
                "gt_intrusion_frames": [min(inside), max(inside)] if inside else []})
    return out


def make_fire(out: Path, seconds=12, fps=15, size=DEF_SIZE) -> Path:
    w, h = size
    wr = _writer(out, fps, size)
    rng = random.Random(0)
    for i in range(seconds * fps):
        t = i / fps
        frame = np.full((h, w, 3), 55, np.uint8)
        phase = "negative"
        if 4 <= t < 8:  # cháy + khói
            phase = "fire+smoke"
            ov = frame.copy()
            cy = int(h * 0.6 - (t - 4) * 28)
            cv2.ellipse(ov, (int(w * 0.5), cy), (90, 130), 0, 0, 360, (160, 160, 160), -1)
            frame = cv2.addWeighted(ov, 0.45, frame, 0.55, 0)
            f = rng.randint(-15, 15)
            cv2.ellipse(frame, (int(w * 0.5), int(h * 0.72)), (55 + f, 85 + f), 0, 0, 360, (0, 140, 255), -1)
            cv2.ellipse(frame, (int(w * 0.5), int(h * 0.74)), (32, 58), 0, 0, 360, (0, 215, 255), -1)
        elif t >= 8 and rng.random() < 0.35:  # tia hàn (negative — test báo giả)
            phase = "welding(neg)"
            for _ in range(25):
                px, py = int(w * 0.5 + rng.randint(-45, 45)), int(h * 0.6 + rng.randint(-45, 45))
                cv2.circle(frame, (px, py), 2, (210, 255, 255), -1)
        cv2.putText(frame, phase, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        wr.write(frame)
    wr.release()
    _meta(out, {"scenario": "fire", "fps": fps, "size": size,
                "gt": {"fire_smoke_s": [4, 8], "welding_negative_s": [8, seconds]}})
    return out


def make_ppe(out: Path, seconds=8, fps=15, size=DEF_SIZE) -> Path:
    w, h = size
    wr = _writer(out, fps, size)

    def figure(frame, cx, helmet):
        cv2.rectangle(frame, (cx - 35, 260), (cx + 35, 470), (140, 110, 90), -1)  # thân
        cv2.circle(frame, (cx, 220), 42, (170, 150, 130), -1)  # đầu
        if helmet:
            cv2.ellipse(frame, (cx, 210), (48, 40), 0, 180, 360, (0, 215, 255), -1)  # mũ vàng
            cv2.putText(frame, "helmet", (cx - 40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        else:
            cv2.putText(frame, "no helmet", (cx - 55, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for _ in range(seconds * fps):
        frame = np.full((h, w, 3), 80, np.uint8)
        figure(frame, int(w * 0.32), helmet=True)
        figure(frame, int(w * 0.68), helmet=False)
        wr.write(frame)
    wr.release()
    _meta(out, {"scenario": "ppe", "fps": fps, "size": size, "note": "tổng hợp — demo pipeline"})
    return out


# --------------------------------------------------------------------------- #
# Chạy pipeline trên 1 video -> overlay + tóm tắt
# --------------------------------------------------------------------------- #
def run_pipeline(source, module, engine, verify, out, config) -> None:
    from vms247.core.registry import build
    from vms247.core.schemas import Engine, ModuleId
    from vms247.shell.input import VideoSource
    from vms247.shell.overlay import draw_detections, draw_event_banner, draw_roi
    from vms247.shell.router import Router

    mid = ModuleId(module)
    router = Router(verifier=_build_verifier(config, force=verify))
    plugin = build(mid, Engine(engine))
    if hasattr(plugin, "configure"):
        plugin.configure(config.get(mid.value, {}))
    router.attach(plugin)
    roi = config.get(mid.value, {}).get("roi", [])

    fps = float(config.get("target_fps", 5.0)) or 5.0
    out = Path(out)
    wr = None
    n_frames, n_dets = 0, 0
    src = VideoSource(source, target_fps=fps)
    print(f"[run] {source} | module={module} engine={engine} verify={verify}")
    for frame, meta in src.frames():
        dets, _events = router.process_frame(mid, frame, meta)
        n_frames += 1
        n_dets += len(dets)
        vis = draw_event_banner(draw_detections(draw_roi(frame, roi), dets), _events)
        if wr is None:
            wr = _writer(out, fps, (frame.shape[1], frame.shape[0]))
        wr.write(vis)
    if wr is not None:
        wr.release()

    # tóm tắt
    events = router.event_store.all()
    by_type: dict[str, int] = {}
    for e in events:
        by_type[e.type.value] = by_type.get(e.type.value, 0) + 1
    print("\n" + "=" * 56)
    print(f"DEMO SUMMARY — {module}/{engine}")
    print("=" * 56)
    print(f"frames xử lý : {n_frames}")
    print(f"detections   : {n_dets}")
    print(f"events       : {len(events)}  {by_type if by_type else ''}")
    if router.verifier is not None:
        print(f"VLM verify   : checked={router.verifier.checked} suppressed={router.verifier.suppressed}")
    for m in router.metrics.snapshot():
        print(f"metric       : {m}")
    print(f"video overlay -> {out}")
    print(f"event log     -> {router.event_store.jsonl}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="VMS247 demo")
    sub = ap.add_subparsers(dest="cmd")

    mk = sub.add_parser("make", help="Tạo video test tổng hợp")
    mk.add_argument("--scenario", choices=["intrusion", "fire", "ppe", "all"], default="all")
    mk.add_argument("--seconds", type=int, default=12)
    mk.add_argument("--fps", type=int, default=15)
    mk.add_argument("--out-dir", default=str(VIDEOS))

    rn = sub.add_parser("run", help="Chạy pipeline trên 1 video")
    rn.add_argument("--source", required=True)
    rn.add_argument("--module", default="M1", choices=["M1", "M2", "M3", "M4"])
    rn.add_argument("--engine", default="baseline", choices=["baseline", "designated"])
    rn.add_argument("--verify", action="store_true")
    rn.add_argument("--out", default="data/videos/demo_annotated.mp4")
    rn.add_argument("--config", default="configs/default.yaml")

    args = ap.parse_args()

    if args.cmd == "make":
        d = Path(args.out_dir)
        todo = ["intrusion", "fire", "ppe"] if args.scenario == "all" else [args.scenario]
        fns = {"intrusion": make_intrusion, "fire": make_fire, "ppe": make_ppe}
        for s in todo:
            p = fns[s](d / f"demo_{s}.mp4", seconds=args.seconds, fps=args.fps)
            print(f"[make] {s} -> {p}")
        return

    if args.cmd == "run":
        run_pipeline(args.source, args.module, args.engine, args.verify, args.out, load_config(args.config))
        return

    # mặc định: end-to-end intrusion + M1
    print("[default] tạo video intrusion + chạy M1 baseline ...")
    vid = make_intrusion(VIDEOS / "demo_intrusion.mp4")
    print(f"[make] intrusion -> {vid}")
    run_pipeline(str(vid), "M1", "baseline", False,
                 "data/videos/demo_intrusion_annotated.mp4", load_config("configs/default.yaml"))


if __name__ == "__main__":
    main()
