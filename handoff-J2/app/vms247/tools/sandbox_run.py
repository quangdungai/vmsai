"""
VMS247 — Runner đơn module (chạy thử ĐÚNG 1 detector trên 1 nguồn video).

Mục đích: chạy thử detector của 1 module trên video. Luồng tối giản:

    decode video -> plugin.process() -> plugin.events() -> vẽ + in event + FPS

Chỉ gồm: đọc video + detector của module + interface core/ — tiện debug detector
không vướng phần còn lại. Muốn chạy đủ (router nhiều module, metric, UI) thì dùng
`python -m vms247.app`.

    python -m vms247.tools.sandbox_run --source data/videos/demo_fire.mp4 --module M2
    python -m vms247.tools.sandbox_run --source webcam0 --module M1 --no-window

Output:
    - Cửa sổ cv2 (q/ESC thoát) trừ khi --no-window.
    - In FPS trung bình + mỗi Event ra stdout.
    - Ghi Event ra <out>/events.jsonl (mặc định ./sandbox_out).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

try:  # in tiếng Việt không lỗi trên console Windows
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load_cfg(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[sandbox] không thấy config {p} — dùng mặc định rỗng.")
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run(
    source: str,
    module: str,
    config_path: str = "configs/sandbox.yaml",
    no_window: bool = False,
    out_dir: str = "sandbox_out",
) -> None:
    import cv2

    from vms247.core.registry import build
    from vms247.core.schemas import Engine, ModuleId
    from vms247.shell.input import VideoSource
    from vms247.shell.overlay import draw_detections, draw_event_banner, draw_roi

    cfg = _load_cfg(config_path)
    mid = ModuleId(module)
    mcfg = cfg.get(mid.value, {}) or {}

    # Runner gọn — chốt baseline cho nhẹ; chạy designated/cascade thì dùng vms247.app.
    plugin = build(mid, Engine.BASELINE)
    if hasattr(plugin, "configure"):
        plugin.configure(mcfg)
    plugin.setup()

    roi = mcfg.get("roi", []) or []
    state: dict = {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev_path = out / "events.jsonl"
    ev_file = ev_path.open("a", encoding="utf-8")

    win = f"VMS247 sandbox - {mid.value}/baseline"
    n_frames = 0
    n_events = 0
    t0 = time.time()
    try:
        with VideoSource(source, target_fps=cfg.get("target_fps", 5.0)) as src:
            for frame, meta in src.frames():
                dets = plugin.process(frame, meta)
                events = plugin.events(dets, state)
                for ev in events:
                    if not ev.camera:
                        ev.camera = meta.camera
                    if not ev.time:
                        ev.time = meta.timestamp
                    n_events += 1
                    row = ev.to_row()
                    print(
                        f"[EVENT] {row['type']} cam={row['camera']} "
                        f"conf={row['confidence']} {row['label']}"
                    )
                    ev_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    ev_file.flush()
                n_frames += 1
                if not no_window:
                    vis = draw_roi(frame, roi)
                    vis = draw_detections(vis, dets)
                    vis = draw_event_banner(vis, events)
                    cv2.imshow(win, vis)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
    finally:
        ev_file.close()
        if not no_window:
            cv2.destroyAllWindows()
        try:
            plugin.teardown()
        except Exception:
            pass

    dt = max(1e-6, time.time() - t0)
    print(
        f"\n[SANDBOX] module={mid.value} frames={n_frames} "
        f"fps={n_frames / dt:.1f} events={n_events} -> {ev_path}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="VMS247 sandbox runner — chạy 1 module baseline (không router/VLM)"
    )
    ap.add_argument("--source", required=True, help="video file / rtsp://... / webcam0")
    ap.add_argument("--module", required=True, choices=["M1", "M2", "M3", "M4"])
    ap.add_argument("--config", default="configs/sandbox.yaml")
    ap.add_argument("--no-window", action="store_true", help="chạy headless, không mở cửa sổ cv2")
    ap.add_argument("--out", default="sandbox_out", help="thư mục xuất events.jsonl")
    args = ap.parse_args()
    run(args.source, args.module, args.config, no_window=args.no_window, out_dir=args.out)


if __name__ == "__main__":
    main()
