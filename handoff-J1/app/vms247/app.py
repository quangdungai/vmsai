"""
VMS247 MVP1 — entrypoint.

    # Mở UI Gradio (demo 4 module + benchmark):
    python -m vms247.app --ui

    # Chạy headless 1 module trên 1 video (cửa sổ cv2 — tiện cho Junior dev):
    python -m vms247.app --source data/videos/demo_fire.mp4 --module M2 --engine baseline

    # Liệt kê plugin đã đăng ký:
    python -m vms247.app --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Console Windows có thể là cp1252 -> ép UTF-8 để in tiếng Việt không lỗi.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_verifier(config: dict, force: bool = False):
    """Tạo CascadeVerifier (Tier-2) từ config['cascade']. None nếu tắt/thiếu endpoint."""
    casc = config.get("cascade", {}) or {}
    if not (force or casc.get("enabled")):
        return None
    endpoint = casc.get("vlm_endpoint")
    if not endpoint:
        print("[cascade] bật verify nhưng thiếu cascade.vlm_endpoint -> bỏ qua.")
        return None
    from vms247.shell.verifier import CascadeVerifier

    return CascadeVerifier(
        endpoint=endpoint,
        model=casc.get("vlm_model", "locate-anything-3b"),
        verify_types=casc.get("verify_types"),
        enabled=True,
    )


def run_headless(source: str, module: str, engine: str, config: dict, verify: bool = False) -> None:
    import cv2

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
    win = f"VMS247 {mid.value}/{engine}"
    with VideoSource(source, target_fps=config.get("target_fps", 5.0)) as src:
        for frame, meta in src.frames():
            dets, events = router.process_frame(mid, frame, meta)
            vis = draw_roi(frame, roi)
            vis = draw_detections(vis, dets)
            vis = draw_event_banner(vis, events)
            cv2.imshow(win, vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    cv2.destroyAllWindows()
    router.teardown()


def main() -> None:
    ap = argparse.ArgumentParser(description="VMS247 MVP1")
    ap.add_argument("--ui", action="store_true", help="Mở Gradio UI")
    ap.add_argument("--list", action="store_true", help="Liệt kê plugin đã đăng ký")
    ap.add_argument("--source", help="Video file / rtsp://... / webcam0")
    ap.add_argument("--module", choices=["M1", "M2", "M3", "M4"])
    ap.add_argument("--engine", choices=["baseline", "designated"], default="baseline")
    ap.add_argument("--verify", action="store_true", help="Bật cascade VLM verify (Tier-2)")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    config = load_config(args.config)

    if args.list:
        from vms247.core.registry import available

        print("Plugin đã đăng ký (module, engine):")
        for m, e in available():
            print(f"  - {m} / {e}")
        return

    if args.ui:
        from vms247.shell.ui import build_ui

        build_ui(config).launch()
        return

    if args.source and args.module:
        run_headless(args.source, args.module, args.engine, config, verify=args.verify)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
