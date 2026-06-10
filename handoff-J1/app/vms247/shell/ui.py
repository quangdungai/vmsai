"""
L6 — Gradio UI: 4 tab module + tab Enroll + bảng Chấm công + panel Metric,
kèm toggle baseline <-> designated.

Đây là SKELETON chạy được: upload video -> xử lý -> stream frame có overlay +
bảng event + metric. Tinh chỉnh webcam/RTSP realtime là việc shell tiếp theo
(đánh dấu TODO). Plugin do Junior cắm vào qua registry, UI không đổi.
"""

from __future__ import annotations

from collections.abc import Iterator

import gradio as gr
import pandas as pd

from vms247.core.registry import build
from vms247.core.schemas import Engine, ModuleId

from .input import VideoSource
from .metrics import MetricLogger
from .overlay import draw_detections, draw_event_banner, draw_roi
from .router import Router

_MODULE_LABELS = {
    ModuleId.M1: "M1 · Xâm nhập",
    ModuleId.M2: "M2 · Cháy/khói",
    ModuleId.M3: "M3 · PPE mũ",
    ModuleId.M4: "M4 · Chấm công",
}


def _new_router(engine: Engine, module: ModuleId, config: dict, verify_on: bool = False) -> Router:
    """Build router 1 module cho 1 engine (+ tuỳ chọn cascade VLM verify)."""
    verifier = None
    casc = config.get("cascade", {}) or {}
    if verify_on and casc.get("vlm_endpoint"):
        from vms247.shell.verifier import CascadeVerifier

        verifier = CascadeVerifier(
            endpoint=casc.get("vlm_endpoint"),
            model=casc.get("vlm_model", "locate-anything-3b"),
            verify_types=casc.get("verify_types"),
            enabled=True,
        )
    router = Router(metrics=MetricLogger(), verifier=verifier)
    plugin = build(module, engine)
    # truyền config (ROI, ngưỡng...) nếu plugin nhận
    if hasattr(plugin, "configure"):
        plugin.configure(config.get(module.value, {}))  # type: ignore[attr-defined]
    router.attach(plugin)
    return router


def _process_video(
    video_path: str, module: ModuleId, engine_str: str, config: dict, verify_on: bool = False
) -> Iterator[tuple]:
    """Generator: yield (frame_RGB, events_df, metrics_df) cho Gradio stream."""
    if not video_path:
        yield None, pd.DataFrame(), pd.DataFrame()
        return

    engine = Engine(engine_str)
    roi = config.get(module.value, {}).get("roi", [])
    try:
        router = _new_router(engine, module, config, verify_on)
    except Exception as e:  # plugin chưa implement -> báo rõ trên UI
        yield None, pd.DataFrame([{"error": f"{module.value}/{engine.value}: {e}"}]), pd.DataFrame()
        return

    src = VideoSource(video_path, target_fps=config.get("target_fps", 5.0))
    try:
        for frame, meta in src.frames():
            dets, events = router.process_frame(module, frame, meta)
            vis = draw_roi(frame, roi)
            vis = draw_detections(vis, dets)
            vis = draw_event_banner(vis, events)
            yield (
                vis[:, :, ::-1],  # BGR -> RGB cho Gradio
                pd.DataFrame(router.event_store.recent(30)),
                pd.DataFrame(router.metrics.snapshot()),
            )
    finally:
        src.release()
        router.teardown()


def build_ui(config: dict | None = None) -> gr.Blocks:
    config = config or {}

    with gr.Blocks(title="VMS247 MVP1 — Feasibility Demo") as demo:
        gr.Markdown("# VMS247 MVP1 — Demo 4 module + Benchmark harness")
        engine = gr.Radio(
            choices=[Engine.BASELINE.value, Engine.DESIGNATED.value],
            value=Engine.BASELINE.value,
            label="Track (engine)",
            info="baseline = lưới an toàn · designated = model đang đánh giá",
        )
        verify_chk = gr.Checkbox(
            value=False,
            label="Cascade VLM verify (Tier-2)",
            info="baseline + VLM hậu kiểm event để cắt báo giả (cần cascade.vlm_endpoint)",
        )

        # --- 4 tab module ---
        def _runner_for(m: ModuleId):
            # generator factory (lambda KHÔNG chứa được yield) — bắt đúng module m
            def _run(video_path, engine_str, verify_on):
                yield from _process_video(video_path, m, engine_str, config, verify_on)

            return _run

        for mid in (ModuleId.M1, ModuleId.M2, ModuleId.M3, ModuleId.M4):
            with gr.Tab(_MODULE_LABELS[mid]):
                with gr.Row():
                    vin = gr.Video(label="Video đầu vào (file)", sources=["upload"])
                    vout = gr.Image(label="Overlay", streaming=True)
                with gr.Row():
                    ev_df = gr.Dataframe(label="Event log", wrap=True)
                    mt_df = gr.Dataframe(label="Metric (FPS/VRAM/latency)")
                run = gr.Button(f"▶ Chạy {mid.value}", variant="primary")
                run.click(
                    fn=_runner_for(mid),
                    inputs=[vin, engine, verify_chk],
                    outputs=[vout, ev_df, mt_df],
                )

        # --- Tab Enroll (M4) ---
        with gr.Tab("➕ Enroll (M4)"):
            gr.Markdown("Đăng ký khuôn mặt cho chấm công (M4.enroll).")
            name = gr.Textbox(label="Tên nhân viên")
            imgs = gr.Gallery(label="Ảnh khuôn mặt (≥1)", type="numpy")
            enroll_btn = gr.Button("Enroll", variant="primary")
            enroll_out = gr.Textbox(label="Kết quả")

            def _do_enroll(nm, images):
                if not nm or not images:
                    return "Thiếu tên hoặc ảnh."
                try:
                    plugin = build(ModuleId.M4, Engine.BASELINE)
                    plugin.setup()
                    faces = [img[0] if isinstance(img, tuple) else img for img in images]
                    ok = plugin.enroll(nm, faces)  # type: ignore[attr-defined]
                    return f"Enroll {'OK' if ok else 'THẤT BẠI'}: {nm}"
                except Exception as e:
                    return f"Lỗi enroll: {e}"

            enroll_btn.click(_do_enroll, [name, imgs], enroll_out)

        # --- Tab Chấm công ---
        with gr.Tab("🗒 Bảng chấm công"):
            gr.Markdown("Log chấm công (event type=attendance). Nối HRM ở giai đoạn sau.")
            att_df = gr.Dataframe(label="Attendance")
            refresh = gr.Button("Làm mới")
            refresh.click(lambda: pd.DataFrame(), outputs=att_df)  # TODO: đọc từ event store dùng chung

    return demo
