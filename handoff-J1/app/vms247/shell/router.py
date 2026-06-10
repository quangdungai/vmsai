"""
L5 — Router. Điều phối: frame -> plugin.process() -> plugin.events()
-> metric + event store. Quản lý `state` bền vững theo (module, camera).

Đây là chỗ shell gặp plugin. Plugin KHÔNG biết gì về router; router chỉ
biết interface ModulePlugin => thay baseline<->designated không đụng code này.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vms247.core.interfaces import ModulePlugin
from vms247.core.schemas import Detection, Engine, Event, FrameMeta, ModuleId

from .events import EventStore
from .metrics import MetricLogger


@dataclass
class ModuleRuntime:
    plugin: ModulePlugin
    ready: bool = False
    state: dict = field(default_factory=dict)  # bền vững giữa các frame (per camera)


class Router:
    def __init__(
        self,
        event_store: EventStore | None = None,
        metrics: MetricLogger | None = None,
        verifier=None,
    ) -> None:
        self.event_store = event_store or EventStore()
        self.metrics = metrics or MetricLogger()
        self.verifier = verifier  # CascadeVerifier (Tier-2 VLM) — tuỳ chọn
        if verifier is not None:
            verifier.setup()
        self._runtimes: dict[ModuleId, ModuleRuntime] = {}

    # --- quản lý plugin ---------------------------------------------------- #
    def attach(self, plugin: ModulePlugin) -> None:
        """Gắn 1 plugin (chưa setup). Tự setup ở lần process đầu."""
        self._runtimes[plugin.module_id] = ModuleRuntime(plugin=plugin)

    def attached_modules(self) -> list[ModuleId]:
        return list(self._runtimes.keys())

    def _ensure_ready(self, rt: ModuleRuntime) -> None:
        if not rt.ready:
            rt.plugin.setup()
            rt.ready = True

    # --- vòng xử lý 1 frame ----------------------------------------------- #
    def process_frame(
        self, module_id: ModuleId, frame: np.ndarray, meta: FrameMeta
    ) -> tuple[list[Detection], list[Event]]:
        rt = self._runtimes.get(module_id)
        if rt is None:
            return [], []
        self._ensure_ready(rt)

        cam_state = rt.state.setdefault(meta.camera, {})

        with self.metrics.measure(module_id, rt.plugin.engine):
            dets = rt.plugin.process(frame, meta)

        events = rt.plugin.events(dets, cam_state)
        for ev in events:
            # điền mặc định nếu plugin để trống
            if not ev.camera:
                ev.camera = meta.camera
            if not ev.time:
                ev.time = meta.timestamp
        # Tier-2 cascade: VLM hậu kiểm event (cắt báo giả) nếu có verifier
        if self.verifier is not None:
            events = self.verifier.filter(frame, events)
        for ev in events:
            self.event_store.add(ev, frame=frame)
        return dets, events

    def teardown(self) -> None:
        for rt in self._runtimes.values():
            try:
                rt.plugin.teardown()
            except Exception:
                pass
