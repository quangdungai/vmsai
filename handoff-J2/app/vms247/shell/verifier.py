"""
CascadeVerifier (Tier-2) — VLM hậu kiểm EVENT của Tier-1 (baseline) để CẮT báo giả.

Đúng kiến trúc 2-tier: detector nhanh (baseline) sinh candidate event → VLM
(LocateAnything-3B) chỉ chạy TRÊN EVENT (không mọi frame) để xác minh ngữ nghĩa.
Khác toggle baseline/designated: đây là CASCADE baseline + VLM-verify đồng thời.

Fail-open: VLM lỗi / chưa cấu hình → GIỮ event (không bỏ sót báo thật).
"""

from __future__ import annotations

from vms247.core.schemas import Event, EventType

# Câu hỏi xác minh theo loại event (chỉ verify loại mà ngữ nghĩa giúp ích).
_QUESTIONS: dict[EventType, str] = {
    EventType.FIRE: "Is there REAL fire/flame here? Answer false for welding sparks, sunlight glare, or red/orange objects.",
    EventType.SMOKE: "Is there REAL smoke here? Answer false for steam, fog, dust, or haze.",
    EventType.INTRUSION: "Is there a real person in this image?",
    EventType.PPE_VIOLATION: "Is there a person NOT wearing a safety helmet in this image?",
}


def _crop(frame, box):
    if frame is None or box is None:
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(max(0, v)) for v in box)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame


def _coerce_types(types) -> set[EventType]:
    out: set[EventType] = set()
    for t in types or ():
        if isinstance(t, EventType):
            out.add(t)
        else:
            try:
                out.add(EventType(str(t)))
            except Exception:
                pass
    return out


class CascadeVerifier:
    def __init__(
        self,
        endpoint: str | None = None,
        model: str = "locate-anything-3b",
        verify_types=None,
        enabled: bool = True,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self.enabled = bool(enabled) and bool(endpoint)
        self.verify_types = _coerce_types(verify_types) or {EventType.FIRE, EventType.SMOKE}
        self._client = None
        self.checked = 0
        self.suppressed = 0

    def setup(self) -> None:
        if not self.enabled:
            return
        from vms247.integrations.vlm_client import VLMClient  # noqa: PLC0415

        self._client = VLMClient(self._endpoint, self._model)

    def filter(self, frame, events: list[Event]) -> list[Event]:
        """Trả về danh sách event GIỮ LẠI (đã bỏ event bị VLM bác)."""
        if not self.enabled or self._client is None or not events:
            return events
        kept: list[Event] = []
        for ev in events:
            if ev.type not in self.verify_types or ev.type not in _QUESTIONS:
                kept.append(ev)
                continue
            self.checked += 1
            try:
                res = self._client.verify(_crop(frame, ev.box), _QUESTIONS[ev.type])
            except Exception:  # pragma: no cover — fail-open
                ev.meta["verified"] = "error_keep"
                kept.append(ev)
                continue
            if res.get("is_true"):
                ev.meta["verified"] = True
                ev.meta["verify_reason"] = res.get("reason", "")
                kept.append(ev)
            else:
                self.suppressed += 1  # báo giả → bỏ
        return kept
