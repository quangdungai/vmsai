"""
Unit test CascadeVerifier — chạy KHÔNG cần cloud (inject client giả).
Kiểm: disabled passthrough · suppress báo giả · giữ báo thật · bỏ qua loại không verify.
"""

from __future__ import annotations

import numpy as np

from vms247.core.schemas import Event, EventType, ModuleId
from vms247.shell.verifier import CascadeVerifier


class _FakeVLM:
    def __init__(self, answer: bool) -> None:
        self.answer = answer

    def verify(self, img, question):
        return {"is_true": self.answer, "reason": "stub"}


def _event(t: EventType) -> Event:
    return Event(type=t, module=ModuleId.M2, time=1.0, camera="c", box=(0, 0, 10, 10))


def _frame():
    return np.zeros((20, 20, 3), dtype=np.uint8)


def test_disabled_passthrough():
    v = CascadeVerifier(endpoint=None)  # không endpoint -> disabled
    assert v.enabled is False
    evs = [_event(EventType.FIRE)]
    assert v.filter(_frame(), evs) == evs


def test_suppress_false_fire():
    v = CascadeVerifier(endpoint="http://x:8000")
    v._client = _FakeVLM(False)  # VLM nói KHÔNG phải cháy
    out = v.filter(_frame(), [_event(EventType.FIRE)])
    assert out == [] and v.suppressed == 1 and v.checked == 1


def test_keep_true_fire():
    v = CascadeVerifier(endpoint="http://x:8000")
    v._client = _FakeVLM(True)
    out = v.filter(_frame(), [_event(EventType.FIRE)])
    assert len(out) == 1 and out[0].meta.get("verified") is True


def test_untracked_type_passthrough():
    v = CascadeVerifier(endpoint="http://x:8000")
    v._client = _FakeVLM(False)
    # ATTENDANCE không nằm trong verify_types -> giữ nguyên, không gọi VLM
    out = v.filter(_frame(), [_event(EventType.ATTENDANCE)])
    assert len(out) == 1 and v.checked == 0
