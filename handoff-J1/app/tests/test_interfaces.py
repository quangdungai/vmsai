"""
Smoke test cho interface contract (L2). Chạy: pytest -q
Không cần model/CUDA — chỉ kiểm registry + shape contract.
"""

from __future__ import annotations

import inspect

import pytest

from vms247.core.interfaces import M4Plugin, ModulePlugin
from vms247.core.registry import available, build
from vms247.core.schemas import Engine, Event, EventType, ModuleId, Severity

ALL = [(m, e) for m in ModuleId for e in Engine]


def test_all_8_registered():
    avail = set(available())
    expected = {(m.value, e.value) for m, e in ALL}
    assert expected.issubset(avail), f"Thiếu plugin: {expected - avail}"


@pytest.mark.parametrize("module,engine", ALL)
def test_build_and_contract(module: ModuleId, engine: Engine):
    plugin = build(module, engine)
    assert isinstance(plugin, ModulePlugin)
    assert plugin.module_id == module
    assert plugin.engine == engine
    for name in ("setup", "process", "events"):
        assert callable(getattr(plugin, name))


@pytest.mark.parametrize("engine", list(Engine))
def test_m4_has_enroll(engine: Engine):
    plugin = build(ModuleId.M4, engine)
    assert isinstance(plugin, M4Plugin)
    sig = inspect.signature(plugin.enroll)
    assert list(sig.parameters) == ["name", "faces"]


def test_event_row_serializable():
    ev = Event(
        type=EventType.INTRUSION,
        module=ModuleId.M1,
        time=1.0,
        camera="cam01",
        confidence=0.9,
        severity=Severity.HIGH,
    )
    row = ev.to_row()
    assert row["module"] == "M1" and row["type"] == "intrusion"
