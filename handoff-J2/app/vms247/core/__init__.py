"""Core contract (L2) — import từ đây trong plugin và shell."""

from .interfaces import M4Plugin, ModulePlugin
from .registry import available, build, register
from .rules import (
    Cooldown,
    Debouncer,
    LineCrosser,
    box_bottom_center,
    box_center,
    point_in_polygon,
)
from .schemas import (
    BBox,
    Detection,
    Engine,
    Event,
    EventType,
    FrameMeta,
    Metric,
    ModuleId,
    Severity,
)

__all__ = [
    # interfaces
    "ModulePlugin",
    "M4Plugin",
    # registry
    "register",
    "build",
    "available",
    # schemas
    "ModuleId",
    "Engine",
    "EventType",
    "Severity",
    "BBox",
    "Detection",
    "Event",
    "Metric",
    "FrameMeta",
    # rules
    "point_in_polygon",
    "box_center",
    "box_bottom_center",
    "LineCrosser",
    "Debouncer",
    "Cooldown",
]
