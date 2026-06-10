"""
VMS247 MVP1 — Rule Engine helpers.

Các tiện ích dùng chung cho plugin.events(): ROI polygon, line-crossing,
debounce theo thời gian, cooldown. Junior import từ đây, không tự viết lại.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import BBox


def box_center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_bottom_center(box: BBox) -> tuple[float, float]:
    """Điểm chân (đáy giữa) — ổn định hơn tâm cho người đứng."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def point_in_polygon(pt: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray casting. polygon = list điểm (x, y) theo pixel."""
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def _side(p, a, b) -> float:
    """Dấu của p so với đường thẳng a->b (>0 một phía, <0 phía kia)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


@dataclass
class LineCrosser:
    """Theo dõi track_id vượt qua 1 đoạn thẳng a->b."""
    a: tuple[float, float]
    b: tuple[float, float]
    _last_side: dict[int, float] = field(default_factory=dict)

    def update(self, track_id: int, point: tuple[float, float]) -> bool:
        """Trả True nếu track này vừa đổi phía (tức vượt line)."""
        s = _side(point, self.a, self.b)
        prev = self._last_side.get(track_id)
        self._last_side[track_id] = s
        if prev is None:
            return False
        return (prev <= 0 < s) or (prev >= 0 > s)


@dataclass
class Debouncer:
    """
    Chỉ phát sự kiện khi điều kiện ĐÚNG liên tục đủ `min_seconds`
    (lọc nhiễu thời gian — nhấp nháy 1-2 frame).
    """
    min_seconds: float = 2.0
    _true_since: float | None = None

    def update(self, condition: bool, now: float) -> bool:
        if not condition:
            self._true_since = None
            return False
        if self._true_since is None:
            self._true_since = now
        return (now - self._true_since) >= self.min_seconds


@dataclass
class Cooldown:
    """Chặn spam: sau khi phát 1 event, im trong `seconds`."""
    seconds: float = 30.0
    _last_fire: float | None = None

    def ready(self, now: float) -> bool:
        return self._last_fire is None or (now - self._last_fire) >= self.seconds

    def fire(self, now: float) -> None:
        self._last_fire = now
