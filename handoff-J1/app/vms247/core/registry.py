"""
VMS247 MVP1 — Plugin registry (L2).

Mỗi plugin tự đăng ký bằng decorator @register(module_id, engine).
Shell build plugin theo (module, engine) mà không cần biết class cụ thể
=> toggle baseline <-> designated chỉ là đổi 1 tham số.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schemas import Engine, ModuleId

if TYPE_CHECKING:
    from .interfaces import ModulePlugin

# (module_id, engine) -> class plugin
_REGISTRY: dict[tuple[ModuleId, Engine], type["ModulePlugin"]] = {}


def register(module_id: ModuleId, engine: Engine):
    """Decorator gắn lên class plugin của Junior."""

    def _decorator(cls: type["ModulePlugin"]) -> type["ModulePlugin"]:
        key = (module_id, engine)
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(
                f"Trùng đăng ký plugin cho {key}: "
                f"{_REGISTRY[key].__name__} vs {cls.__name__}"
            )
        cls.module_id = module_id
        cls.engine = engine
        _REGISTRY[key] = cls
        return cls

    return _decorator


def build(module_id: ModuleId | str, engine: Engine | str) -> "ModulePlugin":
    """Tạo instance plugin theo (module, engine). KHÔNG gọi setup() — caller tự gọi."""
    _ensure_modules_imported()
    key = (ModuleId(module_id), Engine(engine))
    if key not in _REGISTRY:
        raise KeyError(
            f"Chưa có plugin cho {key[0].value}/{key[1].value}. "
            f"Đã đăng ký: {sorted((m.value, e.value) for m, e in _REGISTRY)}"
        )
    return _REGISTRY[key]()


def available() -> list[tuple[str, str]]:
    _ensure_modules_imported()
    return sorted((m.value, e.value) for m, e in _REGISTRY)


_imported = False


def _ensure_modules_imported() -> None:
    """Import package modules để kích hoạt các @register."""
    global _imported
    if _imported:
        return
    import importlib

    importlib.import_module("vms247.modules")
    _imported = True
