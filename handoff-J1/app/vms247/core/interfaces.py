"""
VMS247 MVP1 — Plugin interface (FROZEN, L2).

HỢP ĐỒNG PLUGIN — mọi module (M1..M4), mọi track (baseline/designated)
phải tuân theo đúng 3 method này. Shell chỉ gọi qua interface này, nên
2 Junior code song song KHÔNG chặn nhau.

    setup()    -> load model + resource (import nặng Ở ĐÂY, không ở top file)
    process()  -> infer 1 frame -> list[Detection]
    events()   -> áp luật -> list[Event]

M4 là superset: thêm enroll().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .schemas import Detection, Engine, Event, FrameMeta, ModuleId


class ModulePlugin(ABC):
    """Base cho cả 4 module. Junior kế thừa class này."""

    #: Junior PHẢI set 2 thuộc tính này ở class con (dùng cho registry + metric).
    module_id: ModuleId
    engine: Engine

    @abstractmethod
    def setup(self) -> None:
        """
        Load model/engine và mọi resource cần thiết.
        - Import nặng (torch, ultralytics, insightface...) đặt TRONG hàm này
          để import package shell không kéo theo CUDA.
        - Gọi 1 lần trước vòng lặp frame.
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, frame: np.ndarray, meta: FrameMeta) -> list[Detection]:
        """
        Chạy inference trên 1 frame BGR (numpy HxWx3, uint8) -> danh sách Detection.
        KHÔNG quyết định cảnh báo ở đây — chỉ sinh metadata (box/label/score).
        """
        raise NotImplementedError

    @abstractmethod
    def events(self, detections: list[Detection], state: dict[str, Any]) -> list[Event]:
        """
        Áp Rule Engine (ROI / ngưỡng / đổi-trạng-thái / debounce / cooldown)
        trên detections -> list[Event].
        `state` là dict bền vững theo (module, camera) do router cấp; plugin tự
        lưu timer/lịch sử track vào đây giữa các frame.
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """Giải phóng resource (tuỳ chọn)."""
        return None

    # Tiện ích chung cho metric/log
    @property
    def name(self) -> str:
        return f"{self.module_id.value}:{self.engine.value}"


class M4Plugin(ModulePlugin):
    """M4 Chấm công = superset, thêm luồng enroll khuôn mặt."""

    @abstractmethod
    def enroll(self, name: str, faces: list[np.ndarray]) -> bool:
        """
        Đăng ký 1 người: nhận tên + danh sách ảnh khuôn mặt (BGR) -> lưu embedding.
        Trả True nếu enroll thành công (bắt được mặt + tạo embedding).
        """
        raise NotImplementedError
