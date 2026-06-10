"""
Tải/chuẩn bị weight baseline (Track 0). Chạy sau khi cài requirements-models.txt.

    python scripts/download_baseline_weights.py

Ghi chú:
- YOLO11 (person) tự tải khi gọi YOLO("yolo11m.pt") lần đầu (ultralytics).
- hardhat/fire CẦN train từ dataset (D-Fire, Hard Hat Workers) — xem assignment.
- InsightFace buffalo_l tự tải khi FaceAnalysis(name="buffalo_l").prepare().
"""

from __future__ import annotations

from pathlib import Path

MODELS = Path("models")
MODELS.mkdir(exist_ok=True)


def main() -> None:
    print("== VMS247 baseline weights ==")
    try:
        from ultralytics import YOLO

        print("YOLO11m (COCO, person) ...")
        YOLO("yolo11m.pt")  # tải về cache ultralytics
        print("  OK")
    except Exception as e:
        print(f"  Bỏ qua YOLO (chưa cài ultralytics?): {e}")

    try:
        from insightface.app import FaceAnalysis

        print("InsightFace buffalo_l ...")
        FaceAnalysis(name="buffalo_l")
        print("  OK (sẽ tải khi .prepare())")
    except Exception as e:
        print(f"  Bỏ qua InsightFace: {e}")

    print(
        "\nLƯU Ý: hardhat & fire weights phải TRAIN từ dataset "
        "(Hard Hat Workers, D-Fire) — dùng scripts/train_hardhat.py / train_fire_dfire.py."
    )


if __name__ == "__main__":
    main()
