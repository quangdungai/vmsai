"""
Train baseline M3 PPE mũ — YOLO11 trên dataset Hard Hat Workers.

DATASET (tải thủ công):
  Hard Hat Workers — Roboflow Universe (export định dạng "YOLOv11"/"YOLOv8"):
    https://universe.roboflow.com/  (tìm "Hard Hat Workers")
    Export kèm sẵn data.yaml -> trỏ --data thẳng vào đó.
  Classes thường gặp:  0 = head, 1 = helmet, 2 = person
  (Phương án nhiều lớp PPE hơn: SH17 — 17 lớp bảo hộ.)

CHẠY:
  python scripts/train_hardhat.py --data D:/data/HardHat/data.yaml --epochs 100
  # hoặc tự sinh data.yaml nếu có layout train/ valid/:
  python scripts/train_hardhat.py --root D:/data/HardHat

Output: models/hardhat_yolo.pt  — đường dẫn M3 baseline sẽ load.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

CLASS_NAMES = ["head", "helmet", "person"]  # ⚠️ verify theo data.yaml của bản export


def build_data_yaml(root: Path) -> Path:
    import yaml

    # Roboflow thường dùng train/ valid/ test/
    val = "valid/images" if (root / "valid").exists() else "test/images"
    data = {
        "path": str(root),
        "train": "train/images",
        "val": val,
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    out = root / "data.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[data] Đã sinh {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLO11 hardhat trên Hard Hat Workers")
    ap.add_argument("--root", help="Thư mục gốc dataset (tự sinh data.yaml)")
    ap.add_argument("--data", help="data.yaml có sẵn (Roboflow export) — ưu tiên")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", default="models/hardhat_yolo.pt")
    args = ap.parse_args()

    if not args.data and not args.root:
        ap.error("Cần --data hoặc --root")

    data_yaml = Path(args.data) if args.data else build_data_yaml(Path(args.root))

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name="hardhat",
        exist_ok=True,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    print(f"\n[done] Best weights -> {out}")
    print("Lưu ý M3: 'no_helmet' suy ra từ class 'head' (đầu không mũ) — xử lý trong m3_ppe/baseline.py")


if __name__ == "__main__":
    main()
