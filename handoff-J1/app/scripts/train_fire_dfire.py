"""
Train baseline M2 Cháy/khói — YOLO11 trên dataset D-Fire.

DATASET (tải thủ công, ~1.8GB):
  D-Fire — https://github.com/gaiasd/DFireDataset  (link Google Drive trong README)
  Giải nén ra <root> với layout YOLO:
      <root>/train/images/*.jpg   <root>/train/labels/*.txt
      <root>/test/images/*.jpg    <root>/test/labels/*.txt
  Classes (KIỂM TRA lại theo README D-Fire):  0 = smoke, 1 = fire

CHẠY (sau khi đã: pip install -r requirements-models.txt + torch theo CUDA):
  python scripts/train_fire_dfire.py --root D:/data/DFire --epochs 100 --model yolo11s.pt
  # hoặc trỏ thẳng data.yaml có sẵn:
  python scripts/train_fire_dfire.py --data D:/data/DFire/data.yaml

Output: models/fire_yolo.pt  (best weights) — đúng đường dẫn M2 baseline sẽ load.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

CLASS_NAMES = ["smoke", "fire"]  # ⚠️ verify thứ tự theo dataset


def build_data_yaml(root: Path) -> Path:
    """Sinh data.yaml từ layout D-Fire chuẩn (train/ + test/)."""
    import yaml

    data = {
        "path": str(root),
        "train": "train/images",
        "val": "test/images",  # D-Fire dùng 'test' làm validation
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    out = root / "data.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[data] Đã sinh {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLO11 fire/smoke trên D-Fire")
    ap.add_argument("--root", help="Thư mục gốc D-Fire (tự sinh data.yaml)")
    ap.add_argument("--data", help="Đường dẫn data.yaml có sẵn (ưu tiên hơn --root)")
    ap.add_argument("--model", default="yolo11s.pt", help="Weight khởi tạo")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=0, help="0 / 0,1 / cpu")
    ap.add_argument("--out", default="models/fire_yolo.pt")
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
        name="fire",
        exist_ok=True,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    print(f"\n[done] Best weights -> {out}")
    print("Gắn vào M2: configs/default.yaml > M2.model_path, hoặc load trực tiếp trong m2_fire/baseline.py")


if __name__ == "__main__":
    main()
