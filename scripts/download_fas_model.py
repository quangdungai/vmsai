#!/usr/bin/env python3
"""
Tải ONNX anti-spoof model (tuỳ chọn) vào thư mục models/.
Nếu không có model, hệ thống dùng heuristic DL layer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


def main():
    models_dir = Path(settings.MODELS_PATH)
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "anti_spoof.onnx"

    if target.exists():
        print(f"Model already exists: {target}")
        return

    print("=" * 60)
    print("ONNX Anti-Spoof Model")
    print("=" * 60)
    print()
    print("Đặt file anti_spoof.onnx (MiniFASNet / Silent-Face-Anti-Spoofing)")
    print(f"vào: {target}")
    print()
    print("Nguồn tham khảo:")
    print("  https://github.com/minivision-ai/Silent-Face-Anti-Spoofing")
    print()
    print("Hệ thống vẫn hoạt động với heuristic DL layer nếu không có model.")


if __name__ == "__main__":
    main()
