from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def main() -> None:
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "yolov8n.pt"
    if target.exists():
        print(f"Model already exists: {target}")
        return
    model = YOLO("yolov8n.pt")
    source = Path(getattr(model, "ckpt_path", "yolov8n.pt"))
    if not source.exists():
        source = Path("yolov8n.pt")
    shutil.copy2(source, target)
    print(f"Saved model to: {target}")


if __name__ == "__main__":
    main()
