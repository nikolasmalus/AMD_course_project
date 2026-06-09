from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


MODELS = {
    "yolov8n.pt": "yolov8n.pt",
    "yolov8s.pt": "yolov8s.pt",
    "yolo11m.pt": "yolo11m.pt",
}


def download_model(model_ref: str, target: Path) -> None:
    if target.exists():
        print(f"Model already exists: {target}")
        return
    model = YOLO(model_ref)
    source = Path(getattr(model, "ckpt_path", model_ref))
    if not source.exists():
        source = Path(model_ref)
    shutil.copy2(source, target)
    print(f"Saved model to: {target}")


def main() -> None:
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for filename, model_ref in MODELS.items():
        download_model(model_ref, models_dir / filename)


if __name__ == "__main__":
    main()
