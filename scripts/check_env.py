from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path
from src.hardware_manager import detect_hardware


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    model_path = resolve_path(config, config["paths"]["model_path"])
    model_profiles = {}
    for name, profile in config["model"].get("profiles", {}).items():
        path = resolve_path(config, profile["model_path"])
        model_profiles[name] = {"label": profile.get("label", name), "path": str(path), "exists": path.exists()}
    hardware = detect_hardware().to_dict()
    print(
        json.dumps(
            {"hardware": hardware, "model_path": str(model_path), "model_exists": model_path.exists(), "model_profiles": model_profiles},
            ensure_ascii=False,
            indent=2,
        )
    )
    if hardware.get("warning"):
        print(hardware["warning"])
    if not model_path.exists() or any(not item["exists"] for item in model_profiles.values()):
        print("Model missing. Run: python scripts/download_model.py")


if __name__ == "__main__":
    main()
