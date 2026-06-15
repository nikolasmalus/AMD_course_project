from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import analyze_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local security video analyzer on one MP4.")
    parser.add_argument("video", help="Path to local MP4 video")
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--model-profile", choices=["fast", "enhanced"], default=None)
    args = parser.parse_args()

    result = analyze_video(
        args.video,
        config_path=ROOT / "config.yaml",
        sample_fps=args.sample_fps,
        conf_threshold=args.conf,
        model_profile=args.model_profile,
    )
    print(json.dumps({"result_json_path": result["result_json_path"], "risk_level": result["risk_level"], "alert_text": result["alert_text"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
