from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alert_generator import AlertGenerator
from src.config import load_config
from src.local_llm_client import LocalLLMClient


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    llm_config = config["llm"]
    api_url = str(llm_config.get("api_url", "http://127.0.0.1:11434/api/generate"))
    model_name = str(llm_config.get("model_name", "qwen2.5:3b"))
    timeout_seconds = int(llm_config.get("timeout_seconds", 20))
    allow_external_api = bool(llm_config.get("allow_external_api", False))

    available = False
    reason = None
    try:
        client = LocalLLMClient(
            api_url=api_url,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            allow_external_api=allow_external_api,
        )
        available, reason = client.is_available()
    except Exception as exc:
        reason = str(exc)

    event = {
        "event_type": "restricted_area_intrusion",
        "risk_level": "high",
        "start_time": 12.5,
        "end_time": 16.0,
        "duration": 3.5,
        "track_id": 2,
    }
    generated = AlertGenerator(llm_config).generate(event)
    metadata = {
        "llm_used": generated.get("llm_used"),
        "provider": generated.get("llm_provider"),
        "model_name": generated.get("llm_model"),
        "local_only": not allow_external_api,
        "error": generated.get("llm_fallback_reason"),
    }

    print(f"Ollama API 是否可用: {'是' if available else '否'}")
    if reason:
        print(f"失败原因: {reason}")
    print(f"返回的告警文本: {generated.get('alert_text')}")
    print("metadata:")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
