from __future__ import annotations

from typing import Any

from .local_llm_client import LocalLLMClient


RISK_CN = {"high": "高", "medium": "中", "low": "低", "none": "无"}
EVENT_CN = {
    "restricted_area_intrusion": "禁区闯入",
    "restricted_zone_intrusion": "禁区闯入",
    "loitering": "长时间徘徊",
}


def risk_level(events: list[dict[str, Any]]) -> str:
    severities = {event.get("risk_level") or event.get("severity") for event in events}
    if "high" in severities:
        return "高"
    if "medium" in severities:
        return "中"
    if events:
        return "低"
    return "无"


def _event_type(event: dict[str, Any]) -> str:
    return str(event.get("event_type") or event.get("type") or "unknown")


def _risk(event: dict[str, Any]) -> str:
    return str(event.get("risk_level") or event.get("severity") or "low")


def _duration(event: dict[str, Any]) -> float:
    if "duration" in event:
        return round(float(event["duration"]), 3)
    start = float(event.get("start_time", 0.0))
    end = float(event.get("end_time", start))
    return round(max(0.0, end - start), 3)


class AlertGenerator:
    def __init__(self, llm_config: dict[str, Any]) -> None:
        self.llm_config = llm_config
        self.enabled = bool(llm_config.get("enabled", True))
        self.provider = str(llm_config.get("provider", "ollama"))
        self.model_name = str(llm_config.get("model_name", llm_config.get("model", "qwen2.5:3b")))
        self.api_url = str(llm_config.get("api_url", llm_config.get("ollama_url", "http://127.0.0.1:11434/api/generate")))
        self.timeout_seconds = int(llm_config.get("timeout_seconds", 20))
        self.allow_external_api = bool(llm_config.get("allow_external_api", False))
        self.fallback_to_template = bool(llm_config.get("fallback_to_template", True))
        self._client: LocalLLMClient | None = None
        self._client_error: str | None = None

        if self.enabled:
            try:
                self._client = LocalLLMClient(
                    api_url=self.api_url,
                    model_name=self.model_name,
                    timeout_seconds=self.timeout_seconds,
                    allow_external_api=self.allow_external_api,
                )
            except Exception as exc:
                self._client_error = str(exc)

    def build_prompt(self, event: dict[str, Any]) -> str:
        event_type = _event_type(event)
        risk = _risk(event)
        start_time = round(float(event.get("start_time", 0.0)), 1)
        end_time = round(float(event.get("end_time", start_time)), 1)
        duration = round(_duration(event), 1)
        track_id = event.get("track_id", "-")
        return f"""你是一个本地安防视频告警助手。请根据结构化事件信息生成一句中文告警说明。

要求：
1. 不超过 60 个汉字。
2. 必须包含风险等级、时间范围、事件类型、持续时长和处理建议。
3. 不要编造身份信息、人名、地点名。
4. 不要输出多余解释。
5. 只输出一句话。

事件信息：
事件类型：{EVENT_CN.get(event_type, event_type)}
风险等级：{RISK_CN.get(risk, risk)}
开始时间：{start_time} 秒
结束时间：{end_time} 秒
持续时长：{duration} 秒
轨迹编号：{track_id}

请优先使用这个句式：
{RISK_CN.get(risk, risk)}风险告警：{start_time}秒至{end_time}秒检测到{EVENT_CN.get(event_type, event_type)}，持续约{duration}秒，建议立即查看片段。"""

    def template_alert(self, event: dict[str, Any]) -> str:
        event_type = _event_type(event)
        start_time = round(float(event.get("start_time", 0.0)), 1)
        end_time = round(float(event.get("end_time", start_time)), 1)
        duration = round(_duration(event), 1)
        if event_type in {"restricted_area_intrusion", "restricted_zone_intrusion"}:
            return f"高风险告警：{start_time}秒至{end_time}秒检测到人员进入禁区，持续约{duration}秒，建议立即查看片段。"
        if event_type == "loitering":
            return f"中风险告警：{start_time}秒至{end_time}秒检测到人员长时间徘徊，持续约{duration}秒，建议关注现场情况。"
        return f"告警：{start_time}秒至{end_time}秒检测到异常事件，建议查看视频片段。"

    def _is_valid_alert(self, text: str, event: dict[str, Any]) -> bool:
        event_type = _event_type(event)
        risk = _risk(event)
        start_time = str(round(float(event.get("start_time", 0.0)), 1))
        end_time = str(round(float(event.get("end_time", event.get("start_time", 0.0))), 1))
        duration = str(round(_duration(event), 1))
        event_name = EVENT_CN.get(event_type, event_type)
        risk_name = RISK_CN.get(risk, risk)
        return all(token in text for token in (risk_name, start_time, end_time, duration)) and event_name in text and "建议" in text

    def generate(self, event: dict[str, Any]) -> dict[str, Any]:
        updated = dict(event)
        updated["event_type"] = _event_type(updated)
        updated["risk_level"] = _risk(updated)
        updated["duration"] = _duration(updated)

        fallback_reason: str | None = None
        text = ""
        metadata: dict[str, Any] = {
            "llm_used": False,
            "provider": self.provider,
            "model_name": self.model_name,
            "local_only": True,
            "error": None,
        }

        if not self.enabled:
            fallback_reason = "LLM disabled in config."
        elif self.provider != "ollama":
            fallback_reason = f"Unsupported LLM provider: {self.provider}"
        elif self._client_error:
            fallback_reason = self._client_error
        elif self._client is None:
            fallback_reason = "Local LLM client is unavailable."
        else:
            text, metadata = self._client.generate(self.build_prompt(updated))
            if not text:
                fallback_reason = str(metadata.get("error") or "LLM returned no text.")
            elif not self._is_valid_alert(text, updated):
                fallback_reason = "LLM response did not satisfy required alert fields."
                text = ""

        if not text:
            text = self.template_alert(updated) if self.fallback_to_template else ""

        updated["alert_text"] = text
        updated["llm_used"] = bool(metadata.get("llm_used") and not fallback_reason)
        updated["llm_provider"] = metadata.get("provider", self.provider)
        updated["llm_model"] = metadata.get("model_name", self.model_name)
        updated["llm_fallback_reason"] = fallback_reason
        return updated

    def generate_for_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.generate(event) for event in events]

    def summarize_llm(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        fallback_reasons = sorted({event.get("llm_fallback_reason") for event in events if event.get("llm_fallback_reason")})
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model_name": self.model_name,
            "local_only": not self.allow_external_api,
            "llm_used": any(bool(event.get("llm_used")) for event in events),
            "fallback_count": sum(1 for event in events if event.get("llm_fallback_reason")),
            "fallback_reasons": fallback_reasons,
        }


def summarize_alert_text(events: list[dict[str, Any]]) -> str:
    if not events:
        return "未检测到明显异常事件。"
    first_alerts = [str(event.get("alert_text", "")).strip() for event in events if event.get("alert_text")]
    if first_alerts:
        return first_alerts[0] if len(first_alerts) == 1 else "；".join(first_alerts[:3])
    counts: dict[str, int] = {}
    for event in events:
        name = event.get("type_cn") or EVENT_CN.get(_event_type(event), "异常")
        counts[name] = counts.get(name, 0) + 1
    summary = "，".join(f"{name}{count}起" for name, count in counts.items())
    return f"检测到{summary}，最高风险等级为{risk_level(events)}，请及时复核视频片段。"
