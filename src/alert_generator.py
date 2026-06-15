from __future__ import annotations

from typing import Any


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
    """Generate deterministic template alerts for detected security events."""

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

    def generate(self, event: dict[str, Any]) -> dict[str, Any]:
        updated = dict(event)
        updated["event_type"] = _event_type(updated)
        updated["risk_level"] = _risk(updated)
        updated["duration"] = _duration(updated)
        updated["alert_text"] = self.template_alert(updated)
        return updated

    def generate_for_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.generate(event) for event in events]


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
