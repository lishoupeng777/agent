"""Shared evaluation profile registry."""
from __future__ import annotations

PROFILE_GENERAL = "general"
PROFILE_GOVERNMENT_NOTICE_STRICT = "government_notice_strict"
PROFILE_LEGAL_STRICT = "legal_strict"

_PROFILE_CONFIGS: dict[str, dict[str, object]] = {
    PROFILE_GENERAL: {
        "key": PROFILE_GENERAL,
        "label": "通用文本",
        "description": "适用于普通说明文与常规治理文本",
        "prompt_rules": "保持现有四维度评分逻辑，对明显误改与结构破坏继续严惩。",
        "critical_fact_types": ["number", "date", "range"],
        "penalty_policy": {"review_cap_on_major_fact_loss": False, "fail_cap": 0.35},
    },
    PROFILE_GOVERNMENT_NOTICE_STRICT: {
        "key": PROFILE_GOVERNMENT_NOTICE_STRICT,
        "label": "政务通告/公告（严格保真）",
        "description": "关注日期、期限、罚则区间、适用范围等关键事实",
        "prompt_rules": "数字、日期、罚则、期限、适用范围、例外对象属于关键事实，删除或泛化不能视为高保真。",
        "critical_fact_types": ["number", "date", "range", "deadline", "scope", "penalty"],
        "penalty_policy": {"review_cap_on_major_fact_loss": True, "fail_cap": 0.35},
    },
    PROFILE_LEGAL_STRICT: {
        "key": PROFILE_LEGAL_STRICT,
        "label": "法规合同/条款（严格保真）",
        "description": "关注责任主体、义务、条件、例外与法律后果",
        "prompt_rules": "责任主体、义务、禁止、条件、例外、法律后果不能模糊化。",
        "critical_fact_types": ["number", "date", "range", "obligation", "liability"],
        "penalty_policy": {"review_cap_on_major_fact_loss": True, "fail_cap": 0.35},
    },
}

PROFILE_OPTIONS: list[dict[str, str]] = [
    {
        "key": key,
        "label": str(value["label"]),
        "description": str(value["description"]),
    }
    for key, value in _PROFILE_CONFIGS.items()
]


def validate_profile_key(profile_key: str) -> str:
    if profile_key not in _PROFILE_CONFIGS:
        raise ValueError(f"Unknown evaluation profile: {profile_key}")
    return profile_key


def get_profile_config(profile_key: str) -> dict[str, object]:
    return _PROFILE_CONFIGS[validate_profile_key(profile_key)]
