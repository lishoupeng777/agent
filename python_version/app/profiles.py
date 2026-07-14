"""评测模式注册表 —— 松/中/紧三档

向后兼容旧模式名：
  general → medium
  government_notice_strict → strict
  legal_strict → strict
"""
from __future__ import annotations

# 新模式名
PROFILE_RELAXED = "relaxed"
PROFILE_MEDIUM = "medium"
PROFILE_STRICT = "strict"

# 向后兼容旧模式名
PROFILE_GENERAL = PROFILE_MEDIUM
PROFILE_GOVERNMENT_NOTICE_STRICT = PROFILE_STRICT
PROFILE_LEGAL_STRICT = PROFILE_STRICT

# 向后兼容映射
_COMPAT_MAP = {
    "general": "medium",
    "government_notice_strict": "strict",
    "legal_strict": "strict",
}

_PROFILE_CONFIGS: dict[str, dict[str, object]] = {
    "relaxed": {
        "key": "relaxed",
        "label": "宽松模式",
        "description": "适用于日常聊天、娱乐新闻、创意文案、口语交流。核心大意不变即合格。",
        "prompt_rules": "同义改写和精简不扣分，只看是否完全篡改原意或删掉核心数字/日期。",
        "critical_fact_types": ["number", "date"],
        "penalty_policy": {"review_cap_on_major_fact_loss": False, "fail_cap": 0.25},
    },
    "medium": {
        "key": "medium",
        "label": "普通模式",
        "description": "适用于说明文、日常业务邮件、常规工作汇报。允许删除废话，核心事实不能动。",
        "prompt_rules": "删除冗余不扣分，但不能改动核心时间、人物、数字、金额。约束词删除要扣分。",
        "critical_fact_types": ["number", "date", "range", "constraint"],
        "penalty_policy": {"review_cap_on_major_fact_loss": False, "fail_cap": 0.35},
    },
    "strict": {
        "key": "strict",
        "label": "严格模式",
        "description": "适用于法律合同、政务通告、财务报表、医疗处方。任何细节错误都视为严重瑕疵。",
        "prompt_rules": "错一个数字即为 critical，机构名必须法定全称，约束词删除一律 critical，定义条款逐字保留。",
        "critical_fact_types": ["number", "date", "range", "deadline", "scope", "penalty", "obligation", "liability"],
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
    """验证并归一化 profile key（向后兼容）"""
    # 先做兼容映射
    profile_key = _COMPAT_MAP.get(profile_key, profile_key)
    if profile_key not in _PROFILE_CONFIGS:
        raise ValueError(f"Unknown evaluation profile: {profile_key}")
    return profile_key


def get_profile_config(profile_key: str) -> dict[str, object]:
    """获取 profile 配置"""
    return _PROFILE_CONFIGS[validate_profile_key(profile_key)]
