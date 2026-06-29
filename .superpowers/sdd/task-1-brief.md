### Task 1: Add shared evaluation profile registry and request/response fields

**Files:**
- Create: `python_version/app/profiles.py`
- Modify: `python_version/app/models.py:34-63`
- Test: `python_version/tests/test_evaluation_profiles.py`

**Interfaces:**
- Consumes: existing `EvalRequest` / `EvalResponse` models from `app.models`
- Produces:
  - `PROFILE_GENERAL = "general"`
  - `PROFILE_GOVERNMENT_NOTICE_STRICT = "government_notice_strict"`
  - `PROFILE_LEGAL_STRICT = "legal_strict"`
  - `PROFILE_OPTIONS: list[dict[str, str]]`
  - `get_profile_config(profile_key: str) -> dict[str, object]`
  - `validate_profile_key(profile_key: str) -> str`
  - `EvalRequest.evaluation_profile: str`
  - `EvalResponse.evaluation_profile: str`

- [ ] **Step 1: Write the failing tests**

```python
from app.models import EvalRequest
from app.profiles import (
    PROFILE_GENERAL,
    PROFILE_GOVERNMENT_NOTICE_STRICT,
    PROFILE_LEGAL_STRICT,
    PROFILE_OPTIONS,
    get_profile_config,
    validate_profile_key,
)


def test_eval_request_defaults_to_general_profile():
    req = EvalRequest(request_id="r1", before_text="a", after_text="b")
    assert req.evaluation_profile == PROFILE_GENERAL


def test_validate_profile_key_accepts_builtin_profiles():
    assert validate_profile_key(PROFILE_GENERAL) == PROFILE_GENERAL
    assert validate_profile_key(PROFILE_GOVERNMENT_NOTICE_STRICT) == PROFILE_GOVERNMENT_NOTICE_STRICT
    assert validate_profile_key(PROFILE_LEGAL_STRICT) == PROFILE_LEGAL_STRICT


def test_validate_profile_key_rejects_unknown_profile():
    try:
        validate_profile_key("unknown-profile")
    except ValueError as exc:
        assert "unknown-profile" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown profile")


def test_profile_registry_exposes_labels_for_ui():
    labels = {item["key"]: item["label"] for item in PROFILE_OPTIONS}
    assert labels[PROFILE_GENERAL] == "通用文本"
    assert labels[PROFILE_GOVERNMENT_NOTICE_STRICT] == "政务通告/公告（严格保真）"
    assert labels[PROFILE_LEGAL_STRICT] == "法规合同/条款（严格保真）"


def test_profile_config_contains_prompt_and_penalty_metadata():
    config = get_profile_config(PROFILE_GOVERNMENT_NOTICE_STRICT)
    assert config["key"] == PROFILE_GOVERNMENT_NOTICE_STRICT
    assert "prompt_rules" in config
    assert "critical_fact_types" in config
    assert "penalty_policy" in config
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest python_version/tests/test_evaluation_profiles.py -q`
Expected: FAIL with import errors for `app.profiles` and missing `evaluation_profile` field.

- [ ] **Step 3: Write the minimal implementation**

```python
# python_version/app/profiles.py
PROFILE_GENERAL = "general"
PROFILE_GOVERNMENT_NOTICE_STRICT = "government_notice_strict"
PROFILE_LEGAL_STRICT = "legal_strict"

_PROFILE_CONFIGS = {
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

PROFILE_OPTIONS = [
    {"key": key, "label": value["label"], "description": value["description"]}
    for key, value in _PROFILE_CONFIGS.items()
]


def validate_profile_key(profile_key: str) -> str:
    if profile_key not in _PROFILE_CONFIGS:
        raise ValueError(f"Unknown evaluation profile: {profile_key}")
    return profile_key


def get_profile_config(profile_key: str) -> dict[str, object]:
    return _PROFILE_CONFIGS[validate_profile_key(profile_key)]
```

```python
# python_version/app/models.py
class EvalRequest(BaseModel):
    request_id: str = Field(..., description="请求唯一标识")
    before_text: str = Field(..., description="治理前原文")
    after_text: str = Field(..., description="治理后文本")
    evaluation_profile: str = Field("general", description="评估模式")
    ...


class EvalResponse(BaseModel):
    request_id: str
    evaluation_profile: str = Field("general", description="本次评估使用的评估模式")
    dimensions: list[DimensionScore] = Field(..., description="各维度评分")
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest python_version/tests/test_evaluation_profiles.py -q`
Expected: PASS for all 5 tests.

- [ ] **Step 5: Commit**

```bash
git add python_version/app/profiles.py python_version/app/models.py python_version/tests/test_evaluation_profiles.py
git commit -m "feat: add evaluation profile registry"
```