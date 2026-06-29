# Evaluation Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable, extensible evaluation profile system that tightens scoring for high-fidelity text types, exposes profile choice in both UIs, and keeps history/cache/reproducibility correct per profile.

**Architecture:** Keep a single evaluation pipeline centered on `app.engine.evaluate()`, then thread `evaluation_profile` through request/response models, prompt construction, post-processing, caching, and persistence. Add a focused `profiles.py` registry plus lightweight fact-loss detection helpers so strict profiles can override optimistic LLM scoring without forking the engine.

**Tech Stack:** Python 3, FastAPI, Pydantic v2, Streamlit, static HTML/CSS/JS frontend, LangChain OpenAI client, unittest/pytest-style offline tests

## Global Constraints

- All app commands run from `python_version/`.
- The project has two parallel implementations: FastAPI backend (`main.py` + `app/`) and Streamlit app (`app.py`); changes must be deliberate about which implementation they affect.
- The FastAPI backend is the production path and must support the new profile system end-to-end.
- The Streamlit app already imports and calls `app.engine.evaluate()` and must expose the same profile choices as the backend.
- Do not depend on live API calls for automated tests; add offline tests only.
- Default profile must remain `general` to avoid silent behavior changes for existing callers.
- Unknown profile handling must be explicit and consistent; use clear validation failure rather than silently falling back.
- Include `evaluation_profile` in reproducibility token, history persistence, and cache identity so cross-profile results are never reused incorrectly.

---

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

### Task 2: Make prompt building profile-aware

**Files:**
- Modify: `python_version/app/prompts.py:1-220`
- Test: `python_version/tests/test_evaluation_prompts.py`

**Interfaces:**
- Consumes:
  - `get_profile_config(profile_key: str) -> dict[str, object]`
  - `EvalRequest.evaluation_profile`
- Produces:
  - `build_system_prompt(profile_key: str = "general") -> str`
  - `build_user_prompt(..., evaluation_profile: str = "general") -> str`

- [ ] **Step 1: Write the failing tests**

```python
from app.prompts import build_system_prompt, build_user_prompt
from app.profiles import PROFILE_GENERAL, PROFILE_GOVERNMENT_NOTICE_STRICT, PROFILE_LEGAL_STRICT


def test_general_prompt_mentions_four_dimensions():
    prompt = build_system_prompt(PROFILE_GENERAL)
    assert "语义一致性" in prompt
    assert "过度清洗/误改识别" in prompt
    assert "可读性" in prompt
    assert "结构质量" in prompt


def test_government_prompt_mentions_key_fact_preservation():
    prompt = build_system_prompt(PROFILE_GOVERNMENT_NOTICE_STRICT)
    assert "数字、日期、罚则、期限、适用范围" in prompt
    assert "删除或泛化不能视为高保真" in prompt


def test_legal_prompt_mentions_obligations_and_liability():
    prompt = build_system_prompt(PROFILE_LEGAL_STRICT)
    assert "责任主体" in prompt
    assert "法律后果" in prompt


def test_user_prompt_includes_profile_name_for_traceability():
    prompt = build_user_prompt(
        before_text="原文",
        after_text="改写",
        evaluation_profile=PROFILE_GOVERNMENT_NOTICE_STRICT,
    )
    assert "当前评估模式：government_notice_strict" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest python_version/tests/test_evaluation_prompts.py -q`
Expected: FAIL because prompt builders do not accept a profile argument and do not include profile-specific rules.

- [ ] **Step 3: Write the minimal implementation**

```python
# python_version/app/prompts.py
from .profiles import get_profile_config

_BASE_SYSTEM_PROMPT = """你是一个专业的内容保真度与治理质量评估智能体（LLM-as-Judge）。
你的任务是：比对治理前后的文本，评估治理质量。

你必须严格按以下四个维度打分，并输出 JSON 格式结果：
1. **语义一致性**（权重 0.4）
2. **过度清洗/误改识别**（权重 0.3）
3. **可读性**（权重 0.15）
4. **结构质量**（权重 0.15）
"""


def build_system_prompt(profile_key: str = "general") -> str:
    from .debias import generate_anti_bias_prompt_supplement

    config = get_profile_config(profile_key)
    profile_block = (
        f"\n\n【当前评估模式】{config['key']}\n"
        f"【模式说明】{config['description']}\n"
        f"【模式规则】{config['prompt_rules']}\n"
    )
    return _BASE_SYSTEM_PROMPT + profile_block + "\n" + generate_anti_bias_prompt_supplement()
```

```python
# python_version/app/prompts.py
USER_PROMPT_TEMPLATE = """请评估以下治理前后文本对：

当前评估模式：{evaluation_profile}

=== 治理前原文（BEFORE） ===
{before_text}
...
"""


def build_user_prompt(
    before_text: str,
    after_text: str,
    segments_before: Optional[list[dict[str, Any]]] = None,
    segments_after: Optional[list[dict[str, Any]]] = None,
    evaluation_profile: str = "general",
) -> str:
    before_segments_info = _format_segments(segments_before, "before")
    after_segments_info = _format_segments(segments_after, "after")
    return _safe_format(
        USER_PROMPT_TEMPLATE,
        before_text=before_text,
        after_text=after_text,
        before_segments_info=before_segments_info,
        after_segments_info=after_segments_info,
        evaluation_profile=evaluation_profile,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest python_version/tests/test_evaluation_prompts.py -q`
Expected: PASS for all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add python_version/app/prompts.py python_version/tests/test_evaluation_prompts.py
git commit -m "feat: make prompts profile aware"
```

### Task 3: Add profile-aware fact-loss post-processing in the evaluation engine

**Files:**
- Modify: `python_version/app/engine.py:1-218`
- Test: `python_version/tests/test_evaluation_engine_profiles.py`

**Interfaces:**
- Consumes:
  - `build_system_prompt(profile_key: str = "general") -> str`
  - `build_user_prompt(..., evaluation_profile: str = "general") -> str`
  - `get_profile_config(profile_key: str) -> dict[str, object]`
- Produces:
  - `_extract_fact_markers(text: str) -> list[dict[str, object]]`
  - `_detect_missing_fact_flaws(before_text: str, after_text: str, profile_key: str) -> list[FlawItem]`
  - `_apply_profile_penalties(overall_score: float, flaws: list[FlawItem], profile_key: str) -> float`
  - `evaluate(request: EvalRequest, temperature: float = 0.0) -> EvalResponse` returning `evaluation_profile`

- [ ] **Step 1: Write the failing tests**

```python
from app.engine import _detect_missing_fact_flaws, _apply_profile_penalties
from app.models import FlawItem
from app.profiles import PROFILE_GENERAL, PROFILE_GOVERNMENT_NOTICE_STRICT


def test_detect_missing_fact_flaws_finds_deleted_dates_and_ranges():
    before = "登记费每只每年300元。罚款2000元以上5000元以下。本通告有效期至2027年5月31日。"
    after = "登记费每年300元，违规者处罚款。"
    flaws = _detect_missing_fact_flaws(before, after, PROFILE_GOVERNMENT_NOTICE_STRICT)
    descriptions = [f.description for f in flaws]
    assert any("2027年5月31日" in desc for desc in descriptions)
    assert any("2000元以上5000元以下" in desc for desc in descriptions)


def test_general_profile_keeps_existing_critical_caps():
    score = _apply_profile_penalties(0.92, [
        FlawItem(
            category="over_clean",
            severity="critical",
            description="严重过度清洗",
            location={"segment_id": "1", "start_char": 0, "end_char": 2, "snippet": "abc"},
        )
    ], PROFILE_GENERAL)
    assert score == 0.25


def test_government_profile_caps_to_review_on_major_fact_loss():
    score = _apply_profile_penalties(0.91, [
        FlawItem(
            category="over_clean",
            severity="major",
            description="删除有效期至2027年5月31日",
            location={"segment_id": "1", "start_char": 0, "end_char": 10, "snippet": "2027年5月31日"},
        )
    ], PROFILE_GOVERNMENT_NOTICE_STRICT)
    assert score <= 0.79
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest python_version/tests/test_evaluation_engine_profiles.py -q`
Expected: FAIL because helper functions do not exist and `evaluate()` is not profile-aware.

- [ ] **Step 3: Write the minimal implementation**

```python
# python_version/app/engine.py
_FACT_PATTERNS = [
    ("date", re.compile(r"\d{4}年\d{1,2}月\d{1,2}日")),
    ("percent", re.compile(r"\d+(?:\.\d+)?%")),
    ("deadline", re.compile(r"\d+(?:日|天|小时)内")),
    ("range", re.compile(r"\d+[^，。；]{0,10}(?:以上|以下)\d+[^，。；]{0,10}(?:以下|以上)")),
    ("number", re.compile(r"\d+(?:\.\d+)?(?:元|厘米|公斤|只)")),
]


def _extract_fact_markers(text: str) -> list[dict[str, object]]:
    markers = []
    for fact_type, pattern in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            markers.append(
                {
                    "fact_type": fact_type,
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return markers


def _detect_missing_fact_flaws(before_text: str, after_text: str, profile_key: str) -> list[FlawItem]:
    if profile_key == "general":
        return []

    flaws = []
    for marker in _extract_fact_markers(before_text):
        if marker["text"] in after_text:
            continue
        severity = "critical" if marker["fact_type"] in {"date", "range", "deadline"} else "major"
        flaws.append(
            FlawItem(
                category="over_clean",
                severity=severity,
                description=f"关键事实缺失：{marker['text']}",
                location=AnchorSpan(
                    segment_id="before-1",
                    start_char=int(marker["start"]),
                    end_char=int(marker["end"]),
                    snippet=str(marker["text"]),
                ),
                suggestion="保留原文中的关键事实或使用等价精确表达。",
            )
        )
    return flaws


def _apply_profile_penalties(overall_score: float, flaws: list[FlawItem], profile_key: str) -> float:
    has_critical_structure_flaw = any(f.category == "structure" and f.severity == "critical" for f in flaws)
    has_critical_over_clean = any(f.category == "over_clean" and f.severity == "critical" for f in flaws)
    has_critical_mis_edit = any(f.category == "mis_edit" and f.severity == "critical" for f in flaws)

    if has_critical_structure_flaw or has_critical_over_clean:
        overall_score = min(overall_score, 0.25)
    elif has_critical_mis_edit:
        overall_score = min(overall_score, 0.35)

    if profile_key != "general":
        has_major_fact_loss = any(f.category in {"over_clean", "mis_edit"} and f.severity == "major" for f in flaws)
        if has_major_fact_loss:
            overall_score = min(overall_score, 0.79)
    return _normalize_score(overall_score)
```

```python
# python_version/app/engine.py inside evaluate()
profile_key = validate_profile_key(request.evaluation_profile)
system_prompt = build_system_prompt(profile_key)
user_prompt = build_user_prompt(
    before_text=request.before_text,
    after_text=request.after_text,
    segments_before=request.segments_before,
    segments_after=request.segments_after,
    evaluation_profile=profile_key,
)
...
extra_flaws = _detect_missing_fact_flaws(request.before_text, request.after_text, profile_key)
for flaw in extra_flaws:
    if not any(existing.description == flaw.description for existing in flaws):
        flaws.append(flaw)
overall_score = _apply_profile_penalties(overall_score, flaws, profile_key)
...
return EvalResponse(
    request_id=request.request_id,
    evaluation_profile=profile_key,
    dimensions=dimensions,
    overall_score=round(overall_score, 4),
    flaws=flaws,
    verdict=verdict,
    reproducibility_token=_build_token(request, temperature),
    model_version=DEEPSEEK_MODEL,
    prompt_version=_prompt_version(profile_key),
    raw_llm_output=raw_output,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest python_version/tests/test_evaluation_engine_profiles.py -q`
Expected: PASS for all 3 tests.

- [ ] **Step 5: Commit**

```bash
git add python_version/app/engine.py python_version/tests/test_evaluation_engine_profiles.py
git commit -m "feat: add profile aware engine penalties"
```

### Task 4: Update reproducibility, history, cache, and routes to honor profiles

**Files:**
- Modify: `python_version/app/engine.py:45-67`
- Modify: `python_version/app/storage.py:20-139`
- Modify: `python_version/app/batch.py:20-176`
- Modify: `python_version/app/routes.py:26-145`
- Test: `python_version/tests/test_evaluation_persistence.py`

**Interfaces:**
- Consumes:
  - `EvalRequest.evaluation_profile`
  - `EvalResponse.evaluation_profile`
- Produces:
  - `_prompt_version(profile_key: str) -> str`
  - history records including `evaluation_profile`
  - cache keys including `evaluation_profile`
  - route handlers that preserve profile in normal, stability, calibration, and batch flows

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

from app.batch import _cache_key
from app.models import DimensionScore, EvalRequest, EvalResponse
from app.storage import save_evaluation


def test_cache_key_differs_by_profile():
    req_a = EvalRequest(request_id="1", before_text="a", after_text="b", evaluation_profile="general")
    req_b = EvalRequest(request_id="1", before_text="a", after_text="b", evaluation_profile="government_notice_strict")
    assert _cache_key(req_a) != _cache_key(req_b)


def test_save_evaluation_persists_profile(tmp_path, monkeypatch):
    history_path = tmp_path / "history.jsonl"
    monkeypatch.setattr("app.storage.HISTORY_PATH", history_path)
    response = EvalResponse(
        request_id="r1",
        evaluation_profile="government_notice_strict",
        dimensions=[DimensionScore(dimension="语义一致性", score=0.5, weight=0.4, reason="x")],
        overall_score=0.5,
        flaws=[],
        verdict="review",
        reproducibility_token="tok",
        model_version="m",
        prompt_version="p",
        raw_llm_output="{}",
    )
    save_evaluation(response)
    record = json.loads(history_path.read_text(encoding="utf-8").strip())
    assert record["evaluation_profile"] == "government_notice_strict"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest python_version/tests/test_evaluation_persistence.py -q`
Expected: FAIL because cache key and persisted history do not include profile.

- [ ] **Step 3: Write the minimal implementation**

```python
# python_version/app/engine.py
def _prompt_version(profile_key: str) -> str:
    from .prompts import build_system_prompt
    prompt = build_system_prompt(profile_key)
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]


def _build_token(request: EvalRequest, temperature: float) -> str:
    payload = json.dumps(
        {
            "before": request.before_text,
            "after": request.after_text,
            "temperature": temperature,
            "model": DEEPSEEK_MODEL,
            "evaluation_profile": request.evaluation_profile,
            "prompt_version": _prompt_version(request.evaluation_profile),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
```

```python
# python_version/app/storage.py
record = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "request_id": response.request_id,
    "evaluation_profile": getattr(response, "evaluation_profile", "general"),
    "overall_score": response.overall_score,
    ...
}
```

```python
# python_version/app/batch.py
payload = json.dumps(
    {
        "before": request.before_text,
        "after": request.after_text,
        "evaluation_profile": request.evaluation_profile,
    },
    sort_keys=True,
    ensure_ascii=False,
)
```

```python
# python_version/app/routes.py
@router.post("/evaluate", response_model=EvalResponse)
def evaluate_endpoint(request: EvalRequest) -> EvalResponse:
    ...

@router.post("/stability", response_model=StabilityReport)
def stability_endpoint(request: EvalRequest) -> StabilityReport:
    return run_stability(request, sample_count=request.sample_count)

@router.post("/batch/evaluate")
def batch_evaluate_endpoint(requests: list[EvalRequest], ... ) -> dict[str, Any]:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest python_version/tests/test_evaluation_persistence.py -q`
Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add python_version/app/engine.py python_version/app/storage.py python_version/app/batch.py python_version/app/routes.py python_version/tests/test_evaluation_persistence.py
git commit -m "feat: persist and cache evaluation profiles"
```

### Task 5: Expose evaluation profile selection in the Streamlit app

**Files:**
- Modify: `python_version/app.py:93-149`
- Modify: `python_version/app.py:193-249`
- Modify: `python_version/app.py:254-500`
- Test: none (manual verification only)

**Interfaces:**
- Consumes:
  - `PROFILE_OPTIONS: list[dict[str, str]]`
  - `EvalRequest.evaluation_profile`
- Produces:
  - `render_sidebar() -> tuple[str, str, str, str]`
  - profile-aware `_run_single(before: str, after: str, evaluation_profile: str) -> dict[str, Any]`

- [ ] **Step 1: Add profile options to the sidebar**

```python
from app.profiles import PROFILE_OPTIONS


def render_sidebar() -> tuple[str, str, str, str]:
    with st.sidebar:
        ...
        profile_labels = {item["label"]: item["key"] for item in PROFILE_OPTIONS}
        selected_label = st.selectbox(
            "评估模式",
            list(profile_labels.keys()),
            index=0,
            key="sidebar_profile",
        )
        evaluation_profile = profile_labels[selected_label]
        st.caption(next(item["description"] for item in PROFILE_OPTIONS if item["key"] == evaluation_profile))
    return api_key, base_url, model_name, evaluation_profile
```

- [ ] **Step 2: Thread the profile into single and batch evaluation requests**

```python
def _run_single(before: str, after: str, evaluation_profile: str) -> dict[str, Any]:
    from app.engine import evaluate
    from app.models import EvalRequest

    req = EvalRequest(
        request_id=f"ui_{uuid.uuid4().hex[:8]}",
        before_text=before,
        after_text=after,
        evaluation_profile=evaluation_profile,
    )
    return evaluate(req, temperature=0.0).model_dump()
```

```python
# inside page_single()
result = _run_single(before.strip(), after.strip(), evaluation_profile)
st.caption(f"当前评估模式：{result.get('evaluation_profile', evaluation_profile)}")
```

```python
# inside page_batch()
requests.append(EvalRequest(
    request_id=rid,
    before_text=str(row[before_col]),
    after_text=str(row[after_col]),
    evaluation_profile=evaluation_profile,
))
```

- [ ] **Step 3: Show the chosen profile in the result area**

```python
st.caption(f"本次评估模式：{result.get('evaluation_profile', evaluation_profile)}")
```

- [ ] **Step 4: Run a manual smoke check**

Run: `python app.py`
Expected: Streamlit starts, sidebar shows three profile options, and selecting a profile changes the request payload used by single and batch evaluation.

- [ ] **Step 5: Commit**

```bash
git add python_version/app.py
git commit -m "feat: add profile selector to streamlit app"
```

### Task 6: Expose evaluation profile selection in the static HTML frontend

**Files:**
- Modify: `python_version/static/index.html:447-507`
- Modify: `python_version/static/index.html:511-562`
- Modify: `python_version/static/index.html:681-720`
- Test: none (manual verification only)

**Interfaces:**
- Consumes: `/api/v1/evaluate` and `/api/v1/stability` accepting `evaluation_profile`
- Produces:
  - `getSelectedEvaluationProfile(selectId: string): string`
  - core evaluation and stability requests that include `evaluation_profile`

- [ ] **Step 1: Add profile selectors to the core evaluation and stability forms**

```html
<div class="form-group">
  <label>评估模式</label>
  <select id="evalProfile">
    <option value="general">通用文本</option>
    <option value="government_notice_strict">政务通告/公告（严格保真）</option>
    <option value="legal_strict">法规合同/条款（严格保真）</option>
  </select>
</div>
```

```html
<div class="form-group">
  <label>评估模式</label>
  <select id="stabProfile">
    <option value="general">通用文本</option>
    <option value="government_notice_strict">政务通告/公告（严格保真）</option>
    <option value="legal_strict">法规合同/条款（严格保真）</option>
  </select>
</div>
```

- [ ] **Step 2: Include profile in request payloads and result rendering**

```javascript
const evaluationProfile = document.getElementById('evalProfile').value;
...
body: JSON.stringify({
  request_id: requestId,
  before_text: before,
  after_text: after,
  evaluation_profile: evaluationProfile,
  stabilize: stabilize,
  sample_count: sampleCount,
})
```

```javascript
document.getElementById('evalRid').textContent = data.request_id;
document.getElementById('evalProfileUsed').textContent = data.evaluation_profile || 'general';
```

```javascript
const profile = document.getElementById('stabProfile').value;
body: JSON.stringify({
  request_id: 'stab-' + Date.now(),
  before_text: before,
  after_text: after,
  evaluation_profile: profile,
  stabilize: false,
  sample_count: sampleCount,
})
```

- [ ] **Step 3: Update the usage guide text**

```html
<li><strong>通用文本</strong>：用于普通治理文本与常规润色</li>
<li><strong>政务通告/公告（严格保真）</strong>：用于日期、期限、罚则、范围必须保留的文本</li>
<li><strong>法规合同/条款（严格保真）</strong>：用于义务、责任、条件、后果必须保留的文本</li>
```

- [ ] **Step 4: Run a manual frontend smoke check**

Run: `python main.py`
Expected: browser UI shows profile selectors, requests include `evaluation_profile`, and the result card displays the mode returned by the backend.

- [ ] **Step 5: Commit**

```bash
git add python_version/static/index.html
git commit -m "feat: add profile selector to static frontend"
```

### Task 7: Add offline regression tests for strict-profile behavior

**Files:**
- Create: `python_version/tests/test_evaluation_regressions.py`
- Test: `python_version/tests/test_evaluation_regressions.py`

**Interfaces:**
- Consumes:
  - `_detect_missing_fact_flaws(before_text: str, after_text: str, profile_key: str) -> list[FlawItem]`
  - `_apply_profile_penalties(overall_score: float, flaws: list[FlawItem], profile_key: str) -> float`
- Produces: stable regression coverage for the user’s dog-management notice case plus one normal-text control case

- [ ] **Step 1: Write the regression tests**

```python
from app.engine import _apply_profile_penalties, _detect_missing_fact_flaws
from app.profiles import PROFILE_GENERAL, PROFILE_GOVERNMENT_NOTICE_STRICT

DOG_NOTICE_BEFORE = """近年来，随着我市养犬市民数量持续增长，因养犬引发的噪音扰民、伤人事件、环境卫生等问题日益突出，市民投诉量较去年增长42.3%。
登记费用为每只每年300元。重点管理区域内，每户限养一只犬只，禁止饲养烈性犬和大型犬（肩高超过61厘米或体重超过30公斤）。
违反规定的，由公安机关没收犬只，并处以2000元以上5000元以下罚款。
本通告自2024年6月1日起施行，有效期至2027年5月31日。"""

DOG_NOTICE_AFTER = """近年来，我市养犬数量增长，因养犬引发的扰民、伤人、卫生等问题增多。
登记费每只每年300元。
重点管理区域内，每户限养一只，禁止饲养烈性犬和大型犬。违规者没收犬只并处罚款。
本通告自2024年6月1日起施行。"""


def test_government_notice_profile_flags_multiple_missing_facts():
    flaws = _detect_missing_fact_flaws(DOG_NOTICE_BEFORE, DOG_NOTICE_AFTER, PROFILE_GOVERNMENT_NOTICE_STRICT)
    snippets = {flaw.location.snippet for flaw in flaws}
    assert "42.3%" in snippets
    assert "61厘米" in snippets or "30公斤" in snippets
    assert "2000元以上5000元以下" in snippets
    assert "2027年5月31日" in snippets


def test_government_notice_profile_caps_high_score_after_fact_loss():
    flaws = _detect_missing_fact_flaws(DOG_NOTICE_BEFORE, DOG_NOTICE_AFTER, PROFILE_GOVERNMENT_NOTICE_STRICT)
    score = _apply_profile_penalties(0.885, flaws, PROFILE_GOVERNMENT_NOTICE_STRICT)
    assert score <= 0.35


def test_general_profile_does_not_add_strict_fact_loss_flaws():
    flaws = _detect_missing_fact_flaws("您的订单已发货。预计送达时间：2024年3月18日下午。", "您的订单已发货。预计送达时间：2024年3月18日下午。", PROFILE_GENERAL)
    assert flaws == []
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest python_version/tests/test_evaluation_regressions.py -q`
Expected: PASS for all 3 tests.

- [ ] **Step 3: Run the focused offline suite**

Run: `python -m pytest python_version/tests/test_evaluation_profiles.py python_version/tests/test_evaluation_prompts.py python_version/tests/test_evaluation_engine_profiles.py python_version/tests/test_evaluation_persistence.py python_version/tests/test_evaluation_regressions.py -q`
Expected: PASS for the new offline profile suite.

- [ ] **Step 4: Record the manual verification checklist**

```text
- Streamlit sidebar can switch among general / government_notice_strict / legal_strict
- Static HTML form can switch among the same three modes
- Returned JSON includes evaluation_profile
- Dog-management notice sample is no longer pass under strict profile
- Existing ordinary text sample still scores high under general profile
```

- [ ] **Step 5: Commit**

```bash
git add python_version/tests/test_evaluation_regressions.py
git commit -m "test: add strict profile regression coverage"
```

## Self-Review

- **Spec coverage:**
  - Data model changes are covered in Task 1.
  - Profile registry design is covered in Task 1.
  - Prompt-layer profile rules are covered in Task 2.
  - Engine post-processing and key-fact checks are covered in Task 3.
  - Reproducibility/history/cache/route propagation are covered in Task 4.
  - Streamlit UI is covered in Task 5.
  - Static HTML frontend is covered in Task 6.
  - Offline tests and the dog-notice regression sample are covered in Task 7.
- **Placeholder scan:** No `TODO`, `TBD`, or implicit “write tests later” steps remain; each code-changing task contains concrete snippets and concrete commands.
- **Type consistency:** The plan consistently uses `evaluation_profile: str`, `build_system_prompt(profile_key: str = "general")`, `build_user_prompt(..., evaluation_profile: str = "general")`, `get_profile_config(profile_key: str)`, and `_apply_profile_penalties(overall_score: float, flaws: list[FlawItem], profile_key: str)`.

Plan complete and saved to `docs/superpowers/plans/2026-06-29-evaluation-profile-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**