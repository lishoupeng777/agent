"""LCEL 评估链 —— 用 LangChain Expression Language 重构评估流程

将原来的单体 evaluate() 函数拆分为可组合的 Chain 步骤：
  prompt → llm → parse → post_process → response

每一步独立可测、可替换、可追踪。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Optional, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .models import (
    AnchorSpan,
    DimensionScore,
    EvalRequest,
    EvalResponse,
    FlawItem,
)
from .profiles import get_profile_config, PROFILE_GENERAL


# ============================================================
# 1. LLM 工厂
# ============================================================

_llm_instance: Optional[ChatOpenAI] = None


def create_llm(temperature: float = 0.0, json_mode: bool = False) -> ChatOpenAI:
    """创建 LLM 实例（延迟读取环境变量）

    Args:
        temperature: 温度参数
        json_mode: 是否启用 JSON mode（强制输出合法 JSON）
    """
    global _llm_instance
    if _llm_instance is not None and _llm_instance.temperature == temperature:
        return _llm_instance
    kwargs: dict[str, Any] = {}
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    _llm_instance = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        temperature=temperature,
        max_tokens=4096,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        **kwargs,
    )
    return _llm_instance


# ============================================================
# 2. Prompt 构建（ChatPromptTemplate）
# ============================================================

def build_prompt_messages(
    profile_key: str,
    before_text: str,
    after_text: str,
    segments_before: list[dict[str, Any]] | None = None,
    segments_after: list[dict[str, Any]] | None = None,
) -> list[BaseMessage]:
    """根据 profile 和输入构建消息列表"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from .prompts import build_system_prompt, build_user_prompt

    system_content = build_system_prompt(profile_key)
    user_content = build_user_prompt(
        before_text=before_text,
        after_text=after_text,
        segments_before=segments_before,
        segments_after=segments_after,
    )
    return [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]


# ============================================================
# 3. 输出解析（Pydantic 验证 + 结构化提取）
# ============================================================

from pydantic import BaseModel, Field


class LLMDimensionScore(BaseModel):
    """LLM 输出的维度评分"""
    dimension: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class LLMAnchorSpan(BaseModel):
    """LLM 输出的锚点信息"""
    segment_id: str = ""
    start_char: int = 0
    end_char: int = 0
    snippet: str = ""


class LLMFlawItem(BaseModel):
    """LLM 输出的瑕疵项"""
    category: str
    severity: str = "minor"
    description: str = ""
    location: LLMAnchorSpan = LLMAnchorSpan()
    suggestion: str | None = None


class LLMOutput(BaseModel):
    """LLM 输出的完整结构（Pydantic 验证）"""
    dimensions: list[LLMDimensionScore] = Field(default_factory=list)
    overall_score: float = Field(default=0.5, ge=0.0, le=1.0)
    flaws: list[LLMFlawItem] = Field(default_factory=list)


def parse_llm_output(raw: str) -> dict[str, Any]:
    """从 LLM 原始输出中提取 JSON（三级降级策略 + Pydantic 验证）"""
    # 1) 直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 2) 提取 ```json ... ``` 块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3) 提取最外层 { ... }
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def validate_llm_output(parsed: dict[str, Any]) -> LLMOutput:
    """用 Pydantic 验证 LLM 输出结构，容错处理异常字段"""
    try:
        return LLMOutput(**parsed)
    except Exception:
        # 验证失败时返回默认值
        return LLMOutput()


def _normalize_score(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def extract_dimensions(parsed: dict[str, Any]) -> list[DimensionScore]:
    """从解析结果中提取维度评分"""
    dimensions = []
    for d in parsed.get("dimensions", []):
        if isinstance(d, dict):
            dimensions.append(DimensionScore(
                dimension=d.get("dimension", "未知"),
                score=_normalize_score(d.get("score", 0)),
                weight=float(d.get("weight", 0.25)),
                reason=d.get("reason", ""),
            ))
    if not dimensions:
        overall = _normalize_score(parsed.get("overall_score", 0.5))
        dimensions = [
            DimensionScore(dimension="语义一致性", score=overall, weight=0.5,
                          reason="LLM 未返回分维度得分，使用总分估计"),
            DimensionScore(dimension="可读性与结构", score=overall, weight=0.5,
                          reason="LLM 未返回分维度得分，使用总分估计"),
        ]
    return dimensions


def extract_flaws(parsed: dict[str, Any]) -> list[FlawItem]:
    """从解析结果中提取瑕疵清单"""
    flaws = []
    for f in parsed.get("flaws", []):
        if isinstance(f, dict):
            loc_raw = f.get("location") or f.get("anchor") or {}
            flaws.append(FlawItem(
                category=str(f.get("category", "unknown")),
                severity=str(f.get("severity", "minor")),
                description=str(f.get("description", "")),
                location=AnchorSpan(
                    segment_id=str(loc_raw.get("segment_id", "")),
                    start_char=int(loc_raw.get("start_char", 0)),
                    end_char=int(loc_raw.get("end_char", 0)),
                    snippet=str(loc_raw.get("snippet", "")),
                ),
                suggestion=f.get("suggestion"),
            ))
    return flaws


def compute_overall_score(dimensions: list[DimensionScore]) -> float:
    """计算加权总分"""
    total_weight = sum(d.weight for d in dimensions) or 1.0
    return _normalize_score(sum(d.score * d.weight for d in dimensions) / total_weight)


# ============================================================
# 4. 后处理（veto 规则 + profile 惩罚）
# ============================================================

def apply_veto_rules(
    overall_score: float,
    flaws: list[FlawItem],
) -> float:
    """一票否决规则"""
    has_critical_structure = any(f.category == "structure" and f.severity == "critical" for f in flaws)
    has_critical_over_clean = any(f.category == "over_clean" and f.severity == "critical" for f in flaws)
    has_critical_mis_edit = any(f.category == "mis_edit" and f.severity == "critical" for f in flaws)

    if has_critical_structure or has_critical_over_clean:
        overall_score = min(overall_score, 0.25)
    elif has_critical_mis_edit:
        overall_score = min(overall_score, 0.35)

    return overall_score


def determine_verdict(overall_score: float) -> str:
    """根据总分判定结果"""
    if overall_score >= 0.8:
        return "pass"
    elif overall_score >= 0.5:
        return "review"
    return "fail"


def apply_profile_penalties(
    profile_key: str,
    overall_score: float,
    verdict: str,
    flaws: list[FlawItem],
    before_text: str,
    after_text: str,
) -> tuple[float, str, list[FlawItem]]:
    """Profile-aware 关键事实缺失检查与降分"""
    config = get_profile_config(profile_key)
    penalty_policy = cast(dict[str, Any], config.get("penalty_policy", {}))
    fact_types = [str(t) for t in cast(list[str], config.get("critical_fact_types", []))]

    if profile_key == PROFILE_GENERAL:
        return overall_score, verdict, flaws

    # 提取关键事实
    missing = []
    facts = _extract_key_facts(before_text)
    for fact in facts:
        if fact["type"] not in fact_types and fact["type"] != "number_with_unit":
            continue
        if str(fact["value"]) not in after_text:
            missing.append(fact)

    if not missing:
        return overall_score, verdict, flaws

    # 追加瑕疵
    for fact in missing:
        severity = "critical" if fact["type"] in ("range", "date") else "major"
        flaws.append(FlawItem(
            category="over_clean",
            severity=severity,
            description=f"关键事实缺失：原文中的「{fact['value']}」在改写后未保留",
            location=AnchorSpan(segment_id="auto", start_char=0, end_char=0,
                               snippet=str(fact["value"])),
            suggestion=f"请保留原文中的关键数据「{fact['value']}」",
        ))

    # 分层惩罚
    critical_count = sum(1 for f in flaws if f.severity == "critical")
    major_count = sum(1 for f in flaws if f.severity == "major")

    if critical_count >= 2:
        cap = float(penalty_policy.get("fail_cap", 0.35))
        overall_score = min(overall_score, cap)
        verdict = "fail"
    elif critical_count >= 1:
        overall_score = min(overall_score, 0.45)
        verdict = "review" if overall_score >= 0.35 else "fail"
    elif major_count >= 2:
        if penalty_policy.get("review_cap_on_major_fact_loss"):
            overall_score = min(overall_score, 0.55)
            verdict = "review"
    elif major_count >= 1:
        if penalty_policy.get("review_cap_on_major_fact_loss"):
            verdict = "review"

    return overall_score, verdict, flaws


def _extract_key_facts(text: str) -> list[dict[str, Any]]:
    """从文本中提取高风险事实线索"""
    facts: list[dict[str, Any]] = []
    for m in re.finditer(r"\d{4}[年\-/]\d{1,2}[月\-/]\d{1,2}[日]?", text):
        facts.append({"type": "date", "value": m.group(), "pos": m.start()})
    for m in re.finditer(r"\d+[\d,.]*\s*[元万亿]+\s*[以到至]\s*\d+[\d,.]*\s*[元万亿]+", text):
        facts.append({"type": "range", "value": m.group(), "pos": m.start()})
    for m in re.finditer(r"\d+[\d,.]*\s*(?:元|万元|亿元|%|％|厘米|cm|公斤|kg|日|天|小时|个月|年)", text):
        facts.append({"type": "number_with_unit", "value": m.group(), "pos": m.start()})
    for m in re.finditer(r"\d+\s*(?:日内|小时内|个工作日内|天内)", text):
        facts.append({"type": "deadline", "value": m.group(), "pos": m.start()})
    return facts


# ============================================================
# 5. 可复现令牌
# ============================================================

def build_reproducibility_token(request: EvalRequest, temperature: float) -> str:
    """生成可复现令牌"""
    from .prompts import SYSTEM_PROMPT
    payload = json.dumps({
        "before": request.before_text,
        "after": request.after_text,
        "temperature": temperature,
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "prompt_version": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8],
        "evaluation_profile": request.evaluation_profile,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ============================================================
# 6. Callback：token / 延迟追踪
# ============================================================

class EvalCallbackHandler(BaseCallbackHandler):
    """评估过程追踪回调"""

    def __init__(self) -> None:
        super().__init__()
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_tokens: int = 0
        self.model_name: str = ""

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self.start_time = time.time()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.end_time = time.time()
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.input_tokens = usage.get("prompt_tokens", 0)
            self.output_tokens = usage.get("completion_tokens", 0)
            self.total_tokens = usage.get("total_tokens", 0)

    @property
    def latency_seconds(self) -> float:
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 3)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


# ============================================================
# 7. 主评估函数（LCEL Chain 流程）
# ============================================================

def evaluate(request: EvalRequest, temperature: float = 0.0) -> EvalResponse:
    """
    LCEL 风格的评估流程：
      build_prompt → llm.invoke → parse → post_process → response

    保留与原 evaluate() 相同的接口签名，确保向后兼容。
    """
    callback = EvalCallbackHandler()

    # Step 1: 构建 prompt
    messages = build_prompt_messages(
        profile_key=request.evaluation_profile,
        before_text=request.before_text,
        after_text=request.after_text,
        segments_before=request.segments_before,
        segments_after=request.segments_after,
    )

    # Step 2: 调用 LLM（带 callback，启用 JSON mode）
    llm = create_llm(temperature=temperature, json_mode=True)
    response = llm.invoke(messages, config={"callbacks": [callback]})
    raw_output = str(response.content) if hasattr(response, "content") else str(response)

    # Step 3: 解析输出 + Pydantic 验证
    parsed = parse_llm_output(raw_output)
    validated = validate_llm_output(parsed)
    # 用验证后的数据（Pydantic 已做类型/范围校验）
    dimensions = extract_dimensions(validated.model_dump())
    flaws = extract_flaws(validated.model_dump())
    overall_score = compute_overall_score(dimensions)

    # Step 4: 后处理
    overall_score = apply_veto_rules(overall_score, flaws)
    verdict = determine_verdict(overall_score)
    overall_score, verdict, flaws = apply_profile_penalties(
        request.evaluation_profile, overall_score, verdict, flaws,
        request.before_text, request.after_text,
    )

    # Step 5: 组装响应（含 callback 追踪数据）
    return EvalResponse(
        request_id=request.request_id,
        evaluation_profile=request.evaluation_profile,
        dimensions=dimensions,
        overall_score=round(overall_score, 4),
        flaws=flaws,
        verdict=verdict,
        reproducibility_token=build_reproducibility_token(request, temperature),
        model_version=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        prompt_version=hashlib.sha256(
            __import__("app.prompts", fromlist=["SYSTEM_PROMPT"]).SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()[:8],
        raw_llm_output=raw_output,
        latency_seconds=callback.latency_seconds,
        input_tokens=callback.input_tokens,
        output_tokens=callback.output_tokens,
        total_tokens=callback.total_tokens,
    )
