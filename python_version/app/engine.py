"""核心评判引擎 —— LLM-as-Judge"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Optional, cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .models import (
    AnchorSpan,
    DimensionScore,
    EvalRequest,
    EvalResponse,
    FlawItem,
)
from .prompts import build_system_prompt, build_user_prompt
from .profiles import get_profile_config, PROFILE_GENERAL

# ---------- LLM 配置（可通过环境变量覆盖） ----------
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ---------- LLM 初始化 ----------
_llm: Optional[ChatOpenAI] = None


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """获取 LLM 实例（单例，低温度保证稳定性）"""
    global _llm
    if _llm is None or _llm.temperature != temperature:
        _llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            temperature=temperature,
            max_tokens=4096,
            base_url=DEEPSEEK_BASE_URL,
            api_key=DEEPSEEK_API_KEY,
        )
    return _llm


# ---------- prompt 版本指纹 ----------
def _prompt_version() -> str:
    """返回当前 system prompt 内容的 SHA256 前8位，作为 prompt 版本标识"""
    from .prompts import SYSTEM_PROMPT
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]


# ---------- 可复现令牌 ----------
def _build_token(request: EvalRequest, temperature: float) -> str:
    """基于输入 + 模型 + prompt + profile 哈希生成可复现令牌"""
    payload = json.dumps(
        {
            "before": request.before_text,
            "after": request.after_text,
            "temperature": temperature,
            "model": DEEPSEEK_MODEL,
            "prompt_version": _prompt_version(),
            "evaluation_profile": request.evaluation_profile,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------- 解析 LLM JSON 输出 ----------
def _parse_llm_json(raw: str) -> dict[str, Any]:
    """从 LLM 输出中提取 JSON 块（兼容 Markdown 代码块包裹）"""
    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 尝试提取 ```json ... ``` 块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试提取 { ... } 块（贪婪匹配最外层大括号）
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _normalize_score(val: float) -> float:
    """归一化分数到 0~1"""
    return max(0.0, min(1.0, float(val)))


# ---------- Profile-aware 关键事实缺失检查 ----------
def _extract_key_facts(text: str) -> list[dict[str, Any]]:
    """从文本中提取高风险事实线索（轻量规则，不做复杂 NLP）"""
    facts: list[dict[str, Any]] = []

    # 日期：2024年6月1日、2027-05-31 等
    for m in re.finditer(r"\d{4}[年\-/]\d{1,2}[月\-/]\d{1,2}[日]?", text):
        facts.append({"type": "date", "value": m.group(), "pos": m.start()})

    # 区间：2000元以上5000元以下、2000~5000元 等
    for m in re.finditer(r"\d+[\d,.]*\s*[元万亿]+\s*[以到至]\s*\d+[\d,.]*\s*[元万亿]+", text):
        facts.append({"type": "range", "value": m.group(), "pos": m.start()})

    # 带单位的数字（金额、百分比、阈值）
    for m in re.finditer(r"\d+[\d,.]*\s*(?:元|万元|亿元|%|％|厘米|cm|公斤|kg|日|天|小时|个月|年)", text):
        facts.append({"type": "number_with_unit", "value": m.group(), "pos": m.start()})

    # 时限：15日内、24小时内 等
    for m in re.finditer(r"\d+\s*(?:日内|日内|小时内|个工作日内|天内)", text):
        facts.append({"type": "deadline", "value": m.group(), "pos": m.start()})

    return facts


def _check_fact_preservation(
    before_text: str,
    after_text: str,
    critical_fact_types: list[str],
) -> list[dict[str, Any]]:
    """检查 before 中的关键事实是否在 after 中仍然存在"""
    facts = _extract_key_facts(before_text)
    missing: list[dict[str, Any]] = []

    for fact in facts:
        # 只检查 profile 关注的事实类型
        if fact["type"] not in critical_fact_types and fact["type"] != "number_with_unit":
            continue
        # 在 after_text 中搜索相同的数值/日期
        if str(fact["value"]) not in after_text:
            missing.append(fact)

    return missing


def _apply_profile_penalties(
    profile_key: str,
    overall_score: float,
    verdict: str,
    flaws: list[FlawItem],
    before_text: str,
    after_text: str,
) -> tuple[float, str, list[FlawItem]]:
    """根据 profile 配置执行后处理降分/封顶"""
    config = get_profile_config(profile_key)
    penalty_policy = cast(dict[str, Any], config.get("penalty_policy", {}))
    fact_types = [str(t) for t in cast(list[str], config.get("critical_fact_types", []))]

    # 对 general 模式，只保留原有 veto 逻辑，不做额外事实检查
    if profile_key == PROFILE_GENERAL:
        return overall_score, verdict, flaws

    missing = _check_fact_preservation(before_text, after_text, fact_types)
    if not missing:
        return overall_score, verdict, flaws

    # 追加 fact_missing flaw
    for fact in missing:
        severity = "major"
        # 如果是区间或关键日期缺失，升级为 critical
        if fact["type"] in ("range", "date"):
            severity = "critical"
        flaws.append(
            FlawItem(
                category="over_clean",
                severity=severity,
                description=f"关键事实缺失：原文中的「{fact['value']}」在改写后未保留",
                location=AnchorSpan(
                    segment_id="auto",
                    start_char=0,
                    end_char=0,
                    snippet=str(fact["value"]),
                ),
                suggestion=f"请保留原文中的关键数据「{fact['value']}」",
            )
        )

    # 统计严重程度
    critical_count = sum(1 for f in flaws if f.severity == "critical")
    major_count = sum(1 for f in flaws if f.severity == "major")

    # 分层惩罚
    if critical_count >= 2:
        # 多个 critical → 封顶到 fail_cap，倾向 fail
        cap = float(penalty_policy.get("fail_cap", 0.35))
        overall_score = min(overall_score, cap)
        verdict = "fail"
    elif critical_count >= 1:
        # 1 个 critical → 至少 review，封顶到 0.45
        overall_score = min(overall_score, 0.45)
        verdict = "review" if overall_score >= 0.35 else "fail"
    elif major_count >= 2:
        # 多个 major → 至少 review
        if penalty_policy.get("review_cap_on_major_fact_loss"):
            overall_score = min(overall_score, 0.55)
            verdict = "review"
    elif major_count >= 1:
        if penalty_policy.get("review_cap_on_major_fact_loss"):
            verdict = "review"

    return overall_score, verdict, flaws


# ---------- 核心评估 ----------
def evaluate(request: EvalRequest, temperature: float = 0.0) -> EvalResponse:
    """
    单次评估：
    1. 构建 System/User Prompt
    2. 调用 LLM
    3. 解析 JSON 输出
    4. 组装 EvalResponse
    """
    llm = get_llm(temperature=temperature)
    system_prompt = build_system_prompt(request.evaluation_profile)
    user_prompt = build_user_prompt(
        before_text=request.before_text,
        after_text=request.after_text,
        segments_before=request.segments_before,
        segments_after=request.segments_after,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    raw_output = str(response.content) if hasattr(response, "content") else str(response)

    parsed = _parse_llm_json(raw_output)

    # 解析维度评分
    dimensions = []
    dims_raw = parsed.get("dimensions", [])
    if isinstance(dims_raw, list):
        for d in dims_raw:
            if isinstance(d, dict):
                dimensions.append(
                    DimensionScore(
                        dimension=d.get("dimension", "未知"),
                        score=_normalize_score(d.get("score", 0)),
                        weight=float(d.get("weight", 0.25)),
                        reason=d.get("reason", ""),
                    )
                )

    # 若 LLM 未返回维度，使用默认兜底
    if not dimensions:
        overall = _normalize_score(parsed.get("overall_score", 0.5))
        dimensions = [
            DimensionScore(dimension="语义一致性", score=overall, weight=0.5, reason="LLM 未返回分维度得分，使用总分估计"),
            DimensionScore(dimension="可读性与结构", score=overall, weight=0.5, reason="LLM 未返回分维度得分，使用总分估计"),
        ]

    # 计算加权总分
    total_weight = sum(d.weight for d in dimensions) or 1.0
    overall_score = sum(d.score * d.weight for d in dimensions) / total_weight
    overall_score = _normalize_score(overall_score)

    # 解析瑕疵清单
    flaws = []
    flaws_raw = parsed.get("flaws", [])
    if isinstance(flaws_raw, list):
        for f in flaws_raw:
            if isinstance(f, dict):
                loc_raw = f.get("location") or f.get("anchor") or {}
                location = AnchorSpan(
                    segment_id=str(loc_raw.get("segment_id", "")),
                    start_char=int(loc_raw.get("start_char", 0)),
                    end_char=int(loc_raw.get("end_char", 0)),
                    snippet=str(loc_raw.get("snippet", "")),
                )
                flaws.append(
                    FlawItem(
                        category=str(f.get("category", "unknown")),
                        severity=str(f.get("severity", "minor")),
                        description=str(f.get("description", "")),
                        location=location,
                        suggestion=f.get("suggestion"),
                    )
                )

    # 【一票否决规则】检查是否有严重的结构破坏
    has_critical_structure_flaw = any(
        f.category == "structure" and f.severity == "critical"
        for f in flaws
    )
    has_critical_over_clean = any(
        f.category == "over_clean" and f.severity == "critical"
        for f in flaws
    )
    has_critical_mis_edit = any(
        f.category == "mis_edit" and f.severity == "critical"
        for f in flaws
    )

    # 如果有严重的结构破坏或过度清洗，强制降低整体得分
    if has_critical_structure_flaw or has_critical_over_clean:
        # 强制降低到0.3以下
        overall_score = min(overall_score, 0.25)
    elif has_critical_mis_edit:
        # 强制降低到0.4以下
        overall_score = min(overall_score, 0.35)

    # 综合判定
    if overall_score >= 0.8:
        verdict = "pass"
    elif overall_score >= 0.5:
        verdict = "review"
    else:
        verdict = "fail"

    # Profile-aware 后处理：关键事实缺失检查与降分
    overall_score, verdict, flaws = _apply_profile_penalties(
        profile_key=request.evaluation_profile,
        overall_score=overall_score,
        verdict=verdict,
        flaws=flaws,
        before_text=request.before_text,
        after_text=request.after_text,
    )

    return EvalResponse(
        request_id=request.request_id,
        evaluation_profile=request.evaluation_profile,
        dimensions=dimensions,
        overall_score=round(overall_score, 4),
        flaws=flaws,
        verdict=verdict,
        reproducibility_token=_build_token(request, temperature),
        model_version=DEEPSEEK_MODEL,
        prompt_version=_prompt_version(),
        raw_llm_output=raw_output,
    )