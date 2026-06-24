"""核心评判引擎 —— LLM-as-Judge"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Optional

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


# ---------- 可复现令牌 ----------
def _build_token(request: EvalRequest, temperature: float) -> str:
    """基于输入哈希生成可复现令牌"""
    payload = json.dumps(
        {
            "before": request.before_text,
            "after": request.after_text,
            "temperature": temperature,
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
    system_prompt = build_system_prompt()
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
    raw_output = response.content if hasattr(response, "content") else str(response)

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

    return EvalResponse(
        request_id=request.request_id,
        dimensions=dimensions,
        overall_score=round(overall_score, 4),
        flaws=flaws,
        verdict=verdict,
        reproducibility_token=_build_token(request, temperature),
        raw_llm_output=raw_output,
    )