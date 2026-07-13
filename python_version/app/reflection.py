"""Reflexion 自我反思修正模块

设计动机（扣题：LLM-as-Judge 的"裁判可信度"）：
  第一次评估可能出现"打分与证据自相矛盾"的情况，例如：
    - 检出了 critical 瑕疵，但 overall_score 仍然很高（verdict=pass）
    - 某维度 reason 明确写了"结构严重破坏"，该维度 score 却给了高分
  这类自相矛盾会削弱评估的可解释性与一致性。

  Reflexion 的做法：评估完成后做一次"自我批判"——
    1. 用确定性规则检测评分与证据之间的矛盾点（不额外调用 LLM）
    2. 若存在矛盾，把矛盾点作为反馈喂回 LLM，要求它重新打分（只重打维度分数）
    3. 取更自洽的结果，并记录完整反思轨迹（reflection trace）

  该模块是可开关的（EvalRequest.reflect），默认关闭，不影响既有主评分路径。
"""
from __future__ import annotations

import json
import os
from typing import Any

from .models import DimensionScore, EvalResponse, FlawItem


# 强否定词：出现在某维度 reason 中，通常意味着该维度存在严重问题
_NEGATIVE_CUES = [
    "严重", "完全丢失", "完全破坏", "彻底", "大量删除", "破坏",
    "篡改", "缺失", "丢失", "崩坏", "电报体", "拍平",
]

# 否定前缀：cue 前若紧跟这些词，说明是"没有/无该问题"，属正面表述，应忽略
_NEGATION_PREFIXES = ["没有", "无", "未", "不存在", "不", "非"]


def _cue_is_negated(reason: str, cue_pos: int) -> bool:
    """判断 cue 是否处于否定语境（如"没有篡改""无电报体""未破坏"）。

    检查 cue 前最多 4 个字符内是否出现否定前缀。
    """
    window_start = max(0, cue_pos - 4)
    prefix_window = reason[window_start:cue_pos]
    return any(neg in prefix_window for neg in _NEGATION_PREFIXES)


def _has_active_negative_cue(reason: str) -> str | None:
    """返回 reason 中第一个"非否定语境"的负面 cue，无则返回 None。"""
    for cue in _NEGATIVE_CUES:
        idx = reason.find(cue)
        while idx >= 0:
            if not _cue_is_negated(reason, idx):
                return cue
            idx = reason.find(cue, idx + 1)
    return None


def detect_contradictions(resp: EvalResponse) -> list[dict[str, Any]]:
    """确定性规则检测评分与证据之间的矛盾（不调用 LLM）。

    返回矛盾列表，每项含 type / detail，用于反馈给 LLM 重评。
    """
    contradictions: list[dict[str, Any]] = []

    critical_flaws = [f for f in resp.flaws if f.severity == "critical"]
    major_flaws = [f for f in resp.flaws if f.severity == "major"]

    # 矛盾 1：存在 critical 瑕疵，但总分偏高（verdict 仍可能 pass）
    if critical_flaws and resp.overall_score >= 0.7:
        contradictions.append({
            "type": "critical_flaw_high_score",
            "detail": (
                f"检出 {len(critical_flaws)} 条 critical 瑕疵，"
                f"但 overall_score={resp.overall_score:.3f} 偏高，两者不自洽。"
            ),
        })

    # 矛盾 2：verdict=pass 但存在 critical 瑕疵
    if resp.verdict == "pass" and critical_flaws:
        contradictions.append({
            "type": "pass_with_critical",
            "detail": f"判定为 pass，但存在 {len(critical_flaws)} 条 critical 瑕疵。",
        })

    # 矛盾 3：维度 reason 出现强否定词（非否定语境），但该维度分数却偏高
    for d in resp.dimensions:
        reason = d.reason or ""
        if d.score >= 0.7:
            hit = _has_active_negative_cue(reason)
            if hit:
                contradictions.append({
                    "type": "reason_score_mismatch",
                    "detail": (
                        f"维度 {d.dimension} 的理由含负面表述「{hit}」，"
                        f"但 score={d.score:.2f} 偏高，理由与分数不一致。"
                    ),
                    "dimension": d.dimension,
                })

    # 矛盾 4：瑕疵类别对应维度分数偏高（如 structure 瑕疵但 structure 维度高分）
    _cat_to_dim = {
        "structure": "structure",
        "readability": "readability",
        "over_clean": "semantic",
        "omission": "semantic",
        "mis_edit": "factual",
        "factual": "factual",
        "hallucination": "hallucination",
    }
    dim_map = {d.dimension: d.score for d in resp.dimensions}
    for f in critical_flaws + major_flaws:
        dim = _cat_to_dim.get(f.category)
        if dim and dim_map.get(dim, 0.0) >= 0.8:
            contradictions.append({
                "type": "flaw_dimension_mismatch",
                "detail": (
                    f"存在 {f.severity} 级 {f.category} 瑕疵，"
                    f"但对应维度 {dim} 的 score={dim_map[dim]:.2f} 偏高。"
                ),
                "dimension": dim,
            })

    return contradictions


def _build_reflection_feedback(
    contradictions: list[dict[str, Any]],
    resp: EvalResponse,
) -> str:
    """把矛盾点组织成给 LLM 的反思反馈文本。"""
    lines = ["你在上一轮评估中存在以下自相矛盾之处，请重新审视并修正维度评分：", ""]
    for i, c in enumerate(contradictions, 1):
        lines.append(f"{i}. {c['detail']}")
    lines.append("")
    lines.append("当前各维度评分：")
    for d in resp.dimensions:
        lines.append(f"  - {d.dimension}: {d.score:.2f}（理由：{(d.reason or '')[:60]}）")
    lines.append("")
    lines.append("当前检出瑕疵：")
    for f in resp.flaws:
        lines.append(f"  - [{f.severity}] {f.category}: {(f.description or '')[:60]}")
    lines.append("")
    lines.append(
        "请基于上述矛盾，重新给出更自洽的维度评分。"
        "原则：若确有 critical/major 瑕疵，对应维度分数必须相应降低，"
        "使评分与证据一致。严格输出如下 JSON，不要输出多余内容："
    )
    lines.append(
        '{"dimensions": [{"dimension": "semantic|factual|hallucination|structure|readability", '
        '"score": 0.0, "reason": "修正理由"}]}'
    )
    return "\n".join(lines)


def _reevaluate_dimensions(
    before_text: str,
    after_text: str,
    feedback: str,
    temperature: float,
) -> list[dict[str, Any]] | None:
    """让 LLM 基于反思反馈重新打维度分数。失败返回 None。"""
    from .chain import create_llm, parse_llm_output
    from langchain_core.messages import SystemMessage, HumanMessage

    system = (
        "你是一个严格的评估复核专家。你收到一份存在自相矛盾的评估结果，"
        "需要基于指出的矛盾重新给出自洽的维度评分。只输出 JSON。"
    )
    user = (
        f"=== 治理前原文 ===\n{before_text[:1500]}\n\n"
        f"=== 治理后文本 ===\n{after_text[:1500]}\n\n"
        f"{feedback}"
    )

    llm = create_llm(temperature=temperature, json_mode=True)
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        raw = str(response.content) if hasattr(response, "content") else str(response)
    except Exception:
        return None

    parsed = parse_llm_output(raw)
    dims = parsed.get("dimensions") if isinstance(parsed, dict) else None
    if not dims or not isinstance(dims, list):
        return None
    return dims


def reflect_and_correct(
    resp: EvalResponse,
    before_text: str,
    after_text: str,
    temperature: float = 0.0,
) -> EvalResponse:
    """Reflexion 主入口：检测矛盾 → 按需重评 → 返回更自洽的结果 + 反思轨迹。

    - 无矛盾：原样返回，reflection 记录 "no_contradiction"。
    - 有矛盾：喂回 LLM 重打维度分，重算总分与 verdict，记录前后变化。
    """
    from .chain import (
        _DIM_NAME_MAP,
        _normalize_score,
        compute_overall_score,
        determine_verdict,
        _apply_structural_penalty,
        _score_floor_from_flaws,
    )

    contradictions = detect_contradictions(resp)

    if not contradictions:
        resp.reflection = {
            "triggered": False,
            "status": "no_contradiction",
            "contradictions": [],
            "score_before": resp.overall_score,
            "score_after": resp.overall_score,
        }
        return resp

    feedback = _build_reflection_feedback(contradictions, resp)
    new_dims_raw = _reevaluate_dimensions(before_text, after_text, feedback, temperature)

    score_before = resp.overall_score
    verdict_before = resp.verdict

    if not new_dims_raw:
        # 重评失败：保留原结果，但记录检测到的矛盾（可解释性）
        resp.reflection = {
            "triggered": True,
            "status": "reeval_failed",
            "contradictions": contradictions,
            "score_before": score_before,
            "score_after": score_before,
        }
        return resp

    # 用重评后的维度分数覆盖（保留权重与原理由映射）
    weight_map = {d.dimension: d.weight for d in resp.dimensions}
    new_dimensions: list[DimensionScore] = []
    for d in new_dims_raw:
        if not isinstance(d, dict):
            continue
        canonical = _DIM_NAME_MAP.get(d.get("dimension", ""), d.get("dimension", ""))
        if not canonical:
            continue
        new_dimensions.append(DimensionScore(
            dimension=canonical,
            score=_normalize_score(d.get("score", 0.5)),
            weight=weight_map.get(canonical, 0.2),
            reason=str(d.get("reason", "")) or "（反思修正）",
        ))

    if not new_dimensions:
        resp.reflection = {
            "triggered": True,
            "status": "reeval_empty",
            "contradictions": contradictions,
            "score_before": score_before,
            "score_after": score_before,
        }
        return resp

    # 重算总分：沿用主路径的结构惩罚 + 瑕疵下压，保证与既有口径一致
    adjusted = _apply_structural_penalty(new_dimensions, resp.flaws)
    base = compute_overall_score(adjusted)
    floor = _score_floor_from_flaws(resp.flaws)
    new_score = round(min(base, floor), 4)
    new_verdict = determine_verdict(new_score)

    resp.dimensions = new_dimensions
    resp.overall_score = new_score
    resp.verdict = new_verdict
    resp.reflection = {
        "triggered": True,
        "status": "corrected",
        "contradictions": contradictions,
        "score_before": score_before,
        "score_after": new_score,
        "verdict_before": verdict_before,
        "verdict_after": new_verdict,
    }
    return resp
