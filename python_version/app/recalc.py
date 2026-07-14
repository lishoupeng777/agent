"""热重算引擎 —— 基于 LLM 原始输出的纯数学重算

核心思想：
  LLM 输出的原始数据（维度分数、瑕玼列表）是固定的。
  所有后续计算（加权、惩罚、判定、F1、锚点）都是纯数学运算。
  用户调参数时，不需要重新调 LLM，直接在原始数据上重算。

典型场景：
  用户拖动滑块把 semantic 权重从 0.35 改成 0.4 → 毫秒级重算出新的 overall_score
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .raw_store import load_all_raw


@dataclass
class RecalcParams:
    """重算参数（用户可调）"""
    # 维度权重（自动归一化，与当前 Prompt 五维度一致）
    dimension_weights: dict[str, float] = field(default_factory=lambda: {
        "semantic": 0.30,
        "factual": 0.30,
        "hallucination": 0.20,
        "structure": 0.10,
        "readability": 0.10,
    })
    # 惩罚系数（severity -> 惩罚因子，越小扣分越多）
    penalty_factors: dict[str, float] = field(default_factory=lambda: {
        "critical": 0.6,
        "major": 0.85,
        "minor": 0.95,
    })
    # 判定阈值
    pass_threshold: float = 0.82
    review_threshold: float = 0.5
    # 锚点容差（字符数）
    anchor_tolerance: int = 10


@dataclass
class RecalcResult:
    """单条重算结果"""
    request_id: str
    dimensions: list[dict[str, Any]]
    overall_score: float
    verdict: str
    flaws: list[dict[str, Any]]
    penalty_applied: float


@dataclass
class RecalcReport:
    """批量重算报告"""
    params: RecalcParams
    results: list[RecalcResult]
    # 校准指标（如果有人工标注）
    pearson_r: float | None = None
    spearman_rho: float | None = None
    mae: float | None = None
    rmse: float | None = None
    consistency_rate: float | None = None


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """权重归一化（确保总和为 1）"""
    total = sum(weights.values())
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: v / total for k, v in weights.items()}


def _compute_weighted_score(
    dimensions: list[dict[str, Any]],
    weights: dict[str, float],
) -> float:
    """加权计算总分"""
    score = 0.0
    for d in dimensions:
        dim_name = d.get("dimension", "")
        dim_score = d.get("score", 0.5)
        w = weights.get(dim_name, 0.25)
        score += dim_score * w
    return max(0.0, min(1.0, score))


def _apply_penalty(
    base_score: float,
    flaws: list[dict[str, Any]],
    penalty_factors: dict[str, float],
) -> tuple[float, float]:
    """应用惩罚（取最差惩罚，非累乘）

    Returns:
        (最终分数, 实际惩罚因子)
    """
    worst_penalty = 1.0
    for f in flaws:
        severity = f.get("severity", "minor")
        pf = penalty_factors.get(severity, 0.95)
        worst_penalty = min(worst_penalty, pf)
    return max(0.0, min(1.0, base_score * worst_penalty)), worst_penalty


def _determine_verdict(score: float, pass_threshold: float, review_threshold: float) -> str:
    """判定结果"""
    if score >= pass_threshold:
        return "pass"
    elif score >= review_threshold:
        return "review"
    return "fail"


def recalculate_single(
    raw_dimensions: list[dict[str, Any]],
    raw_flaws: list[dict[str, Any]],
    params: RecalcParams,
) -> RecalcResult:
    """对单条原始数据重算。

    Args:
        raw_dimensions: LLM 原始维度评分
        raw_flaws: LLM 原始瑕玼列表
        params: 重算参数

    Returns:
        RecalcResult
    """
    weights = _normalize_weights(params.dimension_weights)

    # 重算各维度加权分
    overall = _compute_weighted_score(raw_dimensions, weights)

    # 应用惩罚
    overall, penalty = _apply_penalty(overall, raw_flaws, params.penalty_factors)

    # 判定
    verdict = _determine_verdict(overall, params.pass_threshold, params.review_threshold)

    return RecalcResult(
        request_id="",
        dimensions=raw_dimensions,
        overall_score=round(overall, 4),
        verdict=verdict,
        flaws=raw_flaws,
        penalty_applied=penalty,
    )


def recalculate_batch(
    params: RecalcParams,
    human_scores: dict[str, float] | None = None,
) -> RecalcReport:
    """批量重算所有已存储的原始数据。

    Args:
        params: 重算参数
        human_scores: 人工评分 {request_id: score}，用于计算校准指标

    Returns:
        RecalcReport
    """
    raw_records = load_all_raw()
    results = []

    for record in raw_records:
        raw_dims = record.get("raw_dimensions", [])
        raw_flaws = record.get("raw_flaws", [])
        result = recalculate_single(raw_dims, raw_flaws, params)
        result.request_id = record.get("request_id", "")
        results.append(result)

    report = RecalcReport(params=params, results=results)

    # 计算校准指标（如果有人工评分）
    if human_scores and len(results) >= 2:
        llm_scores = []
        gt_scores = []
        for r in results:
            if r.request_id in human_scores:
                llm_scores.append(r.overall_score)
                gt_scores.append(human_scores[r.request_id])

        if len(llm_scores) >= 2:
            import numpy as np
            from scipy.stats import pearsonr, spearmanr

            arr_llm = np.array(llm_scores)
            arr_gt = np.array(gt_scores)

            if np.std(arr_llm) > 1e-8 and np.std(arr_gt) > 1e-8:
                report.pearson_r = round(float(pearsonr(arr_llm, arr_gt)[0]), 4)
                report.spearman_rho = round(float(spearmanr(arr_llm, arr_gt)[0]), 4)
            report.mae = round(float(np.mean(np.abs(arr_llm - arr_gt))), 4)
            report.rmse = round(float(np.sqrt(np.mean((arr_llm - arr_gt) ** 2))), 4)

            consistent = sum(1 for l, g in zip(llm_scores, gt_scores) if abs(l - g) <= 0.1)
            report.consistency_rate = round(consistent / len(llm_scores), 4)

    return report
