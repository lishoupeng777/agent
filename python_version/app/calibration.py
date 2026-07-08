"""一致性校准模块 —— 与人工标注的统计对比"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .engine import evaluate
from .models import CalibrationReport, EvalRequest


def calibrate(
    requests: list[EvalRequest],
    tolerance: float = 0.1,
) -> CalibrationReport:
    """
    对一组带人工标注的评估请求进行一致性校准。

    Args:
        requests: 评估请求列表（每个请求需包含 human_label，其中含 "overall_score" 字段）
        tolerance: 一致率容差阈值（默认 0.1，即 LLM 得分与人工得分差 ≤ 0.1 视为一致）

    Returns:
        CalibrationReport: 含 PPMCC、SRCC、MAE、RMSE、一致率等指标
    """
    if not requests:
        return CalibrationReport(
            pearson_r=0.0,
            spearman_rho=0.0,
            mae=0.0,
            rmse=0.0,
            consistency_rate=0.0,
            sample_count=0,
            details=[],
        )

    llm_scores: list[float] = []
    human_scores: list[float] = []
    details: list[dict[str, Any]] = []

    for req in requests:
        # 人工标注
        human = req.human_label or {}
        human_score = float(human.get("overall_score", 0.5))

        # LLM 评估
        resp = evaluate(req, temperature=0.0)
        llm_score = resp.overall_score

        llm_scores.append(llm_score)
        human_scores.append(human_score)
        details.append(
            {
                "request_id": req.request_id,
                "llm_score": llm_score,
                "human_score": human_score,
                "diff": round(abs(llm_score - human_score), 4),
                "consistent": abs(llm_score - human_score) <= tolerance,
                "evaluation": {
                    "verdict": resp.verdict,
                    "dimensions": [
                        {"dimension": d.dimension, "score": d.score, "reason": d.reason}
                        for d in resp.dimensions
                    ],
                    "flaws": [
                        {"category": f.category, "severity": f.severity, "description": f.description}
                        for f in resp.flaws
                    ],
                },
            }
        )

    arr_llm = np.array(llm_scores)
    arr_human = np.array(human_scores)
    n = len(arr_llm)

    # 皮尔逊相关系数
    if n >= 3 and np.std(arr_llm) > 1e-8 and np.std(arr_human) > 1e-8:
        pearson_r, _ = pearsonr(arr_llm, arr_human)
    else:
        pearson_r = 0.0

    # 斯皮尔曼秩相关系数
    if n >= 3:
        spearman_rho, _ = spearmanr(arr_llm, arr_human)
    else:
        spearman_rho = 0.0

    # MAE
    mae = float(np.mean(np.abs(arr_llm - arr_human)))

    # RMSE
    rmse = float(np.sqrt(np.mean((arr_llm - arr_human) ** 2)))

    # 一致率（容差内一致的比例）
    consistent_count = sum(1 for d in details if d["consistent"])
    consistency_rate = consistent_count / n if n > 0 else 0.0

    return CalibrationReport(
        pearson_r=round(pearson_r, 4),
        spearman_rho=round(spearman_rho, 4),
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        consistency_rate=round(consistency_rate, 4),
        sample_count=n,
        details=details,
    )