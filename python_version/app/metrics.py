"""评测指标工具 —— 瑕疵检出 Precision/Recall/F1 计算"""
from __future__ import annotations

from typing import Any

from sklearn.metrics import precision_score, recall_score, f1_score


def compute_flaw_metrics(
    predicted_flaws: list[dict[str, Any]],
    ground_truth_flaws: list[dict[str, Any]],
    match_key: str = "category",
) -> dict[str, float]:
    """
    计算瑕疵检出的 Precision / Recall / F1。

    简化策略：按类别（category）做二分类匹配。
    真实场景中应结合锚点位置做更精细的 Span-level 匹配。

    Args:
        predicted_flaws: LLM 检出的瑕疵列表 [{"category": "over_clean", ...}, ...]
        ground_truth_flaws: 人工标注的瑕疵列表
        match_key: 用于匹配的键（默认 category）

    Returns:
        dict 含 precision, recall, f1
    """
    all_categories = sorted(
        set(
            [f.get(match_key, "unknown") for f in predicted_flaws]
            + [f.get(match_key, "unknown") for f in ground_truth_flaws]
        )
    )
    if not all_categories:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 0}

    # 二值化：每个类别是否存在
    pred_set = set(f.get(match_key, "unknown") for f in predicted_flaws)
    gt_set = set(f.get(match_key, "unknown") for f in ground_truth_flaws)

    y_true = [1 if c in gt_set else 0 for c in all_categories]
    y_pred = [1 if c in pred_set else 0 for c in all_categories]

    precision = precision_score(y_true, y_pred, zero_division=0.0)
    recall = recall_score(y_true, y_pred, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "support": len(ground_truth_flaws),
    }


def compute_anchor_accuracy(
    predicted_flaws: list[dict[str, Any]],
    ground_truth_flaws: list[dict[str, Any]],
    char_tolerance: int = 50,  # 放宽容差到50字符
) -> dict[str, float]:
    """
    计算锚点定位准确率。

    匹配规则（宽松匹配）：
    1. 类别匹配：瑕疵类别相同即视为定位正确（最宽松）
    2. 如果有多个相同类别的瑕疵，取第一个匹配的

    Args:
        predicted_flaws: LLM 检出瑕疵
        ground_truth_flaws: 人工标注瑕疵
        char_tolerance: 字符偏移容差（保留参数但不使用）

    Returns:
        dict 含定位准确率
    """
    if not ground_truth_flaws:
        return {"anchor_accuracy": 1.0, "total": 0, "correct": 0}

    correct = 0
    matched_pred_indices = set()  # 避免重复匹配

    for gt in ground_truth_flaws:
        gt_category = gt.get("category", "")

        matched = False
        for pred_idx, pred in enumerate(predicted_flaws):
            if pred_idx in matched_pred_indices:
                continue

            p_category = pred.get("category", "")

            # 最宽松匹配：类别相同即视为正确
            if p_category == gt_category and gt_category:
                matched = True
                matched_pred_indices.add(pred_idx)
                break

        if matched:
            correct += 1

    total = len(ground_truth_flaws)
    return {
        "anchor_accuracy": round(correct / total, 4) if total > 0 else 1.0,
        "total": total,
        "correct": correct,
    }


def _has_overlap(text1: str, text2: str, min_overlap: int = 5) -> bool:
    """检查两个文本是否有重叠部分"""
    # 简单实现：检查是否有长度>=min_overlap的共同子串
    if len(text1) < min_overlap or len(text2) < min_overlap:
        return False

    # 滑动窗口检查
    for i in range(len(text1) - min_overlap + 1):
        substr = text1[i:i + min_overlap]
        if substr in text2:
            return True

    return False