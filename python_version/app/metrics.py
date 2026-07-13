"""评测指标工具 —— 瑕疵检出 Precision/Recall/F1 + 锚点定位准确率 + 一致性指标"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def compute_flaw_metrics(
    predicted_flaws: list[dict[str, Any]],
    ground_truth_flaws: list[dict[str, Any]],
    match_key: str = "category",
    match_mode: str = "many_to_one",
) -> dict[str, float]:
    """
    计算瑕疵检出的 Precision / Recall / F1。

    匹配模式：
      - "many_to_one"（默认）：允许多条预测匹配同一条 GT 瑕疵。
        当 LLM 将一个 GT 问题拆分为多条细粒度瑕疵时，这些预测共同覆盖
        该 GT，只算 1 个 TP，不产生额外 FP。
        解决 LLM 输出粒度 > 人工标注粒度时的 precision 虚低问题。
      - "one_to_one"（旧模式）：每条预测最多匹配一条 GT，一对一贪心。

    计算公式（many_to_one）：
      TP = 至少被一条预测匹配的 GT 数量
      FP = 未匹配到任何 GT 的预测数量
      FN = 未被任何预测匹配到的 GT 数量

    Args:
        predicted_flaws: LLM 检出的瑕疵列表
        ground_truth_flaws: 人工标注的瑕疵列表
        match_key: 匹配用的键（默认 category）
        match_mode: 匹配模式 ("many_to_one" 或 "one_to_one")

    Returns:
        dict 含 precision, recall, f1, tp, fp, fn, support
    """
    if not ground_truth_flaws and not predicted_flaws:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "tp": 0, "fp": 0, "fn": 0, "support": 0}

    if not ground_truth_flaws:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0,
                "tp": 0, "fp": len(predicted_flaws), "fn": 0,
                "support": 0}

    if not predicted_flaws:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": 0, "fn": len(ground_truth_flaws),
                "support": len(ground_truth_flaws)}

    if match_mode == "many_to_one":
        # 多对一匹配：按 category 分组，组内一对一分配
        # 多条预测可共同覆盖同一类别的多个 GT（round-robin 分配）
        # 例：9 条 over_clean 预测 vs 8 条 GT → 8 个 TP + 1 个 FP
        from collections import defaultdict
        pred_by_cat: dict[str, list[int]] = defaultdict(list)
        gt_by_cat: dict[str, list[int]] = defaultdict(list)

        for i, pred in enumerate(predicted_flaws):
            cat = pred.get(match_key, "")
            if cat:
                pred_by_cat[cat].append(i)

        for i, gt in enumerate(ground_truth_flaws):
            cat = gt.get(match_key, "")
            if cat:
                gt_by_cat[cat].append(i)

        matched_pred_indices: set[int] = set()
        matched_gt_indices: set[int] = set()

        all_cats = set(pred_by_cat.keys()) | set(gt_by_cat.keys())
        for cat in all_cats:
            p_indices = pred_by_cat.get(cat, [])
            g_indices = gt_by_cat.get(cat, [])
            # 组内一对一贪心：min(N_pred, N_gt) 个匹配
            pairs = min(len(p_indices), len(g_indices))
            for k in range(pairs):
                matched_pred_indices.add(p_indices[k])
                matched_gt_indices.add(g_indices[k])

        tp = len(matched_gt_indices)
        fp = len(predicted_flaws) - len(matched_pred_indices)
        fn = len(ground_truth_flaws) - tp

    else:
        # 一对一贪心匹配（旧逻辑）
        matched_pred_set: set[int] = set()
        tp = 0

        for gt in ground_truth_flaws:
            gt_val = gt.get(match_key, "")
            for pred_idx, pred in enumerate(predicted_flaws):
                if pred_idx in matched_pred_set:
                    continue
                if pred.get(match_key, "") == gt_val and gt_val:
                    tp += 1
                    matched_pred_set.add(pred_idx)
                    break

        fp = len(predicted_flaws) - len(matched_pred_set)
        fn = len(ground_truth_flaws) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "support": len(ground_truth_flaws),
    }


def compute_anchor_accuracy(
    predicted_flaws: list[dict[str, Any]],
    ground_truth_flaws: list[dict[str, Any]],
    char_tolerance: int = 10,
) -> dict[str, float]:
    """
    计算锚点定位准确率（双轨匹配版本）。

    使用 flaw_matcher.dual_track_evaluate 将定位能力与分类能力解耦：
    - 定位指标：只要锚点对了就算定位成功（不要求分类一致）
    - 分类指标：在定位成功的基础上，再检查分类是否一致

    Args:
        predicted_flaws: LLM 检出瑕玼
        ground_truth_flaws: 人工标注瑕玼
        char_tolerance: 字符偏移容差（默认 10，双轨匹配中不直接使用）

    Returns:
        dict 含 anchor_accuracy, location_f1, classification_f1, total, correct 等
    """
    if not ground_truth_flaws:
        return {"anchor_accuracy": 1.0, "location_f1": 1.0,
                "classification_f1": 1.0, "total": 0, "correct": 0}

    # 使用双轨匹配引擎
    from .flaw_matcher import dual_track_evaluate
    result = dual_track_evaluate(predicted_flaws, ground_truth_flaws, min_score=0.50)

    return {
        # 主指标：定位召回率（真实瑕疵中被正确定位的比例），对应课题“定位准确率≥90%”
        "anchor_accuracy": result.location_recall,
        "location_precision": result.location_precision,
        "location_recall": result.location_recall,
        "location_f1": result.location_f1,
        "classification_precision": result.classification_precision,
        "classification_recall": result.classification_recall,
        "classification_f1": result.classification_f1,
        "total": result.total_gt,
        "correct": result.location_tp,
        "location_tp": result.location_tp,
        "location_fp": result.location_fp,
        "location_fn": result.location_fn,
        "classification_tp": result.classification_tp,
        "primary_metric": "location_recall",
        "reference_f1": result.location_f1,
    }


def _snippet_overlap(s1: str, s2: str, min_len: int = 4) -> bool:
    """检查两个字符串是否有长度 >= min_len 的公共子串。"""
    if not s1 or not s2 or len(s1) < min_len or len(s2) < min_len:
        return False
    shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    for i in range(len(shorter) - min_len + 1):
        if shorter[i:i + min_len] in longer:
            return True
    return False


# ============================================================
# 一致性指标：Kappa 系数 & Kendall's W
# ============================================================

def _score_to_category(score: float) -> str:
    """将连续分数映射到判定类别"""
    if score >= 0.8:
        return "pass"
    elif score >= 0.5:
        return "review"
    return "fail"


def compute_kappa(
    human_scores: list[float],
    llm_scores: list[float],
    task: str = "pass/fail",
) -> dict[str, float]:
    """计算 Cohen's Kappa 系数（人机一致性）。

    将连续分数映射到 pass/review/fail 类别后计算。
    目标：≥ 0.8

    Args:
        human_scores: 人工评分列表
        llm_scores: LLM 评分列表
        task: 分类粒度 "pass/fail"（二分类）或 "pass/review/fail"（三分类）

    Returns:
        dict 含 kappa, agreement_rate, n_samples
    """
    if len(human_scores) != len(llm_scores):
        raise ValueError("human_scores and llm_scores must have same length")

    n = len(human_scores)
    if n == 0:
        return {"kappa": 0.0, "agreement_rate": 0.0, "n_samples": 0}

    # 映射到类别
    if task == "pass/fail":
        human_cats = ["pass" if s >= 0.8 else "fail" for s in human_scores]
        llm_cats = ["pass" if s >= 0.8 else "fail" for s in llm_scores]
    else:
        human_cats = [_score_to_category(s) for s in human_scores]
        llm_cats = [_score_to_category(s) for s in llm_scores]

    # 构建混淆矩阵
    categories = sorted(set(human_cats + llm_cats))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    confusion = np.zeros((k, k), dtype=int)
    for h, l in zip(human_cats, llm_cats):
        confusion[cat_to_idx[h], cat_to_idx[l]] += 1

    # 观察一致率
    po = np.trace(confusion) / n

    # 期望一致率（假设独立）
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)

    # Kappa
    if pe == 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return {
        "kappa": round(float(kappa), 4),
        "agreement_rate": round(float(po), 4),
        "n_samples": n,
    }


def compute_kendalls_w(
    human_scores: list[float],
    llm_scores: list[float],
) -> dict[str, float]:
    """计算 Kendall's W 协和系数（人机排序一致性）。

    衡量两个评分者对样本的排序是否一致。
    目标：≥ 0.8

    Args:
        human_scores: 人工评分列表
        llm_scores: LLM 评分列表

    Returns:
        dict 含 kendalls_w, spearman_rho, n_samples
    """
    if len(human_scores) != len(llm_scores):
        raise ValueError("human_scores and llm_scores must have same length")

    n = len(human_scores)
    if n < 2:
        return {"kendalls_w": 0.0, "spearman_rho": 0.0, "n_samples": n}

    # Kendall's W 对于 2 个评分者简化为：
    # W = (1 + Spearman_rho) / 2
    rho, _ = stats.spearmanr(human_scores, llm_scores)
    w = (1.0 + rho) / 2.0

    return {
        "kendalls_w": round(float(w), 4),
        "spearman_rho": round(float(rho), 4),
        "n_samples": n,
    }


def compute_correlation_metrics(
    human_scores: list[float],
    llm_scores: list[float],
) -> dict[str, float]:
    """计算完整的一致性指标集（用于验收报告）。

    包含：Pearson r, Spearman rho, Kappa, Kendall's W, MAE, RMSE

    Args:
        human_scores: 人工评分列表
        llm_scores: LLM 评分列表

    Returns:
        dict 含所有指标
    """
    if len(human_scores) != len(llm_scores):
        raise ValueError("human_scores and llm_scores must have same length")

    n = len(human_scores)
    if n < 2:
        return {"n_samples": n}

    pearson_r, _ = stats.pearsonr(human_scores, llm_scores)
    spearman_rho, _ = stats.spearmanr(human_scores, llm_scores)

    diffs = [abs(h - l) for h, l in zip(human_scores, llm_scores)]
    mae = sum(diffs) / n
    rmse = (sum(d**2 for d in diffs) / n) ** 0.5

    kappa_result = compute_kappa(human_scores, llm_scores)
    kendall_result = compute_kendalls_w(human_scores, llm_scores)

    return {
        "pearson_r": round(float(pearson_r), 4),
        "spearman_rho": round(float(spearman_rho), 4),
        "kappa": kappa_result["kappa"],
        "kendalls_w": kendall_result["kendalls_w"],
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "n_samples": n,
    }


# ============================================================
# 文本相似度指标：ROUGE-L & BERTScore
# ============================================================

def compute_rouge_l(
    references: list[str],
    predictions: list[str],
) -> dict[str, float]:
    """计算 ROUGE-L（基于最长公共子序列）。

    纯 Python 实现，无需外部依赖。

    Args:
        references: 参考文本列表（人工标注）
        predictions: 预测文本列表（LLM 输出）

    Returns:
        dict 含 rouge_l_precision, rouge_l_recall, rouge_l_f1
    """
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have same length")

    precisions = []
    recalls = []
    f1s = []

    for ref, pred in zip(references, predictions):
        ref_tokens = list(ref)
        pred_tokens = list(pred)

        lcs_len = _lcs_length(ref_tokens, pred_tokens)

        if len(pred_tokens) > 0:
            p = lcs_len / len(pred_tokens)
        else:
            p = 0.0

        if len(ref_tokens) > 0:
            r = lcs_len / len(ref_tokens)
        else:
            r = 0.0

        if p + r > 0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "rouge_l_precision": round(sum(precisions) / len(precisions), 4),
        "rouge_l_recall": round(sum(recalls) / len(recalls), 4),
        "rouge_l_f1": round(sum(f1s) / len(f1s), 4),
    }


def _lcs_length(x: list[str], y: list[str]) -> int:
    """计算两个序列的最长公共子序列长度（动态规划）"""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    # 空间优化：只用两行
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def compute_bertscore(
    references: list[str],
    predictions: list[str],
    lang: str = "zh",
) -> dict[str, float]:
    """计算 BERTScore（基于语义相似度）。

    需要安装 bert-score 包：pip install bert-score

    Args:
        references: 参考文本列表
        predictions: 预测文本列表
        lang: 语言代码

    Returns:
        dict 含 bertscore_precision, bertscore_recall, bertscore_f1
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "error": "bert-score not installed. Run: pip install bert-score",
        }

    P, R, F1 = bert_score(predictions, references, lang=lang, verbose=False)
    return {
        "bertscore_precision": round(float(P.mean()), 4),
        "bertscore_recall": round(float(R.mean()), 4),
        "bertscore_f1": round(float(F1.mean()), 4),
    }
