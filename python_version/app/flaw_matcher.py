"""双轨匹配引擎 —— 将定位能力与分类能力解耦

核心思路：
  锚点定位准确率应该衡量"有没有找对地方"，而不是"分类对不对"。
  当 LLM 和人对同一个瑕玼的分类不一致（如 over_clean vs mis_edit），
  不应该因此判定"定位失败"。

三步匹配：
  1. 构建相似度矩阵 S(P, G) = 0.6*锚点 + 0.3*文本 + 0.1*分类
  2. 匈牙利算法求最大权匹配（一对一，防止多对一虚高）
  3. 双阀防线：最小相似度 ≥ 0.50 且 文本必须有交集

输出两组解耦指标：
  - 定位指标：只要锚点对了就算定位成功（不要求分类一致）
  - 分类指标：在定位成功的基础上，再检查分类是否一致
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ============================================================
# 1. 数据结构
# ============================================================

@dataclass
class FlawLocation:
    """瑕玼位置信息"""
    before_anchor: str = ""
    after_anchor: str = ""
    start_char: int = -1
    end_char: int = -1
    snippet: str = ""

    @classmethod
    def from_dict(cls, loc: dict[str, Any] | None) -> FlawLocation:
        """从字典构建，兼容多种格式"""
        if not loc or not isinstance(loc, dict):
            return cls()
        return cls(
            before_anchor=str(loc.get("before_anchor", "") or loc.get("segment_id", "")),
            after_anchor=str(loc.get("after_anchor", "") or loc.get("segment_id", "")),
            start_char=int(loc.get("start_char", -1) or -1),
            end_char=int(loc.get("end_char", -1) or -1),
            snippet=str(loc.get("snippet", "") or ""),
        )

    def anchor_norm(self) -> str:
        """归一化锚点 ID（去掉方括号和空格）"""
        return self.after_anchor.strip().strip("[]").strip()


@dataclass
class Flaw:
    """瑕玼数据"""
    category: str = "unknown"
    severity: str = "minor"
    description: str = ""
    location: FlawLocation = field(default_factory=FlawLocation)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Flaw:
        """从字典构建"""
        return cls(
            category=str(d.get("category", "unknown")),
            severity=str(d.get("severity", "minor")),
            description=str(d.get("description", "")),
            location=FlawLocation.from_dict(d.get("location")),
        )


@dataclass
class MatchPair:
    """一对匹配结果"""
    pred_idx: int
    gt_idx: int
    score: float
    anchor_score: float
    text_score: float
    category_score: float
    category_match: bool


@dataclass
class DualTrackMetrics:
    """双轨指标"""
    # 定位指标（不要求分类一致）
    location_precision: float = 0.0
    location_recall: float = 0.0
    location_f1: float = 0.0
    location_tp: int = 0
    location_fp: int = 0
    location_fn: int = 0
    # 分类指标（在定位成功的基础上检查分类）
    classification_precision: float = 0.0
    classification_recall: float = 0.0
    classification_f1: float = 0.0
    classification_tp: int = 0
    # 匹配详情
    matches: list[MatchPair] = field(default_factory=list)
    total_pred: int = 0
    total_gt: int = 0


# ============================================================
# 2. 子函数
# ============================================================

def _normalize_anchor(anchor: str) -> str:
    """归一化锚点 ID"""
    return anchor.strip().strip("[]").strip()


def _extract_anchor_num(anchor: str) -> str:
    """从锚点 ID 中提取数字部分，用于跨格式匹配。
    
    例：
      "[After 1]"  → "1"
      "[Before 2]" → "2"
      "seg_001"    → "1"  (去除前导零)
      "After 3"    → "3"
      "1"          → "1"
    """
    import re
    m = re.search(r'(\d+)', anchor)
    if m:
        return str(int(m.group(1)))  # 去除前导零
    return ""


def _match_anchor(pred: FlawLocation, gt: FlawLocation) -> float:
    """锚点对齐：先精确匹配，再数字 ID 匹配（跨格式兼容）"""
    p_before = _normalize_anchor(pred.before_anchor)
    g_before = _normalize_anchor(gt.before_anchor)
    p_after = _normalize_anchor(pred.after_anchor)
    g_after = _normalize_anchor(gt.after_anchor)

    # 第一优先：精确字符串匹配
    if p_before and g_before:
        if p_before == g_before and p_after == g_after:
            return 1.0
    if p_after and g_after:
        if p_after == g_after:
            return 1.0

    # 第二优先：数字 ID 匹配（兼容 [After 1] vs seg_001 格式）
    p_num = _extract_anchor_num(pred.after_anchor or pred.before_anchor)
    g_num = _extract_anchor_num(gt.after_anchor or gt.before_anchor)
    if p_num and g_num and p_num == g_num:
        return 1.0

    # 都无法匹配
    return 0.0


def _jaccard_similarity(s1: str, s2: str) -> float:
    """字符级 Jaccard 相似度"""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1)
    set2 = set(s2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _lcs_ratio(s1: str, s2: str) -> float:
    """最长公共子串占最大文本长度的比例"""
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    # 动态规划求 LCS 长度
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    max_len = 0
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])
    return max_len / max(len1, len2)


def _sim_text(pred: FlawLocation, gt: FlawLocation) -> float:
    """文本相似度：取 Jaccard 和 LCS 比例的最大值"""
    s1 = pred.snippet or ""
    s2 = gt.snippet or ""
    if not s1 or not s2:
        return 0.0
    return max(_jaccard_similarity(s1, s2), _lcs_ratio(s1, s2))


def _match_category(pred_cat: str, gt_cat: str) -> float:
    """分类匹配：相同 → 1.0，不同 → 0.0"""
    return 1.0 if pred_cat == gt_cat else 0.0


# ============================================================
# 3. 相似度矩阵
# ============================================================

def build_similarity_matrix(
    preds: list[Flaw],
    gts: list[Flaw],
) -> np.ndarray:
    """构建瑕玼相似度矩阵。

    S(P, G) = 0.6 * Match-anchor + 0.3 * Sim-text + 0.1 * Match-category

    Returns:
        shape (n_pred, n_gt) 的 numpy 数组
    """
    n_pred = len(preds)
    n_gt = len(gts)
    matrix = np.zeros((n_pred, n_gt), dtype=np.float64)

    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            anchor_score = _match_anchor(pred.location, gt.location)
            text_score = _sim_text(pred.location, gt.location)
            cat_score = _match_category(pred.category, gt.category)
            matrix[i, j] = 0.6 * anchor_score + 0.3 * text_score + 0.1 * cat_score

    return matrix


# ============================================================
# 4. 匈牙利算法匹配
# ============================================================

def hungarian_match(
    sim_matrix: np.ndarray,
    min_score: float = 0.50,
) -> list[tuple[int, int, float]]:
    """匈牙利算法求最大权匹配。

    Args:
        sim_matrix: 相似度矩阵 (n_pred, n_gt)
        min_score: 最小相似度门槛

    Returns:
        匹配对列表 [(pred_idx, gt_idx, score), ...]
    """
    from scipy.optimize import linear_sum_assignment

    n_pred, n_gt = sim_matrix.shape
    if n_pred == 0 or n_gt == 0:
        return []

    # linear_sum_assignment 求最小权匹配，取负数转为最大权
    cost_matrix = -sim_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_indices, col_indices):
        score = sim_matrix[r, c]
        if score >= min_score:
            matches.append((int(r), int(c), float(score)))

    return matches


# ============================================================
# 5. 双阀防线
# ============================================================

def double_gate_check(
    pred: Flaw,
    gt: Flaw,
    score: float,
    min_score: float = 0.50,
) -> tuple[bool, str]:
    """双阀防线检查。

    防线 1：最小相似度门槛 ≥ 0.50
    防线 2：文本必须有交集（Sim-text > 0）

    Returns:
        (通过?, 原因)
    """
    # 防线 1
    if score < min_score:
        return False, f"相似度 {score:.4f} < {min_score}"

    # 防线 2
    text_sim = _sim_text(pred.location, gt.location)
    if text_sim <= 0.0:
        return False, "文本片段零交集"

    return True, "通过"


# ============================================================
# 6. 主函数
# ============================================================

def dual_track_evaluate(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    min_score: float = 0.50,
) -> DualTrackMetrics:
    """双轨匹配评估：将定位能力与分类能力解耦。

    Args:
        predicted: 预测瑕玼列表（dict 格式）
        ground_truth: 人工标注瑕玼列表（dict 格式）
        min_score: 最小相似度门槛（默认 0.50）

    Returns:
        DualTrackMetrics：包含定位指标和分类指标
    """
    # 健壮性校验
    preds = [Flaw.from_dict(p) for p in (predicted or []) if isinstance(p, dict)]
    gts = [Flaw.from_dict(g) for g in (ground_truth or []) if isinstance(g, dict)]

    result = DualTrackMetrics(total_pred=len(preds), total_gt=len(gts))

    if not preds or not gts:
        if not preds and not gts:
            result.location_precision = 1.0
            result.location_recall = 1.0
            result.location_f1 = 1.0
            result.classification_precision = 1.0
            result.classification_recall = 1.0
            result.classification_f1 = 1.0
        elif not preds:
            result.location_fn = len(gts)
        else:
            result.location_fp = len(preds)
        return result

    # 第一步：构建相似度矩阵
    sim_matrix = build_similarity_matrix(preds, gts)

    # 第二步：匈牙利算法匹配
    raw_matches = hungarian_match(sim_matrix, min_score=0.0)  # 先不卡门槛，后面双阀检查

    # 第三步：双阀防线
    valid_matches: list[MatchPair] = []
    for pred_idx, gt_idx, score in raw_matches:
        passed, reason = double_gate_check(preds[pred_idx], gts[gt_idx], score, min_score)
        if not passed:
            continue

        anchor_score = _match_anchor(preds[pred_idx].location, gts[gt_idx].location)
        text_score = _sim_text(preds[pred_idx].location, gts[gt_idx].location)
        cat_score = _match_category(preds[pred_idx].category, gts[gt_idx].category)

        valid_matches.append(MatchPair(
            pred_idx=pred_idx,
            gt_idx=gt_idx,
            score=score,
            anchor_score=anchor_score,
            text_score=text_score,
            category_score=cat_score,
            category_match=(cat_score == 1.0),
        ))

    result.matches = valid_matches

    # 计算定位指标（只要匹配成功就算定位正确）
    result.location_tp = len(valid_matches)
    result.location_fp = len(preds) - result.location_tp
    result.location_fn = len(gts) - result.location_tp

    if result.location_tp + result.location_fp > 0:
        result.location_precision = result.location_tp / (result.location_tp + result.location_fp)
    if result.location_tp + result.location_fn > 0:
        result.location_recall = result.location_tp / (result.location_tp + result.location_fn)
    if result.location_precision + result.location_recall > 0:
        result.location_f1 = (
            2 * result.location_precision * result.location_recall
            / (result.location_precision + result.location_recall)
        )

    # 计算分类指标（在定位成功的基础上，再检查分类是否一致）
    result.classification_tp = sum(1 for m in valid_matches if m.category_match)

    if result.location_tp > 0:
        result.classification_precision = result.classification_tp / result.location_tp
        result.classification_recall = result.classification_tp / len(gts) if len(gts) > 0 else 0.0
    if result.classification_precision + result.classification_recall > 0:
        result.classification_f1 = (
            2 * result.classification_precision * result.classification_recall
            / (result.classification_precision + result.classification_recall)
        )

    # 四舍五入
    for attr in [
        "location_precision", "location_recall", "location_f1",
        "classification_precision", "classification_recall", "classification_f1",
    ]:
        setattr(result, attr, round(getattr(result, attr), 4))

    return result
