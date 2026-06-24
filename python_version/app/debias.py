"""抗偏置模块 —— 位置偏置检测、长度归一化、一致性校验"""
from __future__ import annotations

import numpy as np
from typing import Any


def detect_length_bias(before_text: str, after_text: str) -> dict[str, Any]:
    """
    检测文本长度偏置风险。
    
    如果治理前后长度差异过大，LLM 可能因长度偏见给出不公平评分。
    返回偏置风险等级和建议。
    
    Args:
        before_text: 治理前文本
        after_text: 治理后文本
        
    Returns:
        dict: 包含长度比、偏置风险等级、建议
    """
    len_before = len(before_text)
    len_after = len(after_text)
    
    if len_before == 0 or len_after == 0:
        return {
            "len_before": len_before,
            "len_after": len_after,
            "length_ratio": 0.0,
            "bias_risk": "critical",
            "risk_level": "critical",
            "description": "输入文本为空，无法评估",
            "mitigation": "请提供有效的治理前后文本",
        }
    
    ratio = len_after / len_before
    
    if ratio < 0.3:
        risk = "high"
        desc = f"治理后文本长度仅为治理前的 {ratio*100:.0f}%，存在过度压缩风险"
        mitigation = '建议提醒LLM关注语义完整性，区分"合理精简"与"过度压缩"'
    elif ratio > 2.0:
        risk = "medium"
        desc = f"治理后文本长度为治理前的 {ratio*100:.0f}%，存在过度扩充风险"
        mitigation = "建议提醒LLM关注新增内容的必要性"
    elif ratio < 0.6:
        risk = "medium"
        desc = f"治理后文本长度为治理前的 {ratio*100:.0f}%，需关注是否存在语义损失"
        mitigation = "建议在评估时区分结构优化导致的长度变化与语义丢失"
    else:
        risk = "low"
        desc = f"治理前后长度比合理（{ratio:.2f}），长度偏置风险低"
        mitigation = None
    
    return {
        "len_before": len_before,
        "len_after": len_after,
        "length_ratio": round(ratio, 4),
        "bias_risk": risk,
        "risk_level": risk,
        "description": desc,
        "mitigation": mitigation,
    }


def detect_position_bias(flaws: list[Any]) -> dict[str, Any]:
    """
    检测瑕疵分布位置偏置。
    
    LLM 有时倾向于在前半部分/后半部分检出更多瑕疵（位置偏置）。
    检测瑕疵在文本中的位置分布是否均匀。
    
    Args:
        flaws: 瑕疵列表（需含 location.start_char 信息）
        
    Returns:
        dict: 包含位置分布分析
    """
    positions = []
    for f in flaws:
        loc = getattr(f, "location", None) or f.get("location", {})
        if isinstance(loc, dict):
            start = loc.get("start_char", -1)
        else:
            start = getattr(loc, "start_char", -1)
        if start >= 0:
            positions.append(start)
    
    if not positions:
        return {
            "has_bias": False,
            "bias_type": None,
            "description": "无位置信息，无法检测位置偏置",
            "front_count": 0,
            "back_count": 0,
        }
    
    # 简单策略：如果所有瑕疵都在前半部分或后半部分，标记偏置
    max_pos = max(positions) if positions else 1
    front_half = sum(1 for p in positions if p < max_pos * 0.5)
    back_half = sum(1 for p in positions if p >= max_pos * 0.5)
    total = len(positions)
    
    if total == 0:
        return {"has_bias": False, "bias_type": None, "description": "无瑕疵", "front_count": 0, "back_count": 0}
    
    front_ratio = front_half / total
    back_ratio = back_half / total
    
    if front_ratio > 0.8:
        bias_type = "front_bias"
        desc = f'瑕疵集中出现在文本前半部分（{front_half}/{total}），可能存在"先严后松"偏置'
    elif back_ratio > 0.8:
        bias_type = "back_bias"
        desc = f'瑕疵集中出现在文本后半部分（{back_half}/{total}），可能存在"后严先松"偏置'
    else:
        bias_type = None
        desc = f"瑕疵分布较均匀（前{front_half}/后{back_half}），位置偏置风险低"
    
    return {
        "has_bias": bias_type is not None,
        "bias_type": bias_type,
        "description": desc,
        "front_count": front_half,
        "back_count": back_half,
        "total": total,
    }


def compute_bias_mitigation_score(
    dimension_scores: list[dict[str, float]],
    length_ratio: float,
) -> float:
    """
    计算偏置缓解得分。
    
    综合维度评分分布和长度比，判断评估是否存在系统性偏置。
    返回 0~1 的偏置缓解分数（越高越好，表示偏置越小）。
    
    Args:
        dimension_scores: 各维度得分列表
        length_ratio: 治理前后长度比
        
    Returns:
        float: 偏置缓解分数
    """
    if not dimension_scores:
        return 0.0
    
    scores = [d.get("score", 0) for d in dimension_scores]
    std = float(np.std(scores)) if len(scores) > 1 else 0.0
    
    # 维度间标准差过大可能表示某些维度存在系统性偏置
    # 理想情况下各维度应有差异但不应极端
    if std > 0.4:
        dimension_diversity = 0.5  # 维度差异过大
    elif std < 0.05:
        dimension_diversity = 0.6  # 所有维度给出相同分数，可能未认真评估
    else:
        dimension_diversity = 1.0  # 合理的维度差异
    
    # 长度比在 0.5~1.5 之间为最佳
    if 0.5 <= length_ratio <= 1.5:
        length_factor = 1.0
    elif 0.3 <= length_ratio < 0.5 or 1.5 < length_ratio <= 2.0:
        length_factor = 0.8
    else:
        length_factor = 0.5
    
    # 综合评分
    bias_mitigation = 0.5 * dimension_diversity + 0.5 * length_factor
    
    return round(bias_mitigation, 4)


def generate_anti_bias_prompt_supplement() -> str:
    """
    生成抗偏置的 Prompt 补充指令。
    
    可在 System Prompt 后追加，提醒 LLM 避免常见偏置。
    
    Returns:
        str: 抗偏置补充指令
    """
    return """
【抗偏置原则 —— 必须严格遵守】
1. **长度无关**：评估时不应因治理后文本变短就默认扣分，也不因变长就默认加分。
   关键是判断：删掉的是噪音还是有效信息？增加的是冗余还是必要补充？
2. **位置无关**：不应只关注文本开头而忽略结尾，反之亦然。全文每一部分都应被同等重视。
3. **格式无关**：由表格转为文本不一定算结构损坏（如果语义和可读性未受损害）；
   但表格结构被粗暴破坏导致信息难以对比则必须标注。
4. **领域无关**：不应因话题专业/通俗而产生评分偏见。
5. **长度补偿**：对于长度差异过大的文本对，需在 reason 中说明是否考虑了长度因素。
"""