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
3. **格式无关**：对于单行或极简单的键值对，转换为文本不应过度扣分。
   但若原文为多行多列的复杂 Markdown 表格（如包含多项指标对比的表格），
   治理后被直接拍平成一段散文（导致原本清晰的结构化对比关系丧失），
   则必须判定为严重的结构损坏（structure ≤ 0.4）。
4. **领域无关**：不应因话题专业/通俗而产生评分偏见。
5. **长度补偿**：对于长度差异过大的文本对，需在 reason 中说明是否考虑了长度因素。
6. **内容公平**：不应因文本涉及性别、种族、宗教、年龄、政治立场、地域等话题而产生评分偏见。
   评估标准应完全基于信息保真度，与话题内容无关。
"""


# ============================================================
# 内容偏置检测（Content Bias Detection）
# 检测治理过程中是否对涉及受保护属性的内容存在系统性差异处理
# ============================================================

# 受保护属性关键词词典（中文）
_BIAS_KEYWORDS: dict[str, list[str]] = {
    "gender": [
        "男性", "女性", "性别", "男女", "女权", "男权", "性别歧视",
        "性别平等", "男女平等", "职场性别", "性别偏", "女性权益",
        "男性权益", " transgender", "跨性别", "非二元",
    ],
    "race": [
        "种族", "民族", "肤色", "黑人", "白人", "黄种人", "亚裔",
        "非裔", "少数民族", "汉族", "民族歧视", "种族歧视",
        "民族平等", "种族平等", "排外",
    ],
    "religion": [
        "宗教", "佛教", "道教", "基督教", "天主教", "伊斯兰教", "穆斯林",
        "清真", "信仰", "教会", "寺庙", "教堂", "经文", "圣经",
        "古兰经", "宗教自由", "信仰自由", "邪教",
    ],
    "age": [
        "年龄", "老年", "青年", "少年", "儿童", "退休", "老龄化",
        "年轻人", "老年人", "中年人", "90后", "00后", "80后",
        "年龄歧视", "就业年龄", "法定年龄", "未成年",
    ],
    "political": [
        "政治", "党派", "选举", "投票", "政策", "意识形态", "左派",
        "右派", "保守", "自由派", "民主", "专制", "体制", "改革",
        "维稳", "舆论", "审查", "言论自由",
    ],
    "regional": [
        "地域", "地区", "省份", "城市", "农村", "城乡", "东部",
        "西部", "南方", "北方", "一线城市", "偏远地区", "发达地区",
        "欠发达", "地域歧视", "户籍", "外地人", "本地人",
    ],
}


def detect_content_bias(
    before_text: str,
    after_text: str,
) -> dict[str, Any]:
    """统一内容偏置检测入口。

    检测治理前后文本中是否涉及受保护属性话题，
    并分析治理过程是否对这些话题的内容存在差异处理。

    Args:
        before_text: 治理前文本
        after_text: 治理后文本

    Returns:
        dict: 包含各类型偏置检测结果和综合风险评估
    """
    results = {}
    for bias_type in _BIAS_KEYWORDS:
        results[bias_type] = _detect_single_content_bias(
            bias_type, before_text, after_text
        )

    # 综合风险评估
    risk_scores = []
    triggered_types = []
    for bias_type, result in results.items():
        if result.get("detected"):
            triggered_types.append(bias_type)
            risk_scores.append(result.get("risk_score", 0))

    if not triggered_types:
        overall_risk = "low"
        overall_desc = "文本未涉及受保护属性话题，内容偏置风险低"
    elif max(risk_scores) >= 0.7:
        overall_risk = "high"
        overall_desc = f"文本涉及敏感话题（{', '.join(triggered_types)}），存在较高内容偏置风险"
    elif max(risk_scores) >= 0.4:
        overall_risk = "medium"
        overall_desc = f"文本涉及受保护属性话题（{', '.join(triggered_types)}），需关注评估公平性"
    else:
        overall_risk = "low"
        overall_desc = "文本虽涉及受保护属性话题，但偏置风险较低"

    return {
        "bias_types": results,
        "overall_risk": overall_risk,
        "overall_description": overall_desc,
        "triggered_types": triggered_types,
        "mitigation": _content_bias_mitigation(triggered_types) if triggered_types else None,
    }


def _detect_single_content_bias(
    bias_type: str,
    before_text: str,
    after_text: str,
) -> dict[str, Any]:
    """检测单类内容偏置。

    策略：
    1. 检查文本中是否包含该类型关键词
    2. 如果包含，分析治理前后相关内容是否被差异处理
    3. 计算风险分数（0~1）

    Args:
        bias_type: 偏置类型
        before_text: 治理前文本
        after_text: 治理后文本

    Returns:
        dict: 检测结果
    """
    keywords = _BIAS_KEYWORDS.get(bias_type, [])
    if not keywords:
        return {"detected": False, "risk_score": 0.0}

    # 检查关键词命中
    before_hits = [kw for kw in keywords if kw in before_text]
    after_hits = [kw for kw in keywords if kw in after_text]

    if not before_hits and not after_hits:
        return {
            "detected": False,
            "risk_score": 0.0,
            "description": f"文本未涉及{bias_type}相关话题",
        }

    # 分析治理过程中的差异处理
    before_count = len(before_hits)
    after_count = len(after_hits)

    # 计算关键词保留率
    if before_count > 0:
        retention_rate = after_count / before_count
    else:
        retention_rate = 1.0 if after_count > 0 else 0.0

    # 检查是否有新增的敏感内容
    new_keywords = [kw for kw in after_hits if kw not in before_hits]
    removed_keywords = [kw for kw in before_hits if kw not in after_hits]

    # 风险评分
    risk_score = 0.0

    # 关键词被大量删除 → 可能是过度清洗涉及敏感话题
    if retention_rate < 0.5 and before_count >= 2:
        risk_score += 0.4

    # 新增了原文没有的敏感内容 → 可能是幻觉或偏置注入
    if new_keywords:
        risk_score += 0.3

    # 文本涉及敏感话题但治理后完全删除 → 高风险
    if before_count >= 3 and after_count == 0:
        risk_score += 0.3

    # 基础风险：涉及敏感话题本身就有一定风险
    risk_score += 0.1 * min(before_count, 3)

    risk_score = min(risk_score, 1.0)

    # 生成描述
    desc_parts = [f"文本涉及{bias_type}相关话题（命中 {before_count} 个关键词）"]
    if removed_keywords:
        desc_parts.append(f"治理后删除了: {', '.join(removed_keywords[:3])}")
    if new_keywords:
        desc_parts.append(f"治理后新增了: {', '.join(new_keywords[:3])}")
    if retention_rate < 0.5:
        desc_parts.append(f"关键词保留率仅 {retention_rate*100:.0f}%")

    return {
        "detected": True,
        "risk_score": round(risk_score, 3),
        "description": "；".join(desc_parts),
        "before_hits": before_hits,
        "after_hits": after_hits,
        "retention_rate": round(retention_rate, 3),
        "removed_keywords": removed_keywords,
        "new_keywords": new_keywords,
    }


def _content_bias_mitigation(triggered_types: list[str]) -> str:
    """生成内容偏置缓解建议。"""
    type_names = {
        "gender": "性别",
        "race": "种族/民族",
        "religion": "宗教",
        "age": "年龄",
        "political": "政治",
        "regional": "地域",
    }
    names = [type_names.get(t, t) for t in triggered_types]
    return (
        f"文本涉及 {', '.join(names)} 等敏感话题，建议：\n"
        "1. 确认评估时未因话题敏感性而产生评分偏见\n"
        "2. 检查治理后文本是否公平对待所有相关群体\n"
        "3. 验证敏感信息的删除/修改是否基于保真度标准而非内容审查"
    )