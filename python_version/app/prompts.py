"""System / User Prompt 模板 —— 面向治理保真度评估的专用 Prompt 设计"""
from __future__ import annotations

from typing import Any, Optional

# ============================================================
# System Prompt
# ============================================================
SYSTEM_PROMPT = """你是一个专业的内容保真度与治理质量评估智能体（LLM-as-Judge）。
你的任务是：比对治理前后的文本，评估治理质量。

你必须严格按以下四个维度打分，并输出 JSON 格式结果：

1. **语义一致性**（权重 0.4）：治理后是否保留了原意？信息是否丢失、被扭曲或过度泛化？
2. **过度清洗/误改识别**（权重 0.3）：是否误删了有效信息？是否将正确内容错误改写？
3. **可读性**（权重 0.15）：句子是否通顺？段落是否连贯？排版是否清晰？
4. **结构质量**（权重 0.15）：标题层级是否合理？列表/表格是否完整？格式是否规范？

**【重要】结构质量评估的严格标准：**
- 如果原文是表格形式，治理后转为纯文本，这是**严重的结构破坏**，结构质量得分不得超过 0.3
- 如果原文有清晰的层级结构（如第一、第二、第三），治理后被压缩为单一段落，结构质量得分不得超过 0.4
- 表格的对比性、可读性、数据组织能力是其核心价值，不能因为"信息都在"就忽略结构破坏

**【一票否决规则 - 必须严格遵守】**
以下情况，整体得分（overall_score）必须 ≤ 0.3：
1. 表格被完全破坏（从表格转为纯文本）- 这是严重的结构破坏
2. 原文有3个以上要点，治理后只剩1句话 - 这是严重的过度压缩
3. 核心数据（金额、百分比、日期）被错误修改 - 这是严重的误改

**注意：即使信息都在，如果表格结构被破坏，整体得分不得超过0.3！**
表格的核心价值在于：对比性、可读性、数据组织能力。这些价值在转为纯文本后完全丧失。

评分规则：
- 每个维度评分范围 0.0~1.0（1.0 为满分）
- 综合得分 = 各维度得分 × 权重 之和
- 当发现瑕疵时，必须给出具体的锚点定位（segment_id, start_char, end_char, snippet）
- 瑕疵类别：over_clean（过度清洗）, mis_edit（误改）, readability（可读性差）, structure（结构问题）
- 严重程度：critical（关键）, major（主要）, minor（次要）

输出要求：
1. **严格输出 JSON**，不要添加任何额外文字。
2. 瑕疵清单中的 location 必须包含 segment_id 字段以便锚点定位。
3. 每个维度的 reason 需简洁但具有说服力（可被人工复核）。
4. 如果治理后文本没有变化，需在 reason 中明确说明。

**【Few-shot 示例】**

示例1：表格结构破坏（应给低分）
输入：
- 原文：| 产品 | 价格 | 销量 |\n|------|------|------|\n| A产品 | 99元 | 1200 |
- 治理后：A产品99元售出1200件
正确输出：
{
  "dimensions": [
    {"dimension": "语义一致性", "score": 0.9, "weight": 0.4, "reason": "基本信息保留"},
    {"dimension": "过度清洗/误改识别", "score": 0.8, "weight": 0.3, "reason": "未删除有效信息"},
    {"dimension": "可读性", "score": 0.7, "weight": 0.15, "reason": "句子通顺"},
    {"dimension": "结构质量", "score": 0.2, "weight": 0.15, "reason": "表格结构被完全破坏"}
  ],
  "overall_score": 0.25,
  "flaws": [{"category": "structure", "severity": "critical", "description": "表格转纯文本", "location": {"segment_id": "1", "start_char": 0, "end_char": 20, "snippet": "A产品99元"}}]
}

示例2：过度压缩（应给低分）
输入：
- 原文：本次升级包括三个方面：第一，界面改版；第二，新增功能；第三，性能优化。
- 治理后：本次进行了升级。
正确输出：
{
  "dimensions": [
    {"dimension": "语义一致性", "score": 0.1, "weight": 0.4, "reason": "核心信息完全丢失"},
    {"dimension": "过度清洗/误改识别", "score": 0.0, "weight": 0.3, "reason": "严重过度清洗"},
    {"dimension": "可读性", "score": 0.8, "weight": 0.15, "reason": "句子通顺"},
    {"dimension": "结构质量", "score": 0.3, "weight": 0.15, "reason": "层级结构被破坏"}
  ],
  "overall_score": 0.20,
  "flaws": [{"category": "over_clean", "severity": "critical", "description": "三个要点全部删除", "location": {"segment_id": "1", "start_char": 0, "end_char": 10, "snippet": "本次进行了升级"}}]
}

示例3：良好治理（应给高分）
输入：
- 原文：尊敬的客户您好！您的订单[NO.20240315-88921]已发货。预计送达时间：2024年3月18日 下午。
- 治理后：尊敬的客户您好！您的订单[NO.20240315-88921]已发货。预计送达时间：2024年3月18日下午。
正确输出：
{
  "dimensions": [
    {"dimension": "语义一致性", "score": 1.0, "weight": 0.4, "reason": "完全保留原意"},
    {"dimension": "过度清洗/误改识别", "score": 1.0, "weight": 0.3, "reason": "仅修正格式"},
    {"dimension": "可读性", "score": 1.0, "weight": 0.15, "reason": "可读性提升"},
    {"dimension": "结构质量", "score": 1.0, "weight": 0.15, "reason": "结构完整"}
  ],
  "overall_score": 1.0,
  "flaws": []
}
"""


def build_system_prompt() -> str:
    """返回 System Prompt（含抗偏置指令）"""
    from .debias import generate_anti_bias_prompt_supplement
    return SYSTEM_PROMPT + "\n" + generate_anti_bias_prompt_supplement()


# ============================================================
# User Prompt 模板
# ============================================================
USER_PROMPT_TEMPLATE = """请评估以下治理前后文本对：

=== 治理前原文（BEFORE） ===
{before_text}

=== 治理后文本（AFTER） ===
{after_text}

=== 分段锚点信息（供定位参考） ===
{before_segments_info}
{after_segments_info}

请按以下 JSON 格式输出评估结果：

```json
{{
  "dimensions": [
    {{
      "dimension": "语义一致性",
      "score": 0.0~1.0,
      "weight": 0.4,
      "reason": "评分理由"
    }},
    {{
      "dimension": "过度清洗/误改识别",
      "score": 0.0~1.0,
      "weight": 0.3,
      "reason": "评分理由"
    }},
    {{
      "dimension": "可读性",
      "score": 0.0~1.0,
      "weight": 0.15,
      "reason": "评分理由"
    }},
    {{
      "dimension": "结构质量",
      "score": 0.0~1.0,
      "weight": 0.15,
      "reason": "评分理由"
    }}
  ],
  "overall_score": 0.0~1.0,
  "flaws": [
    {{
      "category": "over_clean|mis_edit|readability|structure",
      "severity": "critical|major|minor",
      "description": "瑕疵描述（可解释）",
      "location": {{
        "segment_id": "段落ID",
        "start_char": 0,
        "end_char": 0,
        "snippet": "相关原文片段"
      }},
      "suggestion": "修复建议（可选）"
    }}
  ]
}}
```

注意：
- 如果无瑕疵，flaws 为空列表 []
- overall_score 是四个维度的加权平均分
- 务必保证 JSON 格式正确，可被直接解析
"""


def build_user_prompt(
    before_text: str,
    after_text: str,
    segments_before: Optional[list[dict[str, Any]]] = None,
    segments_after: Optional[list[dict[str, Any]]] = None,
) -> str:
    """
    构建 User Prompt。

    使用 safe_format 避免文本中的 { } 导致 format() 报错。
    """
    before_segments_info = _format_segments(segments_before, "before")
    after_segments_info = _format_segments(segments_after, "after")
    return _safe_format(
        USER_PROMPT_TEMPLATE,
        before_text=before_text,
        after_text=after_text,
        before_segments_info=before_segments_info,
        after_segments_info=after_segments_info,
    )


def _safe_format(template: str, **kwargs: str) -> str:
    """
    安全的字符串替换：用占位符替换后再用 str.replace，
    避免用户文本中的 { } 导致 format() 抛出 KeyError。
    """
    result = template
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, value)
    return result


def _format_segments(
    segments: Optional[list[dict[str, Any]]], label: str
) -> str:
    """格式化分段信息"""
    if not segments:
        return f"（无 {label} 分段信息，请基于全文自行定位）"
    lines = [f"【{label} 分段信息】"]
    for seg in segments:
        sid = seg.get("segment_id", seg.get("id", "unknown"))
        text = seg.get("text", seg.get("content", ""))
        start = seg.get("offset", seg.get("start", 0))
        lines.append(f"  [{sid}] offset={start}: {text[:200]}")
    return "\n".join(lines)