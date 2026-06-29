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

**【信息精简 vs 信息丢失 — 必须区分】**
- **合理精简**（得分 0.6-0.8）：删除冗余修饰、简化表达，但核心信息（数据、结论、关键事实）完整保留。例如删除"构建智能制造体系"的修饰语但保留"数字化转型"方向。
- **过度清洗**（得分 0.3-0.6）：删除了具体措施、量化目标、限定条件等有价值信息，但核心主题仍在。
- **严重丢失**（得分 < 0.3）：核心数据、关键结论、法律要件被删除或篡改。

**区分标准：**
- 删除的是"锦上添花的细节"还是"不可或缺的信息"？
- 改写后读者能否做出与原文相同的决策？
- 法律/合同中的限定条件（时间范围、前提条件、例外条款）具有法律效力，删除属于严重丢失。

**【一票否决规则 - 必须严格遵守】**
以下情况，整体得分（overall_score）必须 ≤ 0.3：
1. 表格被完全破坏（从表格转为纯文本）- 这是严重的结构破坏
2. 核心数据（金额、百分比、日期）被错误修改 - 这是严重的误改
3. 法律条款中的责任主体、义务、后果被删除或弱化

**注意：合理精简不适用一票否决。** 如果原文要点被压缩但核心信息仍在，得分应在 0.5-0.7 之间。
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
3. 每个维度的 reason **不超过 20 个汉字**，简洁有力。
4. 瑕疵的 description **不超过 30 个汉字**。
5. 如果无瑕疵，flaws 为空列表 []，不要输出空对象。
6. 如果治理后文本没有变化，reason 写"完全一致"。
7. **所有字段必须精简输出，禁止冗余描述。**

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

示例4：合理精简（应给中高分，不是低分）
输入：
- 原文：公司2024年度战略重点包括：深化数字化转型，构建智能制造体系；加强供应链韧性建设，优化全球采购网络；持续推进绿色低碳发展，争取2030年实现碳中和目标；积极拓展东南亚市场，预计新增营收15亿元。
- 治理后：公司2024年度战略重点包括：深化数字化转型；加强供应链建设；推进绿色发展；拓展东南亚市场。
正确输出：
{
  "dimensions": [
    {"dimension": "语义一致性", "score": 0.6, "weight": 0.4, "reason": "核心战略方向保留，具体措施和量化目标丢失"},
    {"dimension": "过度清洗/误改识别", "score": 0.5, "weight": 0.3, "reason": "删除了具体措施和量化目标"},
    {"dimension": "可读性", "score": 0.8, "weight": 0.15, "reason": "表达精简通顺"},
    {"dimension": "结构质量", "score": 0.7, "weight": 0.15, "reason": "结构基本完整"}
  ],
  "overall_score": 0.62,
  "flaws": [{"category": "over_clean", "severity": "major", "description": "删除了具体措施和量化目标", "location": {"segment_id": "1", "start_char": 10, "end_char": 50, "snippet": "深化数字化转型；加强供应链建设"}}]
}
"""


# ============================================================
# Profile-specific prompt 补充规则
# ============================================================

_PROFILE_SUPPLEMENTS: dict[str, str] = {
    "government_notice_strict": """
**【政务通告严格保真模式 — 已启用】**

本次评估对象为政务通告/公告/通知类文本，适用更严格的保真标准。

以下信息属于关键事实，删除、泛化或区间抹平**不能视为高保真**：
- 生效日期、失效日期、截止日期、有效期
- 办理时限、报告时限、响应时限
- 金额、罚款区间、收费标准
- 百分比、数量门槛、面积/重量/身高等阈值
- 适用范围、区域范围、对象范围
- 豁免对象、例外对象
- 处罚措施、责任后果

具体示例：
- 将"罚款 2000~5000 元"改成"处罚款" → 属于显著信息损失，至少标为 major
- 将"有效期至 2027年5月31日"删除 → 属于关键信息删失，标为 critical
- 将"肩高 61 厘米或体重 30 公斤"改成"大型犬" → 属于阈值丢失，标为 major 以上
- 将"投诉量增长 42.3%"删除 → 属于数据删失，标为 major
""",

    "legal_strict": """
**【法规合同严格保真模式 — 已启用】**

本次评估对象为法规、合同、条款、协议、制度规范类文本，适用更严格的保真标准。

以下信息属于关键事实，删除或模糊化**不能视为高保真**：
- 责任主体（谁承担义务/责任）
- 义务与禁止性规定（必须做什么、不得做什么）
- 条件触发（在什么条件下生效/适用）
- 例外条款（什么情况除外）
- 法律后果或违约后果（违反后承担什么责任）

具体规则：
- 删除责任主体 → 至少标为 major
- 弱化义务或禁止性规定 → 至少标为 major
- 删除条件限制 → 至少标为 major
- 删除或弱化法律后果 → 标为 critical
- "维持大意"不足以视为高保真，关键条款边界必须精确保留
""",
}

_PROFILE_FEW_SHOTS: dict[str, str] = {
    "government_notice_strict": """
示例4：政务通告关键事实丢失（应给低分）
输入：
- 原文：根据《养犬管理条例》，重点管理区内禁止饲养肩高超过61厘米或体重超过30公斤的犬只。违反规定的，处2000元以上5000元以下罚款。本通告自发布之日起施行，有效期至2027年5月31日。
- 治理后：根据相关规定，重点管理区内禁止饲养大型犬。违反规定的将予以处罚。
正确输出：
{
  "dimensions": [
    {"dimension": "语义一致性", "score": 0.2, "weight": 0.4, "reason": "关键事实大量丢失：犬只判定阈值、罚款区间、有效期均被删除"},
    {"dimension": "过度清洗/误改识别", "score": 0.1, "weight": 0.3, "reason": "严重过度清洗，核心数据被泛化或删除"},
    {"dimension": "可读性", "score": 0.8, "weight": 0.15, "reason": "句子通顺"},
    {"dimension": "结构质量", "score": 0.6, "weight": 0.15, "reason": "结构基本完整但信息密度大幅下降"}
  ],
  "overall_score": 0.25,
  "flaws": [
    {"category": "over_clean", "severity": "critical", "description": "犬只判定阈值（肩高61cm/体重30kg）被删除", "location": {"segment_id": "1", "start_char": 0, "end_char": 50, "snippet": "肩高超过61厘米或体重超过30公斤"}},
    {"category": "over_clean", "severity": "critical", "description": "罚款区间（2000~5000元）被泛化为'处罚'", "location": {"segment_id": "1", "start_char": 80, "end_char": 120, "snippet": "2000元以上5000元以下"}},
    {"category": "mis_edit", "severity": "critical", "description": "有效期截止日被删除", "location": {"segment_id": "1", "start_char": 130, "end_char": 160, "snippet": "有效期至2027年5月31日"}}
  ]
}
""",

    "legal_strict": """
示例5：法律条款责任后果弱化（应给低分）
输入：
- 原文：承租人未经出租人书面同意，不得将租赁物转租给第三方。违反本条规定的，出租人有权解除合同并要求承租人支付违约金人民币伍万元整。
- 治理后：承租人不得擅自转租租赁物。违反规定的，出租人可解除合同。
正确输出：
{
  "dimensions": [
    {"dimension": "语义一致性", "score": 0.3, "weight": 0.4, "reason": "关键法律要件丢失：书面同意要求、违约金金额被删除"},
    {"dimension": "过度清洗/误改识别", "score": 0.2, "weight": 0.3, "reason": "严重误改，法律后果被弱化"},
    {"dimension": "可读性", "score": 0.9, "weight": 0.15, "reason": "句子通顺"},
    {"dimension": "结构质量", "score": 0.7, "weight": 0.15, "reason": "结构完整"}
  ],
  "overall_score": 0.30,
  "flaws": [
    {"category": "mis_edit", "severity": "critical", "description": "违约金（伍万元整）被删除，法律后果被弱化", "location": {"segment_id": "1", "start_char": 60, "end_char": 100, "snippet": "违约金人民币伍万元整"}},
    {"category": "over_clean", "severity": "major", "description": "书面同意要求被简化为'擅自'", "location": {"segment_id": "1", "start_char": 0, "end_char": 30, "snippet": "未经出租人书面同意"}}
  ]
}
""",
}


def build_system_prompt(evaluation_profile: str = "general") -> str:
    """返回 System Prompt（含 profile-specific 规则与抗偏置指令）"""
    from .debias import generate_anti_bias_prompt_supplement
    from .profiles import get_profile_config

    prompt = SYSTEM_PROMPT

    # 叠加 profile-specific 补充规则
    supplement = _PROFILE_SUPPLEMENTS.get(evaluation_profile, "")
    if supplement:
        prompt += "\n" + supplement

    # 叠加 profile-specific few-shot
    extra_few_shot = _PROFILE_FEW_SHOTS.get(evaluation_profile, "")
    if extra_few_shot:
        prompt += "\n" + extra_few_shot

    # 叠加抗偏置指令
    prompt += "\n" + generate_anti_bias_prompt_supplement()

    return prompt


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