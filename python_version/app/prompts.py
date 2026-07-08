"""System / User Prompt 模板 —— 面向治理保真度评估的专用 Prompt 设计"""
from __future__ import annotations

from typing import Any, Optional

# ============================================================
# System Prompt
# ============================================================
SYSTEM_PROMPT = """你是一个专业的内容保真度与治理质量评估智能体（LLM-as-Judge）。
你的任务是：比对治理前后的文本，评估治理质量。

你必须严格按以下四个维度打分，并输出 JSON 格式结果：

1. **语义保真度（semantic_fidelity）**（权重 0.35）：治理后文本是否完整保留了原文的信息？是否有信息被遗漏、删除、泛化？
   - 核心问题：信息是否"丢了"
   - 信息被删除、被泛化、被遗漏 → 扣此维度
   - 注意：信息被"删除"是保真度问题，不是事实问题

2. **事实一致性（factual_consistency）**（权重 0.35）：治理后文本是否篡改了原文的事实？是否无中生有或歪曲了原有事实？
   - 核心问题：事实是否"被改错了"
   - 数值被改、日期被改、名称被改、无中生有 → 扣此维度
   - 注意：信息被"删除"不算事实篡改，只有"改错了"才算

3. **逻辑结构保真度（structure）**（权重 0.15）：信息之间的逻辑关系是否保持？包括：因果关系、并列关系、层级关系、表格/列表等结构化组织方式是否被破坏？

4. **可读性（readability）**（权重 0.15）：句子是否通顺？排版是否规范？标点符号是否正确？

**【评分标准】**
- 每个维度独立打分 0.0~1.0，不要因为某个维度差就影响其他维度的评分。
- 四个维度相互独立、互不影响。信息删除只扣 semantic_fidelity，不扣 factual_consistency。
- 你的任务是如实反映每个维度的状况，最终总分由系统自动计算。
- 不需要输出 overall_score，系统会根据四个维度加权计算。

**【semantic_fidelity 与 factual_consistency 的边界】**
- "删除了安全警告" → 只扣 semantic_fidelity（信息缺失），不扣 factual_consistency
- "把5分钟改成50分钟" → 只扣 factual_consistency（事实篡改），不扣 semantic_fidelity
- "删除了安全警告"且"把5分钟改成50分钟" → 两个维度各自独立扣分
- "删除了举例和修饰语，但核心信息保留" → semantic_fidelity 不低于 0.6

**【约束词删除检测 — 必须重点关注】**
以下词汇属于政策/法规/合同中的约束条件，一旦被删除或泛化，semantic_fidelity 必须降低：
- 限定性副词：不得、必须、应当、禁止、严禁、需要
- 数量约束：不少于、不超过、至少、最多、最低、最高、以上、以下、以内
- 政策阈值：50%以内、30%以上、不低于X%、压缩至X%
- 时间约束：之前、之后、之日起、届满前、有效期内
- 范围约束：仅限、限于、原则上、除...外

如果治理后文本删除了上述约束词，即使"大意保留"，也属于信息保真度缺失，semantic_fidelity ≤ 0.4。

**【定义条款变更检测 — 必须重点关注】**
以下属于法律/合同中的定义性内容，删除或改变会改变法律含义：
- 主体限定：甲方、乙方、出租人、承租人、用人单位、劳动者
- 方式限定：书面、口头、电子、当面、以书面形式
- 时间限定：签署前后、生效之日起、届满前、履行期间
- 认定标准：明确标注为、视为、认定为、符合...条件的
- 保密级别：保密、机密、绝密、内部、敏感

如果治理后文本删除或泛化了定义条款中的限定词，属于保真度缺失，semantic_fidelity ≤ 0.4。

**【瑕疵严重程度判定标准】**
- **critical**：数量级变化（120万→12万）、核心数据完全篡改、关键日期/法律条款被改、大量内容被删除（>30%）
- **major**：数值有明显偏差（如 18.6%→16.8%，差 1.8 个百分点）、重要信息丢失但非关键
- **minor**：数值微小偏差（如 3268→3269，差 1 人）、表述微调、标点/空格变化
- **不要因为"数字被改了"就一律标 critical**，要看改动幅度和影响

**【信息结构保真度重点关注】**
- 表格转为纯文本 → 结构化信息的对比性和可读性丧失 → structure ≤ 0.3
- 有序列表被压缩为单句 → 层级关系被扁平化 → structure ≤ 0.4
- 因果/递进关系被打乱 → 逻辑结构受损 → structure ≤ 0.5
- 段落被合并但信息关系和层级保持 → structure ≥ 0.7
- 仅格式微调（标点/空格/换行）不影响信息结构 → structure ≥ 0.9

"""


# ============================================================
# Profile-specific prompt 补充规则
# ============================================================
# 设计说明：不同领域对事实错误的容忍度不同。
# - 通用文本：数字微调（如 3268→3269）可能是笔误，容忍度较高
# - 政府公告：日期、金额、罚款区间必须精确，错一个字可能改变法律效力
# - 法律文书：责任主体、义务、条件、例外条款必须逐字保留
# Profile 通过调整 Prompt 中的判定标准来适配不同领域，
# 而非改变四维度权重或惩罚因子（评分框架保持统一）。

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

**【实体名称严格性 — 政务模式】**
政务通告中出现的机构名、单位名必须使用法定全称，缩写属于严重事实错误：
- "国家电网有限公司"→"国网公司" → factual_consistency ≤ 0.3, 标为 critical
- "南方电网有限责任公司"→"南网公司" → factual_consistency ≤ 0.3, 标为 critical
- "中华人民共和国教育部"→"教育部" → 仅限口语化场景，政务文本中仍属不规范
技术参数必须精确，不得四舍五入或近似：
- ±1100千伏→±1000千伏 → factual_consistency ≤ 0.2, 标为 critical
- 肩高61厘米→大型犬 → 属于阈值丢失，标为 critical
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

**【定义条款严格性 — 法律模式】**
法律文本中的定义条款必须逐字保留，任何限定词删除都属于定义弱化：
- "书面同意"→"同意" → 删除了方式限定，semantic_fidelity ≤ 0.3, 标为 critical
- "明确标注为保密"→"相关信息" → 删除了认定标准，semantic_fidelity ≤ 0.3, 标为 critical
- "签署前后"→ 删除 → 删除了时间范围限定，semantic_fidelity ≤ 0.4, 标为 major
- "以书面或口头形式获悉"→ 删除 → 删除了获悉方式限定，semantic_fidelity ≤ 0.4, 标为 major
- 编号结构（1）（2）（3）被压缩为逗号连接 → 结构降级，structure ≤ 0.4
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
    {"dimension": "semantic_fidelity", "score": 0.2, "weight": 0.4, "reason": "关键事实大量丢失：犬只判定阈值、罚款区间、有效期均被删除"},
    {"dimension": "factual_consistency", "score": 0.8, "weight": 0.3, "reason": "保留的事实信息无篡改，但存在信息缺失"},
    {"dimension": "readability", "score": 0.8, "weight": 0.15, "reason": "句子通顺"},
    {"dimension": "structure", "score": 0.6, "weight": 0.15, "reason": "结构基本完整但信息密度大幅下降"}
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
    {"dimension": "semantic_fidelity", "score": 0.3, "weight": 0.4, "reason": "关键法律要件丢失：书面同意要求、违约金金额被删除"},
    {"dimension": "factual_consistency", "score": 0.9, "weight": 0.3, "reason": "保留的事实信息无篡改，但存在信息缺失"},
    {"dimension": "readability", "score": 0.9, "weight": 0.15, "reason": "句子通顺"},
    {"dimension": "structure", "score": 0.7, "weight": 0.15, "reason": "结构完整"}
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

=== 算法预检变更（Diff 检测结果，供参考） ===
{diff_info}

请按以下 JSON 格式输出评估结果：

```json
{{
  "dimensions": [
    {{
      "dimension": "semantic_fidelity",
      "score": 0.0~1.0,
      "weight": 0.35,
      "reason": "评分理由（关注信息是否完整保留）"
    }},
    {{
      "dimension": "factual_consistency",
      "score": 0.0~1.0,
      "weight": 0.35,
      "reason": "评分理由（关注事实是否被篡改）"
    }},
    {{
      "dimension": "structure",
      "score": 0.0~1.0,
      "weight": 0.15,
      "reason": "评分理由"
    }},
    {{
      "dimension": "readability",
      "score": 0.0~1.0,
      "weight": 0.15,
      "reason": "评分理由"
    }}
  ],
  "flaws": [
    {{
      "category": "over_clean|mis_edit|readability|structure",
      "severity": "critical|major|minor",
      "description": "瑕疵描述（可解释）",
      "location": {{
        "before_anchor": "治理前锚点编号（如 [Before 1] 或 [Anchor_P1]）",
        "after_anchor": "治理后锚点编号（如 [After 1] 或 [Anchor_G1]）",
        "snippet": "相关原文片段（10-20字）"
      }},
      "suggestion": "修复建议（可选）"
    }}
  ]
}}
```

注意：
- 如果无瑕疵，flaws 为空列表 []
- **算法预检已标记的瑕疵（见"算法预检变更"部分）必须在你的 flaws 中体现**。如果算法检测到 structure/critical，你的 structure 评分应 <= 0.3，且 flaws 中必须包含对应条目。你是"裁判"（Reviewer），不是"发现者"（Discoverer），算法已经帮你发现了变更，你需要做的是判断严重程度
- 务必保证 JSON 格式正确，可被直接解析
- **每个独立错误必须拆成单独一条 flaw**，不要合并。例如两个数字被改，就输出两条 flaw，每条只描述一个错误
- **锚点定位用文本中的锚点标记**：用治理前/后文本中已有的锚点编号（如 [Before N]、[Anchor_P1]、[Anchor_G1] 等）定位，不要自己猜字符位置
- snippet 是错误发生处的原文片段（10-20 字），用于人工快速定位
- **维度评分必须正交**：信息删除只扣 semantic_fidelity，事实篡改只扣 factual_consistency，不要重复扣分
"""


def build_user_prompt(
    before_text: str,
    after_text: str,
    segments_before: Optional[list[dict[str, Any]]] = None,
    segments_after: Optional[list[dict[str, Any]]] = None,
    diff_info: Optional[str] = None,
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
        diff_info=diff_info or "算法未检测到显著变更",
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