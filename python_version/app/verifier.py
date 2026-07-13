"""两阶段瑕疵验证器 —— 第二阶段：高精度校验（Critic / Verifier）

设计思路：
  第一阶段（现有评估流程）：LLM 扮演"搜寻员"，高召回地检出所有候选瑕疵。
  第二阶段（本模块）：LLM 扮演"大法官"，对每条候选瑕疵做二分类判定——
    保留（keep）：确实是治理引入的实质性问题
    剔除（reject）：正常改写、措辞微调、同义替换等不构成瑕疵

核心创新：对比型 Few-shot（Contrastive Examples）
  对每个类别提供"非瑕疵（放行）"与"实质瑕疵（拦截）"的成对反例，
  让 LLM 通过对比学会区分边界，比抽象规则更有效。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .models import FlawItem

logger = logging.getLogger(__name__)


# ============================================================
# Verifier System Prompt
# ============================================================

VERIFIER_SYSTEM_PROMPT = """你是一个瑕疵校验专家（Verifier）。你的任务是判断"候选瑕疵"是否为真正的治理问题。

**背景**：第一阶段评估已经初步检出了一批候选瑕疵，但其中可能包含误报（False Positive）。
你需要逐条审查，判断每条候选瑕疵是"真正的瑕疵"还是"正常改写被误判"。

**【铁律 — 只读审计员，禁止越权】**
你是一个只读的审计员。你的唯一职责是对输入的候选瑕疵列表逐条进行 keep 或 reject 投票。
- 你绝对无权添加任何新的瑕疵
- 你绝对无权修改原瑕疵的类别、严重程度或定位信息
- 如果你认为某条候选瑕疵不是真正的问题，直接输出 reject 即可
- 不要在输出中指出"第一阶段遗漏了哪些问题"——那不是你的职责

**判定标准——以下情况应判定为"非瑕疵"（reject）：**
- 正常的同义词替换（如"有限公司"→"科技公司"中"有限"被删，但主体名称核心未变）
- 合理的措辞精简（如"为了确保"→"为"，"各位用户"→"用户"）
- 标点符号、空格、换行的正常调整
- 语序调整但语义完全保留
- 连接词/过渡词的合理增删（不影响实质内容）

**判定标准——以下情况应判定为"真瑕疵"（keep）：**
- 关键数据/数值被改变或删除
- 责任主体、义务、条件等核心法律要素被删除或弱化
- 信息结构被破坏（表格被拍平、列表被压缩为单句）
- 新增了原文不存在的信息（幻觉）
- 关键限定词/约束条件被删除导致语义范围扩大
- 因果/逻辑关系被打乱

**对比型示例（Contrastive Examples）：**

--- 示例 1：over_clean 类别 ---
【非瑕疵 → reject】
原文："北京泰极科技有限公司发布了通知"
治理后："北京泰极科技公司发布了通知"
候选瑕疵：over_clean - 删除了"有限"二字
判定理由：公司名称核心"北京泰极科技"完整保留，"有限"属于法律形式后缀，在日常语境中删除不影响主体识别和语义传达。→ reject

【实质瑕疵 → keep】
原文："肩高超过61厘米或体重超过30公斤的犬只禁止饲养"
治理后："大型犬禁止饲养"
候选瑕疵：over_clean - 犬只判定阈值（肩高61cm/体重30kg）被删除
判定理由：具体的量化标准被替换为模糊分类，读者无法判断具体标准，属于实质性信息损失。→ keep

--- 示例 2：mis_edit 类别 ---
【非瑕疵 → reject】
原文："投诉量增长42.3%"
治理后："投诉量增长约42%"
候选瑕疵：mis_edit - 42.3%被改为约42%
判定理由：数值精度微调（去掉小数位+加"约"），在摘要/概述场景下属于合理的近似表达，核心数量级和趋势完全一致。→ reject

【实质瑕疵 → keep】
原文："罚款200元"
治理后："罚款500元"
候选瑕疵：mis_edit - 罚款金额从200元被改为500元
判定理由：数值被实质性篡改，直接影响法律效力和经济后果。→ keep

--- 示例 3：structure 类别 ---
【非瑕疵 → reject】
原文段落包含3个并列要点，治理后用一段连贯文字表达了相同内容，要点之间用"此外""同时"等连接词串联。
候选瑕疵：structure - 列表被改为段落
判定理由：虽然形式从列表变为段落，但逻辑关系（并列）通过连接词完整保留，信息层级未丢失，属于合理的格式调整。→ reject

【实质瑕疵 → keep】
原文：一个包含5行3列的对比表格
治理后：表格被拍平为逗号连接的散文
候选瑕疵：structure - 表格结构被破坏
判定理由：表格的行列对比关系完全丧失，读者无法快速对比各项数据，结构化信息的对比性和可读性严重下降。→ keep

--- 示例 4：readability 类别 ---
【非瑕疵 → reject】
原文："请各位用户注意以下事项"
治理后："请用户注意以下事项"
候选瑕疵：readability - 删除了"各位"
判定理由：删去敬语量词"各位"不影响句意和可读性，属于正常的精简。→ reject

【实质瑕疵 → keep】
原文："为确保系统安全稳定运行，请各位用户注意以下事项"
治理后："为保系统安全稳定运行请各用户注意下事"
候选瑕疵：readability - 过度缩写导致阅读困难
判定理由：大量连接词和修饰词被删除，文本变成电报体，严重影响可读性。→ keep

**输出格式（思维链优先 — 必须先推理再判定）：**
严格按以下 JSON 输出，不要输出其他内容。注意：reason 必须在 verdict 之前，先写出你的分析推理过程，再给出最终裁决。
```json
{
  "verifications": [
    {
      "index": 0,
      "reason": "先写分析：该瑕疵涉及的变更是什么，是否影响实质语义，为什么构成/不构成问题",
      "verdict": "keep"
    },
    {
      "index": 1,
      "reason": "先写分析：该变更属于同义替换/措辞精简，未造成实质信息损失",
      "verdict": "reject"
    }
  ]
}
```
"""


# ============================================================
# Verifier User Prompt Template
# ============================================================

VERIFIER_USER_PROMPT_TEMPLATE = """请逐条审查以下候选瑕疵，判断每条是"真瑕疵"还是"误报"。

=== 治理前原文 ===
{before_text}

=== 治理后文本 ===
{after_text}

=== 候选瑕疵列表（共 {n_flaws} 条） ===
{flaws_text}

请逐条判定，严格按 JSON 格式输出。
"""


# ============================================================
# 核心函数
# ============================================================

def _extract_local_context(text: str, snippet: str, window: int = 50) -> str:
    """提取 snippet 在原文中的局部上下文滑窗。

    Args:
        text: 原文全文
        snippet: 瑕疵定位的文本片段
        window: 前后各扩展的字符数

    Returns:
        包含上下文的局部文本片段，用 [...] 标注截断
    """
    if not snippet or not text:
        return snippet or ""
    pos = text.find(snippet[:20]) if len(snippet) >= 20 else text.find(snippet)
    if pos < 0:
        return snippet
    start = max(0, pos - window)
    end = min(len(text), pos + len(snippet) + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def _format_flaws_for_verifier(
    flaws: list[FlawItem],
    before_text: str = "",
    after_text: str = "",
) -> str:
    """将瑕疵列表格式化为 Verifier 可读的文本，含局部上下文滑窗"""
    lines = []
    for i, flaw in enumerate(flaws):
        loc = flaw.location
        # 提取局部上下文：优先在 after_text 中查找，找不到则用 before_text
        local_ctx = _extract_local_context(after_text, loc.snippet)
        if not local_ctx or local_ctx == loc.snippet:
            local_ctx = _extract_local_context(before_text, loc.snippet)
        lines.append(
            f"[候选瑕疵 {i}] 类别={flaw.category} | 严重程度={flaw.severity}\n"
            f"  描述：{flaw.description}\n"
            f"  原文片段：{loc.snippet}\n"
            f"  局部上下文：{local_ctx}"
        )
    return "\n\n".join(lines)


def verify_flaws(
    before_text: str,
    after_text: str,
    candidate_flaws: list[FlawItem],
    temperature: float = 0.0,
) -> list[FlawItem]:
    """
    第二阶段验证：对候选瑕疵做二分类校验，剔除误报。

    Args:
        before_text: 治理前原文
        after_text: 治理后文本
        candidate_flaws: 第一阶段检出的候选瑕疵列表
        temperature: LLM 温度（建议 0.0 保证确定性）

    Returns:
        list[FlawItem]: 通过验证的瑕疵列表（误报已被剔除）
    """
    if not candidate_flaws:
        return []

    print(f"[Verifier] 收到 {len(candidate_flaws)} 条候选瑕疵，开始第二阶段校验...")

    # 构建验证 prompt
    flaws_text = _format_flaws_for_verifier(candidate_flaws, before_text, after_text)
    from .prompts import _safe_format
    user_prompt = _safe_format(
        VERIFIER_USER_PROMPT_TEMPLATE,
        before_text=before_text,
        after_text=after_text,
        n_flaws=str(len(candidate_flaws)),
        flaws_text=flaws_text,
    )

    # 调用 LLM
    from .chain import create_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = create_llm(temperature=temperature, json_mode=True)
    messages = [
        SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw_output = str(response.content) if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning(f"[Verifier] LLM 调用失败，保留全部候选瑕疵: {e}")
        return candidate_flaws

    # 解析结果
    try:
        # 提取 JSON（兼容 markdown code block 包裹）
        json_str = raw_output
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]

        result = json.loads(json_str.strip())
        # 兼容 LLM 返回数组或对象两种格式
        if isinstance(result, list):
            verifications = result
        else:
            verifications = result.get("verifications", [])

        # 按 index 建立判定映射
        verdict_map: dict[int, str] = {}
        for v in verifications:
            idx = v.get("index", -1)
            verdict = v.get("verdict", "keep").lower()
            if idx >= 0:
                verdict_map[idx] = verdict

        # 过滤：只保留 verdict=keep 的瑕疵
        kept = []
        rejected_count = 0
        for i, flaw in enumerate(candidate_flaws):
            verdict = verdict_map.get(i, "keep")  # 默认保留（保守策略）
            if verdict == "keep":
                kept.append(flaw)
            else:
                rejected_count += 1
                logger.info(
                    f"[Verifier] 剔除误报: [{flaw.category}] {flaw.description}"
                )

        print(
            f"[Verifier] 验证完成: {len(candidate_flaws)} 条候选 → "
            f"{len(kept)} 条保留, {rejected_count} 条剔除"
        )
        return kept

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"[Verifier] 解析失败，保留全部候选瑕疵: {e}")
        return candidate_flaws
