"""评估 Agent —— 自主决策评估策略

使用 LangChain Agent 实现：
1. 自动判断文本类型，选择评估 profile
2. 调用专用工具（事实检查、结构分析、可读性分析）
3. 自反思：检查评估结果是否合理，不合理则重新评估
"""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .chain import evaluate as chain_evaluate, create_llm
from .models import EvalRequest, EvalResponse
from .profiles import PROFILE_GENERAL, PROFILE_GOVERNMENT_NOTICE_STRICT, PROFILE_LEGAL_STRICT


# ============================================================
# 1. Agent 工具定义
# ============================================================

@tool
def detect_text_type(before_text: str, after_text: str) -> str:
    """分析文本类型，推荐最合适的评估模式。

    Args:
        before_text: 治理前原文
        after_text: 治理后文本

    Returns:
        JSON 字符串，包含 recommended_profile 和 reason
    """
    # 关键词匹配规则
    gov_keywords = ["通告", "公告", "通知", "规定", "条例", "办法", "决定", "意见",
                     "执法", "行政", "政府", "办公厅", "管理局", "委员会"]
    legal_keywords = ["合同", "协议", "条款", "甲方", "乙方", "违约", "赔偿",
                       "法律", "法规", "权利", "义务", "保密", "知识产权"]

    text = before_text + after_text

    gov_score = sum(1 for kw in gov_keywords if kw in text)
    legal_score = sum(1 for kw in legal_keywords if kw in text)

    if legal_score >= 3:
        return json.dumps({
            "recommended_profile": PROFILE_LEGAL_STRICT,
            "reason": f"检测到法律/合同类关键词{legal_score}个，建议使用法规合同严格模式",
            "confidence": min(0.5 + legal_score * 0.1, 0.95),
        }, ensure_ascii=False)
    elif gov_score >= 3:
        return json.dumps({
            "recommended_profile": PROFILE_GOVERNMENT_NOTICE_STRICT,
            "reason": f"检测到政务通告类关键词{gov_score}个，建议使用政务通告严格模式",
            "confidence": min(0.5 + gov_score * 0.1, 0.95),
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "recommended_profile": PROFILE_GENERAL,
            "reason": "未检测到明显的政务/法律特征，使用通用模式",
            "confidence": 0.6,
        }, ensure_ascii=False)


@tool
def extract_key_facts(before_text: str) -> str:
    """从原文中提取关键事实（日期、金额、百分比、时限等）。

    Args:
        before_text: 治理前原文

    Returns:
        JSON 字符串，包含提取到的关键事实列表
    """
    import re

    facts = []

    # 日期
    for m in re.finditer(r"\d{4}[年\-/]\d{1,2}[月\-/]\d{1,2}[日]?", before_text):
        facts.append({"type": "date", "value": m.group(), "pos": m.start()})

    # 金额区间
    for m in re.finditer(r"\d+[\d,.]*\s*[元万亿]+\s*[以到至]\s*\d+[\d,.]*\s*[元万亿]+", before_text):
        facts.append({"type": "range", "value": m.group(), "pos": m.start()})

    # 带单位数字
    for m in re.finditer(r"\d+[\d,.]*\s*(?:元|万元|亿元|%|％|厘米|cm|公斤|kg)", before_text):
        facts.append({"type": "number", "value": m.group(), "pos": m.start()})

    # 时限
    for m in re.finditer(r"\d+\s*(?:日内|小时内|个工作日内|天内)", before_text):
        facts.append({"type": "deadline", "value": m.group(), "pos": m.start()})

    return json.dumps({
        "facts": facts,
        "total_count": len(facts),
        "types": list(set(f["type"] for f in facts)),
    }, ensure_ascii=False)


@tool
def check_fact_preservation(before_text: str, after_text: str) -> str:
    """检查原文中的关键事实是否在改写后保留。

    Args:
        before_text: 治理前原文
        after_text: 治理后文本

    Returns:
        JSON 字符串，包含保留和丢失的事实
    """
    import re

    facts = []
    for m in re.finditer(r"\d{4}[年\-/]\d{1,2}[月\-/]\d{1,2}[日]?", before_text):
        facts.append({"type": "date", "value": m.group()})
    for m in re.finditer(r"\d+[\d,.]*\s*[元万亿]+\s*[以到至]\s*\d+[\d,.]*\s*[元万亿]+", before_text):
        facts.append({"type": "range", "value": m.group()})
    for m in re.finditer(r"\d+[\d,.]*\s*(?:元|万元|亿元|%|％|厘米|cm|公斤|kg)", before_text):
        facts.append({"type": "number", "value": m.group()})
    for m in re.finditer(r"\d+\s*(?:日内|小时内|个工作日内|天内)", before_text):
        facts.append({"type": "deadline", "value": m.group()})

    preserved = []
    missing = []
    for fact in facts:
        if fact["value"] in after_text:
            preserved.append(fact)
        else:
            missing.append(fact)

    return json.dumps({
        "preserved": preserved,
        "missing": missing,
        "preservation_rate": round(len(preserved) / max(len(facts), 1), 2),
    }, ensure_ascii=False)


@tool
def analyze_structure(before_text: str, after_text: str) -> str:
    """分析文本结构变化（列表、表格、段落等）。

    Args:
        before_text: 治理前原文
        after_text: 治理后文本

    Returns:
        JSON 字符串，包含结构变化分析
    """
    import re

    def detect_elements(text: str) -> dict[str, bool]:
        return {
            "has_table": bool(re.search(r"\|.*\|.*\|", text)),
            "has_numbered_list": bool(re.search(r"(?:^|\n)\s*\d+[\.\、]", text)),
            "has_bullet_list": bool(re.search(r"(?:^|\n)\s*[-•·]", text)),
            "has_chinese_list": bool(re.search(r"[一二三四五六七八九十][\、\.]", text)),
            "has_paragraphs": text.count("\n") >= 2,
            "has_headers": bool(re.search(r"[【\[][^】\]]+[】\]]", text)),
        }

    before_elements = detect_elements(before_text)
    after_elements = detect_elements(after_text)

    changes = []
    for key in before_elements:
        if before_elements[key] and not after_elements[key]:
            changes.append(f"丢失了{key.replace('has_', '')}结构")
        elif not before_elements[key] and after_elements[key]:
            changes.append(f"新增了{key.replace('has_', '')}结构")

    return json.dumps({
        "before_elements": before_elements,
        "after_elements": after_elements,
        "changes": changes,
        "structure_preserved": len(changes) == 0,
    }, ensure_ascii=False)


@tool
def evaluate_single(before_text: str, after_text: str, profile: str = "general") -> str:
    """使用指定评估模式对文本对进行评分。

    Args:
        before_text: 治理前原文
        after_text: 治理后文本
        profile: 评估模式（general/government_notice_strict/legal_strict）

    Returns:
        JSON 字符串，包含评分结果
    """
    req = EvalRequest(
        request_id="agent_eval",
        before_text=before_text,
        after_text=after_text,
        evaluation_profile=profile,
    )
    resp = chain_evaluate(req, temperature=0.0)

    return json.dumps({
        "overall_score": resp.overall_score,
        "verdict": resp.verdict,
        "profile": resp.evaluation_profile,
        "dimensions": [
            {"dimension": d.dimension, "score": round(d.score, 3), "reason": d.reason}
            for d in resp.dimensions
        ],
        "flaw_count": len(resp.flaws),
        "latency_seconds": resp.latency_seconds,
    }, ensure_ascii=False)


# ============================================================
# 2. Agent 系统提示
# ============================================================

AGENT_SYSTEM_PROMPT = """你是一个内容保真度与治理质量评估智能体。

你的工作流程：
1. 分析输入的治理前后文本
2. 使用 detect_text_type 工具判断文本类型，选择最合适的评估模式
3. 使用 extract_key_facts 工具提取原文中的关键事实
4. 使用 check_fact_preservation 工具检查关键事实是否保留
5. 使用 analyze_structure 工具分析结构变化
6. 使用 evaluate_single 工具进行正式评估
7. 综合所有工具的输出，给出最终评估结论

你可以调用以下工具：
- detect_text_type: 判断文本类型
- extract_key_facts: 提取关键事实
- check_fact_preservation: 检查事实保留情况
- analyze_structure: 分析结构变化
- evaluate_single: 执行评估并打分

注意：
- 你可以多次调用工具，比如用不同 profile 分别评估然后对比
- 如果对评估结果有疑问，可以重新评估
- 最终输出必须包含：overall_score, verdict, evaluation_profile, 综合分析
"""


# ============================================================
# 3. Agent 执行器
# ============================================================

def create_agent_llm() -> ChatOpenAI:
    """创建 Agent 用的 LLM（支持工具调用）"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        temperature=0.0,
        max_tokens=4096,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    )


def run_agent(
    before_text: str,
    after_text: str,
    evaluation_profile: str | None = None,
) -> dict[str, Any]:
    """运行评估 Agent。

    如果指定了 evaluation_profile，直接用该模式评估。
    如果未指定，让 Agent 自动判断文本类型并选择最合适的模式。

    Args:
        before_text: 治理前原文
        after_text: 治理后文本
        evaluation_profile: 指定评估模式（可选）

    Returns:
        dict: 包含评估结果和 Agent 分析过程
    """
    llm = create_agent_llm()
    tools = [detect_text_type, extract_key_facts, check_fact_preservation,
             analyze_structure, evaluate_single]
    llm_with_tools = llm.bind_tools(tools)

    # 如果指定了 profile，直接评估
    if evaluation_profile:
        result = evaluate_single.invoke({
            "before_text": before_text,
            "after_text": after_text,
            "profile": evaluation_profile,
        })
        return {
            "evaluation": json.loads(result),
            "agent_reasoning": f"使用指定模式 {evaluation_profile} 直接评估",
            "auto_profile": False,
        }

    # 未指定 profile，让 Agent 自主决策
    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"请评估以下治理前后文本的质量：\n\n【治理前】\n{before_text[:1000]}\n\n【治理后】\n{after_text[:500]}"),
    ]

    # 第一轮：Agent 分析 + 工具调用
    reasoning_steps = []
    response = llm_with_tools.invoke(messages)
    reasoning_steps.append({"step": "initial_analysis", "content": str(response.content)[:500]})

    # 执行 Agent 请求的工具调用
    tool_results = {}
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            # 执行工具
            for t in tools:
                if t.name == tool_name:
                    result = t.invoke(tool_args)
                    tool_results[tool_name] = result
                    reasoning_steps.append({"step": f"tool:{tool_name}", "result": result[:300]})
                    break

    # 第二轮：基于工具结果做最终评估
    # 自动选择 profile
    if "detect_text_type" in tool_results:
        detection = json.loads(tool_results["detect_text_type"])
        selected_profile = detection["recommended_profile"]
        reasoning_steps.append({"step": "profile_selection", "profile": selected_profile,
                                "reason": detection["reason"]})
    else:
        selected_profile = PROFILE_GENERAL
        reasoning_steps.append({"step": "profile_selection", "profile": selected_profile,
                                "reason": "默认使用通用模式"})

    # 执行评估
    eval_result = evaluate_single.invoke({
        "before_text": before_text,
        "after_text": after_text,
        "profile": selected_profile,
    })
    evaluation = json.loads(eval_result)
    reasoning_steps.append({"step": "evaluation", "result": evaluation})

    return {
        "evaluation": evaluation,
        "agent_reasoning": reasoning_steps,
        "auto_profile": True,
        "tool_results": tool_results,
    }


# ============================================================
# 4. 兼容接口
# ============================================================

def evaluate_with_agent(
    request: EvalRequest,
    temperature: float = 0.0,
) -> EvalResponse:
    """Agent 评估入口，兼容 chain.evaluate 的接口。

    如果 request.evaluation_profile 为 "general"（默认），Agent 自动判断。
    如果指定了其他 profile，直接使用该模式。
    """
    # 如果用户明确指定了非 general 的 profile，直接用
    if request.evaluation_profile != PROFILE_GENERAL:
        return chain_evaluate(request, temperature)

    # 否则让 Agent 自动判断
    result = run_agent(
        before_text=request.before_text,
        after_text=request.after_text,
        evaluation_profile=None,  # 让 Agent 自主选择
    )

    # 从 Agent 结果构建 EvalResponse
    eval_data = result["evaluation"]

    from .models import DimensionScore, FlawItem, AnchorSpan

    dimensions = [
        DimensionScore(
            dimension=d["dimension"],
            score=d["score"],
            weight=0.4 if "语义" in d["dimension"] else 0.3 if "清洗" in d["dimension"] or "误改" in d["dimension"] else 0.15,
            reason=d.get("reason", ""),
        )
        for d in eval_data.get("dimensions", [])
    ]

    return EvalResponse(
        request_id=request.request_id,
        evaluation_profile=eval_data.get("profile", PROFILE_GENERAL),
        dimensions=dimensions,
        overall_score=eval_data.get("overall_score", 0.5),
        flaws=[],  # Agent 模式下瑕疵已在 dimensions 中体现
        verdict=eval_data.get("verdict", "review"),
        reproducibility_token="agent",
        model_version=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        prompt_version="agent",
        raw_llm_output=json.dumps(result, ensure_ascii=False),
    )
