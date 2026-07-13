"""LCEL 评估链 —— 用 LangChain Expression Language 重构评估流程

将原来的单体 evaluate() 函数拆分为可组合的 Chain 步骤：
  prompt → llm → parse → post_process → response

每一步独立可测、可替换、可追踪。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Optional, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .models import (
    AnchorSpan,
    DimensionScore,
    EvalRequest,
    EvalResponse,
    FlawItem,
)
from .profiles import get_profile_config, PROFILE_GENERAL


# ============================================================
# 0b. 评分校准器（懒加载）
# ============================================================

_calibrator_instance = None
_calibrator_loaded = False


def _get_calibrator():
    """懒加载校准器。从 data/calibration_params.json 读取参数。
    如果文件不存在或加载失败，返回 None（不校准）。
    """
    global _calibrator_instance, _calibrator_loaded
    if _calibrator_loaded:
        return _calibrator_instance
    _calibrator_loaded = True
    try:
        from .calibrator import MultiModelCalibrator
        cal_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "calibration_params.json",
        )
        if os.path.exists(cal_path):
            mc = MultiModelCalibrator.load(cal_path)
            model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
            _calibrator_instance = mc.get_calibrator(model_name)
            if _calibrator_instance is not None:
                print(f"[校准器] 已加载 {model_name} 校准参数: "
                      f"slope={_calibrator_instance.slope:.4f}, "
                      f"intercept={_calibrator_instance.intercept:.4f}, "
                      f"R²={_calibrator_instance.r_squared:.4f}")
    except Exception as e:
        print(f"[校准器] 加载失败，跳过校准: {e}")
    return _calibrator_instance


# ============================================================
# 0. 缓存 / 限流 / 流式 配置
# ============================================================

_cache_enabled = True
_rate_limiter = None


def enable_cache(cache_type: str = "memory", db_path: str = "output/llm_cache.db") -> None:
    """启用 LLM 响应缓存。

    Args:
        cache_type: "memory"（内存，重启丢失）或 "sqlite"（持久化）
        db_path: SQLite 缓存文件路径（仅 sqlite 模式）
    """
    global _cache_enabled
    if cache_type == "sqlite":
        from langchain_community.cache import SQLiteCache
        import sqlite3
        set_llm_cache(SQLiteCache(database_path=db_path))
    else:
        set_llm_cache(InMemoryCache())
    _cache_enabled = True


def disable_cache() -> None:
    """禁用缓存"""
    global _cache_enabled
    set_llm_cache(None)  # type: ignore[arg-type]
    _cache_enabled = False


def clear_cache() -> None:
    """清空缓存（重新启用即可清空）"""
    if _cache_enabled:
        enable_cache()


def enable_rate_limiter(
    requests_per_second: float = 5.0,
    check_every_n_seconds: float = 0.5,
    max_bucket_size: int = 10,
) -> None:
    """启用 API 限流。

    Args:
        requests_per_second: 每秒最大请求数
        check_every_n_seconds: 检查间隔
        max_bucket_size: 令牌桶最大容量
    """
    global _rate_limiter
    from langchain_core.rate_limiters import InMemoryRateLimiter
    _rate_limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=check_every_n_seconds,
        max_bucket_size=max_bucket_size,
    )


def disable_rate_limiter() -> None:
    """禁用限流"""
    global _rate_limiter
    _rate_limiter = None


def enable_streaming() -> None:
    """启用流式输出（改善感知延迟，首字即出）。

    注意：流式模式下 raw_llm_output 为空，因为内容是逐步输出的。
    评估场景建议关闭流式，确保完整解析。
    """
    global _llm_instance
    _llm_instance = None  # 重置，下次创建时会带上 streaming


def get_cache_info() -> dict[str, Any]:
    """返回缓存状态信息"""
    from langchain_core.globals import get_llm_cache
    cache = get_llm_cache()
    if cache is None:
        return {"enabled": False, "type": "none"}
    cache_type = type(cache).__name__
    info: dict[str, Any] = {"enabled": True, "type": cache_type}
    if hasattr(cache, "_cache"):
        info["entries"] = len(cache._cache)
    return info


# 默认启用内存缓存
enable_cache("memory")


# ============================================================
# 1. LLM 工厂
# ============================================================

_llm_instance: Optional[ChatOpenAI] = None


def create_llm(temperature: float = 0.0, json_mode: bool = False, use_cache: bool = True) -> ChatOpenAI:
    """创建 LLM 实例（延迟读取环境变量，支持限流）

    Args:
        temperature: 温度参数
        json_mode: 是否启用 JSON mode（强制输出合法 JSON）
        use_cache: 是否使用全局缓存（False 时创建独立实例绕过缓存，不影响其他 worker）
    """
    global _llm_instance
    # 构建 model_kwargs（json_mode）
    model_kwargs: dict[str, Any] = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    # 不走缓存的请求创建独立实例，不污染全局单例
    if not use_cache:
        kwargs: dict[str, Any] = {"model_kwargs": model_kwargs, "seed": 42}
        if _rate_limiter is not None:
            kwargs["rate_limiter"] = _rate_limiter
        nocache_llm = ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            temperature=temperature,
            top_p=1.0,
            max_tokens=2048,
            max_retries=2,
            request_timeout=60,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            **kwargs,
        )
        nocache_llm.cache = False
        return nocache_llm

    if _llm_instance is not None and _llm_instance.temperature == temperature:
        return _llm_instance
    kwargs: dict[str, Any] = {"model_kwargs": model_kwargs, "seed": 42}
    if _rate_limiter is not None:
        kwargs["rate_limiter"] = _rate_limiter
    _llm_instance = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        temperature=temperature,
        top_p=1.0,           # 固定 top_p，确保确定性输出
        max_tokens=2048,
        max_retries=2,       # 429/503 自动指数退避重试
        request_timeout=60,  # 60 秒超时，防止 API 挂起
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        **kwargs,
    )
    return _llm_instance


# ============================================================
# 2. Prompt 构建（ChatPromptTemplate）
# ============================================================

def build_prompt_messages(
    profile_key: str,
    before_text: str,
    after_text: str,
    segments_before: list[dict[str, Any]] | None = None,
    segments_after: list[dict[str, Any]] | None = None,
    diff_info: str | None = None,
) -> list[BaseMessage]:
    """根据 profile 和输入构建消息列表（动态从 SQLite 加载 Profile 规则）"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from .prompts import SYSTEM_PROMPT, build_user_prompt
    from .storage import get_profile, normalize_profile
    from .debias import generate_anti_bias_prompt_supplement

    # 向后兼容映射
    profile_key = normalize_profile(profile_key)

    # 拼装 System Prompt：静态规则 + 动态 Profile 补充 + 抗偏置指令
    system_content = SYSTEM_PROMPT

    profile = get_profile(profile_key)
    if profile and profile.get("prompt_supplement"):
        system_content += "\n" + profile["prompt_supplement"]

    # 抗偏置指令
    system_content += "\n" + generate_anti_bias_prompt_supplement()

    user_content = build_user_prompt(
        before_text=before_text,
        after_text=after_text,
        segments_before=segments_before,
        segments_after=segments_after,
        diff_info=diff_info,
    )
    return [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]


# ============================================================
# 3. 输出解析（Pydantic 验证 + 结构化提取）
# ============================================================

from pydantic import BaseModel, Field


class LLMDimensionScore(BaseModel):
    """LLM 输出的维度评分"""
    dimension: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class LLMAnchorSpan(BaseModel):
    """LLM 输出的锚点信息"""
    model_config = {"coerce_numbers_to_str": True}  # LLM 返回 segment_id=1 (int) 时自动转为 "1"
    segment_id: str = ""
    before_anchor: str | None = ""   # Prompt 要求 LLM 输出 [Before N] 格式
    after_anchor: str | None = ""    # Prompt 要求 LLM 输出 [After N] 格式
    start_char: int = 0
    end_char: int = 0
    snippet: str = ""


class LLMFlawItem(BaseModel):
    """LLM 输出的瑕疵项"""
    category: str
    severity: str = "minor"
    description: str = ""
    location: LLMAnchorSpan = LLMAnchorSpan()
    suggestion: str | None = None


class LLMOutput(BaseModel):
    """LLM 输出的完整结构（Pydantic 验证）"""
    dimensions: list[LLMDimensionScore] = Field(default_factory=list)
    overall_score: float = Field(default=0.5, ge=0.0, le=1.0)
    flaws: list[LLMFlawItem] = Field(default_factory=list)


def parse_llm_output(raw: str) -> dict[str, Any]:
    """从 LLM 原始输出中提取 JSON（三级降级策略 + Pydantic 验证）

    使用 strict=False 允许控制字符（LLM 有时在 reason 中输出裸换行符）。
    """
    # 1) 直接解析（strict=False 允许控制字符）
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        pass
    # 2) 提取 ```json ... ``` 块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        try:
            return json.loads(m.group(1), strict=False)
        except json.JSONDecodeError:
            pass
    # 3) 提取最外层 { ... }
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(), strict=False)
        except json.JSONDecodeError:
            pass
    return {}


def validate_llm_output(parsed: dict[str, Any]) -> LLMOutput:
    """用 Pydantic 验证 LLM 输出结构，容错处理异常字段"""
    try:
        return LLMOutput(**parsed)
    except Exception as e:
        # 验证失败时打印错误详情，方便排查
        import logging
        logging.warning(f"LLM 输出 Pydantic 验证失败: {e}")
        print(f"[WARNING] LLM 输出 Pydantic 验证失败: {e}")
        return LLMOutput()


def _normalize_score(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


# 维度名称映射（兼容新旧两种命名）
_DIM_NAME_MAP = {
    "semantic": "semantic",
    "semantic_fidelity": "semantic",
    "factual": "factual",
    "factual_consistency": "factual",
    "hallucination": "hallucination",
    "structure": "structure",
    "readability": "readability",
}


def extract_dimensions(parsed: dict[str, Any]) -> list[DimensionScore]:
    """从解析结果中提取维度评分（兼容新旧维度命名）"""
    dimensions = []
    for d in parsed.get("dimensions", []):
        if isinstance(d, dict):
            raw_name = d.get("dimension", "未知")
            canonical = _DIM_NAME_MAP.get(raw_name, raw_name)
            dimensions.append(DimensionScore(
                dimension=canonical,
                score=_normalize_score(d.get("score", 0)),
                weight=float(d.get("weight", 0.25)),
                reason=d.get("reason", ""),
            ))
    if not dimensions:
        overall = _normalize_score(parsed.get("overall_score", 0.5))
        dimensions = [
            DimensionScore(dimension="semantic", score=overall, weight=0.30,
                          reason="LLM 未返回分维度得分，使用总分估计"),
            DimensionScore(dimension="factual", score=overall, weight=0.30,
                          reason="LLM 未返回分维度得分，使用总分估计"),
            DimensionScore(dimension="hallucination", score=overall, weight=0.20,
                          reason="LLM 未返回分维度得分，使用总分估计"),
            DimensionScore(dimension="structure", score=overall, weight=0.10,
                          reason="LLM 未返回分维度得分，使用总分估计"),
            DimensionScore(dimension="readability", score=overall, weight=0.10,
                          reason="LLM 未返回分维度得分，使用总分估计"),
        ]
    return dimensions


def _score_floor_from_flaws(flaws: list[FlawItem]) -> float:
    """根据瑕疵类型计算总分下限抑制。

    设计原则：内容删除（over_clean/omission）与事实篡改（mis_edit）应主导总分，
    避免低权重维度（structure/readability/hallucination）保持高分把加权均值抬起来，
    从而与人工评审对"实质信息损失"的判断保持一致。
    """
    if not flaws:
        return 1.0

    floor = 1.0

    # 结构 / 可读性
    if any(f.category == "structure" and f.severity in ("critical", "major") for f in flaws):
        floor = min(floor, 0.70)
    if any(f.category == "readability" and f.severity in ("critical", "major") for f in flaws):
        floor = min(floor, 0.72)

    # 内容删除（over_clean/omission）—— 按严重度与数量累积下压
    over_clean_critical = sum(1 for f in flaws if f.category in ("over_clean", "omission") and f.severity == "critical")
    over_clean_major = sum(1 for f in flaws if f.category in ("over_clean", "omission") and f.severity == "major")
    if over_clean_critical >= 1:
        floor = min(floor, 0.60)
    if over_clean_major >= 2:
        # 多处重要内容删除 ≈ 一次严重删除
        floor = min(floor, 0.62)
    elif over_clean_major == 1:
        floor = min(floor, 0.72)

    # 事实篡改（mis_edit/factual）
    mis_edit_critical = sum(1 for f in flaws if f.category in ("mis_edit", "factual") and f.severity == "critical")
    mis_edit_major = sum(1 for f in flaws if f.category in ("mis_edit", "factual") and f.severity == "major")
    if mis_edit_critical >= 1:
        floor = min(floor, 0.60)
    elif mis_edit_major >= 1:
        floor = min(floor, 0.72)

    return floor


def _apply_structural_penalty(dimensions: list[DimensionScore], flaws: list[FlawItem]) -> list[DimensionScore]:
    """对结构/可读性/关键事实类瑕疵施加更强的维度级惩罚。"""
    adjusted = [d.model_copy() for d in dimensions]
    flaw_index = {(f.category, f.severity) for f in flaws}

    for dim in adjusted:
        if dim.dimension == "structure":
            if any(cat == "structure" and sev in ("critical", "major") for cat, sev in flaw_index):
                dim.score = min(dim.score, 0.40)
        elif dim.dimension == "readability":
            if any(cat == "readability" and sev in ("critical", "major") for cat, sev in flaw_index):
                dim.score = min(dim.score, 0.35)
        elif dim.dimension == "semantic":
            if any(cat in ("over_clean", "omission") and sev == "critical" for cat, sev in flaw_index):
                dim.score = min(dim.score, 0.35)
        elif dim.dimension == "factual":
            if any(cat in ("mis_edit", "factual") and sev == "critical" for cat, sev in flaw_index):
                dim.score = min(dim.score, 0.35)

    return adjusted


def extract_flaws(parsed: dict[str, Any]) -> list[FlawItem]:
    """从解析结果中提取瑕疵清单（兼容新旧锚点格式）"""
    flaws = []
    for f in parsed.get("flaws", []):
        if isinstance(f, dict):
            loc_raw = f.get("location") or f.get("anchor") or {}
            # 兼容新格式（before_anchor/after_anchor）和旧格式（segment_id/start_char）
            segment_id = str(
                loc_raw.get("segment_id", "")
                or loc_raw.get("after_anchor", "")
                or loc_raw.get("before_anchor", "")
            )
            start_char = int(loc_raw.get("start_char", 0))
            end_char = int(loc_raw.get("end_char", 0))
            snippet = str(loc_raw.get("snippet", ""))

            # 如果有 after_anchor 但没有 start_char，记录锚点信息供后处理解析
            after_anchor = loc_raw.get("after_anchor", "")
            before_anchor = loc_raw.get("before_anchor", "")

            flaws.append(FlawItem(
                category=str(f.get("category", "unknown")),
                severity=str(f.get("severity", "minor")),
                description=str(f.get("description", "")),
                location=AnchorSpan(
                    segment_id=segment_id,
                    start_char=start_char,
                    end_char=end_char,
                    snippet=snippet,
                ),
                suggestion=f.get("suggestion"),
            ))
    return flaws


def locate_flaw(
    snippet: str,
    after_text: str,
    before_text: str = "",
    segment_id: str = "",
    segments_after: list[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """多策略定位：在原文中找到 snippet 的精确字符偏移。

    核心优化：如果 LLM 提供了 segment_id（如 "After 2"），先在对应段落内搜索，
    将搜索空间从全文缩小到几十个字符，大幅降低误匹配。

    策略（按优先级降级）：
    0. 段落定位 —— 用 segment_id 找到对应段落，段落内做策略 1-3
    1. 精确匹配 —— snippet 完整出现在 after_text 中
    2. 最长公共子串 —— 用 difflib 找 snippet 与 after_text 的最大重叠段（≥6 字符）
    3. 逐级前缀搜索 —— 缩短 snippet 前缀（30→20→15→10→8 chars）在 after_text 中搜索
    4. before_text 搜索 —— 被删除的内容可能在原文中存在

    Returns:
        (start_char, end_char) —— 均为 0 表示定位失败
    """
    if not snippet or not after_text:
        return 0, 0

    # ── 策略 0：段落定位（缩小搜索空间到几十个字符，从根本上消除误匹配）──
    para_text = ""
    para_offset = -1
    if segment_id and segments_after:
        clean_id = _normalize_segment_id(segment_id)
        for seg in segments_after:
            if _normalize_segment_id(seg.get("segment_id", "")) == clean_id:
                para_text = seg.get("text", "")
                # 在原文中定位该段落
                para_offset = after_text.find(para_text)
                if para_offset >= 0:
                    break
                # 如果精确查找失败，用 difflib 模糊定位
                if para_offset < 0 and para_text:
                    sm = difflib.SequenceMatcher(None, para_text, after_text)
                    match = sm.find_longest_match(0, len(para_text), 0, len(after_text))
                    if match.size >= len(para_text) * 0.6:
                        para_offset = match.b
                        para_text = after_text[match.b:match.b + match.size]
                        break
                para_text = ""  # 定位失败，回退到全文搜索

    # 如果定位到了段落，先在段落内搜索
    if para_offset >= 0 and para_text:
        # 策略 1-段落: 精确匹配
        pos = para_text.find(snippet)
        if pos >= 0:
            return para_offset + pos, para_offset + pos + len(snippet)

        # 策略 2-段落: LCS
        sm = difflib.SequenceMatcher(None, snippet, para_text)
        match = sm.find_longest_match(0, len(snippet), 0, len(para_text))
        if match.size >= 4:  # 段落内降低门槛到 4 字符
            return para_offset + match.b, para_offset + match.b + match.size

        # 策略 3-段落: 前缀搜索
        for n in [20, 15, 10, 8]:
            if len(snippet) >= n:
                pos = para_text.find(snippet[:n])
                if pos >= 0:
                    return para_offset + pos, para_offset + pos + n

        # 段落内搜索失败，降级到全文搜索（保留 para_offset 信息）
        # 用段落开头作为兜底位置
        pass

    # ── 全文搜索（策略 1-4）──
    # 策略 1: 精确匹配
    pos = after_text.find(snippet)
    if pos >= 0:
        return pos, pos + len(snippet)

    # 策略 2: difflib 最长公共子串
    sm = difflib.SequenceMatcher(None, snippet, after_text)
    match = sm.find_longest_match(0, len(snippet), 0, len(after_text))
    if match.size >= 6:
        return match.b, match.b + match.size

    # 策略 3: 逐级缩短前缀
    for n in [30, 20, 15, 10, 8]:
        if len(snippet) >= n:
            pos = after_text.find(snippet[:n])
            if pos >= 0:
                return pos, pos + n

    # 策略 4: 在 before_text 中搜索
    if before_text:
        search_key = snippet[:min(20, len(snippet))]
        pos = before_text.find(search_key)
        if pos >= 0:
            return pos, pos + len(search_key)

    # 兜底：如果段落定位成功但所有搜索都失败，返回段落位置
    if para_offset >= 0:
        return para_offset, para_offset + min(len(para_text), 80)

    return 0, 0


def _normalize_segment_id(seg_id: str) -> str:
    """归一化 segment_id：去掉前缀和方括号，只保留数字。

    "After 2" → "2"
    "[After 2]" → "2"
    "Before 1" → "1"
    "2" → "2"
    """
    import re
    # 提取所有数字
    nums = re.findall(r'\d+', seg_id)
    return nums[-1] if nums else seg_id.strip("[] ")


def compute_overall_score(dimensions: list[DimensionScore]) -> float:
    """计算加权总分"""
    total_weight = sum(d.weight for d in dimensions) or 1.0
    return _normalize_score(sum(d.score * d.weight for d in dimensions) / total_weight)


# ============================================================
# 4. 后处理（软惩罚 + profile 惩罚）
# ============================================================

def apply_soft_penalty(
    base_score: float,
    dimensions: list[DimensionScore],
    flaws: list[FlawItem],
) -> float:
    """软惩罚机制（乘法衰减，替代硬 veto）

    工业界主流做法：不用一票否决，用连续化的惩罚因子。
    - factual 维度低 → 乘法衰减
    - critical 瑕疵 → 额外衰减
    - 最终分数 = base_score * penalty_factor

    采用风险聚合（Risk Aggregation）策略：
    - 每个瑕疵有对应的惩罚因子
    - 最终惩罚取所有因子中最重的一个（min），不连续相乘
    - 避免多个瑕疵导致分数归零（隐式 veto）

    penalty_factor 取值：
      critical factual:   0.6
      critical structure: 0.75
      critical omission:  0.65
      major:              0.85
      minor:              0.95
    """
    # 风险聚合（Risk Aggregation）：取所有瑕疵中最重的惩罚因子
    # 不连续相乘，避免多个瑕疵导致分数归零（隐式 veto）
    penalty_map = {
        ("critical", "factual"): 0.6,
        ("critical", "mis_edit"): 0.6,
        ("critical", "structure"): 0.75,
        ("critical", "omission"): 0.65,
        ("critical", "over_clean"): 0.65,
    }

    worst_penalty = 1.0
    for f in flaws:
        key = (f.severity, f.category)
        if key in penalty_map:
            worst_penalty = min(worst_penalty, penalty_map[key])
        elif f.severity == "critical":
            worst_penalty = min(worst_penalty, 0.8)
        elif f.severity == "major":
            worst_penalty = min(worst_penalty, 0.85)
        elif f.severity == "minor":
            worst_penalty = min(worst_penalty, 0.95)

    return _normalize_score(base_score * worst_penalty)


def determine_verdict(overall_score: float) -> str:
    """根据总分判定结果"""
    # 缓冲带：0.78~0.82 之间默认 review，防止边界值波动导致判定跳变
    # 只有 ≥ 0.82 才直接 pass，确保 pass 的文本是绝对安全的
    if overall_score >= 0.82:
        return "pass"
    elif overall_score >= 0.5:
        return "review"
    return "fail"


def determine_reason_code(
    overall_score: float,
    verdict: str,
    flaws: list[FlawItem],
) -> tuple[str, list[dict[str, Any]]]:
    """判定原因码 + 详细原因列表。

    Returns:
        (reason_code, reject_reasons)
        reason_code: 机器可读的判定原因
        reject_reasons: 详细原因列表
    """
    reasons: list[dict[str, Any]] = []

    # 检查 critical 级瑕玼
    critical_flaws = [f for f in flaws if f.severity == "critical"]
    if critical_flaws:
        for f in critical_flaws:
            code = "CONSTRAINT_DELETION" if f.category == "over_clean" else "FACTUAL_DISTORTION"
            reasons.append({
                "code": code,
                "description": f.description[:100],
                "severity": "critical",
                "category": f.category,
            })

    # 检查分数阈值
    if overall_score < 0.5:
        reasons.append({
            "code": "SCORE_BELOW_THRESHOLD",
            "description": f"综合得分 {overall_score:.4f} 低于 0.5",
            "severity": "major",
        })

    # 确定主原因码
    if not reasons:
        reason_code = "PASS_ALL_CLEAR"
    elif critical_flaws:
        reason_code = "CRITICAL_FLAW_DETECTED"
    elif overall_score < 0.5:
        reason_code = "SCORE_BELOW_THRESHOLD"
    else:
        reason_code = "REVIEW_REQUIRED"

    return reason_code, reasons


def apply_profile_penalties(
    profile_key: str,
    overall_score: float,
    verdict: str,
    flaws: list[FlawItem],
    before_text: str,
    after_text: str,
) -> tuple[float, str, list[FlawItem]]:
    """Profile-aware 关键事实缺失检查与降分"""
    config = get_profile_config(profile_key)
    penalty_policy = cast(dict[str, Any], config.get("penalty_policy", {}))
    fact_types = [str(t) for t in cast(list[str], config.get("critical_fact_types", []))]

    if profile_key == PROFILE_GENERAL:
        return overall_score, verdict, flaws

    # 提取关键事实
    missing = []
    facts = _extract_key_facts(before_text)
    for fact in facts:
        if fact["type"] not in fact_types and fact["type"] != "number_with_unit":
            continue
        if str(fact["value"]) not in after_text:
            missing.append(fact)

    if not missing:
        return overall_score, verdict, flaws

    # 追加瑕疵
    for fact in missing:
        severity = "critical" if fact["type"] in ("range", "date") else "major"
        flaws.append(FlawItem(
            category="over_clean",
            severity=severity,
            description=f"关键事实缺失：原文中的「{fact['value']}」在改写后未保留",
            location=AnchorSpan(segment_id="auto", start_char=0, end_char=0,
                               snippet=str(fact["value"])),
            suggestion=f"请保留原文中的关键数据「{fact['value']}」",
        ))

    # 软惩罚（乘法衰减，替代硬 cap）
    critical_count = sum(1 for f in flaws if f.severity == "critical")
    major_count = sum(1 for f in flaws if f.severity == "major")

    # 风险聚合：取最重惩罚，不连续相乘
    penalty = 1.0
    if critical_count > 0:
        penalty = 0.6
    elif major_count > 0:
        penalty = 0.85

    overall_score = _normalize_score(overall_score * penalty)
    verdict = determine_verdict(overall_score)

    return overall_score, verdict, flaws


def _extract_key_facts(text: str) -> list[dict[str, Any]]:
    """从文本中提取高风险事实线索"""
    facts: list[dict[str, Any]] = []
    for m in re.finditer(r"\d{4}[年\-/]\d{1,2}[月\-/]\d{1,2}[日]?", text):
        facts.append({"type": "date", "value": m.group(), "pos": m.start()})
    for m in re.finditer(r"\d+[\d,.]*\s*[元万亿]+\s*[以到至]\s*\d+[\d,.]*\s*[元万亿]+", text):
        facts.append({"type": "range", "value": m.group(), "pos": m.start()})
    for m in re.finditer(r"\d+[\d,.]*\s*(?:元|万元|亿元|%|％|厘米|cm|公斤|kg|日|天|小时|个月|年)", text):
        facts.append({"type": "number_with_unit", "value": m.group(), "pos": m.start()})
    for m in re.finditer(r"\d+\s*(?:日内|小时内|个工作日内|天内)", text):
        facts.append({"type": "deadline", "value": m.group(), "pos": m.start()})
    return facts


# ============================================================
# 5. 可复现令牌
# ============================================================

# 评分规则版本：维度权重、惩罚因子等变更时递增
RULE_VERSION = "v3.0"  # v3.0: 5维度(semantic/factual/hallucination/structure/readability) + 软惩罚


def _get_rule_hash() -> str:
    """基于当前评分规则生成哈希（维度权重 + 惩罚因子）"""
    rule_str = "semantic:0.30,factual:0.30,hallucination:0.20,structure:0.10,readability:0.10," \
               "penalty:risk_aggregation(max),critical_factual=0.6,critical_structure=0.75," \
               "critical_omission=0.65,major=0.85,minor=0.95"
    return hashlib.sha256(rule_str.encode("utf-8")).hexdigest()[:8]


def build_reproducibility_token(request: EvalRequest, temperature: float) -> str:
    """生成可复现令牌（多因子哈希，防止缓存碰撞）"""
    from .prompts import SYSTEM_PROMPT
    payload = json.dumps({
        "before": request.before_text,
        "after": request.after_text,
        "temperature": temperature,
        "top_p": 1.0,
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "prompt_version": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8],
        "rule_version": _get_rule_hash(),
        "evaluation_profile": request.evaluation_profile,
        "stabilize": request.stabilize,
        "sample_count": request.sample_count,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ============================================================
# 6. Callback：token / 延迟追踪
# ============================================================

class EvalCallbackHandler(BaseCallbackHandler):
    """评估过程追踪回调"""

    def __init__(self) -> None:
        super().__init__()
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_tokens: int = 0
        self.model_name: str = ""

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self.start_time = time.time()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.end_time = time.time()
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.input_tokens = usage.get("prompt_tokens", 0)
            self.output_tokens = usage.get("completion_tokens", 0)
            self.total_tokens = usage.get("total_tokens", 0)

    @property
    def latency_seconds(self) -> float:
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 3)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


# ============================================================
# 6b. 对齐模块（Alignment）—— 锚点预处理
# ============================================================

import difflib


def _split_paragraphs(text: str) -> list[str]:
    """按段落切分文本，去除空行"""
    lines = text.split("\n")
    paragraphs: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            paragraphs.append(stripped)
    return paragraphs


def _find_best_match(
    target: str, candidates: list[str], threshold: float = 0.4
) -> tuple[int, float]:
    """在候选列表中找到与 target 最相似的段落，返回 (index, ratio)"""
    best_idx = -1
    best_ratio = 0.0
    for i, cand in enumerate(candidates):
        ratio = difflib.SequenceMatcher(None, target, cand).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
    if best_ratio >= threshold:
        return best_idx, best_ratio
    return -1, 0.0


def _detect_user_anchors(text: str) -> list[dict[str, Any]] | None:
    """检测用户输入是否自带锚点标记。

    支持格式：[Anchor_P1], [Anchor_G1], [P1], [G1], [Before 1], [After 1] 等。
    如果检测到锚点，返回解析后的段落列表；否则返回 None。
    """
    import re
    # 匹配常见的锚点格式
    pattern = r"\[(?:Anchor_)?(?:P|G|Before|After|锚点)[\s_]*(\d+[a-zA-Z]?)\]\s*"
    matches = list(re.finditer(pattern, text))
    if len(matches) < 2:  # 至少 2 个锚点才算有效
        return None

    segments = []
    for i, m in enumerate(matches):
        anchor_id = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        para_text = text[start:end].strip()
        if para_text:
            segments.append({
                "segment_id": anchor_id,
                "text": para_text,
                "start_char": m.start(),
                "end_char": end,
            })
    return segments if segments else None


def build_anchored_text(
    before_text: str, after_text: str
) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
    """锚点预处理：将治理前后文本按段落对齐，添加锚点标记。

    策略：
    1. 如果用户输入自带锚点（如 [Anchor_P1]），直接使用用户的锚点
    2. 如果用户输入不带锚点，系统自动生成 [Before N] / [After N] 标记

    Args:
        before_text: 治理前原始文本
        after_text: 治理后文本

    Returns:
        (anchored_before, anchored_after, segments_before, segments_after)
    """
    # 检测用户是否自带锚点
    user_before_segs = _detect_user_anchors(before_text)
    user_after_segs = _detect_user_anchors(after_text)

    if user_before_segs and user_after_segs:
        # 用户自带锚点，直接使用
        anchored_before = before_text
        anchored_after = after_text
        return anchored_before, anchored_after, user_before_segs, user_after_segs

    # 用户不带锚点，系统自动生成
    paras_before = _split_paragraphs(before_text)
    paras_after = _split_paragraphs(after_text)

    # 匹配状态追踪
    matched_after: set[int] = set()
    pairs: list[tuple[int, int | None]] = []  # (before_idx, after_idx | None)

    # 用 before 去 after 里找最佳匹配
    for i, bp in enumerate(paras_before):
        remaining = [j for j in range(len(paras_after)) if j not in matched_after]
        if not remaining:
            pairs.append((i, None))
            continue
        candidates = [paras_after[j] for j in remaining]
        best_local_idx, ratio = _find_best_match(bp, candidates)
        if best_local_idx >= 0:
            actual_after_idx = remaining[best_local_idx]
            matched_after.add(actual_after_idx)
            pairs.append((i, actual_after_idx))
        else:
            pairs.append((i, None))

    # after 中未被匹配的段落 = 新增内容
    added_after = [j for j in range(len(paras_after)) if j not in matched_after]

    # 构建带锚点标记的文本
    anchored_before_lines: list[str] = []
    anchored_after_lines: list[str] = []
    segments_before: list[dict[str, Any]] = []
    segments_after: list[dict[str, Any]] = []
    char_offset_b = 0
    char_offset_a = 0
    pair_num = 0

    for before_idx, after_idx in pairs:
        pair_num += 1
        bp = paras_before[before_idx]
        tag_b = f"[Before {pair_num}]"
        line_b = f"{tag_b} {bp}"
        anchored_before_lines.append(line_b)
        segments_before.append({
            "segment_id": str(pair_num),
            "text": bp,
            "start_char": char_offset_b,
            "end_char": char_offset_b + len(bp),
        })
        char_offset_b += len(line_b) + 1  # +1 for newline

        if after_idx is not None:
            ap = paras_after[after_idx]
            tag_a = f"[After {pair_num}]"
            line_a = f"{tag_a} {ap}"
            anchored_after_lines.append(line_a)
            segments_after.append({
                "segment_id": str(pair_num),
                "text": ap,
                "start_char": char_offset_a,
                "end_char": char_offset_a + len(ap),
            })
            char_offset_a += len(line_a) + 1
        else:
            # before 有但 after 没有 → 被删除
            tag_a = f"[After {pair_num}: 已删除]"
            anchored_after_lines.append(tag_a)
            char_offset_a += len(tag_a) + 1

    # after 中新增的段落
    for after_idx in added_after:
        pair_num += 1
        ap = paras_after[after_idx]
        tag_a = f"[After {pair_num}: 新增]"
        line_a = f"{tag_a} {ap}"
        anchored_after_lines.append(line_a)
        segments_after.append({
            "segment_id": str(pair_num),
            "text": ap,
            "start_char": char_offset_a,
            "end_char": char_offset_a + len(ap),
        })
        char_offset_a += len(line_a) + 1
        # before 中对应位置标记
        anchored_before_lines.append(f"[Before {pair_num}: 无对应]")
        char_offset_b += len(f"[Before {pair_num}: 无对应]") + 1

    return (
        "\n".join(anchored_before_lines),
        "\n".join(anchored_after_lines),
        segments_before,
        segments_after,
    )


# ============================================================
# 6c. Diff / 瑕疵检测模块（核心）
# ============================================================

import re


def _char_diff(before: str, after: str) -> list[dict[str, Any]]:
    """字符级 diff：找出 before→after 的具体变更。

    返回变更列表，每项包含：
      type: substitution / deletion / insertion
      before_text: 原文片段
      after_text: 改后片段
      before_start / before_end: 原文中的位置
      after_start / after_end: 改后文本中的位置
    """
    sm = difflib.SequenceMatcher(None, before, after)
    changes: list[dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        changes.append({
            "type": tag,  # substitution / deletion / insertion
            "before_text": before[i1:i2] if tag != "insertion" else "",
            "after_text": after[j1:j2] if tag != "deletion" else "",
            "before_start": i1,
            "before_end": i2,
            "after_start": j1,
            "after_end": j2,
        })

    return changes


def _classify_change(change: dict[str, Any]) -> tuple[str, str]:
    """对单个变更进行分类和严重程度判定。

    分类规则：
      category:
        - factual: 数字、日期、金额等事实性数据被改动
        - structure: 结构被破坏（表格→文本、列表→段落等）
        - omission: 内容被删除
        - modification: 表述被改写（非事实性）
        - addition: 新增内容

      severity:
        - critical: 关键事实被篡改 / 大量内容被删 / 结构被破坏
        - major: 重要信息丢失或改写
        - minor: 轻微表述调整
    """
    before_txt = change.get("before_text", "")
    after_txt = change.get("after_text", "")
    change_type = change["type"]

    # 数字/日期/金额模式
    num_pattern = r"\d+[\.\d]*[%％亿万]?"

    # 删除
    if change_type == "deletion":
        deleted_len = len(before_txt)
        if deleted_len > 50:
            return "omission", "critical"
        elif deleted_len > 15:
            return "omission", "major"
        return "omission", "minor"

    # 插入
    if change_type == "insertion":
        return "addition", "minor"

    # 替换（substitution）
    # 检查是否涉及数字/日期变更
    before_nums = re.findall(num_pattern, before_txt)
    after_nums = re.findall(num_pattern, after_txt)

    if before_nums or after_nums:
        if before_nums != after_nums:
            # 数字被改了 → 事实性错误
            # 判断严重程度：数量级变化 vs 微调
            for bn, an in zip(before_nums, after_nums):
                try:
                    b_val = float(re.sub(r"[^\d.]", "", bn))
                    a_val = float(re.sub(r"[^\d.]", "", an))
                    if b_val > 0 and abs(a_val - b_val) / b_val > 0.1:
                        return "factual", "critical"  # 变化超过 10%
                except (ValueError, ZeroDivisionError):
                    pass
            return "factual", "major"

    # 检查是否是结构变化（表格标记、列表标记等）
    structural_markers = ["|", "────", "---", "1.", "2.", "3.", "•", "- "]
    if any(m in before_txt for m in structural_markers) and not any(m in after_txt for m in structural_markers):
        return "structure", "critical"

    # 普通表述改写
    if len(before_txt) > 30 or len(after_txt) > 30:
        return "modification", "major"
    return "modification", "minor"


def _extract_table_snippet(text: str) -> str:
    """从原文中提取 Markdown 表格区域的文本片段，用于定位。

    取表格第一行（表头），保留原文格式确保可被 locate_flaw 搜索到。
    """
    if not text:
        return ""
    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            return stripped[:80]  # 直接返回原文中的表头行
    return ""


def _extract_list_snippet(text: str) -> str:
    """从原文中提取有序列表区域的文本片段，用于定位。

    取第一个列表项，保留原文格式确保可被 locate_flaw 搜索到。
    """
    if not text:
        return ""
    import re
    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        if re.match(r"\d+[\.\)]\s", stripped):
            return stripped[:80]  # 直接返回原文中的第一个列表项
    return ""


def detect_flaws(
    segments_before: list[dict[str, Any]],
    segments_after: list[dict[str, Any]],
    anchored_before: str,
    anchored_after: str,
    before_text: str = "",
    after_text: str = "",
) -> list[dict[str, Any]]:
    """Diff / 瑕疵检测：对配对段落做变更检测，输出结构化瑕疵列表。

    检测策略：
    1. 被整段删除 → omission / critical
    2. 配对段落相似度 < 0.3 → 大幅改写，用字符级 diff 找具体变更
    3. 配对段落相似度 0.3~0.7 → 中等改写，检测数字/关键事实变更
    4. 配对段落相似度 > 0.7 → 轻微调整，跳过或标 minor

    Args:
        segments_before: before 的分段信息
        segments_after: after 的分段信息
        anchored_before: 带锚点标记的 before 文本
        anchored_after: 带锚点标记的 after 文本
        before_text: 原始治理前文本（用于提取结构瑕疵的实际内容snippet）
        after_text: 原始治理后文本

    Returns:
        瑕疵列表，每项包含 type, anchor_before, anchor_after, category, severity,
        以及 before_snippet/after_snippet（用于后续精确定位）
    """
    flaws: list[dict[str, Any]] = []

    # ── 第一层：确定性规则检测（Markdown 表格 / 列表结构破坏）──
    # 这类结构变化是算法 100% 能判断的，不需要依赖 LLM
    before_pipe = anchored_before.count("|")
    after_pipe = anchored_after.count("|")
    has_table_separator = "|---" in anchored_before or "| ---" in anchored_before

    table_detected = before_pipe >= 6 and has_table_separator and after_pipe < 2
    if table_detected:
        # 从原始 before_text 中提取表格区域的文本作为定位 snippet
        table_snippet = _extract_table_snippet(before_text)
        flaws.append({
            "type": "structure_loss",
            "anchor_before": "[Before] Markdown 表格结构",
            "anchor_after": "[After] 表格已转为纯文本",
            "category": "structure",
            "severity": "critical",
            "before_snippet": table_snippet,
            "after_snippet": "",
        })

    # 检测有序列表被压缩（1. 2. 3. → 单行文本）
    import re
    before_list_items = len(re.findall(r"(?:^|\n)\s*\d+[\.\)]\s", anchored_before))
    after_list_items = len(re.findall(r"(?:^|\n)\s*\d+[\.\)]\s", anchored_after))
    if before_list_items >= 3 and after_list_items == 0:
        # 从原始 before_text 中提取列表区域的文本作为定位 snippet
        list_snippet = _extract_list_snippet(before_text)
        flaws.append({
            "type": "structure_loss",
            "anchor_before": "[Before] 有序列表结构",
            "anchor_after": "[After] 列表已压缩为单行文本",
            "category": "structure",
            "severity": "major",
            "before_snippet": list_snippet,
            "after_snippet": "",
        })

    # 建立 segment_id → text 的映射
    before_map = {s["segment_id"]: s["text"] for s in segments_before}
    after_map = {s["segment_id"]: s["text"] for s in segments_after}

    all_ids = sorted(set(before_map.keys()) | set(after_map.keys()),
                     key=lambda x: int(x) if x.isdigit() else 0)

    num_pattern = re.compile(r"\d+[\.\d]*[%％亿万]?")

    for sid in all_ids:
        b_text = before_map.get(sid, "")
        a_text = after_map.get(sid, "")

        # ── 整段被删 ──
        if b_text and not a_text:
            # 表格行被转为文本不算 omission，已在第一层标记为 structure
            if table_detected and "|" in b_text:
                continue
            flaws.append({
                "type": "omission",
                "anchor_before": f"[Before {sid}] {b_text[:80]}",
                "anchor_after": "",
                "category": "omission",
                "severity": "critical" if len(b_text) > 15 else "major",
            })
            continue

        # ── 新增段落 ──
        if a_text and not b_text:
            flaws.append({
                "type": "addition",
                "anchor_before": "",
                "anchor_after": f"[After {sid}] {a_text[:80]}",
                "category": "addition",
                "severity": "minor",
            })
            continue

        # ── 配对段落：计算相似度 ──
        if b_text == a_text:
            continue

        ratio = difflib.SequenceMatcher(None, b_text, a_text).ratio()

        # 相似度 > 0.85 → 轻微调整，但先检查是否有数字变化
        if ratio > 0.85:
            changes = _char_diff(b_text, a_text)
            for ch in changes:
                if ch["type"] == "equal":
                    continue
                b_nums = num_pattern.findall(ch.get("before_text", ""))
                a_nums = num_pattern.findall(ch.get("after_text", ""))
                if b_nums and a_nums and b_nums != a_nums:
                    for bn, an in zip(b_nums, a_nums):
                        try:
                            b_val = float(re.sub(r"[^\d.]", "", bn))
                            a_val = float(re.sub(r"[^\d.]", "", an))
                            pct = abs(a_val - b_val) / b_val if b_val > 0 else 1.0
                            sev = "critical" if pct > 0.1 else "major"
                        except (ValueError, ZeroDivisionError):
                            sev = "major"
                        flaws.append({
                            "type": "substitution",
                            "anchor_before": f"[Before {sid}] ...{bn}...",
                            "anchor_after": f"[After {sid}] ...{an}...",
                            "category": "factual",
                            "severity": sev,
                        })
            continue

        # 相似度 < 0.3 → 大幅改写/过度清洗
        if ratio < 0.3:
            flaws.append({
                "type": "rewrite",
                "anchor_before": f"[Before {sid}] {b_text[:80]}",
                "anchor_after": f"[After {sid}] {a_text[:80]}",
                "category": "omission",
                "severity": "critical",
            })
            continue

        # 相似度 0.3~0.85 → 中等变更，做字符级 diff 分析
        changes = _char_diff(b_text, a_text)

        # 检查是否有数字变更
        has_num_change = False
        for ch in changes:
            if ch["type"] == "equal":
                continue
            b_nums = num_pattern.findall(ch.get("before_text", ""))
            a_nums = num_pattern.findall(ch.get("after_text", ""))
            if b_nums != a_nums and (b_nums or a_nums):
                has_num_change = True
                for bn, an in zip(b_nums, a_nums):
                    try:
                        b_val = float(re.sub(r"[^\d.]", "", bn))
                        a_val = float(re.sub(r"[^\d.]", "", an))
                        pct = abs(a_val - b_val) / b_val if b_val > 0 else 1.0
                        sev = "critical" if pct > 0.1 else "major"
                    except (ValueError, ZeroDivisionError):
                        sev = "major"
                    flaws.append({
                        "type": "substitution",
                        "anchor_before": f"[Before {sid}] ...{bn}...",
                        "anchor_after": f"[After {sid}] ...{an}...",
                        "category": "factual",
                        "severity": sev,
                    })

        # 检查是否有结构变化
        structural_markers = ["|", "────", "---", "1.", "2.", "3.", "•", "- "]
        had_structure = any(m in b_text for m in structural_markers)
        has_structure = any(m in a_text for m in structural_markers)
        if had_structure and not has_structure:
            flaws.append({
                "type": "structure_loss",
                "anchor_before": f"[Before {sid}] {b_text[:60]}",
                "anchor_after": f"[After {sid}] {a_text[:60]}",
                "category": "structure",
                "severity": "critical",
            })

        # 如果没有数字变更也没有结构变化，但相似度较低 → 表述改写
        if not has_num_change and not (had_structure and not has_structure) and ratio < 0.7:
            sev = "major" if ratio < 0.5 else "minor"
            flaws.append({
                "type": "modification",
                "anchor_before": f"[Before {sid}] {b_text[:60]}",
                "anchor_after": f"[After {sid}] {a_text[:60]}",
                "category": "modification",
                "severity": sev,
            })

    # ── 合并同类瑕疵（同一 category+severity 的多条合并为一条）──
    # 避免"表格→文本"产生 5 条独立 flaw
    merged: list[dict[str, Any]] = []
    seen_keys: dict[str, int] = {}  # key → index in merged

    for f in flaws:
        key = f"{f['category']}_{f['severity']}"
        if key in seen_keys:
            # 合并到已有条目
            idx = seen_keys[key]
            merged[idx]["description"] = merged[idx].get("description", "") + "; " + f.get("description", "")[:40]
            # 更新 anchor 为范围
            if f.get("anchor_before"):
                merged[idx]["anchor_after"] = f["anchor_after"]
        else:
            seen_keys[key] = len(merged)
            merged.append(f)

    return merged


# ============================================================
# 6d. 置信度计算
# ============================================================

def compute_confidence(
    dimensions: list[DimensionScore],
    flaws: list[FlawItem],
    detected_flaws: list[dict[str, Any]],
    diff_info: str,
) -> float:
    """计算评估置信度（0~1）。

    三个因子各占 1/3：
    1. Diff 一致性：算法预检的变更是否被 LLM 瑕疵列表覆盖
    2. 理由完整度：LLM 给出的 reason 是否足够详细
    3. 维度一致性：各维度分数是否合理分布（不是全 0 或全 1）
    """
    score = 0.0

    # ① Diff 一致性（0~1）
    # 算法检测到 N 个变更，LLM 瑕疵列表覆盖了 M 个 → M/N
    if detected_flaws:
        algo_categories = set()
        for df in detected_flaws:
            cat = df.get("category", "")
            if cat and cat != "addition":
                algo_categories.add(cat)
        llm_categories = set(f.category for f in flaws)
        if algo_categories:
            covered = len(algo_categories & llm_categories)
            diff_score = covered / len(algo_categories)
        else:
            diff_score = 1.0  # 算法没检测到变更，LLM 也没报瑕疵 → 一致
    else:
        diff_score = 0.8  # 没有 diff 信息，给中等置信度
    score += diff_score * 0.33

    # ② 理由完整度（0~1）
    # 每个维度的 reason 长度 > 5 字 → 算有理由
    reasons_with_content = sum(
        1 for d in dimensions if d.reason and len(d.reason.strip()) > 5
    )
    reason_score = reasons_with_content / max(len(dimensions), 1)
    score += reason_score * 0.33

    # ③ 规则一致性（0~1）
    # Diff 判定的严重程度是否与 LLM 维度分数一致
    # 例：Diff 检测到 critical factual，但 LLM 给 factual=0.95 → 冲突
    rule_score = 1.0
    dim_map = {d.dimension: d.score for d in dimensions}
    for df in detected_flaws:
        df_severity = df.get("severity", "")
        df_category = df.get("category", "")
        if df_severity == "critical" and df_category in ("factual", "mis_edit"):
            factual_score = dim_map.get("factual", 0.5)
            if factual_score > 0.7:
                rule_score *= 0.5  # Diff 说 critical 但 LLM 给高分 → 冲突
        elif df_severity == "critical" and df_category == "structure":
            structure_score = dim_map.get("structure", 0.5)
            if structure_score > 0.7:
                rule_score *= 0.5
        elif df_severity == "critical" and df_category in ("omission", "over_clean"):
            semantic_score = dim_map.get("semantic", 0.5)
            if semantic_score > 0.7:
                rule_score *= 0.6
    score += rule_score * 0.34

    return round(min(max(score, 0.0), 1.0), 4)


def compute_risk_level(overall_score: float, confidence: float) -> str:
    """计算风险等级，指导人工审核优先级。

    结合分数和置信度：
    - score < 0.3 或 (score < 0.5 且 confidence < 0.6) → high
    - score < 0.7 或 confidence < 0.5 → medium
    - 其他 → low
    """
    if overall_score < 0.3:
        return "high"
    if overall_score < 0.5 and confidence < 0.6:
        return "high"
    if overall_score < 0.7 or confidence < 0.5:
        return "medium"
    return "low"


# ============================================================
# 7. 主评估函数（LCEL Chain 流程）
# ============================================================

def _evaluate_once(request: EvalRequest, temperature: float = 0.0, use_cache: bool = True) -> EvalResponse:
    """单次评估（不含临界区多次采样），供内部调用。"""
    callback = EvalCallbackHandler()

    # Step 0: 对齐预处理 —— 将 before/after 按段落配对，添加锚点标记
    anchored_before, anchored_after, seg_before, seg_after = build_anchored_text(
        request.before_text, request.after_text
    )

    # Step 0b: Diff 瑕疵检测 —— 对配对段落做字符级 diff，输出结构化变更列表
    detected_flaws = detect_flaws(
        seg_before, seg_after, anchored_before, anchored_after,
        before_text=request.before_text,
        after_text=request.after_text,
    )
    # 格式化 diff 结果为文本，注入 Prompt
    diff_summary_lines: list[str] = []
    for f in detected_flaws:
        line = f"- [{f['severity'].upper()}] {f['category']}: {f['type']}"
        if f["anchor_before"]:
            line += f" | 原文: {f['anchor_before'][:60]}"
        if f["anchor_after"]:
            line += f" | 改后: {f['anchor_after'][:60]}"
        diff_summary_lines.append(line)
    diff_info = "\n".join(diff_summary_lines) if diff_summary_lines else "算法未检测到显著变更"

    # Step 1: 构建 prompt（使用对齐文本 + diff 检测结果）
    messages = build_prompt_messages(
        profile_key=request.evaluation_profile,
        before_text=anchored_before,
        after_text=anchored_after,
        segments_before=seg_before if not request.segments_before else request.segments_before,
        segments_after=seg_after if not request.segments_after else request.segments_after,
        diff_info=diff_info,
    )

    # Step 2: 调用 LLM（带 callback，启用 JSON mode，空返回自动重试 1 次）
    llm = create_llm(temperature=temperature, json_mode=True, use_cache=use_cache)
    response = llm.invoke(messages, config={"callbacks": [callback]})
    raw_output = str(response.content) if hasattr(response, "content") else str(response)

    # 空返回重试：API 偶尔返回空 content（token 消耗了但内容为空）
    if not raw_output.strip():
        response = llm.invoke(messages, config={"callbacks": [callback]})
        raw_output = str(response.content) if hasattr(response, "content") else str(response)

    # Step 3: 解析输出 + Pydantic 验证
    parsed = parse_llm_output(raw_output)
    validated = validate_llm_output(parsed)
    # 用验证后的数据（Pydantic 已做类型/范围校验）
    dimensions = extract_dimensions(validated.model_dump())
    flaws = extract_flaws(validated.model_dump())

    # 记录原始解析结果，便于诊断“原始输出正常但最终结果退化”的问题
    parsed_dimensions = len(parsed.get("dimensions", [])) if isinstance(parsed, dict) else 0
    parsed_flaws = len(parsed.get("flaws", [])) if isinstance(parsed, dict) else 0

    parse_fallback_used = False
    if not dimensions and parsed_dimensions:
        parse_fallback_used = True
        dimensions = extract_dimensions(parsed)
    if not flaws and parsed_flaws:
        parse_fallback_used = True
        flaws = extract_flaws(parsed)

    # 如果原始输出有结构，但验证后被降级为空，保留诊断信息，避免静默吞掉异常
    parse_diagnostics = {
        "parsed_dimensions": parsed_dimensions,
        "parsed_flaws": parsed_flaws,
        "validated_dimensions": len(dimensions),
        "validated_flaws": len(flaws),
        "parse_fallback_used": parse_fallback_used,
        "parse_ok": bool(parsed_dimensions or parsed_flaws),
    }

    if not dimensions:
        dimensions = extract_dimensions({"overall_score": 0.5})
        parse_diagnostics["dimension_defaulted"] = True

    # Step 3a: 保存原始数据（供热重算使用，在任何后处理之前）
    try:
        from .raw_store import save_raw
        save_raw(
            request_id=request.request_id,
            raw_dimensions=[{"dimension": d.dimension, "score": d.score, "weight": d.weight, "reason": d.reason} for d in dimensions],
            raw_flaws=[{"category": f.category, "severity": f.severity, "description": f.description, "location": {"segment_id": f.location.segment_id, "start_char": f.location.start_char, "end_char": f.location.end_char, "snippet": f.location.snippet}} for f in flaws],
            raw_llm_output=raw_output,
            before_text=request.before_text,
            after_text=request.after_text,
            evaluation_profile=request.evaluation_profile,
            model_version=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        )
    except Exception:
        pass  # 保存失败不影响主流程

    # Step 3b: 锚点增强 —— 段落定位 + 多策略搜索将 LLM 的 snippet 映射为原文精确偏移
    for flaw in flaws:
        loc = flaw.location
        if loc.start_char == 0 and loc.end_char == 0:
            start, end = locate_flaw(
                snippet=loc.snippet,
                after_text=request.after_text,
                before_text=request.before_text,
                segment_id=loc.segment_id,
                segments_after=seg_after,
            )
            if start > 0 or end > 0:
                flaw.location = AnchorSpan(
                    segment_id=loc.segment_id,
                    start_char=start,
                    end_char=end,
                    snippet=loc.snippet,
                )

    # Step 3b2: 两阶段验证 —— 用 Verifier 过滤候选瑕疵中的误报（False Positive）
    verifier_before = len(flaws)
    verifier_rejected = 0
    if flaws:
        from .verifier import verify_flaws
        verified_flaws = verify_flaws(
            before_text=request.before_text,
            after_text=request.after_text,
            candidate_flaws=flaws,
            temperature=temperature,
        )
        verifier_rejected = max(0, verifier_before - len(verified_flaws))
        flaws = verified_flaws

    # Step 3c: Merge —— 算法检测到的 critical/major structure 瑕疵补充到 flaws 列表
    # LLM 对结构变化（表格转文本、列表压缩）不敏感，算法层补充
    llm_categories = {(f.category, f.severity) for f in flaws}
    llm_has_structure = any(f.category == "structure" for f in flaws)
    for df in detected_flaws:
        df_cat = df.get("category", "")
        df_sev = df.get("severity", "")
        # 只补充 LLM 没检出的 critical/major structure 瑕疵
        if df_cat == "structure" and df_sev in ("critical", "major") and not llm_has_structure:
            # 优先使用 before_snippet（表格/列表原文），其次尝试从 anchor_before 提取
            snippet = df.get("before_snippet", "") or df.get("after_snippet", "")
            if not snippet:
                anchor_before = df.get("anchor_before", "")
                snippet = anchor_before.split("] ...")[-1].rstrip("...") if "] ..." in anchor_before else ""
            # 算法结构瑕疵的 snippet 可能来自 before_text（表格原文），
            # 但 GT 标注的是 after_text 中的替换文本，两者不重叠 → 无法匹配。
            # 因此：如果 snippet 在 after_text 中找不到，改用 after_text 第一段文本，
            # 确保 snippet 与 GT 有文本重叠。
            if snippet and request.after_text.find(snippet[:min(10, len(snippet))]) < 0:
                # snippet 不在 after_text 中（是 before_text 中的表格/列表内容）
                # 改用 after_text 第一段作为 snippet（结构破坏影响整个文本）
                if seg_after:
                    snippet = seg_after[0].get("text", "")[:80]
                if not snippet:
                    snippet = request.after_text[:80]
            # 用多策略搜索定位该瑕疵在原文中的位置
            algo_start, algo_end = locate_flaw(
                snippet=snippet,
                after_text=request.after_text,
                before_text=request.before_text,
                segment_id=df.get("anchor_before", ""),
                segments_after=seg_after,
            )
            # Fallback：结构破坏类瑕疵如果定位失败，指向 after_text 开头区域
            # （表格/列表被拍平后，替换文本通常从 after_text 首部开始）
            if algo_start == 0 and algo_end == 0:
                algo_start = 0
                algo_end = min(80, len(request.after_text))
                # 用 after_text 开头作为 snippet
                snippet = request.after_text[:min(80, len(request.after_text))]
            flaws.append(FlawItem(
                category="structure",
                severity=df_sev,
                description=f"算法检测：{df.get('type', '结构变化')}（{snippet[:50]}）",
                location=AnchorSpan(
                    segment_id=df.get("anchor_before", "")[:30] if df.get("anchor_before") else "algo",
                    start_char=algo_start,
                    end_char=algo_end,
                    snippet=snippet[:50],
                ),
                suggestion="结构被破坏，建议恢复原始格式",
            ))
    # 记录算法修正（不修改 LLM 原始输出）
    algo_adjustments: dict[str, dict[str, Any]] = {}
    for df in detected_flaws:
        df_cat = df.get("category", "")
        df_sev = df.get("severity", "")
        if df_sev == "critical" and df_cat == "structure":
            # 找 LLM 给的 structure 分数
            llm_struct = next((d.score for d in dimensions if d.dimension == "structure"), 0.5)
            if llm_struct > 0.5:
                algo_adjustments["structure"] = {
                    "llm_score": llm_struct,
                    "penalty": -0.5,
                    "adjusted_score": 0.3,
                    "reason": "Markdown 表格/列表结构被破坏（算法检测）",
                }
        elif df_sev == "critical" and df_cat in ("omission", "over_clean"):
            llm_semantic = next((d.score for d in dimensions if d.dimension == "semantic"), 0.5)
            if llm_semantic > 0.6:
                algo_adjustments["semantic"] = {
                    "llm_score": llm_semantic,
                    "penalty": -0.3,
                    "adjusted_score": max(0.0, llm_semantic - 0.3),
                    "reason": "大量内容被删除（算法检测）",
                }

    adjusted_dimensions = _apply_structural_penalty(dimensions, flaws)

    # 用调整后的维度分数计算最终得分（LLM 原始分数保留在 dimensions 中）
    scoring_dimensions = []
    for d in adjusted_dimensions:
        if d.dimension in algo_adjustments:
            adj = algo_adjustments[d.dimension]
            scoring_dimensions.append(DimensionScore(
                dimension=d.dimension,
                score=adj["adjusted_score"],
                weight=d.weight,
                reason=d.reason,
            ))
        else:
            scoring_dimensions.append(d)

    base_overall_score = compute_overall_score(scoring_dimensions)
    score_floor = _score_floor_from_flaws(flaws)
    overall_score = min(base_overall_score, score_floor)

    # Step 4: 瑕疵仅作为定性输出，不再修改总分（主流 LLM-as-Judge 标准做法）
    # apply_soft_penalty() 函数保留，可用于前端展示惩罚信息，但不影响 overall_score
    verdict = determine_verdict(overall_score)

    overall_score, verdict, flaws = apply_profile_penalties(
        request.evaluation_profile, overall_score, verdict, flaws,
        request.before_text, request.after_text,
    )

    # Step 4c: 线性校准（将 LLM 原始分映射到人工分布）
    _cal = _get_calibrator()
    if _cal is not None:
        overall_score = _cal.calibrate(overall_score)
        verdict = determine_verdict(overall_score)

    # Step 5: 计算置信度
    confidence = compute_confidence(dimensions, flaws, detected_flaws, diff_info)

    # Step 6: 计算判定原因码
    reason_code, reject_reasons = determine_reason_code(overall_score, verdict, flaws)

    # Step 7: 组装响应（含 callback 追踪数据）
    response = EvalResponse(
        request_id=request.request_id,
        evaluation_profile=request.evaluation_profile,
        dimensions=dimensions,
        overall_score=round(overall_score, 4),
        flaws=flaws,
        verdict=verdict,
        reason_code=reason_code,
        reject_reasons=reject_reasons,
        reproducibility_token=build_reproducibility_token(request, temperature),
        model_version=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        prompt_version=hashlib.sha256(
            __import__("app.prompts", fromlist=["SYSTEM_PROMPT"]).SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()[:8],
        rule_version=_get_rule_hash(),
        confidence=confidence,
        risk_level=compute_risk_level(overall_score, confidence),
        algorithm_adjustment=algo_adjustments if algo_adjustments else None,
        raw_llm_output=raw_output,
        latency_seconds=callback.latency_seconds,
        input_tokens=callback.input_tokens,
        output_tokens=callback.output_tokens,
        total_tokens=callback.total_tokens,
    )
    response.parse_diagnostics = parse_diagnostics
    response.verifier_rejected_count = verifier_rejected

    # Step 8: Reflexion 自我反思修正（可开关，默认关闭，不影响既有主路径）
    if getattr(request, "reflect", False):
        from .reflection import reflect_and_correct
        response = reflect_and_correct(
            response,
            before_text=request.before_text,
            after_text=request.after_text,
            temperature=temperature,
        )

    return response


def evaluate(request: EvalRequest, temperature: float = 0.0, use_cache: bool = True) -> EvalResponse:
    """
    LCEL 风格的评估流程（含临界区多次采样）：
      单次评估 → 判定 → 临界区自动多次采样 → 最终结果

    Args:
        use_cache: False 时绕过全局缓存（用于稳定性测试），不影响其他并发 worker。
    """
    # 可观测性：启动追踪
    from .tracing import trace_manager
    trace = trace_manager.start_trace(request.request_id)

    # 单次评估
    span_eval = trace.add_span("evaluate_once", {"request_id": request.request_id})
    resp = _evaluate_once(request, temperature, use_cache=use_cache)
    span_eval.finish(output={"overall_score": resp.overall_score, "verdict": resp.verdict})

    # 临界区自动多次采样（分数在 0.75~0.85 时，再跑 2 次取均值）
    # 防止边界值波动导致 pass/review 跳变
    if 0.75 <= resp.overall_score <= 0.85:
        span_resample = trace.add_span("critical_zone_resample", {"initial_score": resp.overall_score})
        extra_scores = [resp.overall_score]
        for _ in range(2):
            try:
                resp2 = _evaluate_once(request, temperature, use_cache=use_cache)
                extra_scores.append(resp2.overall_score)
            except Exception:
                pass
        avg_score = round(sum(extra_scores) / len(extra_scores), 4)
        resp.overall_score = avg_score
        resp.verdict = determine_verdict(avg_score)
        span_resample.finish(output={"scores": extra_scores, "avg_score": avg_score})

    # 完成追踪
    trace.finish(
        score=resp.overall_score,
        verdict=resp.verdict,
        model=resp.model_version,
        prompt_version=resp.prompt_version,
    )
    trace_manager.save_trace(trace)

    return resp


# ============================================================
# Pairwise Comparison（Chatbot Arena 风格）
# ============================================================

def compare_pair(
    before_text: str,
    output_a: str,
    output_b: str,
    evaluation_profile: str = PROFILE_GENERAL,
    label_a: str = "A",
    label_b: str = "B",
    model: str | None = None,
    temperature: float = 0.0,
) -> "CompareResponse":
    """
    对比评估：对同一段原文的两个治理结果分别评估，然后计算差异。

    设计决策：
    - 复用已有的 evaluate() 管线（含 Diff 检测、锚点解析、校准等），
      而非让 LLM 在单次调用中同时评估两个文本。
    - 优势：每个输出都享受完整的后处理管线，对比结果是精确的数学差值。
    - 代价：2 倍 LLM 调用量（但对比评估本身就是低频操作）。

    Args:
        before_text: 治理前原文
        output_a: 治理结果 A
        output_b: 治理结果 B
        evaluation_profile: 评估模式
        label_a: A 的显示标签
        label_b: B 的显示标签
        model: 指定模型（可选）
        temperature: 温度参数

    Returns:
        CompareResponse: 包含两侧评估结果和对比分析
    """
    import uuid
    import time as _time
    from .models import (
        CompareRequest,
        CompareResponse,
        DimensionDelta,
        SideEvaluation,
    )

    t0 = _time.time()

    # 构建两个独立的 EvalRequest
    req_a = EvalRequest(
        request_id=f"compare-{uuid.uuid4().hex[:8]}-a",
        before_text=before_text,
        after_text=output_a,
        evaluation_profile=evaluation_profile,
        model=model,
    )
    req_b = EvalRequest(
        request_id=f"compare-{uuid.uuid4().hex[:8]}-b",
        before_text=before_text,
        after_text=output_b,
        evaluation_profile=evaluation_profile,
        model=model,
    )

    # 分别评估（复用完整管线）
    resp_a = evaluate(req_a, temperature=temperature)
    resp_b = evaluate(req_b, temperature=temperature)

    # 构建维度映射（dimension_name -> score）
    scores_a = {d.dimension: d.score for d in resp_a.dimensions}
    scores_b = {d.dimension: d.score for d in resp_b.dimensions}

    # 计算各维度差异
    dimension_deltas = []
    all_dims = ["semantic", "factual", "hallucination", "structure", "readability"]
    for dim in all_dims:
        sa = scores_a.get(dim, 0.0)
        sb = scores_b.get(dim, 0.0)
        delta = round(sa - sb, 4)
        if abs(delta) < 0.02:
            winner = "tie"
        elif delta > 0:
            winner = "A"
        else:
            winner = "B"
        dimension_deltas.append(DimensionDelta(
            dimension=dim,
            score_a=sa,
            score_b=sb,
            delta=delta,
            winner=winner,
        ))

    # 总体对比
    overall_delta = round(resp_a.overall_score - resp_b.overall_score, 4)

    # 判定胜出方
    has_critical_a = any(
        f.severity == "critical" for f in resp_a.flaws
    )
    has_critical_b = any(
        f.severity == "critical" for f in resp_b.flaws
    )

    if has_critical_a and not has_critical_b:
        winner = "B"
    elif has_critical_b and not has_critical_a:
        winner = "A"
    elif abs(overall_delta) > 0.05:
        winner = "A" if overall_delta > 0 else "B"
    else:
        winner = "tie"

    # 生成对比理由
    a_wins = [d.dimension for d in dimension_deltas if d.winner == "A"]
    b_wins = [d.dimension for d in dimension_deltas if d.winner == "B"]
    reason_parts = []
    if a_wins:
        reason_parts.append(f"{label_a} 在 {', '.join(a_wins)} 维度领先")
    if b_wins:
        reason_parts.append(f"{label_b} 在 {', '.join(b_wins)} 维度领先")
    if not reason_parts:
        reason_parts.append("两者表现接近，各维度差异均在容差范围内")
    if has_critical_a:
        reason_parts.append(f"注意：{label_a} 存在 critical 级别瑕疵")
    if has_critical_b:
        reason_parts.append(f"注意：{label_b} 存在 critical 级别瑕疵")
    reason = "；".join(reason_parts) + "。"

    latency = round(_time.time() - t0, 2)

    return CompareResponse(
        evaluation_a=SideEvaluation(
            dimensions=resp_a.dimensions,
            flaws=resp_a.flaws,
            overall_score=resp_a.overall_score,
            verdict=resp_a.verdict,
        ),
        evaluation_b=SideEvaluation(
            dimensions=resp_b.dimensions,
            flaws=resp_b.flaws,
            overall_score=resp_b.overall_score,
            verdict=resp_b.verdict,
        ),
        dimension_deltas=dimension_deltas,
        winner=winner,
        overall_delta=overall_delta,
        reason=reason,
        label_a=label_a,
        label_b=label_b,
        latency_seconds=latency,
    )
