"""请求与响应数据模型"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from .profiles import PROFILE_GENERAL, validate_profile_key


class AnchorSpan(BaseModel):
    """锚点定位信息"""
    segment_id: str = Field(..., description="段落/句子唯一标识")
    start_char: int = Field(..., description="起始字符偏移")
    end_char: int = Field(..., description="结束字符偏移")
    snippet: str = Field(..., description="原文片段供人工复核")


class FlawItem(BaseModel):
    """单条瑕疵记录"""
    category: str = Field(..., description="瑕疵类别: over_clean | mis_edit | readability | structure")
    severity: str = Field(..., description="严重程度: critical | major | minor")
    description: str = Field(..., description="判定理由（可解释）")
    location: AnchorSpan = Field(..., description="锚点定位")
    suggestion: Optional[str] = Field(None, description="修复建议")


class DimensionScore(BaseModel):
    """单维度评分"""
    dimension: str = Field(..., description="评估维度")
    score: float = Field(..., ge=0.0, le=1.0, description="得分 0~1")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="维度权重")
    reason: str = Field(..., description="评分理由")


class EvalRequest(BaseModel):
    """评估请求"""
    request_id: str = Field(..., description="请求唯一标识")
    before_text: str = Field(..., description="治理前原文")
    after_text: str = Field(..., description="治理后文本")
    evaluation_profile: str = Field(PROFILE_GENERAL, description="评估模式")
    segments_before: Optional[list[dict[str, Any]]] = Field(
        None, description="治理前分段信息（含 segment_id, offset 等锚点）"
    )
    segments_after: Optional[list[dict[str, Any]]] = Field(
        None, description="治理后分段信息"
    )
    human_label: Optional[dict[str, Any]] = Field(
        None, description="人工标注（用于一致性校准，可选）"
    )
    model: Optional[str] = Field(None, description="指定模型（deepseek/mimo/gpt），None 则使用默认模型")
    stabilize: bool = Field(False, description="是否启用多次采样稳定化")
    sample_count: int = Field(3, ge=1, le=5, description="稳定化采样次数")
    reflect: bool = Field(False, description="是否启用 Reflexion 自我反思修正（检测评分与证据矛盾后按需重评）")

    @field_validator("evaluation_profile")
    @classmethod
    def validate_evaluation_profile(cls, value: str) -> str:
        return validate_profile_key(value)


class EvalResponse(BaseModel):
    """评估响应"""
    request_id: str
    evaluation_profile: str = Field(PROFILE_GENERAL, description="本次评估使用的评估模式")
    dimensions: list[DimensionScore] = Field(..., description="各维度评分")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="加权总分")
    flaws: list[FlawItem] = Field(default_factory=list, description="瑕疵清单（可锚点定位）")
    verdict: str = Field(..., description="综合判定: pass | review | fail")
    reason_code: str = Field("PASS_ALL_CLEAR", description="判定原因码: PASS_ALL_CLEAR | CRITICAL_FLAW_DETECTED | SCORE_BELOW_THRESHOLD | REVIEW_REQUIRED")
    reject_reasons: list[dict[str, Any]] = Field(default_factory=list, description="详细判定原因列表")
    reproducibility_token: str = Field(..., description="可复现令牌（输入+模型+prompt 指纹）")
    model_version: str = Field("", description="评估时使用的模型标识")
    prompt_version: str = Field("", description="评估时使用的 prompt 版本哈希（前8位）")
    rule_version: str = Field("", description="评分规则版本（维度权重/惩罚因子等变更时递增）")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="评估置信度（0~1，基于Diff一致性/理由完整度/规则一致性）")
    risk_level: str | None = Field(None, description="风险等级：high(0~0.3) / medium(0.3~0.7) / low(0.7+)")
    algorithm_adjustment: dict[str, dict[str, float | str]] | None = Field(None, description="算法修正记录（保留LLM原始分数，单独记录修正值和原因）")
    raw_llm_output: Optional[str] = Field(None, description="LLM 原始输出（留存判定依据）")
    latency_seconds: float | None = Field(None, description="LLM 响应延迟（秒）")
    input_tokens: int | None = Field(None, description="输入 token 数")
    output_tokens: int | None = Field(None, description="输出 token 数")
    total_tokens: int | None = Field(None, description="总 token 数")
    parse_diagnostics: dict[str, Any] | None = Field(None, description="解析诊断信息（原始解析/验证降级情况）")
    verifier_rejected_count: int | None = Field(None, description="二阶段验证剔除的候选瑕疵数")
    reflection: dict[str, Any] | None = Field(None, description="Reflexion 反思轨迹（检测到的矛盾 + 是否重评 + 分数变化）")

    @field_validator("evaluation_profile")
    @classmethod
    def validate_evaluation_profile(cls, value: str) -> str:
        return validate_profile_key(value)


class StabilityReport(BaseModel):
    """评分稳定性报告"""
    request_id: str
    samples: list[dict[str, float]] = Field(..., description="各次采样得分详情")
    mean_score: float
    variance: float
    std_dev: float
    is_stable: bool = Field(..., description="方差是否在阈值内")


class CalibrationReport(BaseModel):
    """一致性校准报告"""
    pearson_r: float = Field(..., description="皮尔逊相关系数")
    spearman_rho: float = Field(..., description="斯皮尔曼秩相关系数")
    mae: float = Field(..., description="平均绝对误差")
    rmse: float = Field(..., description="均方根误差")
    consistency_rate: float = Field(..., description="一致率（阈值容差内一致的比例）")
    sample_count: int
    details: list[dict[str, Any]] = Field(default_factory=list, description="逐条对比详情")


class RecalcParams(BaseModel):
    """热重算参数（用户可调）"""
    dimension_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "semantic": 0.35,
            "factual": 0.35,
            "structure": 0.15,
            "readability": 0.15,
        },
        description="维度权重（自动归一化）"
    )
    penalty_factors: dict[str, float] = Field(
        default_factory=lambda: {
            "critical": 0.6,
            "major": 0.85,
            "minor": 0.95,
        },
        description="惩罚系数（severity -> 惩罚因子）"
    )
    pass_threshold: float = Field(0.82, ge=0.0, le=1.0, description="通过阈值")
    review_threshold: float = Field(0.5, ge=0.0, le=1.0, description="复审阈值")
    anchor_tolerance: int = Field(10, ge=0, le=100, description="锚点容差（字符数）")


class RecalcResultItem(BaseModel):
    """单条重算结果"""
    request_id: str
    dimensions: list[dict[str, Any]]
    overall_score: float
    verdict: str
    flaws: list[dict[str, Any]]
    penalty_applied: float


class RecalcReport(BaseModel):
    """批量重算报告"""
    params: RecalcParams
    results: list[RecalcResultItem]
    pearson_r: float | None = None
    spearman_rho: float | None = None
    mae: float | None = None
    rmse: float | None = None
    consistency_rate: float | None = None


# ============================================================
# Pairwise Comparison Models（Chatbot Arena 风格）
# ============================================================

class CompareRequest(BaseModel):
    """对比评估请求：同一段原文 + 两个治理结果"""
    before_text: str = Field(..., description="治理前原文")
    output_a: str = Field(..., description="治理结果 A")
    output_b: str = Field(..., description="治理结果 B")
    evaluation_profile: str = Field(PROFILE_GENERAL, description="评估模式")
    label_a: str = Field("A", description="结果 A 的显示标签")
    label_b: str = Field("B", description="结果 B 的显示标签")
    model: Optional[str] = Field(None, description="指定模型，None 则使用默认模型")

    @field_validator("evaluation_profile")
    @classmethod
    def validate_evaluation_profile(cls, value: str) -> str:
        return validate_profile_key(value)


class DimensionDelta(BaseModel):
    """单维度对比差异"""
    dimension: str
    score_a: float = Field(..., ge=0.0, le=1.0)
    score_b: float = Field(..., ge=0.0, le=1.0)
    delta: float = Field(..., description="score_a - score_b")
    winner: str = Field(..., description="A / B / tie")


class SideEvaluation(BaseModel):
    """单侧评估结果（A 或 B）"""
    dimensions: list[DimensionScore] = Field(default_factory=list)
    flaws: list[FlawItem] = Field(default_factory=list)
    overall_score: float = 0.0
    verdict: str = "pass"


class CompareResponse(BaseModel):
    """对比评估响应"""
    evaluation_a: SideEvaluation = Field(..., description="结果 A 的评估")
    evaluation_b: SideEvaluation = Field(..., description="结果 B 的评估")
    dimension_deltas: list[DimensionDelta] = Field(default_factory=list, description="各维度差异")
    winner: str = Field(..., description="胜出方: A / B / tie")
    overall_delta: float = Field(..., description="加权总分差值 (A - B)")
    reason: str = Field("", description="综合对比理由")
    label_a: str = "A"
    label_b: str = "B"
    latency_seconds: float | None = Field(None, description="总耗时（秒）")