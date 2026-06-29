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
    stabilize: bool = Field(False, description="是否启用多次采样稳定化")
    sample_count: int = Field(3, ge=1, le=5, description="稳定化采样次数")

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
    reproducibility_token: str = Field(..., description="可复现令牌（输入+模型+prompt 指纹）")
    model_version: str = Field("", description="评估时使用的模型标识")
    prompt_version: str = Field("", description="评估时使用的 prompt 版本哈希（前8位）")
    raw_llm_output: Optional[str] = Field(None, description="LLM 原始输出（留存判定依据）")

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