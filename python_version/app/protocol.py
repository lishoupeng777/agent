"""统一评估协议（Unified Evaluation Protocol）

定义所有模型实现必须遵守的接口规范。
上层代码（routes、reporter、agent）只依赖此协议，不关心底层模型。

协议核心：
- 输入：EvalRequest（before_text + after_text + evaluation_profile）
- 输出：EvalResponse（overall_score + dimensions + flaws + verdict）
- 所有模型实现返回相同结构的结果，确保可比性
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import EvalRequest, EvalResponse


class EvaluationProtocol(ABC):
    """评估协议抽象基类。

    任何模型适配器必须继承此类并实现 evaluate() 方法。
    确保：
    1. 输入输出结构统一
    2. 分数范围统一（0.0-1.0）
    3. 判定标准统一（pass/review/fail）
    4. 瑕疵格式统一（category + severity + location）
    """

    @abstractmethod
    def evaluate(self, request: EvalRequest, temperature: float = 0.0) -> EvalResponse:
        """单次评估。

        Args:
            request: 评估请求（包含 before_text, after_text, evaluation_profile）
            temperature: 温度参数（0.0 = 确定性输出）

        Returns:
            EvalResponse：标准化评估结果
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """返回当前使用的模型名称。

        用于：
        - reproducibility_token 生成
        - 报告中的模型标识
        - 多模型对比时区分不同模型
        """
        ...

    @abstractmethod
    def get_model_version(self) -> str:
        """返回模型版本标识。

        用于：
        - 可复现性验证
        - Prompt 版本管理
        """
        ...

    def batch_evaluate(
        self,
        requests: list[EvalRequest],
        temperature: float = 0.0,
    ) -> list[EvalResponse]:
        """批量评估（默认逐条调用 evaluate，子类可覆盖优化）。

        Args:
            requests: 评估请求列表
            temperature: 温度参数

        Returns:
            list[EvalResponse]：评估结果列表
        """
        return [self.evaluate(req, temperature) for req in requests]


class ProtocolResult:
    """协议执行结果包装。

    用于统一返回评估结果 + 执行元数据（延迟、token 等）。
    """

    def __init__(
        self,
        response: EvalResponse,
        latency_seconds: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        model_name: str = "",
    ) -> None:
        self.response = response
        self.latency_seconds = latency_seconds
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.model_name = model_name

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "evaluation": self.response.model_dump(),
            "execution": {
                "model": self.model_name,
                "latency_seconds": self.latency_seconds,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
        }
