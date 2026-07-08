"""模型注册表（Model Registry）

管理多个模型适配器的注册、查询和调度。
上层代码通过注册表获取模型，不直接实例化适配器。
"""
from __future__ import annotations

from typing import Any

from .protocol import EvaluationProtocol, ProtocolResult
from .models import EvalRequest, EvalResponse


class ModelRegistry:
    """模型注册表。

    使用方式：
        registry = ModelRegistry()
        registry.register("deepseek", DeepSeekAdapter())
        registry.register("mimo", MimoAdapter())

        # 用指定模型评估
        result = registry.evaluate("deepseek", request)

        # 用所有注册模型评估（对比）
        results = registry.evaluate_all(request)
    """

    def __init__(self) -> None:
        self._adapters: dict[str, EvaluationProtocol] = {}
        self._default: str | None = None

    def register(self, name: str, adapter: EvaluationProtocol, default: bool = False) -> None:
        """注册模型适配器。

        Args:
            name: 模型名称（如 "deepseek", "mimo", "gpt"）
            adapter: 评估协议实现
            default: 是否设为默认模型
        """
        self._adapters[name] = adapter
        if default or self._default is None:
            self._default = name

    def unregister(self, name: str) -> None:
        """注销模型适配器"""
        self._adapters.pop(name, None)
        if self._default == name:
            self._default = next(iter(self._adapters), None)

    def get(self, name: str | None = None) -> EvaluationProtocol:
        """获取模型适配器。

        Args:
            name: 模型名称，None 则返回默认模型

        Returns:
            EvaluationProtocol 实例

        Raises:
            KeyError: 模型未注册
        """
        key = name or self._default
        if key is None or key not in self._adapters:
            raise KeyError(f"Model '{key}' not registered. Available: {list(self._adapters.keys())}")
        return self._adapters[key]

    def list_models(self) -> list[str]:
        """返回所有已注册模型名称"""
        return list(self._adapters.keys())

    @property
    def default_model(self) -> str | None:
        return self._default

    def evaluate(
        self,
        model_name: str | None,
        request: EvalRequest,
        temperature: float = 0.0,
    ) -> ProtocolResult:
        """用指定模型评估。

        Args:
            model_name: 模型名称（None = 默认模型）
            request: 评估请求
            temperature: 温度参数

        Returns:
            ProtocolResult
        """
        adapter = self.get(model_name)
        resp = adapter.evaluate(request, temperature)
        return ProtocolResult(
            response=resp,
            model_name=adapter.get_model_name(),
        )

    def evaluate_all(
        self,
        request: EvalRequest,
        temperature: float = 0.0,
    ) -> dict[str, ProtocolResult]:
        """用所有注册模型评估（用于多模型对比）。

        Args:
            request: 评估请求
            temperature: 温度参数

        Returns:
            dict：模型名称 → ProtocolResult
        """
        results = {}
        for name, adapter in self._adapters.items():
            try:
                resp = adapter.evaluate(request, temperature)
                results[name] = ProtocolResult(
                    response=resp,
                    model_name=adapter.get_model_name(),
                )
            except Exception as e:
                results[name] = ProtocolResult(
                    response=EvalResponse(
                        request_id=request.request_id,
                        evaluation_profile=request.evaluation_profile,
                        dimensions=[],
                        overall_score=0.0,
                        flaws=[],
                        verdict="fail",
                        reproducibility_token="error",
                        model_version=adapter.get_model_name(),
                        prompt_version="error",
                        raw_llm_output=f"Error: {e}",
                    ),
                    model_name=adapter.get_model_name(),
                )
        return results


# 全局注册表实例
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """获取全局模型注册表（懒加载）"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def register_default_models() -> None:
    """注册所有可用模型适配器。

    默认注册 DeepSeek，其他模型按环境变量可用性注册。
    """
    from .adapters import DeepSeekAdapter, MimoAdapter, GPTAdapter
    import os

    registry = get_registry()

    # DeepSeek（默认）
    registry.register("deepseek", DeepSeekAdapter(), default=True)

    # Mimo 2.5 Pro
    registry.register("mimo", MimoAdapter())

    # GPT（如果配置了 API Key）
    if os.getenv("GPT_API_KEY"):
        registry.register("gpt", GPTAdapter())
