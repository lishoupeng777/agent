"""核心评判引擎 —— LLM-as-Judge

保留此模块作为向后兼容入口。
实际评估逻辑已迁移到 chain.py（LCEL 架构）。
"""
from __future__ import annotations

from typing import Any

from .models import EvalRequest, EvalResponse

# 向后兼容：从 chain.py 导入所有公开接口
from .chain import (
    create_llm as get_llm,
    evaluate,
    parse_llm_output as _parse_llm_json,
    EvalCallbackHandler,
)

__all__ = ["get_llm", "evaluate", "_parse_llm_json", "EvalCallbackHandler"]
