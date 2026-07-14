"""可观测性模块 —— 评估链路全链路追踪

支持两种模式：
1. 本地追踪：将每次评估的完整链路写入 JSON 文件（默认）
2. LangFuse 追踪：接入 LangFuse 云平台（需配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY）

追踪内容：
- 每个步骤的输入/输出/耗时
- LLM 调用的 token 数/延迟
- 中间结果（维度分数、瑕疵列表）
- 最终输出
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class TraceSpan:
    """单个追踪跨度"""
    span_id: str
    name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    input: Optional[dict] = None
    output: Optional[dict] = None
    metadata: dict = field(default_factory=dict)
    status: str = "ok"  # ok / error

    def finish(self, output: dict | None = None, status: str = "ok") -> None:
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.output = output
        self.status = status


@dataclass
class EvaluationTrace:
    """一次完整评估的追踪记录"""
    trace_id: str
    request_id: str
    start_time: float
    end_time: float = 0.0
    total_duration_ms: float = 0.0
    spans: list[TraceSpan] = field(default_factory=list)
    final_score: float = 0.0
    final_verdict: str = ""
    model: str = ""
    prompt_version: str = ""

    def add_span(self, name: str, input_data: dict | None = None) -> TraceSpan:
        span = TraceSpan(
            span_id=str(uuid.uuid4())[:8],
            name=name,
            start_time=time.time(),
            input=input_data,
        )
        self.spans.append(span)
        return span

    def finish(self, score: float, verdict: str, model: str = "", prompt_version: str = "") -> None:
        self.end_time = time.time()
        self.total_duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.final_score = score
        self.final_verdict = verdict
        self.model = model
        self.prompt_version = prompt_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "final_score": self.final_score,
            "final_verdict": self.final_verdict,
            "spans": [
                {
                    "span_id": s.span_id,
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "status": s.status,
                    "input_keys": list(s.input.keys()) if s.input else [],
                    "output_keys": list(s.output.keys()) if s.output else [],
                    "metadata": s.metadata,
                }
                for s in self.spans
            ],
        }


class TraceManager:
    """追踪管理器 —— 管理当前追踪上下文"""

    def __init__(self, enabled: bool = True, trace_dir: str | None = None) -> None:
        self.enabled = enabled
        self.trace_dir = trace_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "traces",
        )
        self._current_trace: Optional[EvaluationTrace] = None

    def start_trace(self, request_id: str) -> EvaluationTrace:
        trace = EvaluationTrace(
            trace_id=str(uuid.uuid4())[:12],
            request_id=request_id,
            start_time=time.time(),
        )
        self._current_trace = trace
        return trace

    @property
    def current(self) -> Optional[EvaluationTrace]:
        return self._current_trace

    def save_trace(self, trace: EvaluationTrace) -> str:
        """保存追踪记录到 JSON 文件"""
        if not self.enabled:
            return ""
        os.makedirs(self.trace_dir, exist_ok=True)
        filename = f"{trace.trace_id}_{trace.request_id}.json"
        filepath = os.path.join(self.trace_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)
        return filepath

    def list_traces(self, limit: int = 20) -> list[dict]:
        """列出最近的追踪记录"""
        if not os.path.exists(self.trace_dir):
            return []
        files = sorted(
            [f for f in os.listdir(self.trace_dir) if f.endswith(".json")],
            key=lambda f: os.path.getmtime(os.path.join(self.trace_dir, f)),
            reverse=True,
        )[:limit]
        traces = []
        for f in files:
            try:
                with open(os.path.join(self.trace_dir, f), "r", encoding="utf-8") as fh:
                    traces.append(json.load(fh))
            except Exception:
                pass
        return traces


# 全局追踪管理器
trace_manager = TraceManager(enabled=True)


def setup_langfuse() -> bool:
    """尝试初始化 LangFuse（可选）。

    需要环境变量：LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
    如果未配置，返回 False，系统回退到本地追踪。
    """
    try:
        from langfuse.callback import CallbackHandler
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if public_key and secret_key:
            handler = CallbackHandler(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            print(f"[可观测性] LangFuse 已连接: {host}")
            return True
    except ImportError:
        pass
    return False
