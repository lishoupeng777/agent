"""实验追踪模块（Experiment Tracking）

自动记录每次评估的完整元数据，支持：
- 本地 JSONL 持久化（默认，零依赖）
- MLflow 集成（可选，配置 MLFLOW_TRACKING_URI 后自动启用）
- Weights & Biases 集成（可选，配置 WANDB_API_KEY 后自动启用）

设计目标：
- 对主链零侵入：通过 hook 机制在评估完成后自动记录
- 支持查询、统计、趋势分析
- 数据格式兼容 MLflow/W&B 的导入规范
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import EvalRequest, EvalResponse


# ============================================================
# 数据存储路径
# ============================================================

_DATA_DIR = Path(__file__).parent.parent / "data"
_EXPERIMENTS_FILE = _DATA_DIR / "experiments.jsonl"


# ============================================================
# 核心记录类
# ============================================================

class ExperimentRecord:
    """单条实验记录。"""

    def __init__(
        self,
        request_id: str,
        timestamp: str,
        model_version: str,
        evaluation_profile: str,
        overall_score: float,
        verdict: str,
        dimensions: dict[str, float],
        flaw_count: int,
        flaw_severities: dict[str, int],
        latency_seconds: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        confidence: float | None = None,
        risk_level: str | None = None,
        prompt_version: str = "",
        rule_version: str = "",
        before_text_length: int = 0,
        after_text_length: int = 0,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.request_id = request_id
        self.timestamp = timestamp
        self.model_version = model_version
        self.evaluation_profile = evaluation_profile
        self.overall_score = overall_score
        self.verdict = verdict
        self.dimensions = dimensions
        self.flaw_count = flaw_count
        self.flaw_severities = flaw_severities
        self.latency_seconds = latency_seconds
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.confidence = confidence
        self.risk_level = risk_level
        self.prompt_version = prompt_version
        self.rule_version = rule_version
        self.before_text_length = before_text_length
        self.after_text_length = after_text_length
        self.tags = tags or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "evaluation_profile": self.evaluation_profile,
            "overall_score": self.overall_score,
            "verdict": self.verdict,
            "dimensions": self.dimensions,
            "flaw_count": self.flaw_count,
            "flaw_severities": self.flaw_severities,
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "prompt_version": self.prompt_version,
            "rule_version": self.rule_version,
            "before_text_length": self.before_text_length,
            "after_text_length": self.after_text_length,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__init__.__code__.co_varnames})

    @classmethod
    def from_evaluation(cls, request: EvalRequest, response: EvalResponse) -> "ExperimentRecord":
        """从评估请求和响应中自动构建实验记录。"""
        # 提取维度分数
        dimensions = {d.dimension: d.score for d in response.dimensions}

        # 统计瑕疵严重程度
        flaw_severities: dict[str, int] = {"critical": 0, "major": 0, "minor": 0}
        for flaw in response.flaws:
            sev = flaw.severity
            if sev in flaw_severities:
                flaw_severities[sev] += 1

        return cls(
            request_id=response.request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_version=response.model_version,
            evaluation_profile=response.evaluation_profile,
            overall_score=response.overall_score,
            verdict=response.verdict,
            dimensions=dimensions,
            flaw_count=len(response.flaws),
            flaw_severities=flaw_severities,
            latency_seconds=response.latency_seconds,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            confidence=response.confidence,
            risk_level=response.risk_level,
            prompt_version=response.prompt_version,
            rule_version=response.rule_version,
            before_text_length=len(request.before_text),
            after_text_length=len(request.after_text),
        )


# ============================================================
# 本地 JSONL 存储
# ============================================================

def _ensure_data_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def log_experiment(record: ExperimentRecord) -> None:
    """记录一条实验数据到本地 JSONL 文件。"""
    _ensure_data_dir()
    with open(_EXPERIMENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_experiments(
    limit: int = 100,
    offset: int = 0,
    model_filter: str | None = None,
    profile_filter: str | None = None,
    verdict_filter: str | None = None,
) -> list[ExperimentRecord]:
    """加载实验记录。

    Args:
        limit: 最大返回条数
        offset: 偏移量
        model_filter: 按模型过滤
        profile_filter: 按评估模式过滤
        verdict_filter: 按判定结果过滤

    Returns:
        list[ExperimentRecord]
    """
    if not _EXPERIMENTS_FILE.exists():
        return []

    records = []
    with open(_EXPERIMENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                record = ExperimentRecord.from_dict(data)
                # 过滤
                if model_filter and record.model_version != model_filter:
                    continue
                if profile_filter and record.evaluation_profile != profile_filter:
                    continue
                if verdict_filter and record.verdict != verdict_filter:
                    continue
                records.append(record)
            except (json.JSONDecodeError, TypeError):
                continue

    # 按时间倒序
    records.sort(key=lambda r: r.timestamp, reverse=True)
    return records[offset:offset + limit]


def experiment_stats() -> dict[str, Any]:
    """实验统计汇总。"""
    records = load_experiments(limit=10000)
    if not records:
        return {
            "total": 0,
            "models": {},
            "profiles": {},
            "verdicts": {"pass": 0, "review": 0, "fail": 0},
            "avg_score": 0.0,
            "avg_latency": 0.0,
            "total_tokens": 0,
        }

    models: dict[str, int] = {}
    profiles: dict[str, int] = {}
    verdicts: dict[str, int] = {"pass": 0, "review": 0, "fail": 0}
    total_score = 0.0
    total_latency = 0.0
    latency_count = 0
    total_tokens = 0

    for r in records:
        models[r.model_version] = models.get(r.model_version, 0) + 1
        profiles[r.evaluation_profile] = profiles.get(r.evaluation_profile, 0) + 1
        verdicts[r.verdict] = verdicts.get(r.verdict, 0) + 1
        total_score += r.overall_score
        if r.latency_seconds is not None:
            total_latency += r.latency_seconds
            latency_count += 1
        if r.total_tokens:
            total_tokens += r.total_tokens

    return {
        "total": len(records),
        "models": models,
        "profiles": profiles,
        "verdicts": verdicts,
        "avg_score": round(total_score / len(records), 4),
        "avg_latency": round(total_latency / max(latency_count, 1), 2),
        "total_tokens": total_tokens,
    }


def experiment_trend(
    model_filter: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """实验趋势数据（用于前端图表）。

    Returns:
        list of {timestamp, overall_score, verdict, model}
    """
    records = load_experiments(limit=limit, model_filter=model_filter)
    # 按时间正序
    records.sort(key=lambda r: r.timestamp)
    return [
        {
            "timestamp": r.timestamp,
            "overall_score": r.overall_score,
            "verdict": r.verdict,
            "model": r.model_version,
            "profile": r.evaluation_profile,
            "latency": r.latency_seconds,
            "flaw_count": r.flaw_count,
        }
        for r in records
    ]


# ============================================================
# MLflow 集成（可选）
# ============================================================

_mlflow_client = None


def _get_mlflow_client():
    """懒加载 MLflow client。"""
    global _mlflow_client
    if _mlflow_client is not None:
        return _mlflow_client

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return None

    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("fidelity-eval")
        _mlflow_client = mlflow
        print(f"[ExperimentTracking] MLflow 已启用: {tracking_uri}")
        return _mlflow_client
    except ImportError:
        print("[ExperimentTracking] MLflow 未安装，跳过（pip install mlflow）")
        return None
    except Exception as e:
        print(f"[ExperimentTracking] MLflow 初始化失败: {e}")
        return None


def log_to_mlflow(record: ExperimentRecord) -> None:
    """将实验记录同步到 MLflow。"""
    mlflow = _get_mlflow_client()
    if mlflow is None:
        return

    try:
        with mlflow.start_run(run_name=f"eval-{record.request_id[:8]}", nested=True):
            # 记录参数
            mlflow.log_param("model", record.model_version)
            mlflow.log_param("profile", record.evaluation_profile)
            mlflow.log_param("prompt_version", record.prompt_version)
            mlflow.log_param("rule_version", record.rule_version)

            # 记录指标
            mlflow.log_metric("overall_score", record.overall_score)
            mlflow.log_metric("flaw_count", record.flaw_count)
            for dim, score in record.dimensions.items():
                mlflow.log_metric(f"dim_{dim}", score)
            if record.latency_seconds is not None:
                mlflow.log_metric("latency_seconds", record.latency_seconds)
            if record.confidence is not None:
                mlflow.log_metric("confidence", record.confidence)
            if record.total_tokens is not None:
                mlflow.log_metric("total_tokens", record.total_tokens)

            # 记录标签
            tags = {"verdict": record.verdict, "risk_level": record.risk_level or "unknown"}
            mlflow.set_tags(tags)
    except Exception as e:
        print(f"[ExperimentTracking] MLflow 记录失败: {e}")


# ============================================================
# Weights & Biases 集成（可选）
# ============================================================

_wandb_run = None


def _get_wandb():
    """懒加载 W&B。"""
    global _wandb_run
    if _wandb_run is not None:
        return _wandb_run

    api_key = os.getenv("WANDB_API_KEY", "")
    if not api_key:
        return None

    try:
        import wandb
        _wandb_run = wandb.init(
            project="fidelity-eval",
            name="evaluation-tracking",
            resume="allow",
            config={"service": "llm-quality-judge"},
        )
        print("[ExperimentTracking] W&B 已启用")
        return _wandb_run
    except ImportError:
        print("[ExperimentTracking] W&B 未安装，跳过（pip install wandb）")
        return None
    except Exception as e:
        print(f"[ExperimentTracking] W&B 初始化失败: {e}")
        return None


def log_to_wandb(record: ExperimentRecord) -> None:
    """将实验记录同步到 W&B。"""
    wandb_run = _get_wandb()
    if wandb_run is None:
        return

    try:
        import wandb
        log_data = {
            "eval/overall_score": record.overall_score,
            "eval/flaw_count": record.flaw_count,
            "eval/latency": record.latency_seconds or 0,
        }
        for dim, score in record.dimensions.items():
            log_data[f"eval/dim_{dim}"] = score
        if record.total_tokens:
            log_data["eval/tokens"] = record.total_tokens
        wandb.log(log_data)
    except Exception as e:
        print(f"[ExperimentTracking] W&B 记录失败: {e}")


# ============================================================
# 统一记录入口
# ============================================================

def track_evaluation(request: EvalRequest, response: EvalResponse) -> None:
    """统一实验记录入口。

    自动执行：
    1. 写入本地 JSONL（始终执行）
    2. 同步到 MLflow（如果已配置）
    3. 同步到 W&B（如果已配置）

    此函数应在每次评估完成后调用。
    异常不影响主流程（静默失败）。
    """
    try:
        record = ExperimentRecord.from_evaluation(request, response)

        # 1. 本地持久化（始终执行）
        log_experiment(record)

        # 2. MLflow（可选）
        log_to_mlflow(record)

        # 3. W&B（可选）
        log_to_wandb(record)

    except Exception as e:
        # 实验追踪失败不应影响主流程
        print(f"[ExperimentTracking] 记录失败（不影响评估）: {e}")
