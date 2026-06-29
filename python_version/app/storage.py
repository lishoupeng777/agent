"""评估历史持久化 —— JSONL 格式追加写入，支持按 token/request_id 查询"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

# 默认历史文件路径（可通过环境变量覆盖）
_DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_history.jsonl"
HISTORY_PATH = Path(os.getenv("EVAL_HISTORY_PATH", str(_DEFAULT_HISTORY_PATH)))


def _ensure_dir() -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_evaluation(response: Any) -> None:
    """
    将 EvalResponse 追加写入历史文件。

    每行一条 JSON 记录，包含：
      ts, request_id, before_hash, after_hash,
      overall_score, verdict, model_version, prompt_version,
      reproducibility_token
    """
    _ensure_dir()
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "request_id": response.request_id,
        "overall_score": response.overall_score,
        "verdict": response.verdict,
        "model_version": getattr(response, "model_version", ""),
        "prompt_version": getattr(response, "prompt_version", ""),
        "reproducibility_token": response.reproducibility_token,
        "flaw_count": len(response.flaws),
        "dimension_scores": {d.dimension: d.score for d in response.dimensions},
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_history(
    request_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    读取历史记录，支持按 request_id 过滤。

    Args:
        request_id: 若提供，只返回该 request_id 的记录
        limit: 最多返回条数（默认100）
        offset: 跳过前 N 条（用于分页）

    Returns:
        记录列表（按时间倒序）
    """
    if not HISTORY_PATH.exists():
        return []

    records: list[dict[str, Any]] = []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if request_id is None or rec.get("request_id") == request_id:
                    records.append(rec)
            except json.JSONDecodeError:
                continue

    # 倒序（最新在前）
    records.reverse()
    return records[offset: offset + limit]


def find_by_token(token: str) -> dict[str, Any] | None:
    """
    根据可复现令牌查找历史评估记录。

    Args:
        token: reproducibility_token

    Returns:
        匹配的记录，未找到返回 None
    """
    if not HISTORY_PATH.exists():
        return None

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("reproducibility_token") == token:
                    return rec
            except json.JSONDecodeError:
                continue
    return None


def history_stats() -> dict[str, Any]:
    """返回历史记录的汇总统计"""
    if not HISTORY_PATH.exists():
        return {"total": 0, "pass_count": 0, "review_count": 0, "fail_count": 0}

    total = pass_c = review_c = fail_c = 0
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                total += 1
                v = rec.get("verdict", "")
                if v == "pass":
                    pass_c += 1
                elif v == "review":
                    review_c += 1
                elif v == "fail":
                    fail_c += 1
            except json.JSONDecodeError:
                continue

    return {
        "total": total,
        "pass_count": pass_c,
        "review_count": review_c,
        "fail_count": fail_c,
        "pass_rate": round(pass_c / total, 4) if total else 0.0,
    }
