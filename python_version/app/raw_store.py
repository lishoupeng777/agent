"""原始数据存储 —— 保存 LLM 评估的原始输出，供热重算使用

核心思想：LLM 只跑一次，后续改参数（权重/惩罚/阈值）时
从原始数据重算，不需要重新调 LLM。

存储格式：JSONL 文件，每行一条评估的原始数据。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

RAW_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "raw_evaluations.jsonl",
)


def save_raw(
    request_id: str,
    raw_dimensions: list[dict[str, Any]],
    raw_flaws: list[dict[str, Any]],
    raw_llm_output: str = "",
    before_text: str = "",
    after_text: str = "",
    evaluation_profile: str = "general",
    model_version: str = "",
) -> None:
    """保存一条评估的原始数据。

    Args:
        request_id: 请求 ID
        raw_dimensions: LLM 返回的原始维度评分（未经过惩罚/修正）
        raw_flaws: LLM 返回的原始瑕玼列表（未经合并/投票）
        raw_llm_output: LLM 原始输出文本
        before_text: 治理前原文（用于回溯）
        after_text: 治理后文本（用于回溯）
        evaluation_profile: 评估模式
        model_version: 模型版本
    """
    record = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "raw_dimensions": raw_dimensions,
        "raw_flaws": raw_flaws,
        "raw_llm_output": raw_llm_output,
        "before_text": before_text[:500],  # 截断，避免文件过大
        "after_text": after_text[:500],
        "evaluation_profile": evaluation_profile,
        "model_version": model_version,
    }

    os.makedirs(os.path.dirname(RAW_STORE_PATH), exist_ok=True)
    with open(RAW_STORE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_raw(request_id: str) -> dict[str, Any] | None:
    """按 request_id 加载原始数据。"""
    if not os.path.exists(RAW_STORE_PATH):
        return None
    with open(RAW_STORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("request_id") == request_id:
                    return record
            except json.JSONDecodeError:
                continue
    return None


def load_all_raw() -> list[dict[str, Any]]:
    """加载所有原始数据（用于批量重算）。"""
    if not os.path.exists(RAW_STORE_PATH):
        return []
    records = []
    with open(RAW_STORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def clear_raw() -> int:
    """清空原始数据存储，返回清除的条数。"""
    if not os.path.exists(RAW_STORE_PATH):
        return 0
    count = sum(1 for line in open(RAW_STORE_PATH, encoding="utf-8") if line.strip())
    os.remove(RAW_STORE_PATH)
    return count
