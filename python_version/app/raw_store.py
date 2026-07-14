"""原始数据存储 —— SQLite 存储，保存 LLM 评估的原始输出，供热重算使用

核心思想：LLM 只跑一次，后续改参数（权重/惩罚/阈值）时
从原始数据重算，不需要重新调 LLM。
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from typing import Any

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "raw_evaluations.db",
)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS raw_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    raw_dimensions TEXT,
    raw_flaws TEXT,
    raw_llm_output TEXT,
    before_text TEXT,
    after_text TEXT,
    evaluation_profile TEXT,
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_raw_request_id ON raw_evaluations(request_id);
"""


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(_CREATE_SQL)


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
    """保存一条评估的原始数据。"""
    conn = _get_conn()
    try:
        _ensure_table(conn)
        conn.execute(
            """INSERT INTO raw_evaluations
               (request_id, timestamp, raw_dimensions, raw_flaws, raw_llm_output,
                before_text, after_text, evaluation_profile, model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request_id,
                datetime.now().isoformat(),
                json.dumps(raw_dimensions, ensure_ascii=False),
                json.dumps(raw_flaws, ensure_ascii=False),
                raw_llm_output,
                before_text[:500],
                after_text[:500],
                evaluation_profile,
                model_version,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_raw(request_id: str) -> dict[str, Any] | None:
    """按 request_id 加载原始数据。"""
    conn = _get_conn()
    try:
        _ensure_table(conn)
        row = conn.execute(
            "SELECT * FROM raw_evaluations WHERE request_id = ? ORDER BY id DESC LIMIT 1",
            (request_id,),
        ).fetchone()
        if not row:
            return None
        return _row_to_dict(row)
    finally:
        conn.close()


def load_all_raw() -> list[dict[str, Any]]:
    """加载所有原始数据（用于批量重算）。"""
    conn = _get_conn()
    try:
        _ensure_table(conn)
        rows = conn.execute("SELECT * FROM raw_evaluations ORDER BY id").fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def clear_raw() -> int:
    """清空原始数据存储，返回清除的条数。"""
    conn = _get_conn()
    try:
        _ensure_table(conn)
        row = conn.execute("SELECT COUNT(*) as cnt FROM raw_evaluations").fetchone()
        count = row["cnt"] if row else 0
        conn.execute("DELETE FROM raw_evaluations")
        conn.commit()
        return count
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """将 SQLite Row 转为字典，解析 JSON 字段。"""
    d = dict(row)
    for key in ("raw_dimensions", "raw_flaws"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except json.JSONDecodeError:
                d[key] = []
    return d
