"""存储层 —— SQLite 实现（evaluations + eval_profiles 双表）

表 1: evaluations —— 评估历史记录
表 2: eval_profiles —— 评测模式（松/中/紧三档）
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 数据库路径
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_history.db"
DB_PATH = Path(os.getenv("EVAL_HISTORY_DB", str(_DEFAULT_DB_PATH)))

# ============================================================
# 建表 SQL
# ============================================================

_CREATE_EVALUATIONS = """
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT NOT NULL,
    overall_score REAL,
    verdict TEXT,
    human_verdict TEXT,
    model_version TEXT,
    prompt_version TEXT,
    rule_version TEXT,
    reproducibility_token TEXT,
    flaw_count INTEGER,
    dimension_scores TEXT,
    raw_report TEXT
);
CREATE INDEX IF NOT EXISTS idx_token ON evaluations(reproducibility_token);
CREATE INDEX IF NOT EXISTS idx_request_id ON evaluations(request_id);
CREATE INDEX IF NOT EXISTS idx_ts ON evaluations(ts);
"""

_CREATE_PROFILES = """
CREATE TABLE IF NOT EXISTS eval_profiles (
    profile_id TEXT PRIMARY KEY,
    profile_name TEXT NOT NULL,
    description TEXT,
    prompt_supplement TEXT NOT NULL,
    few_shot_supplement TEXT DEFAULT '[]',
    created_at TEXT,
    is_builtin INTEGER DEFAULT 0
);
"""

# ============================================================
# 种子数据：三档评估模式
# ============================================================

_SEED_PROFILES = [
    {
        "profile_id": "relaxed",
        "profile_name": "宽松模式",
        "description": "适用于日常聊天、娱乐新闻、创意文案、口语交流。核心大意不变即合格。",
        "prompt_supplement": (
            "**【本条文本采用 宽松模式 评估】**\n"
            "- 正常的同义词改写和精简不算瑕疵，不扣分。\n"
            "- 只要核心大意没变，任何删词、缩写、口语化转换（如'老旧空调轰隆隆'精简为'空调噪音大'）均予以放行。\n"
            "- 重点关注：是否完全篡改了原文大意？是否删掉了核心数字/日期？"
            "如果仅是表达方式变了但意思相同，各个维度得分不低于 0.8。"
        ),
        "few_shot_supplement": "[]",
        "is_builtin": 1,
    },
    {
        "profile_id": "medium",
        "profile_name": "普通模式",
        "description": "适用于说明文、日常业务邮件、常规工作汇报。允许删除废话，但核心事实不能动。",
        "prompt_supplement": (
            "**【本条文本采用 普通模式 评估】**\n"
            "- 删除冗余语气词和重复信息不扣分。\n"
            "- 但不能改动、歪曲或遗漏核心的时间、人物、数字和金额。\n"
            "- 约束词（如：必须、不得、至少）如果被无端删除或泛化，"
            "semantic_fidelity 必须降分（通常在 0.5~0.7 区间）。\n"
            "- 格式调整（如标点规范化、段落合并）不影响评分。"
        ),
        "few_shot_supplement": "[]",
        "is_builtin": 1,
    },
    {
        "profile_id": "strict",
        "profile_name": "严格模式",
        "description": "适用于法律合同、政务通告、财务报表、医疗处方。任何细节错误都视为严重瑕疵。",
        "prompt_supplement": (
            "**【本条文本采用 严格保真模式 评估】**\n"
            "- 错一个数字、篡改一个计量单位或百分比，立即判定为严重事实错误（critical mis_edit），"
            "该维度得分降至 0.4 以下。\n"
            "- 机构名、主体名称必须使用法定全称，任何缩写均属于事实错误。\n"
            "- 任何关键约束词（如：必须、不得、至少、不超过）如果被删除或泛化，"
            "一律判定为 critical 瑕疵，且相关维度得分不得高于 0.5。\n"
            "- 法律合同中的定义性条款（如：甲方、乙方、书面同意、生效之日起）必须逐字保留，"
            "任何限定词的删除一律标为 critical 瑕疵。"
        ),
        "few_shot_supplement": "[]",
        "is_builtin": 1,
    },
]

# 向后兼容映射
_COMPAT_MAP = {
    "general": "medium",
    "government_notice_strict": "strict",
    "legal_strict": "strict",
}


# ============================================================
# 连接管理
# ============================================================

def _get_conn() -> sqlite3.Connection:
    """获取数据库连接（每次新建，线程安全）"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """确保表和索引存在，并初始化种子数据"""
    conn.executescript(_CREATE_EVALUATIONS)
    conn.executescript(_CREATE_PROFILES)
    # 检查是否需要初始化种子数据
    row = conn.execute("SELECT COUNT(*) as cnt FROM eval_profiles").fetchone()
    if row["cnt"] == 0:
        for p in _SEED_PROFILES:
            conn.execute(
                """INSERT INTO eval_profiles
                   (profile_id, profile_name, description, prompt_supplement,
                    few_shot_supplement, created_at, is_builtin)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    p["profile_id"],
                    p["profile_name"],
                    p["description"],
                    p["prompt_supplement"],
                    p["few_shot_supplement"],
                    datetime.now().isoformat(),
                    p["is_builtin"],
                ),
            )
        conn.commit()


def normalize_profile(profile_id: str) -> str:
    """向后兼容：旧模式名映射到新模式名"""
    return _COMPAT_MAP.get(profile_id, profile_id)


# ============================================================
# Profile 查询接口
# ============================================================

def get_profile(profile_id: str) -> dict[str, Any] | None:
    """查询单个评测模式"""
    profile_id = normalize_profile(profile_id)
    conn = _get_conn()
    try:
        _ensure_tables(conn)
        row = conn.execute(
            "SELECT * FROM eval_profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_profiles() -> list[dict[str, Any]]:
    """列出所有评测模式"""
    conn = _get_conn()
    try:
        _ensure_tables(conn)
        rows = conn.execute("SELECT * FROM eval_profiles ORDER BY is_builtin DESC, profile_id").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_profile(profile_id: str, **kwargs) -> bool:
    """更新评测模式（仅非 builtin 可更新）"""
    conn = _get_conn()
    try:
        _ensure_tables(conn)
        allowed = {"profile_name", "description", "prompt_supplement", "few_shot_supplement"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [profile_id]
        cursor = conn.execute(
            f"UPDATE eval_profiles SET {set_clause} WHERE profile_id = ? AND is_builtin = 0",
            values,
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ============================================================
# 评估记录接口（对外 API 不变）
# ============================================================

def save_evaluation(response: Any) -> None:
    """将 EvalResponse 写入数据库"""
    conn = _get_conn()
    try:
        _ensure_tables(conn)
        conn.execute(
            """INSERT INTO evaluations
               (ts, request_id, overall_score, verdict, human_verdict,
                model_version, prompt_version, rule_version,
                reproducibility_token, flaw_count, dimension_scores, raw_report)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                response.request_id,
                response.overall_score,
                response.verdict,
                None,
                getattr(response, "model_version", ""),
                getattr(response, "prompt_version", ""),
                getattr(response, "rule_version", ""),
                response.reproducibility_token,
                len(response.flaws),
                json.dumps(
                    {d.dimension: d.score for d in response.dimensions},
                    ensure_ascii=False,
                ),
                json.dumps(
                    response.model_dump() if hasattr(response, "model_dump") else {},
                    ensure_ascii=False,
                ),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_history(
    request_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """读取历史记录，支持按 request_id 过滤和分页"""
    conn = _get_conn()
    try:
        _ensure_tables(conn)
        if request_id:
            rows = conn.execute(
                "SELECT * FROM evaluations WHERE request_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (request_id, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM evaluations ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def find_by_token(token: str) -> dict[str, Any] | None:
    """根据可复现令牌查找历史评估记录"""
    conn = _get_conn()
    try:
        _ensure_tables(conn)
        row = conn.execute(
            "SELECT * FROM evaluations WHERE reproducibility_token = ? LIMIT 1",
            (token,),
        ).fetchone()
        if not row:
            return None
        d = _row_to_dict(row)
        # 反序列化 raw_report 提供完整详情
        if d.get("raw_report"):
            try:
                d["raw_report"] = json.loads(d["raw_report"])
            except json.JSONDecodeError:
                pass
        return d
    finally:
        conn.close()


def update_human_verdict(request_id: str, human_verdict: str) -> bool:
    """人工覆写 verdict"""
    conn = _get_conn()
    try:
        _ensure_tables(conn)
        cursor = conn.execute(
            "UPDATE evaluations SET human_verdict = ? WHERE request_id = ?",
            (human_verdict, request_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def history_stats() -> dict[str, Any]:
    """返回历史记录的汇总统计"""
    conn = _get_conn()
    try:
        _ensure_tables(conn)
        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN verdict = 'pass' THEN 1 ELSE 0 END) as pass_count,
                SUM(CASE WHEN verdict = 'review' THEN 1 ELSE 0 END) as review_count,
                SUM(CASE WHEN verdict = 'fail' THEN 1 ELSE 0 END) as fail_count
            FROM evaluations"""
        ).fetchone()
        total = row["total"] or 0
        pass_c = row["pass_count"] or 0
        review_c = row["review_count"] or 0
        fail_c = row["fail_count"] or 0
        return {
            "total": total,
            "pass_count": pass_c,
            "review_count": review_c,
            "fail_count": fail_c,
            "pass_rate": round(pass_c / total, 4) if total else 0.0,
        }
    finally:
        conn.close()


# ============================================================
# 工具函数
# ============================================================

def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """将 SQLite Row 转为字典，解析 JSON 字段"""
    d = dict(row)
    if d.get("dimension_scores"):
        try:
            d["dimension_scores"] = json.loads(d["dimension_scores"])
        except json.JSONDecodeError:
            pass
    # raw_report 保持字符串，由调用方按需反序列化
    return d
