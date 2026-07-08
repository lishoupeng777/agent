"""异步任务管理器 —— 批量评估的任务状态机

核心功能：
1. 提交即返回 task_id，后台异步执行
2. 任务状态机：pending → running → success/failed
3. 进度追踪（已完成数/总数）
4. 错误隔离（单条失败不影响整体）
5. 断点续评（跳过已完成的样本）
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable

TASK_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "tasks.jsonl",
)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Task:
    """任务对象"""

    def __init__(self, task_id: str, total: int, requests: list[dict[str, Any]]):
        self.task_id = task_id
        self.status = TaskStatus.PENDING
        self.total = total
        self.progress = 0
        self.results: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self._requests = requests  # 原始请求数据（用于断点续评）
        self._completed_indices: set[int] = set()  # 已完成的索引

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "total": self.total,
            "progress": self.progress,
            "results_count": len(self.results),
            "errors_count": len(self.errors),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_full_dict(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "results": self.results,
            "errors": self.errors,
        }

    def update_progress(self) -> None:
        self.updated_at = datetime.now().isoformat()
        self.progress = len(self.results) + len(self.errors)

    def mark_completed_index(self, idx: int) -> None:
        self._completed_indices.add(idx)

    def is_completed(self, idx: int) -> bool:
        return idx in self._completed_indices


# 全局任务注册表
_tasks: dict[str, Task] = {}


def create_task(requests: list[dict[str, Any]]) -> Task:
    """创建新任务"""
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    task = Task(task_id=task_id, total=len(requests), requests=requests)
    _tasks[task_id] = task
    _persist_task(task)
    return task


def get_task(task_id: str) -> Task | None:
    """获取任务"""
    return _tasks.get(task_id)


def list_tasks() -> list[dict[str, Any]]:
    """列出所有任务"""
    return [t.to_dict() for t in _tasks.values()]


def _persist_task(task: Task) -> None:
    """持久化任务状态"""
    os.makedirs(os.path.dirname(TASK_STORE_PATH), exist_ok=True)
    record = {
        **task.to_dict(),
        "results": task.results,
        "errors": task.errors,
        "completed_indices": list(task._completed_indices),
        "requests": task._requests,
    }
    # 追加写入（简单实现，生产环境应用 SQLite）
    with open(TASK_STORE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_tasks_from_disk() -> None:
    """从磁盘恢复任务状态（进程重启后调用）"""
    if not os.path.exists(TASK_STORE_PATH):
        return
    with open(TASK_STORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                task_id = record.get("task_id", "")
                if not task_id or task_id in _tasks:
                    continue
                task = Task(
                    task_id=task_id,
                    total=record.get("total", 0),
                    requests=record.get("requests", []),
                )
                task.status = TaskStatus(record.get("status", "pending"))
                task.progress = record.get("progress", 0)
                task.results = record.get("results", [])
                task.errors = record.get("errors", [])
                task.created_at = record.get("created_at", "")
                task.updated_at = record.get("updated_at", "")
                for idx in record.get("completed_indices", []):
                    task._completed_indices.add(idx)
                _tasks[task_id] = task
            except (json.JSONDecodeError, ValueError):
                continue


def resume_task(task_id: str) -> Task | None:
    """恢复未完成的任务（断点续评）"""
    task = _tasks.get(task_id)
    if task is None:
        return None
    if task.status in (TaskStatus.SUCCESS,):
        return task  # 已完成，无需恢复
    task.status = TaskStatus.PENDING
    task.updated_at = datetime.now().isoformat()
    return task


async def run_task(
    task: Task,
    evaluate_fn: Callable,
    max_concurrency: int = 3,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    """异步执行批量评估任务。

    Args:
        task: 任务对象
        evaluate_fn: 评估函数 (request_dict) -> result_dict
        max_concurrency: 最大并发数
        on_progress: 进度回调 (task_id, done, total)
    """
    task.status = TaskStatus.RUNNING
    task.updated_at = datetime.now().isoformat()

    semaphore = asyncio.Semaphore(max_concurrency)
    loop = asyncio.get_event_loop()

    async def evaluate_one(idx: int, req: dict[str, Any]) -> None:
        # 跳过已完成的样本（断点续评）
        if task.is_completed(idx):
            return

        async with semaphore:
            try:
                # 在线程池中运行同步评估函数
                result = await loop.run_in_executor(None, evaluate_fn, req)
                task.results.append({
                    "index": idx,
                    "request_id": req.get("request_id", f"req_{idx}"),
                    "status": "ok",
                    "result": result,
                })
                task.mark_completed_index(idx)
            except Exception as e:
                task.errors.append({
                    "index": idx,
                    "request_id": req.get("request_id", f"req_{idx}"),
                    "status": "error",
                    "error": str(e),
                })
                task.mark_completed_index(idx)

            task.update_progress()
            if on_progress:
                on_progress(task.task_id, task.progress, task.total)

    # 并发执行所有任务
    tasks = [evaluate_one(i, req) for i, req in enumerate(task._requests)]
    await asyncio.gather(*tasks)

    # 最终状态
    if len(task.errors) == 0:
        task.status = TaskStatus.SUCCESS
    elif len(task.results) == 0:
        task.status = TaskStatus.FAILED
    else:
        task.status = TaskStatus.SUCCESS  # 部分成功也算成功

    task.updated_at = datetime.now().isoformat()
    _persist_task(task)
