"""异步批量评估 —— 并发控制 + 内存缓存 + 指数退避重试"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from typing import Any

from .engine import evaluate
from .models import EvalRequest, EvalResponse
from .storage import save_evaluation

# 内存缓存：{ cache_key: EvalResponse }
_cache: dict[str, EvalResponse] = {}
_MAX_CACHE = 256


def _cache_key(request: EvalRequest) -> str:
    payload = json.dumps(
        {"before": request.before_text, "after": request.after_text},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _get_cached(request: EvalRequest) -> EvalResponse | None:
    return _cache.get(_cache_key(request))


def _set_cached(request: EvalRequest, resp: EvalResponse) -> None:
    key = _cache_key(request)
    if len(_cache) >= _MAX_CACHE:
        # 简单 LRU：删除最早插入的一条
        oldest = next(iter(_cache))
        del _cache[oldest]
    _cache[key] = resp


def _evaluate_with_retry(request: EvalRequest, max_retries: int = 3) -> EvalResponse:
    """同步评估，带指数退避重试"""
    last_exc: Exception = RuntimeError("未知错误")
    for attempt in range(max_retries):
        try:
            return evaluate(request, temperature=0.0)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s / 2s / 4s
    raise last_exc


async def _evaluate_one(
    request: EvalRequest,
    semaphore: asyncio.Semaphore,
    use_cache: bool,
    persist: bool,
) -> dict[str, Any]:
    """单条异步评估（在线程池中运行同步 LLM 调用）"""
    # 缓存命中
    if use_cache:
        cached = _get_cached(request)
        if cached is not None:
            return {
                "request_id": request.request_id,
                "status": "ok",
                "from_cache": True,
                "result": cached.model_dump(),
            }

    async with semaphore:
        loop = asyncio.get_event_loop()
        try:
            resp: EvalResponse = await loop.run_in_executor(
                None, _evaluate_with_retry, request
            )
            if use_cache:
                _set_cached(request, resp)
            if persist:
                try:
                    save_evaluation(resp)
                except Exception:
                    pass  # 持久化失败不影响主流程
            return {
                "request_id": request.request_id,
                "status": "ok",
                "from_cache": False,
                "result": resp.model_dump(),
            }
        except Exception as exc:
            return {
                "request_id": request.request_id,
                "status": "error",
                "from_cache": False,
                "error": str(exc),
            }


async def batch_evaluate_async(
    requests: list[EvalRequest],
    max_concurrency: int = 5,
    use_cache: bool = True,
    persist: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """
    异步并发批量评估。

    Args:
        requests: 评估请求列表（最多50条）
        max_concurrency: 最大并发数（默认5，避免 API 限流）
        use_cache: 是否启用内存缓存（默认开启）
        persist: 是否持久化到 JSONL（默认开启）
        on_progress: 进度回调 on_progress(done, total)

    Returns:
        每条请求的评估结果列表，顺序与输入一致
    """
    if len(requests) > 50:
        raise ValueError("单次批量评估最多支持50条请求")

    semaphore = asyncio.Semaphore(max_concurrency)
    total = len(requests)
    results: list[dict[str, Any]] = [{}] * total
    done_count = 0

    async def _wrapped(idx: int, req: EvalRequest) -> None:
        nonlocal done_count
        result = await _evaluate_one(req, semaphore, use_cache, persist)
        results[idx] = result
        done_count += 1
        if on_progress:
            on_progress(done_count, total)

    tasks = [_wrapped(i, req) for i, req in enumerate(requests)]
    await asyncio.gather(*tasks)
    return results


def batch_evaluate(
    requests: list[EvalRequest],
    max_concurrency: int = 5,
    use_cache: bool = True,
    persist: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """
    同步入口（内部运行 asyncio 事件循环）。
    适合在非异步上下文（如 FastAPI 同步路由、脚本）中调用。
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已在异步上下文中（如 Jupyter / FastAPI async 路由），使用 nest_asyncio
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return asyncio.get_event_loop().run_until_complete(
        batch_evaluate_async(requests, max_concurrency, use_cache, persist, on_progress)
    )


def clear_cache() -> int:
    """清空内存缓存，返回清除的条目数"""
    count = len(_cache)
    _cache.clear()
    return count


def cache_stats() -> dict[str, int]:
    return {"size": len(_cache), "max_size": _MAX_CACHE}
