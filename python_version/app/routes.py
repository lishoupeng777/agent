"""FastAPI 路由 —— 对外标准接口"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from .engine import evaluate as single_evaluate
from .models import (
    CalibrationReport,
    EvalRequest,
    EvalResponse,
    StabilityReport,
)
from .calibration import calibrate
from .metrics import compute_anchor_accuracy, compute_flaw_metrics
from .stability import run_stability
from .debias import detect_length_bias, detect_position_bias
from .reporter import run_full_evaluation
from .storage import save_evaluation, load_history, find_by_token, history_stats
from .batch import batch_evaluate, cache_stats, clear_cache

router = APIRouter(prefix="/api/v1", tags=["quality-judge"])


@router.post("/evaluate", response_model=EvalResponse)
def evaluate_endpoint(request: EvalRequest) -> EvalResponse:
    """
    核心评估接口：
    输入治理前后文本对，输出维度评分 + 瑕疵清单 + 判定理由。
    结果自动持久化到历史记录。
    """
    if request.stabilize:
        report = run_stability(request, sample_count=request.sample_count)
        resp = single_evaluate(request, temperature=0.0)
        resp.overall_score = report.mean_score
        if resp.overall_score >= 0.8:
            resp.verdict = "pass"
        elif resp.overall_score >= 0.5:
            resp.verdict = "review"
        else:
            resp.verdict = "fail"
    else:
        resp = single_evaluate(request, temperature=0.0)

    # 持久化历史
    try:
        save_evaluation(resp)
    except Exception:
        pass

    return resp


@router.post("/stability", response_model=StabilityReport)
def stability_endpoint(request: EvalRequest) -> StabilityReport:
    """评分稳定性分析：对同一输入多次评估，返回方差/标准差等指标。"""
    return run_stability(request, sample_count=request.sample_count)


@router.post("/calibrate", response_model=CalibrationReport)
def calibrate_endpoint(requests: list[EvalRequest]) -> CalibrationReport:
    """一致性校准：传入带 human_label 的请求，返回 Pearson/Spearman/MAE/RMSE/一致率。"""
    return calibrate(requests)


@router.post("/metrics/flaw-detection")
def flaw_metrics_endpoint(
    predicted: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
) -> dict[str, float]:
    """瑕疵检出指标：返回 Precision、Recall、F1（样本级 TP/FP/FN）。"""
    return compute_flaw_metrics(predicted, ground_truth)


@router.post("/metrics/anchor-accuracy")
def anchor_accuracy_endpoint(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    char_tolerance: int = 10,
) -> dict[str, float]:
    """锚点定位准确率：snippet 重叠或 start_char 偏差在容差内即为正确。"""
    return compute_anchor_accuracy(predicted, ground_truth, char_tolerance)


@router.post("/debias/detect")
def debias_detect_endpoint(request: EvalRequest) -> dict:
    """偏置检测：检测长度偏置和位置偏置风险。"""
    try:
        length_bias = detect_length_bias(request.before_text, request.after_text)
        resp = single_evaluate(request, temperature=0.0)
        position_bias = detect_position_bias(resp.flaws)
        return {
            "request_id": request.request_id,
            "length_bias": length_bias,
            "position_bias": position_bias,
        }
    except Exception as e:
        return {"request_id": request.request_id, "error": str(e)}


@router.post("/report/full")
def full_report_endpoint(
    requests: list[EvalRequest],
    run_stability_analysis: bool = Query(False, description="是否启用稳定性分析（默认关闭，启用后 API 调用次数增加 3 倍）"),
) -> dict:
    """
    综合评估报告：一次性完成评估 + 校准 + 瑕疵指标 + 锚点准确率 + 偏置分析。
    run_stability_analysis=true 时额外执行多次采样稳定性分析。
    """
    return run_full_evaluation(requests, run_stability=run_stability_analysis)


# ---------- 批量评估 ----------

@router.post("/batch/evaluate")
def batch_evaluate_endpoint(
    requests: list[EvalRequest],
    max_concurrency: int = Query(5, ge=1, le=10, description="最大并发数"),
    use_cache: bool = Query(True, description="是否启用内存缓存"),
) -> dict[str, Any]:
    """
    批量评估接口：并发评估最多50条请求，自动缓存与持久化。
    返回每条请求的评估结果及汇总统计。
    """
    if len(requests) > 50:
        return {"error": "单次批量评估最多支持50条请求", "count": len(requests)}

    results = batch_evaluate(
        requests,
        max_concurrency=max_concurrency,
        use_cache=use_cache,
        persist=True,
    )

    ok = [r for r in results if r.get("status") == "ok"]
    errors = [r for r in results if r.get("status") == "error"]
    cached = [r for r in ok if r.get("from_cache")]

    return {
        "total": len(results),
        "success": len(ok),
        "errors": len(errors),
        "from_cache": len(cached),
        "results": results,
    }


# ---------- 历史记录 ----------

@router.get("/history")
def history_endpoint(
    request_id: str | None = Query(None, description="按 request_id 过滤"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """查询历史评估记录（按时间倒序）。"""
    records = load_history(request_id=request_id, limit=limit, offset=offset)
    stats = history_stats()
    return {
        "total_in_db": stats["total"],
        "returned": len(records),
        "offset": offset,
        "records": records,
    }


@router.get("/history/stats")
def history_stats_endpoint() -> dict[str, Any]:
    """历史记录汇总统计（总数、pass/review/fail 分布）。"""
    return history_stats()


@router.get("/reproduce/{token}")
def reproduce_endpoint(token: str) -> dict[str, Any]:
    """
    根据可复现令牌查找历史评估记录。
    令牌由 SHA256(before + after + temperature + model + prompt_version) 生成。
    """
    record = find_by_token(token)
    if record is None:
        return {"found": False, "token": token, "message": "未找到对应历史记录，可能尚未入库或令牌已过期"}
    return {"found": True, "token": token, "record": record}


# ---------- 缓存管理 ----------

@router.get("/cache/stats")
def cache_stats_endpoint() -> dict[str, int]:
    """查询内存缓存状态。"""
    return cache_stats()


@router.delete("/cache/clear")
def cache_clear_endpoint() -> dict[str, Any]:
    """清空内存缓存。"""
    cleared = clear_cache()
    return {"cleared": cleared, "message": f"已清除 {cleared} 条缓存记录"}


@router.get("/health")
def health_check() -> dict[str, str]:
    """健康检查"""
    return {"status": "ok", "service": "llm-quality-judge"}
