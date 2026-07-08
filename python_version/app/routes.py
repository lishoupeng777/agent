"""FastAPI 路由 —— 对外标准接口"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Query

from .engine import evaluate as single_evaluate
from .models import (
    CalibrationReport,
    EvalRequest,
    EvalResponse,
    RecalcParams,
    StabilityReport,
)
from .calibration import calibrate
from .metrics import compute_anchor_accuracy, compute_flaw_metrics
from .stability import run_stability
from .debias import detect_length_bias, detect_position_bias
from .reporter import run_full_evaluation
from .storage import save_evaluation, load_history, find_by_token, history_stats, update_human_verdict
from .batch import batch_evaluate, cache_stats, clear_cache
from .model_registry import get_registry, register_default_models

router = APIRouter(prefix="/api/v1", tags=["quality-judge"])

# 启动时注册所有可用模型
register_default_models()


@router.get("/models")
def list_models() -> dict:
    """返回可用模型列表"""
    registry = get_registry()
    return {
        "models": registry.list_models(),
        "default": registry.default_model,
    }


@router.post("/evaluate")
def evaluate_endpoint(request: EvalRequest) -> dict[str, Any]:
    """
    核心评估接口：
    输入治理前后文本对，输出维度评分 + 瑕疵清单 + 判定理由。
    支持通过 model 字段指定模型（deepseek/mimo/gpt），None 则使用默认模型。
    启用 stabilize 时，额外返回 stability 字段（方差分析）。
    结果自动持久化到历史记录。
    """
    stability = None

    # 稳定性分析优先：多次采样取均值（无论是否指定模型）
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
        # 附加稳定性数据
        stability = {
            "mean_score": report.mean_score,
            "variance": report.variance,
            "std_dev": report.std_dev,
            "is_stable": report.is_stable,
            "samples": report.samples,
            "sample_count": request.sample_count,
        }
    elif request.model:
        registry = get_registry()
        try:
            result = registry.evaluate(request.model, request, temperature=0.0)
            resp = result.response
        except KeyError:
            resp = single_evaluate(request, temperature=0.0)
    else:
        resp = single_evaluate(request, temperature=0.0)

    # 持久化历史
    try:
        save_evaluation(resp)
    except Exception:
        pass

    # 返回评估结果 + 稳定性数据（如果有）
    result = resp.model_dump()
    if stability:
        result["stability"] = stability
    return result


@router.post("/recalculate")
def recalculate_endpoint(params: RecalcParams) -> dict[str, Any]:
    """热重算：基于已存储的 LLM 原始数据，用新参数毫秒级重算。

    不调用 LLM，纯数学运算。用于参数调优场景。
    """
    from .recalc import recalculate_batch, RecalcParams as _Params

    recalc_params = _Params(
        dimension_weights=params.dimension_weights,
        penalty_factors=params.penalty_factors,
        pass_threshold=params.pass_threshold,
        review_threshold=params.review_threshold,
        anchor_tolerance=params.anchor_tolerance,
    )
    report = recalculate_batch(recalc_params)

    return {
        "params": {
            "dimension_weights": report.params.dimension_weights,
            "penalty_factors": report.params.penalty_factors,
            "pass_threshold": report.params.pass_threshold,
            "review_threshold": report.params.review_threshold,
        },
        "results": [
            {
                "request_id": r.request_id,
                "overall_score": r.overall_score,
                "verdict": r.verdict,
                "penalty_applied": r.penalty_applied,
                "dimensions": r.dimensions,
            }
            for r in report.results
        ],
        "summary": {
            "total": len(report.results),
            "pass_count": sum(1 for r in report.results if r.verdict == "pass"),
            "review_count": sum(1 for r in report.results if r.verdict == "review"),
            "fail_count": sum(1 for r in report.results if r.verdict == "fail"),
            "mean_score": round(sum(r.overall_score for r in report.results) / max(len(report.results), 1), 4),
        },
        "calibration": {
            "pearson_r": report.pearson_r,
            "spearman_rho": report.spearman_rho,
            "mae": report.mae,
            "rmse": report.rmse,
            "consistency_rate": report.consistency_rate,
        } if report.pearson_r is not None else None,
    }


@router.get("/raw/{request_id}")
def get_raw_endpoint(request_id: str) -> dict[str, Any]:
    """获取某条评估的原始数据（供热重算调试用）"""
    from .raw_store import load_raw
    record = load_raw(request_id)
    if record is None:
        return {"found": False, "request_id": request_id}
    return {"found": True, **record}


@router.post("/stability", response_model=StabilityReport)
def stability_endpoint(request: EvalRequest) -> StabilityReport:
    """评分稳定性分析：对同一输入多次评估，返回方差/标准差等指标。"""
    return run_stability(request, sample_count=request.sample_count)


@router.post("/calibrate", response_model=CalibrationReport)
def calibrate_endpoint(requests: list[EvalRequest]) -> CalibrationReport:
    """一致性校准：传入带 human_label 的请求，返回 Pearson/Spearman/MAE/RMSE/一致率。"""
    return calibrate(requests)


@router.post("/calibrate/visual")
def calibrate_visual_endpoint(requests: list[EvalRequest]) -> dict[str, Any]:
    """一致性校准 + 可视化数据（散点图 + 混淆矩阵）"""
    report = calibrate(requests)

    # 散点图数据
    scatter_data = []
    for d in report.details:
        scatter_data.append({
            "request_id": d["request_id"],
            "human": d["human_score"],
            "llm": d["llm_score"],
            "diff": d["diff"],
            "outlier": d["diff"] > 0.2,
        })

    # 混淆矩阵（pass/review/fail 交叉）
    def to_verdict(score):
        if score >= 0.82:
            return "pass"
        elif score >= 0.5:
            return "review"
        return "fail"

    matrix = {"pass": {"pass": 0, "review": 0, "fail": 0},
              "review": {"pass": 0, "review": 0, "fail": 0},
              "fail": {"pass": 0, "review": 0, "fail": 0}}

    for d in report.details:
        h_verdict = to_verdict(d["human_score"])
        l_verdict = to_verdict(d["llm_score"])
        matrix[h_verdict][l_verdict] += 1

    return {
        "calibration": {
            "pearson_r": report.pearson_r,
            "spearman_rho": report.spearman_rho,
            "mae": report.mae,
            "rmse": report.rmse,
            "consistency_rate": report.consistency_rate,
            "sample_count": report.sample_count,
        },
        "scatter": scatter_data,
        "confusion_matrix": matrix,
        "details": report.details,
    }


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


@router.post("/batch/submit")
async def batch_submit_endpoint(
    requests: list[EvalRequest],
    max_concurrency: int = Query(3, ge=1, le=10, description="最大并发数"),
) -> dict[str, Any]:
    """异步批量评估：提交即返回 task_id，后台执行，前端轮询进度。"""
    if len(requests) > 100:
        return {"error": "单次批量评估最多支持100条请求", "count": len(requests)}

    from .task_manager import create_task, run_task

    # 将 EvalRequest 转为 dict
    req_dicts = [r.model_dump(mode="json") for r in requests]
    task = create_task(req_dicts)

    # 启动后台任务
    def eval_fn(req_dict):
        eval_req = EvalRequest(**req_dict)
        return single_evaluate(eval_req, temperature=0.0).model_dump()

    asyncio.create_task(run_task(task, eval_fn, max_concurrency=max_concurrency))

    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "total": task.total,
        "message": f"任务已提交，共 {task.total} 条样本",
    }


@router.get("/batch/status/{task_id}")
def batch_status_endpoint(task_id: str) -> dict[str, Any]:
    """查询批量评估任务状态"""
    from .task_manager import get_task

    task = get_task(task_id)
    if task is None:
        return {"error": f"任务 {task_id} 不存在"}

    return task.to_full_dict()


@router.post("/batch/resume/{task_id}")
async def batch_resume_endpoint(task_id: str) -> dict[str, Any]:
    """断点续评：恢复未完成的任务，跳过已评估的样本。"""
    from .task_manager import get_task, resume_task, run_task

    task = resume_task(task_id)
    if task is None:
        return {"error": f"任务 {task_id} 不存在"}

    remaining = task.total - task.progress
    if remaining <= 0:
        return {"message": "任务已完成，无需续评", "task_id": task_id, "status": task.status.value}

    def eval_fn(req_dict):
        eval_req = EvalRequest(**req_dict)
        return single_evaluate(eval_req, temperature=0.0).model_dump()

    asyncio.create_task(run_task(task, eval_fn, max_concurrency=3))

    return {
        "task_id": task.task_id,
        "status": "running",
        "remaining": remaining,
        "message": f"已恢复，剩余 {remaining} 条样本待评估",
    }


@router.get("/batch/tasks")
def list_tasks_endpoint() -> dict[str, Any]:
    """列出所有批量评估任务"""
    from .task_manager import list_tasks

    return {"tasks": list_tasks()}


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


@router.post("/review/{request_id}")
def review_endpoint(request_id: str, human_verdict: str = Query(..., description="人工判定: pass 或 fail")) -> dict[str, Any]:
    """人工审核覆写：对 verdict=review 的记录进行人工判定。"""
    if human_verdict not in ("pass", "fail"):
        return {"success": False, "message": "human_verdict 只能是 pass 或 fail"}
    ok = update_human_verdict(request_id, human_verdict)
    if ok:
        return {"success": True, "request_id": request_id, "human_verdict": human_verdict}
    return {"success": False, "message": f"未找到 request_id={request_id} 的记录"}


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
