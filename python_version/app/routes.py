"""FastAPI 路由 —— 对外标准接口"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

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

router = APIRouter(prefix="/api/v1", tags=["quality-judge"])


@router.post("/evaluate", response_model=EvalResponse)
def evaluate_endpoint(request: EvalRequest) -> EvalResponse:
    """
    核心评估接口：
    输入治理前后文本对，输出维度评分 + 瑕疵清单 + 判定理由。

    如果 stabilize=True，自动走多次采样稳定化。
    """
    if request.stabilize:
        # 多次采样取均值
        report = run_stability(request, sample_count=request.sample_count)
        # 同时执行一次完整评估获取瑕疵清单等
        resp = single_evaluate(request, temperature=0.0)
        # 用稳定化均值替换 overall_score
        resp.overall_score = report.mean_score
        # 更新 verdict
        if resp.overall_score >= 0.8:
            resp.verdict = "pass"
        elif resp.overall_score >= 0.5:
            resp.verdict = "review"
        else:
            resp.verdict = "fail"
        return resp
    return single_evaluate(request, temperature=0.0)


@router.post("/stability", response_model=StabilityReport)
def stability_endpoint(request: EvalRequest) -> StabilityReport:
    """
    评分稳定性分析接口：
    对同一输入多次评估，返回方差/标准差等稳定性指标。
    """
    return run_stability(request, sample_count=request.sample_count)


@router.post("/calibrate", response_model=CalibrationReport)
def calibrate_endpoint(requests: list[EvalRequest]) -> CalibrationReport:
    """
    一致性校准接口：
    传入一组带人工标注（human_label）的评估请求，
    返回 Pearson/Spearman 相关系数、MAE、RMSE、一致率。
    """
    return calibrate(requests)


@router.post("/metrics/flaw-detection")
def flaw_metrics_endpoint(
    predicted: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
) -> dict[str, float]:
    """
    瑕疵检出指标接口：
    输入预测瑕疵列表和人工标注瑕疵列表，
    返回 Precision、Recall、F1。
    """
    return compute_flaw_metrics(predicted, ground_truth)


@router.post("/metrics/anchor-accuracy")
def anchor_accuracy_endpoint(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    char_tolerance: int = 10,
) -> dict[str, float]:
    """
    锚点定位准确率接口：
    输入预测瑕疵和人工标注瑕疵（均含 location 信息），
    返回定位准确率。
    """
    return compute_anchor_accuracy(predicted, ground_truth, char_tolerance)


@router.post("/debias/detect")
def debias_detect_endpoint(request: EvalRequest) -> dict:
    """
    偏置检测接口：
    检测长度偏置和位置偏置风险。
    """
    try:
        length_bias = detect_length_bias(request.before_text, request.after_text)
        # 先快速评估获取瑕疵
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
def full_report_endpoint(requests: list[EvalRequest]) -> dict:
    """
    综合评估报告接口：
    一次性完成评估 + 一致性校准 + 稳定性 + 瑕疵指标 + 锚点准确率 + 偏置分析 + 可复现性。
    
    返回符合课题12验收标准的完整综合报告。
    """
    return run_full_evaluation(requests)


@router.get("/health")
def health_check() -> dict[str, str]:
    """健康检查"""
    return {"status": "ok", "service": "llm-quality-judge"}
