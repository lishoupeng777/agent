"""评分稳定性模块 —— 多次采样 + 方差分析"""
from __future__ import annotations

from .engine import evaluate
from .models import EvalRequest, StabilityReport


def run_stability(
    request: EvalRequest, sample_count: int = 3
) -> StabilityReport:
    """
    对同一输入多次评估，分析评分稳定性。

    注意：执行期间会临时关闭 LLM 响应缓存，确保每次评估都是
    真实调用 API 而非命中缓存（否则方差恒为 0，失去测试意义）。
    测试完成后自动恢复原有缓存配置。

    Args:
        request: 评估请求（sample_count 控制采样次数）
        sample_count: 采样次数（默认 3，最多 5）

    Returns:
        StabilityReport: 包含各次采样得分、均值、方差、标准差
    """
    from .chain import disable_cache, enable_cache

    # 临时关闭 LLM 响应缓存，确保每次真实调用 API
    disable_cache()
    try:
        samples: list[dict[str, float]] = []
        scores: list[float] = []

        for i in range(sample_count):
            resp = evaluate(request, temperature=0.0)  # 固定低温度保证可复现
            sample = {d.dimension: d.score for d in resp.dimensions}
            sample["overall"] = resp.overall_score
            samples.append(sample)
            scores.append(resp.overall_score)
    finally:
        # 无论成功失败，恢复内存缓存
        enable_cache("memory")

    n = len(scores)
    mean_score = sum(scores) / n if n > 0 else 0.0
    variance = sum((s - mean_score) ** 2 for s in scores) / n if n > 0 else 0.0
    std_dev = variance**0.5

    # 方差阈值：小于 0.005 视为稳定（即标准差 < 0.07）
    is_stable = variance < 0.005

    return StabilityReport(
        request_id=request.request_id,
        samples=samples,
        mean_score=round(mean_score, 4),
        variance=round(variance, 6),
        std_dev=round(std_dev, 4),
        is_stable=is_stable,
    )