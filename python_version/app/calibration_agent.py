"""辅助校准 Agent（Calibration Agent）

独立 agent，负责：
1. 评分校准：用 Gold Dataset 训练校准器
2. 一致性检验：验证标注员间/人机一致性
3. 质量保证：检测评分异常
4. 校准报告：生成完整的校准结果

不参与正常评估流程，只在离线阶段运行。
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from scipy import stats

from .models import EvalRequest, EvalResponse
from .calibrator import ScoreCalibrator, MultiModelCalibrator, auto_calibrate
from .metrics import compute_kappa, compute_kendalls_w, compute_correlation_metrics


class CalibrationAgent:
    """辅助校准 Agent。

    工作流程：
    1. 加载 Gold Dataset
    2. 用指定模型对 Gold Dataset 评分
    3. 计算人机一致性指标
    4. 训练校准器
    5. 生成校准报告
    """

    def __init__(self) -> None:
        self.gold_data: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.calibrator: ScoreCalibrator | None = None
        self.consistency_report: dict[str, Any] = {}

    def load_gold_dataset(self, filepath: str) -> int:
        """加载 Gold Dataset。

        Args:
            filepath: Gold Dataset JSON 文件路径

        Returns:
            int: 加载的样本数
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self.gold_data = json.load(f)
        return len(self.gold_data)

    def run_calibration(
        self,
        adapter: Any,
        model_name: str,
        max_samples: int = 0,
    ) -> dict[str, Any]:
        """运行完整校准流程。

        Args:
            adapter: EvaluationProtocol 实例
            model_name: 模型名称
            max_samples: 最大使用样本数（0 = 全部）

        Returns:
            dict: 校准报告
        """
        if not self.gold_data:
            raise ValueError("No gold dataset loaded. Call load_gold_dataset() first.")

        data = self.gold_data[:max_samples] if max_samples > 0 else self.gold_data

        # Step 1: 用模型评分
        print(f"[CalibrationAgent] Running {len(data)} samples with {model_name}...")
        self.results = []
        for i, item in enumerate(data):
            req = EvalRequest(
                request_id=item["id"],
                before_text=item["before_text"][:1500],
                after_text=item["after_text"][:800],
            )
            try:
                resp = adapter.evaluate(req, temperature=0.0)
                self.results.append({
                    "id": item["id"],
                    "gold_score": item["overall"]["weighted_score"],
                    "gold_verdict": item["overall"]["verdict"],
                    "model_score": resp.overall_score,
                    "model_verdict": resp.verdict,
                    "category": item.get("category", ""),
                    "latency": resp.latency_seconds,
                    "tokens": resp.total_tokens,
                })
            except Exception as e:
                self.results.append({
                    "id": item["id"],
                    "gold_score": item["overall"]["weighted_score"],
                    "gold_verdict": item["overall"]["verdict"],
                    "model_score": None,
                    "model_verdict": "error",
                    "category": item.get("category", ""),
                    "error": str(e),
                })

        valid = [r for r in self.results if r["model_score"] is not None]
        print(f"[CalibrationAgent] {len(valid)}/{len(data)} successful")

        if len(valid) < 3:
            return {"error": "Not enough valid results for calibration", "valid": len(valid)}

        # Step 2: 计算一致性指标
        gold_scores = [r["gold_score"] for r in valid]
        model_scores = [r["model_score"] for r in valid]

        self.consistency_report = compute_correlation_metrics(gold_scores, model_scores)
        self.consistency_report["kappa_pass_fail"] = compute_kappa(gold_scores, model_scores, task="pass/fail")
        self.consistency_report["kappa_ternary"] = compute_kappa(gold_scores, model_scores, task="pass/review/fail")
        self.consistency_report["kendalls_w"] = compute_kendalls_w(gold_scores, model_scores)

        # Step 3: 训练校准器
        self.calibrator = ScoreCalibrator(model_name)
        self.calibrator.fit(model_scores, gold_scores)

        # Step 4: 按类别分析
        category_analysis = {}
        for r in valid:
            cat = r["category"]
            if cat not in category_analysis:
                category_analysis[cat] = {"count": 0, "diffs": []}
            category_analysis[cat]["count"] += 1
            category_analysis[cat]["diffs"].append(abs(r["model_score"] - r["gold_score"]))

        for cat, data_cat in category_analysis.items():
            diffs = data_cat["diffs"]
            data_cat["avg_diff"] = round(sum(diffs) / len(diffs), 4)
            data_cat["max_diff"] = round(max(diffs), 4)

        # Step 5: 检测异常（diff > 0.3 的样本）
        anomalies = [r for r in valid if abs(r["model_score"] - r["gold_score"]) > 0.3]

        # 组装报告
        report = {
            "model_name": model_name,
            "total_samples": len(data),
            "valid_samples": len(valid),
            "consistency": self.consistency_report,
            "calibrator": self.calibrator.to_dict(),
            "category_analysis": category_analysis,
            "anomalies": {
                "count": len(anomalies),
                "threshold": 0.3,
                "samples": [
                    {
                        "id": a["id"],
                        "gold": a["gold_score"],
                        "model": a["model_score"],
                        "diff": round(abs(a["model_score"] - a["gold_score"]), 4),
                        "category": a["category"],
                    }
                    for a in anomalies[:10]
                ],
            },
            "pass_criteria": {
                "spearman_ge_0_8": self.consistency_report.get("spearman_rho", 0) >= 0.8,
                "kappa_ge_0_6": self.consistency_report.get("kappa", 0) >= 0.6,
                "calibration_passed": (
                    self.consistency_report.get("spearman_rho", 0) >= 0.8
                    and self.consistency_report.get("kappa", 0) >= 0.6
                ),
            },
        }

        return report

    def generate_report_text(self, report: dict[str, Any]) -> str:
        """生成可读的校准报告文本。

        Args:
            report: run_calibration() 返回的报告

        Returns:
            str: 格式化的报告文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"  Calibration Report - {report.get('model_name', 'unknown')}")
        lines.append("=" * 60)

        lines.append(f"\n[Basic Info]")
        lines.append(f"  Total samples: {report.get('total_samples', 0)}")
        lines.append(f"  Valid samples: {report.get('valid_samples', 0)}")

        cm = report.get("consistency", {})
        lines.append(f"\n[Consistency Metrics]")
        lines.append(f"  Pearson r:    {cm.get('pearson_r', 0):.4f}")
        lines.append(f"  Spearman rho: {cm.get('spearman_rho', 0):.4f}")
        lines.append(f"  Kappa:        {cm.get('kappa', 0):.4f}")
        lines.append(f"  Kendall's W:  {cm.get('kendalls_w', 0):.4f}")
        lines.append(f"  MAE:          {cm.get('mae', 0):.4f}")
        lines.append(f"  RMSE:         {cm.get('rmse', 0):.4f}")

        cal = report.get("calibrator", {})
        lines.append(f"\n[Calibrator]")
        lines.append(f"  Slope:     {cal.get('slope', 1.0):.4f}")
        lines.append(f"  Intercept: {cal.get('intercept', 0.0):.4f}")
        lines.append(f"  R-squared: {cal.get('r_squared', 0.0):.4f}")

        lines.append(f"\n[Category Analysis]")
        for cat, data_cat in report.get("category_analysis", {}).items():
            lines.append(f"  {cat}: n={data_cat['count']} avg_diff={data_cat['avg_diff']:.4f} max_diff={data_cat['max_diff']:.4f}")

        anomalies = report.get("anomalies", {})
        lines.append(f"\n[Anomalies (diff > {anomalies.get('threshold', 0.3)})]")
        lines.append(f"  Count: {anomalies.get('count', 0)}")
        for a in anomalies.get("samples", []):
            lines.append(f"    {a['id']}: gold={a['gold']:.2f} model={a['model']:.4f} diff={a['diff']:.4f} [{a['category']}]")

        pc = report.get("pass_criteria", {})
        lines.append(f"\n[Pass Criteria]")
        lines.append(f"  Spearman >= 0.8: {'PASS' if pc.get('spearman_ge_0_8') else 'FAIL'}")
        lines.append(f"  Kappa >= 0.6:    {'PASS' if pc.get('kappa_ge_0_6') else 'FAIL'}")
        lines.append(f"  Overall:         {'PASS' if pc.get('calibration_passed') else 'FAIL'}")

        lines.append(f"\n{'=' * 60}")
        return "\n".join(lines)
