"""评分校准器（Score Calibrator）

解决不同模型评分尺度不一致的问题。
例如：DeepSeek 偏严（给 0.3），GLM 偏松（给 0.7），但实际都是"中等问题"。

校准流程：
1. 用 Calibration Set（带 Gold Label）分别让各模型评分
2. 为每个模型训练一个线性回归：raw_score → calibrated_score
3. 生产环境中，模型输出经校准后再返回

公式：calibrated = slope * raw + intercept

当前实现说明：
- 训练数据：gold_dataset_v1.json（34 条人工标注样本）
- 方法：简单线性回归（scipy.stats.linregress）
- 局限：数据量较小（34 条），回归参数（slope/intercept）基于有限样本
- 适用场景：当引入新模型时，用于快速对齐评分尺度
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from scipy import stats


class ScoreCalibrator:
    """单模型评分校准器。

    使用简单线性回归将模型原始分数映射到标准尺度。
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.slope: float = 1.0
        self.intercept: float = 0.0
        self.r_squared: float = 0.0
        self.is_fitted: bool = False
        self.n_samples: int = 0

    def fit(self, raw_scores: list[float], gold_scores: list[float]) -> None:
        """用校准数据拟合线性回归。

        Args:
            raw_scores: 模型原始评分列表
            gold_scores: Gold Label 评分列表
        """
        if len(raw_scores) != len(gold_scores):
            raise ValueError("raw_scores and gold_scores must have same length")

        n = len(raw_scores)
        if n < 3:
            raise ValueError("Need at least 3 samples for calibration")

        x = np.array(raw_scores)
        y = np.array(gold_scores)

        # 线性回归：y = slope * x + intercept
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        self.slope = slope
        self.intercept = intercept
        self.r_squared = r_value ** 2
        self.n_samples = n
        self.is_fitted = True

    def calibrate(self, raw_score: float) -> float:
        """校准单个分数。

        Args:
            raw_score: 模型原始评分

        Returns:
            float: 校准后的分数（裁剪到 0.0-1.0）
        """
        if not self.is_fitted:
            return raw_score

        calibrated = self.slope * raw_score + self.intercept
        return max(0.0, min(1.0, round(calibrated, 4)))

    def calibrate_batch(self, raw_scores: list[float]) -> list[float]:
        """批量校准"""
        return [self.calibrate(s) for s in raw_scores]

    def to_dict(self) -> dict[str, Any]:
        """导出校准参数"""
        return {
            "model_name": self.model_name,
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "n_samples": self.n_samples,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoreCalibrator:
        """从字典加载校准参数"""
        cal = cls(data["model_name"])
        cal.slope = data["slope"]
        cal.intercept = data["intercept"]
        cal.r_squared = data["r_squared"]
        cal.n_samples = data["n_samples"]
        cal.is_fitted = data["is_fitted"]
        return cal


class MultiModelCalibrator:
    """多模型校准管理器。

    为每个模型维护一个独立的 ScoreCalibrator。
    """

    def __init__(self) -> None:
        self._calibrators: dict[str, ScoreCalibrator] = {}

    def fit_model(
        self,
        model_name: str,
        raw_scores: list[float],
        gold_scores: list[float],
    ) -> None:
        """为指定模型拟合校准器。

        Args:
            model_name: 模型名称
            raw_scores: 模型原始评分列表
            gold_scores: Gold Label 评分列表
        """
        cal = ScoreCalibrator(model_name)
        cal.fit(raw_scores, gold_scores)
        self._calibrators[model_name] = cal

    def calibrate(self, model_name: str, raw_score: float) -> float:
        """用指定模型的校准器校准分数。

        Args:
            model_name: 模型名称
            raw_score: 原始分数

        Returns:
            float: 校准后分数（如果模型无校准器，返回原始分数）
        """
        cal = self._calibrators.get(model_name)
        if cal is None:
            return raw_score
        return cal.calibrate(raw_score)

    def get_calibrator(self, model_name: str) -> ScoreCalibrator | None:
        """获取指定模型的校准器"""
        return self._calibrators.get(model_name)

    def list_models(self) -> list[str]:
        """返回所有已校准的模型名称"""
        return list(self._calibrators.keys())

    def to_dict(self) -> dict[str, Any]:
        """导出所有校准参数"""
        return {
            name: cal.to_dict()
            for name, cal in self._calibrators.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiModelCalibrator:
        """从字典加载"""
        mc = cls()
        for name, cal_data in data.items():
            mc._calibrators[name] = ScoreCalibrator.from_dict(cal_data)
        return mc

    def save(self, filepath: str) -> None:
        """保存到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> MultiModelCalibrator:
        """从文件加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def auto_calibrate(
    model_name: str,
    adapter: Any,
    calibration_data: list[dict[str, Any]],
) -> ScoreCalibrator:
    """自动校准：用 calibration 数据对模型评分，然后拟合校准器。

    Args:
        model_name: 模型名称
        adapter: EvaluationProtocol 实例
        calibration_data: 校准数据集（每条需有 gold_scores.overall）

    Returns:
        ScoreCalibrator: 拟合好的校准器
    """
    from .models import EvalRequest

    raw_scores = []
    gold_scores = []

    for item in calibration_data:
        req = EvalRequest(
            request_id=item["id"],
            before_text=item["before_text"],
            after_text=item["after_text"],
        )
        try:
            resp = adapter.evaluate(req, temperature=0.0)
            raw_scores.append(resp.overall_score)
            gold_scores.append(item["overall"]["weighted_score"])
        except Exception as e:
            print(f"  Skip {item['id']}: {e}")

    if len(raw_scores) < 3:
        raise ValueError(f"Not enough valid samples for calibration: {len(raw_scores)}")

    cal = ScoreCalibrator(model_name)
    cal.fit(raw_scores, gold_scores)
    return cal
