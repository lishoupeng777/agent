"""DSPy 评估模块 —— 声明式评估接口 + 自动 prompt 优化

使用 DSPy 的 Signature / Module / Teleprompter 实现：
1. Signature：声明式定义评估输入输出
2. Module：封装评估逻辑为可优化的模块
3. Assertions：约束输出格式（分数范围、枚举值）
4. Teleprompter：基于评测数据自动优化 prompt
"""
from __future__ import annotations

import os
from typing import Any

import dspy
from pydantic import BaseModel, Field


# ============================================================
# 1. 配置 DSPy LM
# ============================================================

def configure_dspy_lm() -> None:
    """配置 DSPy 使用 DeepSeek V4 Flash"""
    lm = dspy.LM(
        model=f"openai/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.0,
        max_tokens=4096,
    )
    dspy.configure(lm=lm)


# ============================================================
# 2. Signature 定义
# ============================================================

class TextEvalSignature(dspy.Signature):
    """评估文本治理前后的内容保真度与治理质量。

    对比治理前后的文本，从四个维度打分：
    1. 语义一致性（权重0.4）：治理后是否保留了原意？
    2. 过度清洗/误改识别（权重0.3）：是否误删或错误改写？
    3. 可读性（权重0.15）：句子是否通顺？
    4. 结构质量（权重0.15）：格式是否规范？

    综合判定：pass（≥0.8）、review（0.5-0.8）、fail（<0.5）
    """

    before_text: str = dspy.InputField(description="治理前原始文本")
    after_text: str = dspy.InputField(description="治理后文本")
    evaluation_profile: str = dspy.InputField(description="评估模式：general / government_notice_strict / legal_strict")

    overall_score: float = dspy.OutputField(description="综合得分，0.0到1.0之间")
    verdict: str = dspy.OutputField(description="综合判定：pass、review或fail")
    semantic_score: float = dspy.OutputField(description="语义一致性得分，0.0到1.0")
    over_clean_score: float = dspy.OutputField(description="过度清洗/误改得分，0.0到1.0")
    readability_score: float = dspy.OutputField(description="可读性得分，0.0到1.0")
    structure_score: float = dspy.OutputField(description="结构质量得分，0.0到1.0")
    flaws_summary: str = dspy.OutputField(description="发现的瑕疵摘要，无瑕疵则写'无'")


class StrictEvalSignature(dspy.Signature):
    """严格模式评估签名，增加关键事实检查指令。

    在基础评估之上，额外关注：
    - 政务通告：日期、期限、罚则区间、适用范围
    - 法规合同：责任主体、义务、条件、例外、法律后果
    """

    before_text: str = dspy.InputField(description="治理前原始文本")
    after_text: str = dspy.InputField(description="治理后文本")
    evaluation_profile: str = dspy.InputField(description="评估模式")
    critical_facts: str = dspy.InputField(description="需要检查的关键事实列表")

    overall_score: float = dspy.OutputField(description="综合得分，0.0到1.0之间")
    verdict: str = dspy.OutputField(description="综合判定：pass、review或fail")
    missing_facts: str = dspy.OutputField(description="缺失的关键事实，无则写'无'")
    flaws_summary: str = dspy.OutputField(description="发现的瑕疵摘要")


# ============================================================
# 3. Module 定义
# ============================================================

class TextEvaluator(dspy.Module):
    """文本治理质量评估模块"""

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = dspy.ChainOfThought(TextEvalSignature)

    def forward(
        self,
        before_text: str,
        after_text: str,
        evaluation_profile: str = "general",
    ) -> dspy.Prediction:
        result = self.evaluate(
            before_text=before_text,
            after_text=after_text,
            evaluation_profile=evaluation_profile,
        )
        return result


class StrictEvaluator(dspy.Module):
    """严格模式评估模块（带关键事实检查）"""

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = dspy.ChainOfThought(StrictEvalSignature)

    def forward(
        self,
        before_text: str,
        after_text: str,
        evaluation_profile: str,
        critical_facts: str,
    ) -> dspy.Prediction:
        result = self.evaluate(
            before_text=before_text,
            after_text=after_text,
            evaluation_profile=evaluation_profile,
            critical_facts=critical_facts,
        )
        return result


# ============================================================
# 4. 输出验证（Assertions）
# ============================================================

def validate_score_range(score: float, name: str = "score") -> float:
    """验证分数在 0-1 范围内"""
    if not 0.0 <= score <= 1.0:
        raise dspy.AssertionError(
            f"{name} must be between 0.0 and 1.0, got {score}"
        )
    return score


def validate_verdict(verdict: str) -> str:
    """验证判定值为 pass/review/fail"""
    valid = {"pass", "review", "fail"}
    if verdict not in valid:
        raise dspy.AssertionError(
            f"verdict must be one of {valid}, got '{verdict}'"
        )
    return verdict


def validate_eval_output(prediction: dspy.Prediction) -> bool:
    """验证评估输出的完整性"""
    try:
        validate_score_range(prediction.overall_score, "overall_score")
        validate_verdict(prediction.verdict)
        validate_score_range(prediction.semantic_score, "semantic_score")
        validate_score_range(prediction.overall_score, "over_clean_score")
        validate_score_range(prediction.readability_score, "readability_score")
        validate_score_range(prediction.structure_score, "structure_score")
        return True
    except dspy.AssertionError:
        return False


# ============================================================
# 5. 评估指标（用于 Teleprompter 优化）
# ============================================================

def eval_metric(example: dspy.Example, prediction: dspy.Prediction, trace: Any = None) -> float:
    """评估指标：比较预测分数与人工标注分数的一致性

    用于 Teleprompter 优化时衡量 prompt 质量。
    """
    try:
        human_score = float(example.get("human_score", 0.5))
        pred_score = float(prediction.overall_score)
        # 使用 1 - |差值| 作为得分（越接近 1 越好）
        score = 1.0 - abs(human_score - pred_score)
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


# ============================================================
# 6. Teleprompter 优化
# ============================================================

def optimize_prompt(
    trainset: list[dspy.Example],
    metric: Any = eval_metric,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 2,
) -> TextEvaluator:
    """使用 Teleprompter 自动优化评估 prompt

    Args:
        trainset: 训练数据集（dspy.Example 列表）
        metric: 评估指标函数
        max_bootstrapped_demos: 最大自举示例数
        max_labeled_demos: 最大标注示例数

    Returns:
        优化后的 TextEvaluator 模块
    """
    from dspy.teleprompt import BootstrapFewShot

    optimizer = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )

    evaluator = TextEvaluator()
    optimized = optimizer.compile(evaluator, trainset=trainset)
    return optimized


# ============================================================
# 7. 工具函数：将 eval_dataset 转为 DSPy Example
# ============================================================

def dataset_to_examples(dataset_path: str) -> list[dspy.Example]:
    """将 eval_dataset.json 转为 DSPy Example 列表"""
    import json

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        ex = dspy.Example(
            before_text=item["before_text"],
            after_text=item["after_text"],
            evaluation_profile="general",
            human_score=item["human_score"],
        ).with_inputs("before_text", "after_text", "evaluation_profile")
        examples.append(ex)

    return examples
