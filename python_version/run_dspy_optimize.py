"""DSPy 自动优化测试脚本

用法：python run_dspy_optimize.py

功能：
1. 加载金标准数据集（34 条）
2. 用 DSPy BootstrapFewShot 自动挑选最优 few-shot 组合
3. 对比优化前后的评分一致性（Pearson r）
4. 输出最优 few-shot 示例，可手动写入 prompts.py

注意：需要 DEEPSEEK_API_KEY，会调用 DeepSeek API（消耗约 30-50 次调用）
"""
import json
import os
import sys

# 确保能 import app 模块
sys.path.insert(0, os.path.dirname(__file__))

import dspy
from dspy.teleprompt import BootstrapFewShot


def load_gold_dataset(path: str, max_samples: int = 34) -> list[dspy.Example]:
    """加载金标准数据集，转为 DSPy Example"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data[:max_samples]:
        # 取 overall 分数作为 human_score
        overall = item.get("overall", {})
        if isinstance(overall, dict):
            human_score = overall.get("weighted_score", 0.5)
        else:
            human_score = float(overall)

        ex = dspy.Example(
            before_text=item["before_text"],
            after_text=item["after_text"],
            evaluation_profile="general",
            human_score=human_score,
        ).with_inputs("before_text", "after_text", "evaluation_profile")
        examples.append(ex)

    return examples


def eval_metric(example, prediction, trace=None):
    """评估指标：1 - |人工分 - LLM分|"""
    try:
        human_score = float(example.human_score)
        pred_score = float(prediction.overall_score)
        score = 1.0 - abs(human_score - pred_score)
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def main():
    # 1. 配置 DSPy 使用 DeepSeek
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return

    print("=" * 60)
    print("DSPy 自动优化 — BootstrapFewShot")
    print("=" * 60)

    lm = dspy.LM(
        model=f"openai/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}",
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.0,
        max_tokens=2048,
    )
    dspy.configure(lm=lm)

    # 2. 加载数据集
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "gold_dataset_v1.json")
    print(f"\n加载数据集: {dataset_path}")
    all_examples = load_gold_dataset(dataset_path)
    print(f"总样本数: {len(all_examples)}")

    # 分训练集和测试集（80/20）
    split = int(len(all_examples) * 0.8)
    trainset = all_examples[:split]
    testset = all_examples[split:]
    print(f"训练集: {len(trainset)} 条, 测试集: {len(testset)} 条")

    # 3. 先用未优化的模块跑一遍基线
    print("\n" + "=" * 60)
    print("基线测试（未优化，无 few-shot）...")
    print("=" * 60)

    from app.dspy_eval import TextEvaluator, validate_eval_output

    baseline = TextEvaluator()
    baseline_scores = []
    human_scores = []

    for ex in testset:
        try:
            pred = baseline(
                before_text=ex.before_text,
                after_text=ex.after_text,
                evaluation_profile=ex.evaluation_profile,
            )
            pred_score = float(pred.overall_score)
            baseline_scores.append(pred_score)
            human_scores.append(float(ex.human_score))
            print(f"  人工={ex.human_score:.2f}  LLM={pred_score:.2f}  差={abs(ex.human_score - pred_score):.2f}")
        except Exception as e:
            print(f"  跳过（错误: {e}）")

    if baseline_scores:
        import numpy as np
        from scipy import stats
        baseline_mae = np.mean(np.abs(np.array(baseline_scores) - np.array(human_scores)))
        baseline_r, _ = stats.pearsonr(human_scores, baseline_scores)
        print(f"\n基线结果: MAE={baseline_mae:.4f}, Pearson r={baseline_r:.4f}")

    # 4. 运行 DSPy 优化
    print("\n" + "=" * 60)
    print("开始 DSPy BootstrapFewShot 优化...")
    print("（这会调用多次 API，请耐心等待）")
    print("=" * 60)

    optimizer = BootstrapFewShot(
        metric=eval_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=2,
    )

    optimized = optimizer.compile(baseline, trainset=trainset)

    # 5. 用优化后的模块跑测试集
    print("\n" + "=" * 60)
    print("优化后测试...")
    print("=" * 60)

    optimized_scores = []
    human_scores_opt = []

    for ex in testset:
        try:
            pred = optimized(
                before_text=ex.before_text,
                after_text=ex.after_text,
                evaluation_profile=ex.evaluation_profile,
            )
            pred_score = float(pred.overall_score)
            optimized_scores.append(pred_score)
            human_scores_opt.append(float(ex.human_score))
            print(f"  人工={ex.human_score:.2f}  LLM={pred_score:.2f}  差={abs(ex.human_score - pred_score):.2f}")
        except Exception as e:
            print(f"  跳过（错误: {e}）")

    if optimized_scores:
        optimized_mae = np.mean(np.abs(np.array(optimized_scores) - np.array(human_scores_opt)))
        optimized_r, _ = stats.pearsonr(human_scores_opt, optimized_scores)
        print(f"\n优化后结果: MAE={optimized_mae:.4f}, Pearson r={optimized_r:.4f}")

    # 6. 对比
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    if baseline_scores and optimized_scores:
        print(f"  基线    : MAE={baseline_mae:.4f}, Pearson r={baseline_r:.4f}")
        print(f"  优化后  : MAE={optimized_mae:.4f}, Pearson r={optimized_r:.4f}")
        print(f"  MAE 改善: {baseline_mae - optimized_mae:+.4f}")
        print(f"  r 改善  : {optimized_r - baseline_r:+.4f}")

    # 7. 输出优化后的 few-shot 信息
    print("\n" + "=" * 60)
    print("优化后的模块结构")
    print("=" * 60)
    print(f"模块类型: {type(optimized).__name__}")
    if hasattr(optimized, 'evaluate') and hasattr(optimized.evaluate, 'demos'):
        demos = optimized.evaluate.demos
        print(f"选出的 few-shot 示例数: {len(demos)}")
        for i, demo in enumerate(demos):
            print(f"\n  示例 {i+1}:")
            if hasattr(demo, 'before_text'):
                print(f"    before: {demo.before_text[:80]}...")
            if hasattr(demo, 'overall_score'):
                print(f"    score: {demo.overall_score}")

    print("\n完成！如果优化效果明显，可以将选出的 few-shot 写入 prompts.py")


if __name__ == "__main__":
    main()
