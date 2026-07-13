"""金标数据集评测脚本（v1/v2）——关闭稳定性，输出课题12综合报告。

金标格式: {id, before_text, after_text, dimensions[], flaws[], overall{weighted_score, verdict}}
用法: python run_gold_eval.py v1   或   python run_gold_eval.py v2
"""
from __future__ import annotations

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from app.models import EvalRequest
from app.reporter import run_full_evaluation, print_report_summary, export_report_json

# 金标维度名 → 标准维度名
_DIM_MAP = {
    "semantic_fidelity": "semantic",
    "factual_consistency": "factual",
    "hallucination": "hallucination",
    "structure": "structure",
    "readability": "readability",
}


def build_requests(dataset: list[dict]) -> list[EvalRequest]:
    """金标格式 → EvalRequest + human_label。"""
    requests = []
    for item in dataset:
        overall = item.get("overall", {})
        human_score = overall.get("weighted_score", 0.5)
        # 维度分数映射
        dim_scores = {}
        for d in item.get("dimensions", []):
            canonical = _DIM_MAP.get(d.get("dimension", ""), d.get("dimension", ""))
            if canonical:
                dim_scores[canonical] = d.get("score", 0.5)
        human_label = {
            "overall_score": human_score,
            "label": overall.get("verdict", "review"),
            "flaws": item.get("flaws", []),
            "dimension_scores": dim_scores,
        }
        requests.append(EvalRequest(
            request_id=item["id"],
            before_text=item["before_text"],
            after_text=item["after_text"],
            human_label=human_label,
        ))
    return requests


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", f"gold_dataset_{version}.json",
    )
    if not os.path.exists(data_path):
        print(f"数据集不存在: {data_path}")
        return

    dataset = json.load(open(data_path, encoding="utf-8"))
    print("=" * 60)
    print(f"金标数据集 {version} 评测（关闭稳定性）")
    print("=" * 60)
    print(f"加载: {len(dataset)} 条样本")

    requests = build_requests(dataset)
    report = run_full_evaluation(
        requests,
        run_stability=False,       # 按要求关闭稳定性
        consistency_samples=3,     # 一致性 3 次采样取均值
    )
    print_report_summary(report)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"gold_{version}_eval_report.json",
    )
    export_report_json(report, out_path)
    print(f"[报告已导出] {out_path}")


if __name__ == "__main__":
    main()
