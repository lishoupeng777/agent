"""评测脚本 —— 使用评测数据集进行批量评估 + 一致性校准 + 完整验收报告"""
from __future__ import annotations

import json
import os
import sys

# 将项目根目录加入 path 以便导入 app 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from app.models import EvalRequest
from app.engine import evaluate
from app.calibration import calibrate
from app.metrics import compute_flaw_metrics, compute_anchor_accuracy
from app.reporter import run_full_evaluation, print_report_summary, export_report_json


def load_dataset(path: str) -> list[dict]:
    """加载评测数据集"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_requests(dataset: list[dict]) -> list[EvalRequest]:
    """将数据集转为 EvalRequest 列表"""
    requests = []
    for item in dataset:
        req = EvalRequest(
            request_id=item["id"],
            before_text=item["before_text"],
            after_text=item["after_text"],
            human_label={
                "overall_score": item["human_score"],
                "label": item["label"],
                "flaws": item.get("flaws_gt", []),
            },
        )
        requests.append(req)
    return requests


def main():
    # 数据集路径
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "eval_dataset.json",
    )

    print("=" * 60)
    print("内容保真度与治理质量评估智能体 — 评测运行")
    print("=" * 60)

    dataset = load_dataset(data_path)
    print(f"\n加载评测数据集: {len(dataset)} 条样本")

    requests = build_requests(dataset)

    # ========= 模式选择 =========
    # 方式一：逐条评估（适合调试）
    # 方式二：综合评估报告（符合课题12验收标准，推荐）
    
    use_full_report = True  # 设为 False 则使用逐条评估模式

    if use_full_report:
        # ---- 方式二：综合评估报告 ----
        print("\n" + "=" * 60)
        print("运行综合评估报告（课题12验收标准）...")
        print("=" * 60)
        
        report = run_full_evaluation(requests)
        print_report_summary(report)
        
        # 导出 JSON 报告
        report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "eval_report.json",
        )
        export_report_json(report, report_path)
        print(f"[报告已导出] 综合报告已导出至: {report_path}")
        return

    # ---- 方式一：逐条评估（原版） ----
    print("\n" + "-" * 40)
    print("逐条评估结果")
    print("-" * 40)

    all_predicted_flaws: list[dict] = []
    all_gt_flaws: list[dict] = []

    for i, req in enumerate(requests):
        print(f"\n[{i+1}/{len(requests)}] {req.request_id} — {dataset[i]['label']}")
        try:
            resp = evaluate(req, temperature=0.0)

            # 打印结果
            print(f"  人工评分: {dataset[i]['human_score']:.2f} → LLM评分: {resp.overall_score:.4f}")
            print(f"  判定: {resp.verdict}")
            for d in resp.dimensions:
                print(f"    [{d.dimension}] 得分: {d.score:.2f} | {d.reason[:80]}...")
            if resp.flaws:
                print(f"  检出瑕疵: {len(resp.flaws)} 条")
                for f in resp.flaws:
                    print(f"    [{f.severity}] {f.category}: {f.description[:80]}...")
            else:
                print(f"  检出瑕疵: 0 条")

            # 收集瑕疵数据用于指标计算
            pred_flaws_dicts = [
                {
                    "category": f.category,
                    "severity": f.severity,
                    "description": f.description,
                    "location": {
                        "segment_id": f.location.segment_id,
                        "start_char": f.location.start_char,
                        "end_char": f.location.end_char,
                        "snippet": f.location.snippet,
                    },
                }
                for f in resp.flaws
            ]
            all_predicted_flaws.extend(pred_flaws_dicts)
            all_gt_flaws.extend(dataset[i].get("flaws_gt", []))

        except Exception as e:
            print(f"  ❌ 评估失败: {e}")

    # 一致性校准
    print("\n" + "=" * 40)
    print("一致性校准报告")
    print("=" * 40)
    try:
        report = calibrate(requests)
        print(f"  Pearson r:      {report.pearson_r:.4f}  (目标 ≥ 0.8)")
        print(f"  Spearman ρ:     {report.spearman_rho:.4f}")
        print(f"  MAE:            {report.mae:.4f}")
        print(f"  RMSE:           {report.rmse:.4f}")
        print(f"  一致率:          {report.consistency_rate:.4f}  (容差 ±0.1)")
        print(f"  样本数:          {report.sample_count}")
        print(f"\n  逐条对比:")
        for d in report.details:
            status = "✓" if d["consistent"] else "✗"
            print(f"    {status} {d['request_id']}: LLM={d['llm_score']:.4f}, Human={d['human_score']:.4f}, diff={d['diff']:.4f}")
    except Exception as e:
        print(f"  ❌ 校准失败: {e}")

    # 瑕疵检出指标
    print("\n" + "=" * 40)
    print("瑕疵检出指标")
    print("=" * 40)
    try:
        flaw_metrics = compute_flaw_metrics(all_predicted_flaws, all_gt_flaws)
        print(f"  Precision:  {flaw_metrics['precision']:.4f}")
        print(f"  Recall:     {flaw_metrics['recall']:.4f}")
        print(f"  F1-score:   {flaw_metrics['f1']:.4f}  (目标 ≥ 0.8)")
        print(f"  Support:    {flaw_metrics['support']}")
    except Exception as e:
        print(f"  ❌ 指标计算失败: {e}")

    # 锚点定位准确率
    print("\n" + "=" * 40)
    print("锚点定位准确率")
    print("=" * 40)
    try:
        anchor_acc = compute_anchor_accuracy(all_predicted_flaws, all_gt_flaws)
        print(f"  准确率:     {anchor_acc['anchor_accuracy']:.4f}  (目标 ≥ 0.9)")
        print(f"  正确/总数:  {anchor_acc['correct']}/{anchor_acc['total']}")
    except Exception as e:
        print(f"  ❌ 计算失败: {e}")

    print("\n" + "=" * 60)
    print("评测完成")
    print("=" * 60)


if __name__ == "__main__":
    main()