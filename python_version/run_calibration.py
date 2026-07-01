"""校准测试脚本 —— 统一调用 app/engine.py，20条评测数据集"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from app.engine import evaluate
from app.models import EvalRequest
from app.calibration import calibrate
from app.metrics import compute_flaw_metrics, compute_anchor_accuracy

# ── 加载数据集 ────────────────────────────────────────────────────────────
data_path = PROJECT_ROOT / "data" / "eval_dataset.json"
with open(data_path, "r", encoding="utf-8") as f:
    samples = json.load(f)

print("=" * 70)
print(f"  校准测试 — 共 {len(samples)} 条样本（统一调用 engine.py）")
print("=" * 70)

# ── 构建请求 ──────────────────────────────────────────────────────────────
requests: list[EvalRequest] = []
for s in samples:
    # 从四维度 human_scores 计算加权总分（与系统权重一致）
    hs = s.get("human_scores", {})
    if hs:
        overall = (
            hs.get("semantic_consistency", 0.5) * 0.35
            + hs.get("factual_accuracy", 0.5) * 0.35
            + hs.get("readability_structure", 0.5) * 0.15
            + hs.get("over_cleaning", 0.5) * 0.15
        )
    else:
        overall = s.get("human_score", 0.5)

    requests.append(EvalRequest(
        request_id=s["id"],
        before_text=s["before_text"],
        after_text=s["after_text"],
        human_label={
            "overall_score": round(overall, 4),
            "label": s.get("label", ""),
            "flaws": s.get("flaws_gt", []),
        },
    ))

# ── 逐条评估 ──────────────────────────────────────────────────────────────
print("\n逐条评估中（每条调用 DeepSeek API，约 5-15 秒）…")

llm_scores: list[float] = []
human_scores: list[float] = []
details: list[dict] = []
all_predicted_flaws: list[dict] = []
all_gt_flaws: list[dict] = []

for i, (req, s) in enumerate(zip(requests, samples)):
    human = (req.human_label or {}).get("overall_score", 0.5)
    print(f"  [{i+1}/{len(samples)}] {req.request_id} ({s.get('label','')}) ... ", end="", flush=True)

    try:
        resp = evaluate(req, temperature=0.0)
        llm = resp.overall_score
        diff = abs(llm - human)
        consistent = diff <= 0.15
        print(f"LLM={llm:.3f}  Human={human:.3f}  |diff|={diff:.3f}  {'OK' if consistent else 'GAP'}")

        llm_scores.append(llm)
        human_scores.append(human)
        details.append({
            "request_id": req.request_id,
            "label": s.get("label", ""),
            "llm_score": llm,
            "human_score": human,
            "verdict": resp.verdict,
            "diff": round(diff, 4),
            "consistent": consistent,
            "model_version": resp.model_version,
            "prompt_version": resp.prompt_version,
            "reproducibility_token": resp.reproducibility_token,
        })

        pred_flaws = [
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
        all_predicted_flaws.extend(pred_flaws)
        all_gt_flaws.extend(s.get("flaws_gt", []))

    except Exception as e:
        print(f"ERROR: {e}")
        llm_scores.append(0.0)
        human_scores.append(human)

# ── 一致性校准 ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  一致性校准（调用 calibration.py）")
print("=" * 70)

try:
    cal = calibrate(requests)
    pr = cal.pearson_r
    sr = cal.spearman_rho
    mae = cal.mae
    rmse = cal.rmse
    cr = cal.consistency_rate
    print(f"  Pearson r      = {pr:.4f}  {'✅ 达标(>=0.8)' if pr >= 0.8 else '❌ 未达标'}")
    print(f"  Spearman rho   = {sr:.4f}")
    print(f"  MAE            = {mae:.4f}")
    print(f"  RMSE           = {rmse:.4f}")
    print(f"  一致率(±0.1)   = {cr:.2%}")
    print(f"  样本数         = {cal.sample_count}")
except Exception as e:
    print(f"  校准失败: {e}")

# ── 瑕疵检出指标 ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  瑕疵检出指标（样本级 TP/FP/FN）")
print("=" * 70)

if all_gt_flaws:
    try:
        fm = compute_flaw_metrics(all_predicted_flaws, all_gt_flaws)
        print(f"  Precision  = {fm['precision']:.4f}")
        print(f"  Recall     = {fm['recall']:.4f}")
        print(f"  F1-score   = {fm['f1']:.4f}  {'✅ 达标(>=0.8)' if fm['f1'] >= 0.8 else '❌ 未达标'}")
        print(f"  TP/FP/FN   = {fm['tp']}/{fm['fp']}/{fm['fn']}")
        print(f"  GT 支持数  = {fm['support']}")
    except Exception as e:
        print(f"  指标计算失败: {e}")
else:
    print("  无 GT 瑕疵数据，跳过")

# ── 锚点定位准确率 ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  锚点定位准确率（snippet overlap + char tolerance=10）")
print("=" * 70)

if all_gt_flaws:
    try:
        am = compute_anchor_accuracy(all_predicted_flaws, all_gt_flaws, char_tolerance=10)
        print(f"  anchor_accuracy  = {am['anchor_accuracy']:.4f}  {'✅ 达标(>=0.9)' if am['anchor_accuracy'] >= 0.9 else '❌ 未达标'}")
        print(f"  snippet_accuracy = {am['snippet_accuracy']:.4f}")
        print(f"  char_accuracy    = {am['char_accuracy']:.4f}")
        print(f"  正确/总数        = {am['correct']}/{am['total']}")
    except Exception as e:
        print(f"  计算失败: {e}")
else:
    print("  无 GT 瑕疵数据，跳过")

# ── 逐条对比 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  逐条对比详情")
print("=" * 70)
for d in details:
    arrow = "✅" if d["consistent"] else "⚠️ "
    print(f"  {arrow} {d['request_id']:14s} ({d['label']:12s})  "
          f"LLM={d['llm_score']:.3f}  Human={d['human_score']:.3f}  "
          f"|diff|={d['diff']:.3f}  [{d['verdict']}]")

# ── 保存结果 ──────────────────────────────────────────────────────────────
output = {
    "calibration": {
        "pearson_r": round(float(pr) if 'pr' in dir() else 0, 4),
        "spearman_rho": round(float(sr) if 'sr' in dir() else 0, 4),
        "mae": round(float(mae) if 'mae' in dir() else 0, 4),
        "rmse": round(float(rmse) if 'rmse' in dir() else 0, 4),
        "consistency_rate": round(float(cr) if 'cr' in dir() else 0, 4),
        "sample_count": len(details),
    },
    "details": details,
}
out_path = PROJECT_ROOT / "data" / "calibration_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存至 {out_path}")
print("Done!")
