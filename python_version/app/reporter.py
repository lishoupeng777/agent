"""综合评估报告生成器 —— 符合课题12验收标准的完整评估流程"""
from __future__ import annotations

import json
import time
from typing import Any

from .engine import evaluate
from .models import EvalRequest, EvalResponse
from .calibration import calibrate
from .metrics import compute_flaw_metrics, compute_anchor_accuracy
from .stability import run_stability
from .debias import detect_length_bias, detect_position_bias, compute_bias_mitigation_score


def run_full_evaluation(
    requests: list[EvalRequest],
    stability_samples: int = 3,
    char_tolerance: int = 10,
) -> dict[str, Any]:
    """
    运行完整评估流程，输出符合课题12验收标准的综合报告。
    
    包含：
    1. 逐条评估（维度评分 + 瑕疵清单 + 可复现令牌）
    2. 一致性校准（Pearson/Spearman/MAE/RMSE/一致率）
    3. 稳定性分析（多次采样方差）
    4. 瑕疵检出指标（Precision/Recall/F1）
    5. 锚点定位准确率
    6. 偏置分析（长度偏置、位置偏置）
    7. 可复现性验证
    
    Args:
        requests: 评估请求列表
        stability_samples: 稳定性采样次数
        char_tolerance: 锚点容差
        
    Returns:
        dict: 综合评估报告
    """
    report: dict[str, Any] = {
        "report_meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(requests),
            "stability_samples": stability_samples,
            "char_tolerance": char_tolerance,
        },
        "per_sample_results": [],
        "calibration": {},
        "stability_summary": {},
        "flaw_metrics": {},
        "anchor_metrics": {},
        "bias_analysis": {},
        "reproducibility_verification": {},
        "overall_pass": True,
        "checklist": _generate_checklist(),
    }

    all_predicted_flaws: list[dict] = []
    all_gt_flaws: list[dict] = []
    stability_reports: list[dict] = []
    bias_reports: list[dict] = []
    reproducibility_ok = True

    # 1. 逐条评估 + 偏置分析 + 可复现性
    for req in requests:
        sample_result = {
            "request_id": req.request_id,
            "human_label": req.human_label,
            "evaluation": None,
            "bias": None,
            "stability": None,
            "reproducibility": None,
        }

        # LLM 评估
        try:
            resp: EvalResponse = evaluate(req, temperature=0.0)
            sample_result["evaluation"] = {
                "overall_score": resp.overall_score,
                "verdict": resp.verdict,
                "dimensions": [
                    {
                        "dimension": d.dimension,
                        "score": d.score,
                        "weight": d.weight,
                        "reason": d.reason,
                    }
                    for d in resp.dimensions
                ],
                "flaws": [
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
                        "suggestion": f.suggestion,
                    }
                    for f in resp.flaws
                ],
                "reproducibility_token": resp.reproducibility_token,
                "raw_llm_output": resp.raw_llm_output,
            }

            # 偏置分析
            lb = detect_length_bias(req.before_text, req.after_text)
            pb = detect_position_bias(resp.flaws)
            dim_scores = [{"score": d.score} for d in resp.dimensions]
            bms = compute_bias_mitigation_score(
                dim_scores, 
                lb.get("length_ratio", 1.0)
            )
            sample_result["bias"] = {
                "length_bias": lb,
                "position_bias": pb,
                "bias_mitigation_score": bms,
            }
            bias_reports.append({
                "request_id": req.request_id,
                "length_ratio": lb.get("length_ratio", 0),
                "length_bias_risk": lb.get("bias_risk", "unknown"),
                "position_bias_type": pb.get("bias_type"),
                "bias_mitigation_score": bms,
            })

            # 稳定性分析
            try:
                stab = run_stability(req, sample_count=stability_samples)
                sample_result["stability"] = {
                    "mean_score": stab.mean_score,
                    "variance": stab.variance,
                    "std_dev": stab.std_dev,
                    "is_stable": stab.is_stable,
                    "samples": stab.samples,
                }
                stability_reports.append({
                    "request_id": req.request_id,
                    "mean_score": stab.mean_score,
                    "variance": stab.variance,
                    "is_stable": stab.is_stable,
                })
            except Exception as e:
                sample_result["stability"] = {"error": str(e)}

            # 可复现性验证：同输入再评估一次
            try:
                resp2 = evaluate(req, temperature=0.0)
                sample_result["reproducibility"] = {
                    "token_match": resp.reproducibility_token == resp2.reproducibility_token,
                    "score_diff": abs(resp.overall_score - resp2.overall_score),
                    "token1": resp.reproducibility_token,
                    "token2": resp2.reproducibility_token,
                }
                if not sample_result["reproducibility"]["token_match"]:
                    reproducibility_ok = False
            except Exception as e:
                sample_result["reproducibility"] = {"error": str(e)}

            # 收集瑕疵数据
            all_predicted_flaws.extend(sample_result["evaluation"]["flaws"])
            if req.human_label:
                gt_flaws = req.human_label.get("flaws", [])
                if isinstance(gt_flaws, list):
                    all_gt_flaws.extend(gt_flaws)

        except Exception as e:
            sample_result["evaluation"] = {"error": str(e)}

        report["per_sample_results"].append(sample_result)

    # 2. 一致性校准
    if requests:
        try:
            cal = calibrate(requests)
            report["calibration"] = {
                "pearson_r": cal.pearson_r,
                "spearman_rho": cal.spearman_rho,
                "mae": cal.mae,
                "rmse": cal.rmse,
                "consistency_rate": cal.consistency_rate,
                "sample_count": cal.sample_count,
                "details": cal.details,
                "pass_threshold_0_8": cal.pearson_r >= 0.8,
            }
        except Exception as e:
            report["calibration"] = {"error": str(e)}

    # 3. 稳定性汇总
    if stability_reports:
        stable_count = sum(1 for s in stability_reports if s.get("is_stable", False))
        variances = [s.get("variance", 0) for s in stability_reports]
        report["stability_summary"] = {
            "total": len(stability_reports),
            "stable_count": stable_count,
            "stable_rate": round(stable_count / len(stability_reports), 4),
            "avg_variance": round(sum(variances) / len(variances), 6) if variances else 0,
            "details": stability_reports,
        }

    # 4. 瑕疵检出指标
    if all_gt_flaws:
        try:
            fm = compute_flaw_metrics(all_predicted_flaws, all_gt_flaws)
            report["flaw_metrics"] = {
                "precision": fm["precision"],
                "recall": fm["recall"],
                "f1": fm["f1"],
                "support": fm["support"],
                "pass_threshold_0_8": fm["f1"] >= 0.8,
            }
        except Exception as e:
            report["flaw_metrics"] = {"error": str(e)}

    # 5. 锚点定位准确率
    if all_gt_flaws:
        try:
            am = compute_anchor_accuracy(all_predicted_flaws, all_gt_flaws, char_tolerance)
            report["anchor_metrics"] = {
                "accuracy": am["anchor_accuracy"],
                "total": am["total"],
                "correct": am["correct"],
                "pass_threshold_0_9": am["anchor_accuracy"] >= 0.9,
            }
        except Exception as e:
            report["anchor_metrics"] = {"error": str(e)}

    # 6. 偏置汇总
    if bias_reports:
        high_bias = [b for b in bias_reports if b.get("length_bias_risk") == "high"]
        pos_bias = [b for b in bias_reports if b.get("position_bias_type")]
        avg_bms = sum(b.get("bias_mitigation_score", 0) for b in bias_reports) / len(bias_reports)
        report["bias_analysis"] = {
            "avg_bias_mitigation_score": round(avg_bms, 4),
            "high_length_bias_count": len(high_bias),
            "position_bias_count": len(pos_bias),
            "details": bias_reports,
        }

    # 7. 可复现性验证
    report["reproducibility_verification"] = {
        "all_reproducible": reproducibility_ok,
        "note": "固定策略（temperature=0.0、固定 prompt、相同模型）下所有评估结果可复现",
    }

    # 综合判定
    checks = []
    # 相关性检查
    if report["calibration"].get("pearson_r", 0) >= 0.8:
        checks.append(("一致性（Pearson r ≥ 0.8）", True, report["calibration"]["pearson_r"]))
    else:
        checks.append(("一致性（Pearson r ≥ 0.8）", False, report["calibration"].get("pearson_r", 0)))

    # F1 检查
    if report["flaw_metrics"].get("f1", 0) >= 0.8:
        checks.append(("瑕疵检出 F1 ≥ 0.8", True, report["flaw_metrics"]["f1"]))
    else:
        checks.append(("瑕疵检出 F1 ≥ 0.8", False, report["flaw_metrics"].get("f1", 0)))

    # 锚点准确率检查
    if report["anchor_metrics"].get("accuracy", 0) >= 0.9:
        checks.append(("锚点定位准确率 ≥ 90%", True, report["anchor_metrics"]["accuracy"]))
    else:
        checks.append(("锚点定位准确率 ≥ 90%", False, report["anchor_metrics"].get("accuracy", 0)))

    # 稳定性检查
    if report["stability_summary"].get("stable_rate", 0) >= 0.8:
        checks.append(("评分稳定性（稳定率 ≥ 80%）", True, report["stability_summary"]["stable_rate"]))
    else:
        checks.append(("评分稳定性（稳定率 ≥ 80%）", False, report["stability_summary"].get("stable_rate", 0)))

    # 可复现性检查
    checks.append(("评估可复现", reproducibility_ok, reproducibility_ok))

    report["checklist"] = [
        {"item": c[0], "passed": c[1], "value": c[2]} for c in checks
    ]
    report["overall_pass"] = all(c[1] for c in checks)

    return report


def _generate_checklist() -> list[dict[str, Any]]:
    """课题12验收标准清单"""
    return [
        {
            "requirement": "与人工一致性",
            "metric": "Pearson r ≥ 0.8",
            "status": "待验证",
            "note": "核心考核项",
        },
        {
            "requirement": "评分稳定性",
            "metric": "同一输入多次评估方差 < 0.005",
            "status": "待验证",
            "note": "低温度/固定策略下可复现",
        },
        {
            "requirement": "瑕疵检出",
            "metric": "Precision/Recall → F1 ≥ 0.8",
            "status": "待验证",
            "note": "过度清洗/误改识别",
        },
        {
            "requirement": "可解释性",
            "metric": "判定理由合理可被人工复核",
            "status": "已实现",
            "note": "每维度含 reason 字段，每瑕疵含 description",
        },
        {
            "requirement": "瑕疵可定位",
            "metric": "锚点定位准确率 ≥ 90%",
            "status": "待验证",
            "note": "segment_id + start_char + end_char + snippet",
        },
        {
            "requirement": "可复现",
            "metric": "固定策略下可复现",
            "status": "已实现",
            "note": "temperature=0.0 + SHA256 令牌",
        },
        {
            "requirement": "抗偏置",
            "metric": "长度/位置偏置检测 + 缓解策略",
            "status": "已实现",
            "note": "debias 模块 + anti_bias_prompt",
        },
    ]


def export_report_json(report: dict[str, Any], filepath: str) -> None:
    """导出评估报告为 JSON 文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def print_report_summary(report: dict[str, Any]) -> None:
    """打印评估报告摘要到控制台（兼容Windows GBK编码）"""
    print("\n" + "=" * 70)
    print("  课题12 内容保真度与治理质量评估智能体 -- 综合评估报告")
    print("=" * 70)

    meta = report.get("report_meta", {})
    print(f"\n[报告信息]")
    print(f"   生成时间：{meta.get('generated_at', 'N/A')}")
    print(f"   评估样本：{meta.get('total_samples', 0)} 条")
    print(f"   稳定性采样：{meta.get('stability_samples', 3)} 次")

    cal = report.get("calibration", {})
    print(f"\n[一致性校准]")
    print(f"   Pearson r：{cal.get('pearson_r', 'N/A'):.4f}  {'[达标]' if cal.get('pass_threshold_0_8') else '[未达标]'}（目标 >= 0.8）")
    print(f"   Spearman rho：{cal.get('spearman_rho', 'N/A'):.4f}")
    print(f"   MAE：{cal.get('mae', 'N/A'):.4f}")
    print(f"   RMSE：{cal.get('rmse', 'N/A'):.4f}")
    print(f"   一致率：{cal.get('consistency_rate', 0)*100:.1f}%")

    fm = report.get("flaw_metrics", {})
    print(f"\n[瑕疵检出指标]")
    print(f"   Precision：{fm.get('precision', 'N/A'):.4f}")
    print(f"   Recall：{fm.get('recall', 'N/A'):.4f}")
    print(f"   F1：{fm.get('f1', 'N/A'):.4f}  {'[达标]' if fm.get('pass_threshold_0_8') else '[未达标]'}（目标 >= 0.8）")

    am = report.get("anchor_metrics", {})
    print(f"\n[锚点定位准确率]")
    print(f"   准确率：{am.get('accuracy', 0)*100:.1f}%  {'[达标]' if am.get('pass_threshold_0_9') else '[未达标]'}（目标 >= 90%）")
    print(f"   正确/总数：{am.get('correct', 0)}/{am.get('total', 0)}")

    ss = report.get("stability_summary", {})
    print(f"\n[评分稳定性]")
    print(f"   稳定率：{ss.get('stable_rate', 0)*100:.1f}%")
    print(f"   平均方差：{ss.get('avg_variance', 0):.6f}")
    print(f"   稳定/总数：{ss.get('stable_count', 0)}/{ss.get('total', 0)}")

    ba = report.get("bias_analysis", {})
    print(f"\n[偏置分析]")
    print(f"   平均偏置缓解得分：{ba.get('avg_bias_mitigation_score', 0):.4f}")
    print(f"   高长度偏置风险：{ba.get('high_length_bias_count', 0)} 条")
    print(f"   位置偏置检出：{ba.get('position_bias_count', 0)} 条")

    rv = report.get("reproducibility_verification", {})
    print(f"\n[可复现性验证]")
    print(f"   {'[所有评估可复现]' if rv.get('all_reproducible') else '[存在不可复现的评估]'}")

    print(f"\n{'=' * 70}")
    print(f"  综合判定：{'[全部达标]' if report.get('overall_pass') else '[存在未达标项]'}")
    print(f"{'=' * 70}\n")