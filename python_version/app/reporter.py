"""综合评估报告生成器 —— 符合课题12验收标准的完整评估流程"""
from __future__ import annotations

import json
import time
from typing import Any

from .engine import evaluate
from .models import EvalRequest, EvalResponse
from .calibration import calibrate
from .metrics import (
    compute_flaw_metrics,
    compute_anchor_accuracy,
    compute_kappa,
    compute_kendalls_w,
    compute_correlation_metrics,
    compute_rouge_l,
    compute_bertscore,
)
from .stability import run_stability as _run_stability_fn
from .debias import detect_length_bias, detect_position_bias, compute_bias_mitigation_score


def run_full_evaluation(
    requests: list[EvalRequest],
    stability_samples: int = 3,
    char_tolerance: int = 10,
    run_stability: bool = False,
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
            "stability_samples": stability_samples if run_stability else 0,
            "char_tolerance": char_tolerance,
            "run_stability": run_stability,
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
                "latency_seconds": resp.latency_seconds,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "total_tokens": resp.total_tokens,
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

            # 稳定性分析（默认关闭，run_stability=True 时启用）
            if run_stability:
                try:
                    stab = _run_stability_fn(req, sample_count=stability_samples)
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

    # 2b. 多维一致性指标（Kappa, Kendall's W, Spearman, ICC 等）
    human_scores = []
    llm_scores = []
    for sample in report["per_sample_results"]:
        ev = sample.get("evaluation")
        if ev and not ev.get("error") and sample.get("human_label"):
            llm_scores.append(ev["overall_score"])
            human_scores.append(sample["human_label"].get("overall_score", 0.5))

    if len(human_scores) >= 2:
        try:
            report["consistency_metrics"] = compute_correlation_metrics(human_scores, llm_scores)
        except Exception as e:
            report["consistency_metrics"] = {"error": str(e)}

        try:
            report["kappa_metrics"] = {
                "pass_fail": compute_kappa(human_scores, llm_scores, task="pass/fail"),
                "pass_review_fail": compute_kappa(human_scores, llm_scores, task="pass/review/fail"),
            }
        except Exception as e:
            report["kappa_metrics"] = {"error": str(e)}

        try:
            report["kendalls_w"] = compute_kendalls_w(human_scores, llm_scores)
        except Exception as e:
            report["kendalls_w"] = {"error": str(e)}

    # 2c. ROUGE-L（基于瑕疵描述文本）
    pred_descriptions = [f["description"] for f in all_predicted_flaws[:50]]
    gt_descriptions = [f["description"] for f in all_gt_flaws[:50]]
    if pred_descriptions and gt_descriptions:
        try:
            min_len = min(len(pred_descriptions), len(gt_descriptions))
            report["rouge_l"] = compute_rouge_l(gt_descriptions[:min_len], pred_descriptions[:min_len])
        except Exception as e:
            report["rouge_l"] = {"error": str(e)}

    # 2d. 延迟统计
    latencies = []
    token_counts = []
    for sample in report["per_sample_results"]:
        ev = sample.get("evaluation")
        if ev and not ev.get("error"):
            if ev.get("latency_seconds"):
                latencies.append(ev["latency_seconds"])
            if ev.get("total_tokens"):
                token_counts.append(ev["total_tokens"])

    if latencies:
        report["latency_stats"] = {
            "mean_seconds": round(sum(latencies) / len(latencies), 3),
            "min_seconds": round(min(latencies), 3),
            "max_seconds": round(max(latencies), 3),
            "samples": len(latencies),
            "pass_below_3s": sum(1 for l in latencies if l < 3.0),
            "pass_rate_below_3s": round(sum(1 for l in latencies if l < 3.0) / len(latencies), 4),
        }

    if token_counts:
        report["token_stats"] = {
            "mean_tokens": round(sum(token_counts) / len(token_counts)),
            "total_tokens": sum(token_counts),
            "samples": len(token_counts),
        }

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

    # 相关性检查（Spearman 为主，Pearson 辅助）
    spearman = report.get("consistency_metrics", {}).get("spearman_rho", 0)
    pearson = report.get("consistency_metrics", {}).get("pearson_r", 0)
    if not spearman and not pearson:
        spearman = report["calibration"].get("spearman_rho", 0)
        pearson = report["calibration"].get("pearson_r", 0)
    checks.append(("一致性（Spearman rho ≥ 0.8）", spearman >= 0.8, spearman))
    checks.append(("一致性（Pearson r ≥ 0.8）[辅助]", pearson >= 0.8, pearson))

    # Kappa 检查
    kappa_val = report.get("consistency_metrics", {}).get("kappa", 0)
    checks.append(("Kappa ≥ 0.6", kappa_val >= 0.6, kappa_val))

    # Kendall's W 检查
    kw_val = report.get("kendalls_w", {}).get("kendalls_w", 0)
    checks.append(("Kendall's W ≥ 0.8", kw_val >= 0.8, kw_val))

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

    # 延迟检查
    lat = report.get("latency_stats", {})
    if lat:
        checks.append(("LLM 延迟 < 3s（平均）", lat.get("mean_seconds", 999) < 3.0, lat.get("mean_seconds", 0)))

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
    """课题12验收标准清单（I1-I8）"""
    return [
        {
            "id": "I1",
            "requirement": "与人工一致性",
            "metric": "Pearson r ≥ 0.8",
            "status": "待验证",
            "note": "核心考核项",
        },
        {
            "id": "I2",
            "requirement": "评分稳定性",
            "metric": "同一输入多次评估方差 < 0.005",
            "status": "待验证",
            "note": "低温度/固定策略下可复现",
        },
        {
            "id": "I3",
            "requirement": "瑕疵检出 F1",
            "metric": "Precision/Recall → F1 ≥ 0.8",
            "status": "待验证",
            "note": "过度清洗/误改识别",
        },
        {
            "id": "I4",
            "requirement": "锚点定位准确率",
            "metric": "锚点定位准确率 ≥ 90%",
            "status": "待验证",
            "note": "segment_id + start_char + end_char + snippet，字符容差 ≤ 10",
        },
        {
            "id": "I5",
            "requirement": "可解释性",
            "metric": "每维度含 reason 字段",
            "status": "已实现",
            "note": "每维度含 reason 字段，每瑕疵含 description",
        },
        {
            "id": "I6",
            "requirement": "瑕疵可定位",
            "metric": "行级锚点定位",
            "status": "已实现",
            "note": "flaw.location 含 segment_id + start_char + end_char",
        },
        {
            "id": "I7",
            "requirement": "可复现",
            "metric": "temperature=0.0 + SHA256 令牌",
            "status": "已实现",
            "note": "固定策略下同输入同输出",
        },
        {
            "id": "I8",
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
    if cal.get("error"):
        print(f"   校准失败：{cal['error']}")
    else:
        print(f"   Pearson r：{cal.get('pearson_r', 0):.4f}  {'[达标]' if cal.get('pass_threshold_0_8') else '[未达标]'}（目标 >= 0.8）")
        print(f"   Spearman rho：{cal.get('spearman_rho', 0):.4f}")
        print(f"   MAE：{cal.get('mae', 0):.4f}")
        print(f"   RMSE：{cal.get('rmse', 0):.4f}")
        print(f"   一致率：{cal.get('consistency_rate', 0)*100:.1f}%")

    fm = report.get("flaw_metrics", {})
    print(f"\n[瑕疵检出指标]")
    if fm.get("error"):
        print(f"   指标计算失败：{fm['error']}")
    else:
        print(f"   Precision：{fm.get('precision', 0):.4f}")
        print(f"   Recall：{fm.get('recall', 0):.4f}")
        print(f"   F1：{fm.get('f1', 0):.4f}  {'[达标]' if fm.get('pass_threshold_0_8') else '[未达标]'}（目标 >= 0.8）")

    am = report.get("anchor_metrics", {})
    print(f"\n[锚点定位准确率]")
    if am.get("error"):
        print(f"   计算失败：{am['error']}")
    else:
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

    # 新增：Kappa / Kendall's W / 综合一致性
    cm = report.get("consistency_metrics", {})
    if cm and not cm.get("error"):
        print(f"\n[多维一致性指标]")
        print(f"   Pearson r：{cm.get('pearson_r', 0):.4f}")
        print(f"   Spearman rho：{cm.get('spearman_rho', 0):.4f}")
        print(f"   Kappa（pass/fail）：{cm.get('kappa', 0):.4f}")
        print(f"   Kendall's W：{cm.get('kendalls_w', 0):.4f}")
        print(f"   MAE：{cm.get('mae', 0):.4f}")
        print(f"   RMSE：{cm.get('rmse', 0):.4f}")

    km = report.get("kappa_metrics", {})
    if km and not km.get("error"):
        pf = km.get("pass_fail", {})
        prf = km.get("pass_review_fail", {})
        print(f"\n[Kappa 系数]")
        print(f"   二分类（pass/fail）：kappa={pf.get('kappa', 0):.4f}，一致率={pf.get('agreement_rate', 0):.4f}")
        print(f"   三分类（pass/review/fail）：kappa={prf.get('kappa', 0):.4f}，一致率={prf.get('agreement_rate', 0):.4f}")

    kw = report.get("kendalls_w", {})
    if kw and not kw.get("error"):
        print(f"\n[Kendall's W]")
        print(f"   W：{kw.get('kendalls_w', 0):.4f}  {'[达标]' if kw.get('kendalls_w', 0) >= 0.8 else '[未达标]'}（目标 >= 0.8）")
        print(f"   Spearman rho：{kw.get('spearman_rho', 0):.4f}")

    rl = report.get("rouge_l", {})
    if rl and not rl.get("error"):
        print(f"\n[ROUGE-L（瑕疵描述）]")
        print(f"   Precision：{rl.get('rouge_l_precision', 0):.4f}")
        print(f"   Recall：{rl.get('rouge_l_recall', 0):.4f}")
        print(f"   F1：{rl.get('rouge_l_f1', 0):.4f}")

    # 新增：延迟统计
    ls = report.get("latency_stats", {})
    if ls:
        print(f"\n[效率指标]")
        print(f"   平均延迟：{ls.get('mean_seconds', 0):.3f}s")
        print(f"   最小延迟：{ls.get('min_seconds', 0):.3f}s")
        print(f"   最大延迟：{ls.get('max_seconds', 0):.3f}s")
        print(f"   < 3s 达标率：{ls.get('pass_rate_below_3s', 0)*100:.1f}%（{ls.get('pass_below_3s', 0)}/{ls.get('samples', 0)}）")

    ts = report.get("token_stats", {})
    if ts:
        print(f"   平均 token 数：{ts.get('mean_tokens', 0)}")
        print(f"   总 token 消耗：{ts.get('total_tokens', 0)}")

    print(f"\n{'=' * 70}")
    print(f"  综合判定：{'[全部达标]' if report.get('overall_pass') else '[存在未达标项]'}")
    print(f"{'=' * 70}\n")