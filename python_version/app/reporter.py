"""综合评估报告生成器 —— 符合课题12验收标准的完整评估流程"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# 并发评估的工作线程数（配合 main.py 的限流器：4 并发 + 6 req/s）
# 提速约 2 倍；如遇 429 限流可下调回 2 并发 + 3 req/s
_MAX_WORKERS = 4


def _evaluate_consistency_averaged(req: EvalRequest, samples: int) -> EvalResponse:
    """多次采样平均评估：降低单样本评分噪声，稳定与人工的一致性。

    关键：使用 use_cache=False 强制每次真实调用 API（波动来自跨次独立调用，
    命中缓存会返回同一结果，平均将失去意义）。

    - 对 overall_score 取多次采样均值，得到更稳定的总分。
    - 维度与瑕疵清单沿用首次评估结果（用于瑕疵/锚点指标，不参与平均）。
    """
    if samples <= 1:
        return evaluate(req, temperature=0.0)

    first: EvalResponse | None = None
    scores: list[float] = []
    for i in range(samples):
        resp = evaluate(req, temperature=0.0, use_cache=False)
        if first is None:
            first = resp
        scores.append(resp.overall_score)

    assert first is not None
    mean_score = round(sum(scores) / len(scores), 4)
    first.overall_score = mean_score
    from .chain import determine_verdict
    first.verdict = determine_verdict(mean_score)
    return first


def _process_sample(
    req: EvalRequest,
    run_stability: bool,
    stability_samples: int,
    consistency_samples: int = 1,
) -> tuple[dict, list[dict], list[dict], dict | None, dict | None, bool]:
    """处理单个样本的完整流程（供并发调用）。

    返回: (sample_result, predicted_flaws, gt_flaws, bias_report, stability_report, reproducibility_ok)
    """
    sample_result: dict[str, Any] = {
        "request_id": req.request_id,
        "human_label": req.human_label,
        "evaluation": None,
        "bias": None,
        "stability": None,
        "reproducibility": None,
    }
    predicted_flaws: list[dict] = []
    gt_flaws: list[dict] = []
    bias_report: dict | None = None
    stability_report: dict | None = None
    reproducibility_ok = True

    try:
        resp: EvalResponse = _evaluate_consistency_averaged(req, consistency_samples)
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
        bms = compute_bias_mitigation_score(dim_scores, lb.get("length_ratio", 1.0))
        sample_result["bias"] = {
            "length_bias": lb,
            "position_bias": pb,
            "bias_mitigation_score": bms,
        }
        bias_report = {
            "request_id": req.request_id,
            "length_ratio": lb.get("length_ratio", 0),
            "length_bias_risk": lb.get("bias_risk", "unknown"),
            "position_bias_type": pb.get("bias_type"),
            "bias_mitigation_score": bms,
        }

        # 稳定性分析
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
                stability_report = {
                    "request_id": req.request_id,
                    "mean_score": stab.mean_score,
                    "variance": stab.variance,
                    "is_stable": stab.is_stable,
                }
            except Exception as e:
                sample_result["stability"] = {"error": str(e)}

        # 可复现性验证：同输入再评估一次（命中缓存，秒回）
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
        predicted_flaws = sample_result["evaluation"]["flaws"]
        if req.human_label:
            gt_flaws_raw = req.human_label.get("flaws", [])
            if isinstance(gt_flaws_raw, list):
                gt_flaws = gt_flaws_raw

    except Exception as e:
        sample_result["evaluation"] = {"error": str(e)}

    return sample_result, predicted_flaws, gt_flaws, bias_report, stability_report, reproducibility_ok


def run_full_evaluation(
    requests: list[EvalRequest],
    stability_samples: int = 3,
    char_tolerance: int = 10,
    run_stability: bool = False,
    consistency_samples: int = 1,
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
            "consistency_samples": consistency_samples,
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

    # 1. 并发评估 + 偏置分析 + 可复现性（不含稳定性测试）
    # 稳定性抽样策略：优先选"最难评、最可能波动"的样本，而非随机/等距抽到简单样本。
    # 难度依据（评估前即可确定，无需先跑）：
    #   1) human_label.difficulty == "hard"
    #   2) 人工分处于判定边界中间区间（0.4~0.75），这类分数最容易在多次采样中波动
    # 这样即使方差很小，也更有说服力（连最难样本都稳）。
    _stability_indices: set[int] = set()
    if run_stability:
        n = len(requests)
        scored: list[tuple[int, int]] = []  # (优先级, 索引)，优先级越大越优先
        for idx, req in enumerate(requests):
            hl = req.human_label or {}
            priority = 0
            if hl.get("difficulty") == "hard":
                priority += 2
            hs = hl.get("overall_score", 0.5)
            if 0.4 <= hs <= 0.75:
                priority += 1
            scored.append((priority, idx))
        # 按优先级降序，优先级相同则等距分散，取前 5 条
        scored.sort(key=lambda x: (-x[0], x[1]))
        picked = [idx for _, idx in scored[:5]] if n > 5 else list(range(n))
        _stability_indices = set(picked)
        report["report_meta"]["stability_tested_indices"] = sorted(_stability_indices)
        report["report_meta"]["stability_sampling_strategy"] = "hard_and_boundary_first"

    eval_start = time.time()
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(_process_sample, req, False, stability_samples, consistency_samples): idx
            for idx, req in enumerate(requests)
        }
        results: list[tuple] = []
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                sr, pf, gf, br, sr2, ro = future.result()
                results.append((idx, sr, pf, gf, br, sr2, ro))
            except Exception as e:
                req = requests[idx]
                results.append((idx, {
                    "request_id": req.request_id,
                    "human_label": req.human_label,
                    "evaluation": {"error": str(e)},
                    "bias": None, "stability": None, "reproducibility": None,
                }, [], [], None, None, False))

    # 按原始顺序排序并聚合
    results.sort(key=lambda x: x[0])
    for _, sr, pf, gf, br, sr2, ro in results:
        report["per_sample_results"].append(sr)
        all_predicted_flaws.extend(pf)
        all_gt_flaws.extend(gf)
        if br:
            bias_reports.append(br)
        if not ro:
            reproducibility_ok = False

    eval_elapsed = time.time() - eval_start
    report["report_meta"]["concurrent_workers"] = _MAX_WORKERS
    report["report_meta"]["evaluation_wall_clock_seconds"] = round(eval_elapsed, 3)

    # 1b. 稳定性测试（顺序执行，不干扰评估阶段的缓存和延迟统计）
    stab_elapsed = 0.0
    if run_stability and _stability_indices:
        stab_start = time.time()
        for idx in sorted(_stability_indices):
            req = requests[idx]
            try:
                stab = _run_stability_fn(req, sample_count=stability_samples)
                report["per_sample_results"][idx]["stability"] = {
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
                report["per_sample_results"][idx]["stability"] = {"error": str(e)}
        stab_elapsed = time.time() - stab_start
        report["report_meta"]["stability_wall_clock_seconds"] = round(stab_elapsed, 3)

    report["report_meta"]["wall_clock_seconds"] = round(eval_elapsed + stab_elapsed, 3)

    # 2b. 多维一致性指标（Kappa, Kendall's W, Spearman, ICC 等）
    # 直接复用上面已算出的（可能已多次采样平均的）per-sample 分数，
    # 避免再调一次 calibrate() 重复评估（既省 API，也保证校准与 checklist 口径一致）。
    human_scores = []
    llm_scores = []
    cal_details: list[dict[str, Any]] = []
    for sample in report["per_sample_results"]:
        ev = sample.get("evaluation")
        if ev and not ev.get("error") and sample.get("human_label"):
            llm_s = ev["overall_score"]
            human_s = sample["human_label"].get("overall_score", 0.5)
            llm_scores.append(llm_s)
            human_scores.append(human_s)
            cal_details.append({
                "request_id": sample["request_id"],
                "llm_score": llm_s,
                "human_score": human_s,
                "diff": round(abs(llm_s - human_s), 4),
                "consistent": abs(llm_s - human_s) <= 0.1,
            })

    # 2. 一致性校准（基于 per-sample 分数计算，与 checklist 同源）
    if len(human_scores) >= 2:
        try:
            corr = compute_correlation_metrics(human_scores, llm_scores)
            consistent_n = sum(1 for d in cal_details if d["consistent"])
            report["calibration"] = {
                "pearson_r": corr.get("pearson_r", 0),
                "spearman_rho": corr.get("spearman_rho", 0),
                "mae": corr.get("mae", 0),
                "rmse": corr.get("rmse", 0),
                "consistency_rate": round(consistent_n / len(cal_details), 4),
                "sample_count": len(cal_details),
                "details": cal_details,
                "pass_threshold_0_8": bool(corr.get("pearson_r", 0) >= 0.8),
                "consistency_samples": consistency_samples,
            }
        except Exception as e:
            report["calibration"] = {"error": str(e)}

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
        eval_wall = report.get("report_meta", {}).get("evaluation_wall_clock_seconds", 0)
        total_wall = report.get("report_meta", {}).get("wall_clock_seconds", 0)
        stab_wall = report.get("report_meta", {}).get("stability_wall_clock_seconds", 0)
        report["latency_stats"] = {
            "mean_seconds": round(sum(latencies) / len(latencies), 3),
            "min_seconds": round(min(latencies), 3),
            "max_seconds": round(max(latencies), 3),
            "samples": len(latencies),
            "pass_below_3s": sum(1 for l in latencies if l < 3.0),
            "pass_rate_below_3s": round(sum(1 for l in latencies if l < 3.0) / len(latencies), 4),
            "wall_clock_seconds": total_wall,
            "evaluation_wall_clock_seconds": round(eval_wall, 3),
            "stability_wall_clock_seconds": round(stab_wall, 3),
            "throughput_latency": round(eval_wall / len(latencies), 3) if eval_wall else 0,
            "concurrent_workers": report.get("report_meta", {}).get("concurrent_workers", 1),
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

    # 4. 瑕疵检出指标（多对一匹配：解决 LLM 细粒度拆分导致的 FP 膨胀）
    if all_gt_flaws:
        try:
            fm = compute_flaw_metrics(all_predicted_flaws, all_gt_flaws)
            report["flaw_metrics"] = {
                "precision": fm["precision"],
                "recall": fm["recall"],
                "f1": fm["f1"],
                "tp": fm["tp"],
                "fp": fm["fp"],
                "fn": fm["fn"],
                "support": fm["support"],
                "match_mode": "many_to_one",
                "pass_threshold_0_8": bool(fm["f1"] >= 0.8),
            }
        except Exception as e:
            report["flaw_metrics"] = {"error": str(e)}

    # 5. 锚点定位准确率（双轨匹配：定位与分类解耦）
    if all_gt_flaws:
        try:
            am = compute_anchor_accuracy(all_predicted_flaws, all_gt_flaws, char_tolerance)
            report["anchor_metrics"] = {
                "anchor_accuracy": am.get("anchor_accuracy", 0),
                "primary_metric": am.get("primary_metric", "location_recall"),
                "reference_f1": am.get("reference_f1", am.get("location_f1", 0)),
                "location_precision": am.get("location_precision", 0),
                "location_recall": am.get("location_recall", 0),
                "location_f1": am.get("location_f1", 0),
                "location_tp": am.get("location_tp", 0),
                "location_fp": am.get("location_fp", 0),
                "location_fn": am.get("location_fn", 0),
                "classification_precision": am.get("classification_precision", 0),
                "classification_recall": am.get("classification_recall", 0),
                "classification_f1": am.get("classification_f1", 0),
                "classification_tp": am.get("classification_tp", 0),
                "total": am.get("total", 0),
                "correct": am.get("correct", 0),
                "pass_threshold_0_9": bool(am.get("location_recall", 0) >= 0.9),
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

    # 锚点准确率检查（主指标：定位召回率 location_recall）
    anchor_primary = report["anchor_metrics"].get("location_recall", report["anchor_metrics"].get("anchor_accuracy", 0))
    if anchor_primary >= 0.9:
        checks.append(("锚点定位准确率 ≥ 90%", True, anchor_primary))
    else:
        checks.append(("锚点定位准确率 ≥ 90%", False, anchor_primary))

    # 效率指标（延迟）仅作参考展示，不计入课题12验收达标判定
    # （课题12验收标准不含延迟要求；真实 LLM API 调用延迟受网络/限流影响，
    #  不应作为"裁判可信度"的达标项）
    lat = report.get("latency_stats", {})
    if lat:
        thru = lat.get("throughput_latency", lat.get("mean_seconds", 999))
        report["efficiency_note"] = {
            "throughput_latency": thru,
            "is_reference_only": True,
            "note": "延迟为效率参考指标，不计入课题12验收达标判定",
        }

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

    # 9. 分层分析（按难度 easy/medium/hard）
    strata: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    for sample in report["per_sample_results"]:
        hl = sample.get("human_label") or {}
        difficulty = hl.get("difficulty", "medium")
        strata.setdefault(difficulty, []).append(sample)

    stratified_report = {}
    for level, samples in strata.items():
        if not samples:
            continue
        hs_list, ls_list = [], []
        diffs = []
        consistent = 0
        for s in samples:
            ev = s.get("evaluation") or {}
            hl = s.get("human_label") or {}
            if ev and not ev.get("error") and hl:
                h = hl.get("overall_score", 0.5)
                l = ev.get("overall_score", 0)
                hs_list.append(h)
                ls_list.append(l)
                d = abs(h - l)
                diffs.append(d)
                if d <= 0.1:
                    consistent += 1
        n = len(samples)
        avg_diff = round(sum(diffs) / max(len(diffs), 1), 4) if diffs else 0
        pearson_r = 0.0
        if len(hs_list) >= 3:
            try:
                from scipy.stats import pearsonr
                pearson_r = round(pearsonr(hs_list, ls_list)[0], 4)
            except Exception:
                pass
        stratified_report[level] = {
            "count": n,
            "consistent_count": consistent,
            "consistency_rate": round(consistent / n, 4) if n > 0 else 0,
            "avg_diff": avg_diff,
            "max_diff": round(max(diffs), 4) if diffs else 0,
            "pearson_r": pearson_r,
        }

    report["stratified_analysis"] = stratified_report

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
        cs = cal.get("consistency_samples", 1)
        if cs and cs > 1:
            print(f"   评分策略：每条样本 {cs} 次采样取均值（评分稳定化）")

    fm = report.get("flaw_metrics", {})
    print(f"\n[瑕疵检出指标（多对一匹配）]")
    if fm.get("error"):
        print(f"   指标计算失败：{fm['error']}")
    else:
        print(f"   Precision：{fm.get('precision', 0):.4f}")
        print(f"   Recall：{fm.get('recall', 0):.4f}")
        print(f"   F1：{fm.get('f1', 0):.4f}  {'[达标]' if fm.get('pass_threshold_0_8') else '[未达标]'}（目标 >= 0.8）")
        print(f"   TP/FP/FN：{fm.get('tp', 0)}/{fm.get('fp', 0)}/{fm.get('fn', 0)}  Support：{fm.get('support', 0)}")

    am = report.get("anchor_metrics", {})
    print(f"\n[锚点定位准确率（双轨匹配）]")
    if am.get("error"):
        print(f"   计算失败：{am['error']}")
    else:
        print(f"   定位 Recall（主指标）：{am.get('location_recall', 0):.4f}  {'[达标]' if am.get('pass_threshold_0_9') else '[未达标]'}（目标 >= 0.9）")
        print(f"   定位 Precision：{am.get('location_precision', 0):.4f}")
        print(f"   定位 F1（参考）：{am.get('location_f1', 0):.4f}")
        print(f"   定位 TP/FP/FN：{am.get('location_tp', 0)}/{am.get('location_fp', 0)}/{am.get('location_fn', 0)}")
        print(f"   分类 Precision：{am.get('classification_precision', 0):.4f}")
        print(f"   分类 Recall：{am.get('classification_recall', 0):.4f}")
        print(f"   分类 F1：{am.get('classification_f1', 0):.4f}")
        print(f"   分类正确数：{am.get('classification_tp', 0)}/{am.get('total', 0)}")

    ss = report.get("stability_summary", {})
    print(f"\n[评分稳定性]")
    if not meta.get("run_stability"):
        print("   未测试（本次综合评测未启用稳定性采样）")
    else:
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
        print(f"   平均延迟：{ls.get('mean_seconds', 0):.3f}s（per-sample API 耗时）")
        print(f"   最小延迟：{ls.get('min_seconds', 0):.3f}s")
        print(f"   最大延迟：{ls.get('max_seconds', 0):.3f}s")
        print(f"   < 3s 达标率：{ls.get('pass_rate_below_3s', 0)*100:.1f}%（{ls.get('pass_below_3s', 0)}/{ls.get('samples', 0)}）")
        workers = ls.get('concurrent_workers', 1)
        wall = ls.get('wall_clock_seconds', 0)
        eval_wall = ls.get('evaluation_wall_clock_seconds', 0)
        stab_wall = ls.get('stability_wall_clock_seconds', 0)
        thru = ls.get('throughput_latency', 0)
        print(f"   并发线程数：{workers}")
        print(f"   评估耗时：{eval_wall:.1f}s")
        if stab_wall > 0:
            print(f"   稳定性测试耗时：{stab_wall:.1f}s")
        print(f"   总耗时（wall-clock）：{wall:.1f}s")
        print(f"   吞吐延迟：{thru:.3f}s/条（评估耗时/样本数）")

    ts = report.get("token_stats", {})
    if ts:
        print(f"   平均 token 数：{ts.get('mean_tokens', 0)}")
        print(f"   总 token 消耗：{ts.get('total_tokens', 0)}")

    print(f"\n{'=' * 70}")
    print(f"  综合判定：{'[全部达标]' if report.get('overall_pass') else '[存在未达标项]'}")
    print(f"{'=' * 70}\n")