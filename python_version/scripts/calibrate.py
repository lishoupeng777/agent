"""校准脚本：用 eval_dataset.json 拟合 LLM → 人工 的线性校准参数。

运行方式：python scripts/calibrate.py
输出：data/calibration_params.json
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from app.models import EvalRequest
from app.engine import evaluate
from app.calibrator import ScoreCalibrator


def compute_overall(hs: dict) -> float:
    weights = {"semantic": 0.35, "factual": 0.35, "structure": 0.15, "readability": 0.15}
    return round(sum(hs.get(k, 0.5) * w for k, w in weights.items()), 4)


def main():
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "eval_dataset.json",
    )
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "calibration_params.json",
    )

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"加载 {len(dataset)} 条校准样本")

    raw_scores = []
    gold_scores = []

    for i, item in enumerate(dataset):
        hs = item["human_scores"]
        human_overall = compute_overall(hs)

        req = EvalRequest(
            request_id=item["id"],
            before_text=item["before_text"],
            after_text=item["after_text"],
            human_label={"overall_score": human_overall},
        )

        try:
            resp = evaluate(req, temperature=0.0)
            llm_score = resp.overall_score
            raw_scores.append(llm_score)
            gold_scores.append(human_overall)
            diff = abs(llm_score - human_overall)
            mark = "OK" if diff <= 0.1 else "MISS"
            print(f"  [{i+1}/{len(dataset)}] {item['id']}: LLM={llm_score:.4f} 人工={human_overall:.4f} 差={diff:.4f} [{mark}]")
        except Exception as e:
            print(f"  [{i+1}/{len(dataset)}] {item['id']}: ERROR - {e}")

    if len(raw_scores) < 3:
        print(f"有效样本不足（{len(raw_scores)}），无法拟合")
        sys.exit(1)

    # 拟合线性回归
    cal = ScoreCalibrator("deepseek-v4-flash")
    cal.fit(raw_scores, gold_scores)

    print(f"\n=== 校准参数 ===")
    print(f"  slope:     {cal.slope:.6f}")
    print(f"  intercept: {cal.intercept:.6f}")
    print(f"  R²:        {cal.r_squared:.4f}")
    print(f"  样本数:    {cal.n_samples}")

    # 验证校准效果
    calibrated_scores = cal.calibrate_batch(raw_scores)
    ok_before = sum(1 for r, g in zip(raw_scores, gold_scores) if abs(r - g) <= 0.1)
    ok_after = sum(1 for c, g in zip(calibrated_scores, gold_scores) if abs(c - g) <= 0.1)
    print(f"\n=== 校准效果 ===")
    print(f"  校准前一致率: {ok_before}/{len(raw_scores)} = {ok_before*100//len(raw_scores)}%")
    print(f"  校准后一致率: {ok_after}/{len(raw_scores)} = {ok_after*100//len(raw_scores)}%")

    # 保存
    cal.save(out_path)
    print(f"\n[已保存] {out_path}")


if __name__ == "__main__":
    main()
