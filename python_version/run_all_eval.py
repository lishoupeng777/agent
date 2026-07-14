"""一次性跑完三个数据集的全量评测"""
from __future__ import annotations
import json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from app.models import EvalRequest
from app.reporter import run_full_evaluation, print_report_summary, export_report_json


def build_eval_requests(dataset_path: str) -> list[EvalRequest]:
    """eval_dataset 格式"""
    data = json.load(open(dataset_path, encoding="utf-8"))
    requests = []
    for item in data:
        if "human_scores" in item:
            w = {"semantic":0.30,"factual":0.30,"hallucination":0.20,"structure":0.10,"readability":0.10}
            overall = round(sum(item["human_scores"].get(k,0.5)*v for k,v in w.items()),4)
            hl = {"overall_score": overall, "label": item["label"],
                  "flaws": item.get("flaws_gt",[]), "dimension_scores": item["human_scores"],
                  "rationale": item.get("rationale",""), "difficulty": item.get("difficulty","medium")}
        else:
            hl = {"overall_score": item["human_score"], "label": item["label"],
                  "flaws": item.get("flaws_gt",[])}
        requests.append(EvalRequest(request_id=item["id"], before_text=item["before_text"],
                                     after_text=item["after_text"], human_label=hl))
    return requests


def build_gold_requests(dataset_path: str) -> list[EvalRequest]:
    """gold v1/v2 格式"""
    DIM_MAP = {"semantic_fidelity":"semantic","factual_consistency":"factual",
               "hallucination":"hallucination","structure":"structure","readability":"readability"}
    data = json.load(open(dataset_path, encoding="utf-8"))
    requests = []
    for item in data:
        overall = item.get("overall",{})
        dim_scores = {DIM_MAP.get(d["dimension"],d["dimension"]):d["score"]
                      for d in item.get("dimensions",[]) if d.get("dimension") in DIM_MAP}
        hl = {"overall_score": overall.get("weighted_score",0.5), "label": overall.get("verdict","review"),
              "flaws": item.get("flaws",[]), "dimension_scores": dim_scores}
        requests.append(EvalRequest(request_id=item["id"], before_text=item["before_text"],
                                     after_text=item["after_text"], human_label=hl))
    return requests


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    jobs = [
        ("eval_dataset", os.path.join(base,"data","eval_dataset.json"), build_eval_requests, True),
        ("gold_v1", os.path.join(base,"data","gold_dataset_v1.json"), build_gold_requests, False),
        ("gold_v2", os.path.join(base,"data","gold_dataset_v2.json"), build_gold_requests, False),
    ]

    for name, path, builder, run_stab in jobs:
        print("\n" + "="*60)
        print(f"  开始评测: {name}")
        print("="*60)
        t0 = time.time()
        requests = builder(path)
        report = run_full_evaluation(requests, run_stability=run_stab, stability_samples=3, consistency_samples=3)
        elapsed = round(time.time()-t0, 1)
        print_report_summary(report)
        out = os.path.join(base, f"{name}_eval_report.json")
        export_report_json(report, out)
        print(f"[报告已导出] {out}  耗时 {elapsed}s")

    print("\n" + "="*60)
    print("  三个数据集全部评测完成")
    print("="*60)


if __name__ == "__main__":
    main()
