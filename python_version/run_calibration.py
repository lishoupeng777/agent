"""校准测试脚本 v2 —— 使用 app.py 内置的评估逻辑（绕过 debias.py 的编码问题）"""
import json, sys, os, re
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---- 复用 app.py 中的评估调用（不依赖 debias 模块） ----
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

JUDGE_SYSTEM_PROMPT = r"""你是一名拥有10年以上经验的资深数据质量与文本治理审计专家。你的任务是严格比对"治理前"和"治理后"的文本对。

**重要：你必须严格审计，不要心软。有错必扣，有多处严重错误必须给出极低分。**
**参考以下评分锚点：完全正确=1.0，存在关键事实错误=0.3以下，大量信息丢失=0.2以下。**

请严格遵循以下规则进行审计与打分：

1. 语义一致性评分 (semantic_consistency_score) - 总分 1.0
   - 严重瑕疵（每次扣 0.5-0.6分）：误删、误改了核心事实、实体或关键量化指标（如金额、百分比、日期、人名等）。
   - 中等瑕疵（每次扣 0.25-0.35分）：语义偏离，误改修饰词或特定业务专有名词。
   - 轻微瑕疵（每次扣 0.1-0.15分）：非必要的同义词替换，语意没变但属于过度清洗。
   - **累计扣分规则**：多个瑕疵累计扣分。如果存在2个以上的严重瑕疵，总分不得超过0.3。如果关键数据被错误修改，总分不得超过0.4。如果原文核心信息被大幅删除（压缩至一句话），总分不得超过0.2。

2. 可读性与结构质量评分 (readability_structure_score) - 总分 1.0
   - 严重瑕疵（每次扣 0.4-0.5分）：排版结构被破坏（如表格完全散架），导致无法阅读。
   - 中等瑕疵（每次扣 0.2-0.25分）：语句不通顺，或表格对齐混乱。
   - **加分项**：如果格式比原文更规范、标点/空格修正合理，可加 0.05 分（加分后不超过1.0）。

3. 瑕疵锚点定位要求 (Anchored Localization)
   发现瑕疵时，必须指出是原文本哪一行变化到了后文本哪一行（例如: [Before X] -> [After Y]）。

4. 瑕疵类型必须严格限定为以下四种之一：过度清洗 / 语义误改 / 可读性下降 / 结构破坏
5. 严重程度必须严格限定为以下三种之一：High / Medium / Low

输出要求：
- 你必须严格且仅输出一个 JSON 对象，不要使用代码块包裹。
- 字段名必须与下方 Schema 完全一致。

期望输出的 JSON 格式：
{
  "semantic_consistency_score": 0.0到1.0的浮点数,
  "readability_structure_score": 0.0到1.0的浮点数,
  "flaws": [
    {
      "anchor": "如 [Before X] -> [After Y]",
      "type": "过度清洗 | 语义误改 | 可读性下降 | 结构破坏",
      "severity": "High | Medium | Low",
      "detail": "具体说明瑕疵细节"
    }
  ],
  "overall_explanation": "一句话总结本次治理比对的总体质量和考量"
}
"""

JUDGE_USER_TEMPLATE = """请严格比对并评估以下治理前后的文本对：

【治理前文本 (Original)】
{before_anchored}

【治理后文本 (Governed)】
{after_anchored}
"""


def build_anchored_text(text: str, prefix: str) -> str:
    lines = text.split("\n")
    anchored_lines = [f"[{prefix} {i}] {line}" for i, line in enumerate(lines, 1)]
    return "\n".join(anchored_lines)


def evaluate_single(before_text: str, after_text: str) -> dict:
    """单次评估（独立版，不依赖 debias）"""
    before_anchored = build_anchored_text(before_text, "Before")
    after_anchored = build_anchored_text(after_text, "After")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com/v1",
        temperature=0.0,
        max_retries=3,
        timeout=120,
    )

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=JUDGE_USER_TEMPLATE.format(
            before_anchored=before_anchored,
            after_anchored=after_anchored,
        )),
    ]

    raw = llm.invoke(messages)
    raw_text = str(raw.content) if hasattr(raw, "content") else str(raw)

    # 清洗 think 标签
    clean = re.sub(r"<think[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()

    # 提取 JSON
    json_str = clean
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
    if m:
        json_str = m.group(1).strip()
    brace_start = json_str.find("{")
    brace_end = json_str.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        json_str = json_str[brace_start:brace_end + 1]

    result = json.loads(json_str)
    result.setdefault("semantic_consistency_score", 0.0)
    result.setdefault("readability_structure_score", 0.0)
    result.setdefault("flaws", [])
    result.setdefault("overall_explanation", "")
    return result


# ═══════════════════════════════════════════════════════════════════════
# 加载数据集
# ═══════════════════════════════════════════════════════════════════════
from dotenv import load_dotenv
load_dotenv()

with open(PROJECT_ROOT / 'data' / 'eval_dataset.json', 'r', encoding='utf-8') as f:
    samples = json.load(f)

print("=" * 70)
print(f"  校准测试 v2 - 共 {len(samples)} 条样本")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════
# 逐条评估
# ═══════════════════════════════════════════════════════════════════════
llm_scores = []
human_scores = []
details = []

print("\n逐条评估中（每条调用 DeepSeek API，约 5-15 秒）...")
for i, s in enumerate(samples):
    sid = s['id']
    label = s.get('label', '?')
    human = s.get('human_score', 0.5)

    print(f"  [{i+1}/{len(samples)}] {sid} ({label}) ... ", end="", flush=True)

    try:
        result = evaluate_single(s['before_text'], s['after_text'])
        # overall_score = 语义和可读性的加权平均（各 0.5）
        sem = result.get('semantic_consistency_score', 0.0)
        read = result.get('readability_structure_score', 0.0)
        llm = round((sem + read) / 2, 4)
    except Exception as e:
        print(f"ERROR: {e}")
        sem = 0.0
        read = 0.0
        llm = 0.0

    diff = abs(llm - human)
    consistent = diff <= 0.15

    print(f"LLM={llm:.3f} Human={human:.3f} |diff|={diff:.3f} {'OK' if consistent else 'GAP'}")

    llm_scores.append(llm)
    human_scores.append(human)
    details.append({
        "request_id": sid,
        "label": label,
        "llm_score": llm,
        "human_score": human,
        "semantic": sem,
        "readability": read,
        "diff": round(diff, 4),
        "consistent": consistent,
    })

# ═══════════════════════════════════════════════════════════════════════
# 计算校准指标
# ═══════════════════════════════════════════════════════════════════════
arr_llm = np.array(llm_scores)
arr_human = np.array(human_scores)
n = len(arr_llm)

# Pearson r
from scipy.stats import pearsonr, spearmanr

if n >= 3 and np.std(arr_llm) > 1e-8 and np.std(arr_human) > 1e-8:
    pr, pp = pearsonr(arr_llm, arr_human)
else:
    pr = 0.0

# Spearman
if n >= 3:
    sr, sp = spearmanr(arr_llm, arr_human)
else:
    sr = 0.0

mae = float(np.mean(np.abs(arr_llm - arr_human)))
rmse = float(np.sqrt(np.mean((arr_llm - arr_human) ** 2)))
consistent_count = sum(1 for d in details if d["consistent"])
consistency_rate = consistent_count / n if n > 0 else 0.0

# ═══════════════════════════════════════════════════════════════════════
# 输出结果
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  校准结果汇总")
print("=" * 70)
print(f"  Pearson r      = {pr:.4f}  {'✅ 达标 (>=0.8)' if pr >= 0.8 else '❌ 未达标 (需调 Prompt)'}")
print(f"  Spearman rho   = {sr:.4f}")
print(f"  MAE            = {mae:.4f}")
print(f"  RMSE           = {rmse:.4f}")
print(f"  一致率         = {consistency_rate:.2%}  (|diff| <= 0.15)")
print(f"  样本数         = {n}")

print("\n  逐条详情：")
for d in details:
    arrow = "✅" if d['consistent'] else "⚠️"
    print(f"  {arrow} {d['request_id']:12s} ({d['label']:10s})  "
          f"LLM={d['llm_score']:.3f}  Human={d['human_score']:.3f}  "
          f"Sem={d['semantic']:.2f}  Read={d['readability']:.2f}  "
          f"|diff|={d['diff']:.3f}")

# 保存结果
output = {
    "calibration": {
        "pearson_r": round(pr, 4),
        "spearman_rho": round(sr, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "consistency_rate": round(consistency_rate, 4),
        "sample_count": n,
    },
    "details": details,
}
with open(PROJECT_ROOT / 'data' / 'calibration_result.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存到 data/calibration_result.json")
print("Done!")