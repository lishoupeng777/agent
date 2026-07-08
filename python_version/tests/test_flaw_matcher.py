"""双轨匹配引擎测试

验证：定位能力与分类能力正确解耦。
场景：LLM 和人对同一个瑕玼的分类不同（mis_edit vs over_clean），
      但锚点一致 → 定位准确率应为 100%，分类准确率应较低。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.flaw_matcher import dual_track_evaluate


def test_category_mismatch_but_location_correct():
    """场景：分类不一致但定位正确"""
    predicted = [
        {
            "category": "mis_edit",
            "severity": "critical",
            "description": "将'严禁饮酒'改为'建议避免饮酒'",
            "location": {
                "before_anchor": "[Anchor_B3]",
                "after_anchor": "[Anchor_A3]",
                "start_char": 0,
                "end_char": 0,
                "snippet": "严禁饮酒或饮用含有酒精的饮料",
            },
        },
        {
            "category": "mis_edit",
            "severity": "major",
            "description": "将'禁用'改为'需遵医嘱'",
            "location": {
                "before_anchor": "[Anchor_B4]",
                "after_anchor": "[Anchor_A3]",
                "start_char": 0,
                "end_char": 0,
                "snippet": "严重肝肾功能不全者禁用本品",
            },
        },
    ]

    ground_truth = [
        {
            "category": "over_clean",
            "severity": "critical",
            "description": "安全警告被弱化",
            "location": {
                "before_anchor": "[Anchor_B3]",
                "after_anchor": "[Anchor_A3]",
                "start_char": 131,
                "end_char": 169,
                "snippet": "【严禁饮酒或饮用含有酒精的饮料】（否则极易导致急性肝衰竭甚至死亡！）",
            },
        },
        {
            "category": "mis_edit",
            "severity": "critical",
            "description": "禁用被篡改为遵医嘱",
            "location": {
                "before_anchor": "[Anchor_B4]",
                "after_anchor": "[Anchor_A3]",
                "start_char": 185,
                "end_char": 210,
                "snippet": "孕妇、哺乳期妇女以及严重肝肾功能不全者禁用本品",
            },
        },
    ]

    result = dual_track_evaluate(predicted, ground_truth)

    print("=" * 60)
    print("测试：分类不一致但定位正确")
    print("=" * 60)
    print()
    print("预测瑕玼:")
    for i, p in enumerate(predicted):
        print(f"  [{i}] {p['category']} @ {p['location']['after_anchor']}")
    print()
    print("人工标注:")
    for i, g in enumerate(ground_truth):
        print(f"  [{i}] {g['category']} @ {g['location']['after_anchor']}")
    print()

    print("--- 定位指标（不要求分类一致）---")
    print(f"  Precision: {result.location_precision:.4f}")
    print(f"  Recall:    {result.location_recall:.4f}")
    print(f"  F1:        {result.location_f1:.4f}")
    print(f"  TP/FP/FN:  {result.location_tp}/{result.location_fp}/{result.location_fn}")
    print()

    print("--- 分类指标（定位成功后再检查分类）---")
    print(f"  Precision: {result.classification_precision:.4f}")
    print(f"  Recall:    {result.classification_recall:.4f}")
    print(f"  F1:        {result.classification_f1:.4f}")
    print(f"  分类正确数: {result.classification_tp}/{result.location_tp}")
    print()

    print("--- 匹配详情 ---")
    for m in result.matches:
        cat_status = "分类一致" if m.category_match else "分类不一致"
        print(f"  pred[{m.pred_idx}] -> gt[{m.gt_idx}]  "
              f"score={m.score:.4f}  anchor={m.anchor_score:.1f}  "
              f"text={m.text_score:.4f}  cat={m.category_score:.1f}  [{cat_status}]")
    print()

    # 验证
    assert result.location_f1 == 1.0, f"定位 F1 应为 1.0，实际 {result.location_f1}"
    assert result.classification_tp == 1, f"分类正确数应为 1，实际 {result.classification_tp}"
    print("[OK] 测试通过：定位准确率 100%，分类准确率正确反映不一致")


def test_perfect_match():
    """场景：完全匹配"""
    predicted = [
        {
            "category": "over_clean",
            "severity": "critical",
            "description": "删除安全警告",
            "location": {
                "before_anchor": "[Anchor_B2]",
                "after_anchor": "[Anchor_A2]",
                "start_char": 10,
                "end_char": 30,
                "snippet": "断开电源并静置至少5分钟",
            },
        },
    ]

    ground_truth = [
        {
            "category": "over_clean",
            "severity": "critical",
            "description": "安全步骤被删除",
            "location": {
                "before_anchor": "[Anchor_B2]",
                "after_anchor": "[Anchor_A2]",
                "start_char": 15,
                "end_char": 35,
                "snippet": "断开电源并静置至少5分钟（非常关键）",
            },
        },
    ]

    result = dual_track_evaluate(predicted, ground_truth)

    print("=" * 60)
    print("测试：完全匹配")
    print("=" * 60)
    print(f"  定位 F1: {result.location_f1:.4f}")
    print(f"  分类 F1: {result.classification_f1:.4f}")
    assert result.location_f1 == 1.0
    assert result.classification_f1 == 1.0
    print("[OK] 测试通过")


def test_empty():
    """场景：空输入"""
    result = dual_track_evaluate([], [])
    assert result.location_f1 == 1.0
    assert result.classification_f1 == 1.0
    print("[OK] 空输入测试通过")

    result2 = dual_track_evaluate([{"category": "x", "location": {}}], [])
    assert result2.location_fp == 1
    print("[OK] 有预测无标注测试通过")


if __name__ == "__main__":
    test_category_mismatch_but_location_correct()
    print()
    test_perfect_match()
    print()
    test_empty()
    print()
    print("=" * 60)
    print("所有测试通过")
    print("=" * 60)
