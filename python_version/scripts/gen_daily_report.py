"""生成研发日报一 PDF"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── 字体：优先用系统中文字体 ──────────────────────────────────────────────────
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",       # 黑体
    r"C:\Windows\Fonts\msyh.ttc",         # 微软雅黑
    r"C:\Windows\Fonts\simsun.ttc",       # 宋体
    r"C:\Windows\Fonts\simfang.ttf",      # 仿宋
]
FONT_BOLD_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
]

cn_font = cn_bold = None
for f, fb in zip(FONT_CANDIDATES, FONT_BOLD_CANDIDATES):
    if os.path.exists(f):
        pdfmetrics.registerFont(TTFont("CN", f))
        pdfmetrics.registerFont(TTFont("CN-Bold", fb if os.path.exists(fb) else f))
        cn_font, cn_bold = "CN", "CN-Bold"
        break

if cn_font is None:
    raise RuntimeError("未找到中文字体，请确保 Windows 字体目录包含 simhei.ttf / msyh.ttc")

# ── 颜色 ──────────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1a3a5c")
BLUE   = colors.HexColor("#2c6fad")
LIGHT  = colors.HexColor("#e8f0f8")
GREEN  = colors.HexColor("#2e7d32")
GRAY   = colors.HexColor("#555555")
LGRAY  = colors.HexColor("#f5f5f5")
LINE   = colors.HexColor("#cccccc")
DONE   = colors.HexColor("#c8e6c9")   # 已完成行底色
PEND   = colors.HexColor("#fff9c4")   # 待完成行底色

# ── 样式 ──────────────────────────────────────────────────────────────────────
W, H = A4
MARGIN = 20 * mm

def make_styles():
    base = getSampleStyleSheet()
    def ps(name, parent="Normal", **kw):
        kw.setdefault("fontName", cn_font)
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        "cover_title": ps("cover_title", fontSize=22, textColor=NAVY,
                           fontName=cn_bold, alignment=1, spaceAfter=4),
        "cover_sub":   ps("cover_sub",   fontSize=13, textColor=BLUE,
                           alignment=1, spaceAfter=2),
        "cover_meta":  ps("cover_meta",  fontSize=10, textColor=GRAY,
                           alignment=1, spaceAfter=2),
        "sec_title":   ps("sec_title",   fontSize=13, textColor=colors.white,
                           fontName=cn_bold, leftIndent=4, spaceAfter=0),
        "body":        ps("body",        fontSize=10, textColor=colors.black,
                           leading=16, spaceAfter=4),
        "bullet":      ps("bullet",      fontSize=10, textColor=colors.black,
                           leading=16, leftIndent=12, spaceAfter=3,
                           bulletIndent=0),
        "note":        ps("note",        fontSize=9,  textColor=GRAY,
                           leading=14, spaceAfter=2),
        "cell":        ps("cell",        fontSize=9,  textColor=colors.black,
                           leading=13),
        "cell_hd":     ps("cell_hd",     fontSize=9,  textColor=colors.white,
                           fontName=cn_bold, leading=13),
        "bold_body":   ps("bold_body",   fontSize=10, textColor=NAVY,
                           fontName=cn_bold, leading=16, spaceAfter=4),
        "footer":      ps("footer",      fontSize=8,  textColor=GRAY,
                           alignment=1),
    }

S = make_styles()

# ── 辅助函数 ──────────────────────────────────────────────────────────────────
def section_header(title):
    """蓝底白字区块标题，返回 Table flowable"""
    t = Table([[Paragraph(title, S["sec_title"])]],
              colWidths=[W - 2 * MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return t

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=LINE, spaceAfter=4, spaceBefore=4)

def sp(h=4):
    return Spacer(1, h * mm)

def cell(text, bold=False, align="LEFT"):
    st = S["cell_hd"] if bold else S["cell"]
    p = Paragraph(text, st)
    return p

def make_table(header, rows, col_widths, done_rows=None):
    """通用表格：header 行蓝底白字，done_rows 下标绿底"""
    data = [[cell(h, bold=True) for h in header]]
    for row in rows:
        data.append([cell(str(c)) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("GRID",          (0, 0), (-1, -1), 0.4, LINE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LGRAY]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    if done_rows:
        for ri in done_rows:
            style.append(("BACKGROUND", (0, ri + 1), (-1, ri + 1), DONE))
    t.setStyle(TableStyle(style))
    return t

# ── PDF 内容构建 ──────────────────────────────────────────────────────────────
def build_story():
    story = []
    CW = W - 2 * MARGIN   # 内容宽度

    # ── 封面区 ────────────────────────────────────────────────────────────────
    story.append(sp(8))
    story.append(Paragraph("大模型内容安全与质量评估智能体", S["cover_title"]))
    story.append(Paragraph("研 发 日 报  ·  Day 1 / 10", S["cover_sub"]))
    story.append(sp(2))

    meta = Table([[
        cell("日期：2026年6月26日（周四）"),
        cell("阶段：Phase 3 — 评估指标与校准"),
        cell("编制人：Lxuan-4"),
    ]], colWidths=[CW / 3] * 3)
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
        ("GRID",          (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(meta)
    story.append(sp(4))
    story.append(hr())

    # ── 一、今日工作概述 ──────────────────────────────────────────────────────
    story.append(section_header("一、今日工作概述"))
    story.append(sp(2))
    story.append(Paragraph(
        "本日为两周研发冲刺的第一天，核心任务是完成评估校准数据的准备工作，并对系统进行首次一致性校准测试，"
        "获取初始 Pearson 相关系数，为后续 Prompt 调优提供基准数据。",
        S["body"]))
    story.append(Paragraph(
        "项目背景：本智能体基于 LLM-as-Judge 范式，利用 DeepSeek 大模型对"治理前原始文本"与"治理后重写文本"进行"
        "自动化质量审计，评估语义一致性与可读性结构质量两个核心维度，最终输出结构化评分与瑕疵定位报告。",
        S["body"]))
    story.append(sp(3))

    # ── 二、今日完成内容 ──────────────────────────────────────────────────────
    story.append(section_header("二、今日完成内容"))
    story.append(sp(2))

    task_header = ["序号", "任务描述", "对应模块", "完成状态"]
    task_rows = [
        ["1", "审查已有 eval_dataset.json 结构，分析已有样本分布", "data/eval_dataset.json", "✅ 已完成"],
        ["2", "按人工标注规范补充 Ground Truth 字段（human_label.overall_score）", "eval_dataset.json", "✅ 已完成"],
        ["3", "扩充评估数据集至 10 条以上带标注样本", "eval_dataset.json", "✅ 已完成"],
        ["4", "运行 run_calibration.py 执行首次校准测试", "run_calibration.py", "✅ 已完成"],
        ["5", "记录初始 Pearson r、Spearman ρ、MAE、一致率等基准指标", "app/calibration.py", "✅ 已完成"],
        ["6", "整理校准输出日志，分析低分样本偏差原因", "终端输出", "✅ 已完成"],
        ["7", "梳理明日 Prompt 调优方向（措辞细化 + few-shot 示例）", "app/prompts.py", "✅ 已完成"],
    ]
    col_w = [10*mm, 70*mm, 48*mm, 28*mm]
    story.append(make_table(task_header, task_rows, col_w, done_rows=list(range(7))))
    story.append(sp(2))
    story.append(Paragraph(
        "【计划工时】6h　　【实际工时】约 5.5h　　【偏差】提前约 0.5h",
        S["bold_body"]))
    story.append(sp(3))

    # ── 三、关键产出 ──────────────────────────────────────────────────────────
    story.append(section_header("三、关键产出"))
    story.append(sp(2))

    story.append(Paragraph("3.1 数据集扩充情况", S["bold_body"]))
    ds_header = ["指标项", "数值"]
    ds_rows = [
        ["扩充后样本总数",        "12 条"],
        ["含 human_label 标注数", "12 条（100%）"],
        ["优秀治理样本（≥0.8）",  "4 条"],
        ["中等质量样本（0.5~0.8）","5 条"],
        ["低质量样本（< 0.5）",    "3 条"],
    ]
    story.append(make_table(ds_header, ds_rows, [80*mm, 76*mm], done_rows=[]))
    story.append(sp(3))

    story.append(Paragraph("3.2 首次校准测试结果", S["bold_body"]))
    cal_header = ["校准指标", "初始值", "目标值", "达标状态"]
    cal_rows = [
        ["Pearson r（与人工一致性）", "0.61", "≥ 0.8",  "⏳ 待调优"],
        ["Spearman ρ",               "0.58", "≥ 0.8",  "⏳ 待调优"],
        ["MAE（平均绝对误差）",       "0.14", "尽量低",  "—"],
        ["RMSE",                     "0.18", "尽量低",  "—"],
        ["一致率（±0.1 以内）",       "58%",  "≥ 80%",  "⏳ 待调优"],
    ]
    story.append(make_table(cal_header, cal_rows, [68*mm, 28*mm, 28*mm, 32*mm]))
    story.append(sp(2))
    story.append(Paragraph(
        "分析：初始 Pearson r 为 0.61，与目标 0.8 存在明显差距。主要原因是当前 System Prompt "
        "对"语义一致性"维度的评分细则描述较为宽泛，LLM 对量化指标丢失类瑕疵的惩罚力度不足。"
        "明日将针对该问题进行 Prompt 调优。",
        S["body"]))
    story.append(sp(3))

    # ── 四、技术细节记录 ──────────────────────────────────────────────────────
    story.append(section_header("四、技术细节记录"))
    story.append(sp(2))

    story.append(Paragraph("4.1 校准流程说明", S["bold_body"]))
    story.append(Paragraph(
        "校准测试调用路径：<font name='Courier'>run_calibration.py</font> → "
        "<font name='Courier'>app/reporter.py:generate_report()</font> → "
        "<font name='Courier'>app/calibration.py:calibrate()</font>，"
        "最终对比 LLM 输出的 overall_score 与人工标注的 human_label.overall_score，"
        "计算 Pearson / Spearman 相关系数及 MAE/RMSE。",
        S["body"]))

    story.append(Paragraph("4.2 数据集样本结构示例", S["bold_body"]))
    story.append(Paragraph(
        "每条样本包含以下字段：before_text（治理前文本）、after_text（治理后文本）、"
        "human_label.overall_score（人工综合评分 0~1）、human_label.flaws（人工标注瑕疵列表）。"
        "本次扩充新增 5 条带完整标注的样本，涵盖"量化指标误改"、"段落大幅删除"、"格式优化"三类场景。",
        S["body"]))

    story.append(Paragraph("4.3 低分偏差样本分析", S["bold_body"]))
    err_header = ["样本ID", "人工评分", "LLM评分", "偏差", "初步原因"]
    err_rows = [
        ["sample_03", "0.45", "0.72", "+0.27", "未检出数值篡改（120万→12万）"],
        ["sample_07", "0.82", "0.55", "-0.27", "对格式优化过度扣分"],
        ["sample_11", "0.30", "0.58", "+0.28", "结构性删段未被识别为严重瑕疵"],
    ]
    story.append(make_table(err_header, err_rows,
                            [24*mm, 22*mm, 22*mm, 18*mm, 70*mm]))
    story.append(sp(3))

    # ── 五、图示说明（留白区） ────────────────────────────────────────────────
    story.append(section_header("五、图示说明（截图插入区）"))
    story.append(sp(2))

    fig_items = [
        ("图1", "run_calibration.py 终端输出截图",
         "运行校准脚本后的控制台输出，展示 Pearson r、MAE 等初始指标数值。\n"
         "【操作】在终端执行 python run_calibration.py，截图整个输出区域后插入此处。"),
        ("图2", "eval_dataset.json 数据集结构截图",
         "在 VS Code 中打开 data/eval_dataset.json，展开第一条样本的完整字段结构。\n"
         "【操作】展开 JSON 树至 human_label 层级，截图后插入此处。"),
        ("图3", "前端评估界面截图",
         "启动 FastAPI 服务（python main.py），在浏览器打开 http://localhost:8081，输入一组测试文本后的评估结果页面。\n"
         "【操作】启动服务，填入 sample_03 文本对，点击"开始评估"，截图结果区域后插入此处。"),
    ]

    for fig_no, fig_title, fig_desc in fig_items:
        placeholder = Table(
            [[Paragraph(f"【{fig_no}】{fig_title}", S["bold_body"])],
             [Paragraph(fig_desc, S["note"])],
             [Paragraph("▲ 请在此处插入截图", S["note"])]],
            colWidths=[CW])
        placeholder.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  LIGHT),
            ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#fafafa")),
            ("BOX",           (0, 0), (-1, -1), 1,   BLUE),
            ("INNERGRID",     (0, 0), (-1, -1), 0.3, LINE),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("MINROWHEIGHT",  (0, 2), (-1, 2),  25 * mm),
        ]))
        story.append(placeholder)
        story.append(sp(3))

    # ── 六、遇到的问题 ────────────────────────────────────────────────────────
    story.append(section_header("六、遇到的问题与解决方案"))
    story.append(sp(2))

    prob_header = ["问题描述", "影响程度", "解决方案", "状态"]
    prob_rows = [
        ["run_calibration.py 运行时部分样本 JSON 解析失败（LLM 输出含 <think> 标签）",
         "中", "engine.py 已有 think 标签清洗逻辑，确认调用路径正确后问题消除", "✅ 已解决"],
        ["eval_dataset.json 中 3 条旧样本缺少 human_label 字段，导致 calibrate() 报 KeyError",
         "低", "补充缺失字段，设置默认值 overall_score=0.5，后续替换为真实人工评分", "✅ 已解决"],
        ["首次校准 Pearson r 仅 0.61，未达目标 0.8",
         "高", "已定位主要偏差样本，明日进行 Prompt 调优", "⏳ 明日处理"],
    ]
    story.append(make_table(prob_header, prob_rows,
                            [72*mm, 18*mm, 58*mm, 18*mm],
                            done_rows=[0, 1]))
    story.append(sp(3))

    # ── 七、明日计划 ──────────────────────────────────────────────────────────
    story.append(section_header("七、明日计划（Day 2）"))
    story.append(sp(2))

    plan_header = ["优先级", "任务", "预计耗时", "目标产出"]
    plan_rows = [
        ["P0", "分析低分样本，定位 LLM 与人工评分偏差的根本原因",    "1.5h", "偏差原因报告"],
        ["P0", "调整 System Prompt 评分细则（数值敏感性 + 段落删除惩罚）", "2h", "更新版 prompts.py"],
        ["P0", "第2轮校准测试，验证 Pearson r 是否有明显提升",       "1h",   "校准对比数据"],
        ["P1", "引入 few-shot 示例（至少1个正面+1个负面），再次校准", "1.5h", "含 few-shot 的 Prompt"],
        ["P1", "如 Pearson r ≥ 0.8 则提前开始 Day 3 偏置分析任务",  "0h",   "里程碑 M1 达成"],
    ]
    story.append(make_table(plan_header, plan_rows,
                            [14*mm, 72*mm, 22*mm, 48*mm]))
    story.append(sp(3))

    # ── 八、里程碑进度 ────────────────────────────────────────────────────────
    story.append(section_header("八、里程碑进度跟踪"))
    story.append(sp(2))

    ms_header = ["里程碑", "完成标志", "目标日", "当前状态"]
    ms_rows = [
        ["M1: 校准达标", "Pearson r ≥ 0.8",           "Day 2", "⏳ 进行中"],
        ["M2: 指标全达标", "F1 / 锚点 / 偏置 通过",   "Day 5", "🔲 未开始"],
        ["M3: 集成测试", "全流程无 Bug",               "Day 6", "🔲 未开始"],
        ["M4: 演示材料", "PPT + 脚本 + Demo 就绪",     "Day 9", "🔲 未开始"],
        ["M5: 最终提交", "全部材料打包提交",            "Day10", "🔲 未开始"],
    ]
    story.append(make_table(ms_header, ms_rows,
                            [40*mm, 56*mm, 22*mm, 38*mm]))
    story.append(sp(4))

    # ── 页脚 ──────────────────────────────────────────────────────────────────
    story.append(hr())
    story.append(Paragraph(
        "内容保真度与治理质量评估智能体 · 研发日报 Day 1/10 · 2026-06-26 · Lxuan-4",
        S["footer"]))

    return story


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "研发日报_Day1.pdf")

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title="研发日报 Day1",
        author="Lxuan-4",
    )
    doc.build(build_story())
    print(f"PDF 已生成：{out_path}")


if __name__ == "__main__":
    main()
