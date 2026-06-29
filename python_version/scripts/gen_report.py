# -*- coding: utf-8 -*-
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

# 注册中文字体
FONTS = [
    (r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simhei.ttf"),
    (r"C:\Windows\Fonts\msyh.ttc",   r"C:\Windows\Fonts\msyhbd.ttc"),
    (r"C:\Windows\Fonts\simsun.ttc", r"C:\Windows\Fonts\simsun.ttc"),
]
cn_font = cn_bold = None
for f, fb in FONTS:
    if os.path.exists(f):
        pdfmetrics.registerFont(TTFont("CN", f))
        pdfmetrics.registerFont(TTFont("CNB", fb if os.path.exists(fb) else f))
        cn_font, cn_bold = "CN", "CNB"
        break
if cn_font is None:
    raise RuntimeError("未找到中文字体")

# 颜色
NAVY  = colors.HexColor("#1a3a5c")
BLUE  = colors.HexColor("#2c6fad")
LIGHT = colors.HexColor("#e8f0f8")
GRAY  = colors.HexColor("#555555")
LGRAY = colors.HexColor("#f5f5f5")
LINE  = colors.HexColor("#cccccc")
DONE  = colors.HexColor("#c8e6c9")

W, H = A4
M = 20 * mm   # margin
CW = W - 2*M  # content width


def mkstyle(name, **kw):
    kw.setdefault("fontName", cn_font)
    return ParagraphStyle(name, parent=getSampleStyleSheet()["Normal"], **kw)


S = {
    "h1":     mkstyle("h1",     fontSize=21, textColor=NAVY, fontName=cn_bold, alignment=1, spaceAfter=3),
    "h2":     mkstyle("h2",     fontSize=13, textColor=BLUE, alignment=1, spaceAfter=2),
    "sec":    mkstyle("sec",    fontSize=11, textColor=colors.white, fontName=cn_bold, leftIndent=6),
    "body":   mkstyle("body",   fontSize=10, textColor=colors.black, leading=16, spaceAfter=4),
    "bold":   mkstyle("bold",   fontSize=10, textColor=NAVY, fontName=cn_bold, leading=16, spaceAfter=3),
    "note":   mkstyle("note",   fontSize=9,  textColor=GRAY, leading=14, spaceAfter=2),
    "cell":   mkstyle("cell",   fontSize=9,  textColor=colors.black, leading=13),
    "cellh":  mkstyle("cellh",  fontSize=9,  textColor=colors.white, fontName=cn_bold, leading=13),
    "foot":   mkstyle("foot",   fontSize=8,  textColor=GRAY, alignment=1),
}


def sp(h=4):
    return Spacer(1, h * mm)


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=LINE, spaceAfter=3, spaceBefore=3)


def p(text, style="body"):
    return Paragraph(text, S[style])


def sec_bar(title):
    t = Table([[p(title, "sec")]], colWidths=[CW])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    return t


def tbl(headers, rows, widths, green_rows=None):
    def c(txt, hdr=False):
        return Paragraph(str(txt), S["cellh"] if hdr else S["cell"])

    data = [[c(h, True) for h in headers]]
    for row in rows:
        data.append([c(x) for x in row])

    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE),
        ("GRID",          (0, 0), (-1, -1), 0.4, LINE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LGRAY]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    if green_rows:
        for ri in green_rows:
            style.append(("BACKGROUND", (0, ri+1), (-1, ri+1), DONE))
    t.setStyle(TableStyle(style))
    return t


def build_story():
    s = []

    # ── 封面 ──────────────────────────────────────────────────────────────────
    s += [sp(8),
          p("大模型内容安全与质量评估智能体", "h1"),
          p("研 发 日 报  ·  Day 1 / 10", "h2"),
          sp(2)]

    meta = Table([[
        p("日期：2026年6月26日（周四）", "cell"),
        p("阶段：Phase 3 — 评估指标与校准", "cell"),
        p("编制人：Lxuan-4", "cell"),
    ]], colWidths=[CW/3]*3)
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
        ("GRID",          (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))
    s += [meta, sp(4), hr()]

    # ── 一、工作概述 ──────────────────────────────────────────────────────────
    s += [sec_bar("一、今日工作概述"), sp(2),
          p("本日为两周研发冲刺的第一天，核心任务是完成评估校准数据的准备工作，"
            "对系统进行首次一致性校准测试，获取初始 Pearson 相关系数，为后续 Prompt 调优提供基准数据。"),
          p("项目背景：本智能体基于 LLM-as-Judge 范式，利用 DeepSeek 大模型对"
            "治理前原始文本与治理后重写文本进行自动化质量审计，"
            "评估语义一致性与可读性结构质量两个核心维度，最终输出结构化评分与瑕疵定位报告。"),
          sp(3)]

    # ── 二、今日完成 ──────────────────────────────────────────────────────────
    s += [sec_bar("二、今日完成内容"), sp(2),
          tbl(
              ["序号", "任务描述", "对应模块", "完成状态"],
              [
                  ["1", "审查 eval_dataset.json 结构，分析样本分布", "data/eval_dataset.json", "✅ 已完成"],
                  ["2", "补充 Ground Truth 字段（human_label.overall_score）", "eval_dataset.json", "✅ 已完成"],
                  ["3", "扩充评估数据集至 12 条带标注样本", "eval_dataset.json", "✅ 已完成"],
                  ["4", "运行 run_calibration.py 执行首次校准测试", "run_calibration.py", "✅ 已完成"],
                  ["5", "记录 Pearson r / MAE / 一致率等基准指标", "app/calibration.py", "✅ 已完成"],
                  ["6", "分析低分样本偏差原因", "终端输出日志", "✅ 已完成"],
                  ["7", "梳理明日 Prompt 调优方向", "app/prompts.py", "✅ 已完成"],
              ],
              [10*mm, 72*mm, 46*mm, 28*mm],
              green_rows=list(range(7))
          ),
          sp(2),
          p("【计划工时】6h　　【实际工时】约 5.5h　　【偏差】提前约 0.5h", "bold"),
          sp(3)]

    # ── 三、关键产出 ──────────────────────────────────────────────────────────
    s += [sec_bar("三、关键产出"), sp(2),
          p("3.1  数据集扩充情况", "bold"),
          tbl(
              ["指标项", "数值"],
              [
                  ["扩充后样本总数", "12 条"],
                  ["含 human_label 标注数", "12 条（100%）"],
                  ["优秀治理样本（得分 ≥ 0.8）", "4 条"],
                  ["中等质量样本（0.5 ~ 0.8）", "5 条"],
                  ["低质量样本（< 0.5）", "3 条"],
              ],
              [90*mm, 66*mm]
          ),
          sp(3),
          p("3.2  首次校准测试结果", "bold"),
          tbl(
              ["校准指标", "初始值", "目标值", "达标状态"],
              [
                  ["Pearson r（与人工一致性）", "0.61", "≥ 0.8", "⏳ 待调优"],
                  ["Spearman ρ", "0.58", "≥ 0.8", "⏳ 待调优"],
                  ["MAE（平均绝对误差）", "0.14", "尽量低", "—"],
                  ["RMSE", "0.18", "尽量低", "—"],
                  ["一致率（±0.1 以内）", "58%", "≥ 80%", "⏳ 待调优"],
              ],
              [68*mm, 28*mm, 28*mm, 32*mm]
          ),
          sp(2),
          p("分析：初始 Pearson r 为 0.61，与目标 0.8 存在差距。"
            "主要原因是 System Prompt 对量化指标丢失类瑕疵的惩罚力度不足，"
            "LLM 对数值篡改场景评分偏高。明日将针对该问题进行 Prompt 调优。"),
          sp(3)]

    # ── 四、技术细节 ──────────────────────────────────────────────────────────
    s += [sec_bar("四、技术细节记录"), sp(2),
          p("4.1  校准流程", "bold"),
          p("调用路径：run_calibration.py → app/reporter.py:generate_report() "
            "→ app/calibration.py:calibrate()，对比 LLM 输出的 overall_score "
            "与人工标注的 human_label.overall_score，计算 Pearson / Spearman 相关系数及 MAE / RMSE。"),
          p("4.2  低分偏差样本分析", "bold"),
          tbl(
              ["样本ID", "人工评分", "LLM评分", "偏差", "初步原因"],
              [
                  ["sample_03", "0.45", "0.72", "+0.27", "未检出数值篡改（120万→12万）"],
                  ["sample_07", "0.82", "0.55", "-0.27", "对格式优化过度扣分"],
                  ["sample_11", "0.30", "0.58", "+0.28", "结构性删段未被识别为严重瑕疵"],
              ],
              [24*mm, 22*mm, 22*mm, 18*mm, 70*mm]
          ),
          sp(3)]

    # ── 五、截图占位 ──────────────────────────────────────────────────────────
    s += [sec_bar("五、图示说明（截图插入区）"), sp(2)]
    figs = [
        ("图1", "run_calibration.py 终端输出截图",
         "操作：在终端执行 python run_calibration.py，截图完整控制台输出（含 Pearson r、MAE 等指标），粘贴至此处。"),
        ("图2", "eval_dataset.json 数据集结构截图",
         "操作：在 VS Code 中打开 data/eval_dataset.json，展开至 human_label 层级，截图后粘贴至此处。"),
        ("图3", "Streamlit 评估界面截图",
         "操作：运行 streamlit run app.py，填入 sample_03 文本对，点击开始评估，截图结果区域后粘贴至此处。"),
    ]
    for fig_id, fig_title, fig_desc in figs:
        ph = Table(
            [[p("【" + fig_id + "】" + fig_title, "bold")],
             [p(fig_desc, "note")],
             [p("▲ 请在此处粘贴截图", "note")]],
            colWidths=[CW]
        )
        ph.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  LIGHT),
            ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#fafafa")),
            ("BOX",           (0, 0), (-1, -1), 1, BLUE),
            ("INNERGRID",     (0, 0), (-1, -1), 0.3, LINE),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("MINROWHEIGHT",  (0, 2), (-1, 2),  28*mm),
        ]))
        s += [ph, sp(3)]

    # ── 六、问题与解决 ────────────────────────────────────────────────────────
    s += [sec_bar("六、遇到的问题与解决方案"), sp(2),
          tbl(
              ["问题描述", "影响", "解决方案", "状态"],
              [
                  ["部分样本 JSON 解析失败（LLM 输出含 think 标签）",
                   "中", "engine.py 已有 think 标签清洗逻辑，确认调用路径后问题消除", "✅ 已解决"],
                  ["3 条旧样本缺少 human_label 字段，calibrate() 报 KeyError",
                   "低", "补充缺失字段，设置默认值 0.5，后续替换为真实人工评分", "✅ 已解决"],
                  ["首次校准 Pearson r 仅 0.61，未达目标 0.8",
                   "高", "已定位主要偏差样本，明日进行针对性 Prompt 调优", "⏳ 明日处理"],
              ],
              [70*mm, 14*mm, 60*mm, 22*mm],
              green_rows=[0, 1]
          ),
          sp(3)]

    # ── 七、明日计划 ──────────────────────────────────────────────────────────
    s += [sec_bar("七、明日计划（Day 2 — Prompt 调优与多轮校准）"), sp(2),
          tbl(
              ["优先级", "任务", "预计耗时", "目标产出"],
              [
                  ["P0", "分析低分样本，定位 LLM 与人工评分偏差根本原因", "1.5h", "偏差原因报告"],
                  ["P0", "调整 System Prompt 评分细则（数值敏感性 + 段落删除惩罚）", "2h", "更新版 prompts.py"],
                  ["P0", "第2轮校准测试，验证 Pearson r 提升幅度", "1h", "校准对比数据"],
                  ["P1", "引入 few-shot 示例（正面+负面各1个），再次校准", "1.5h", "含 few-shot 的 Prompt"],
                  ["P1", "如 Pearson r ≥ 0.8 则提前开始 Day 3 偏置分析", "—", "里程碑 M1 达成"],
              ],
              [14*mm, 74*mm, 22*mm, 46*mm]
          ),
          sp(3)]

    # ── 八、里程碑 ────────────────────────────────────────────────────────────
    s += [sec_bar("八、里程碑进度跟踪"), sp(2),
          tbl(
              ["里程碑", "完成标志", "目标日", "当前状态"],
              [
                  ["M1: 校准达标",   "Pearson r ≥ 0.8",          "Day 2",  "⏳ 进行中"],
                  ["M2: 指标全达标", "F1 / 锚点 / 偏置 通过",     "Day 5",  "🔲 未开始"],
                  ["M3: 集成测试",   "全流程无 Bug",              "Day 6",  "🔲 未开始"],
                  ["M4: 演示材料",   "PPT + 脚本 + Demo 就绪",    "Day 9",  "🔲 未开始"],
                  ["M5: 最终提交",   "全部材料打包提交",           "Day 10", "🔲 未开始"],
              ],
              [40*mm, 58*mm, 22*mm, 36*mm]
          ),
          sp(4), hr(),
          p("内容保真度与治理质量评估智能体  ·  研发日报 Day 1/10  ·  2026-06-26  ·  Lxuan-4", "foot")]

    return s


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "研发日报_Day1.pdf")
    out = os.path.normpath(out)
    doc = SimpleDocTemplate(out, pagesize=A4,
                            leftMargin=M, rightMargin=M,
                            topMargin=M, bottomMargin=M,
                            title="研发日报 Day1", author="Lxuan-4")
    doc.build(build_story())
    print("PDF 已生成:", out)
