"""
File: app.py
Streamlit Web UI – 内容保真度与治理质量评估智能体（LLM-as-Judge）
========================================================================
Usage:
    streamlit run app.py

功能：
  - 侧边栏：输入/修改 DEEPSEEK_API_KEY 和 LANGSMITH_API_KEY
  - 左侧：两个大文本域（治理前原始文本、治理后重写文本）
  - 自动锚点预处理（按行标记 [Before N] / [After N]）
  - DeepSeek 裁判模型评估：语义一致性 + 可读性与结构质量
  - 右侧：metric 卡片、瑕疵 DataFrame 表格、判定理由
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ---- 确保项目根目录在 sys.path 上 ----
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ===========================================================================
# 页面基础配置
# ===========================================================================
st.set_page_config(
    page_title="内容保真度与治理质量评估智能体",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# 自定义 CSS
# ===========================================================================
st.markdown(
    """
<style>
    html, body, .stApp {
        font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", sans-serif;
    }
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .badge-pass {
        display: inline-block;
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #fff;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 10px 28px;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(67, 233, 123, 0.4);
    }
    .badge-fail {
        display: inline-block;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: #fff;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 10px 28px;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
    }
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #495057;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stTextArea label {
        font-weight: 600 !important;
        color: #343a40 !important;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #fff !important;
        font-weight: 700 !important;
        font-size: 1.08rem !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 14px 56px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.55) !important;
    }
    .footer {
        text-align: center;
        color: #adb5bd;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
    /* Metric 卡片自定义 */
    [data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    [data-testid="stMetricLabel"] p {
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    [data-testid="stMetricValue"] {
        font-weight: 900 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ===========================================================================
# 锚点预处理算法
# ===========================================================================
def build_anchored_text(text: str, prefix: str) -> str:
    """将文本按行分割并自动标记行号标尺。

    Args:
        text: 原始文本。
        prefix: 行前缀，如 "Before" 或 "After"。

    Returns:
        带行号标尺的文本，每行格式为 [prefix N] 内容。
    """
    lines = text.split("\n")
    anchored_lines = []
    for i, line in enumerate(lines, 1):
        anchored_lines.append(f"[{prefix} {i}] {line}")
    return "\n".join(anchored_lines)


# ===========================================================================
# 裁判大模型 System Prompt
# ===========================================================================
JUDGE_SYSTEM_PROMPT = r"""你是一名拥有10年以上经验的资深数据质量与文本治理审计专家。你的任务是严格比对"治理前"和"治理后"的文本对。

请严格遵循以下规则进行审计与打分：

1. 语义一致性评分 (semantic_consistency_score) - 总分 1.0
   - 严重瑕疵（每次扣 0.4-0.5分）：误删、误改了核心事实、实体或关键量化指标（如金额、百分比、日期等）。
   - 中等瑕疵（每次扣 0.2-0.3分）：语义偏离，误改修饰词或特定业务专有名词。
   - 轻微瑕疵（每次扣 0.05-0.1分）：非必要的同义词替换，语意没变但属于过度清洗。

2. 可读性与结构质量评分 (readability_structure_score) - 总分 1.0
   - 严重瑕疵（每次扣 0.3-0.4分）：排版结构被破坏，导致无法阅读。
   - 中等瑕疵（每次扣 0.15-0.2分）：语句不通顺，或表格对齐混乱。

3. 瑕疵锚点定位要求 (Anchored Localization)
   发现瑕疵时，必须指出是原文本哪一行变化到了后文本哪一行（例如: [Before X] -> [After Y]）。

4. 瑕疵类型必须严格限定为以下四种之一（不允许使用其他类型名称）：
   - 过度清洗
   - 语义误改
   - 可读性下降
   - 结构破坏

5. 严重程度必须严格限定为以下三种之一（不允许使用其他等级名称）：
   - High
   - Medium
   - Low

输出要求：
- 你必须严格且仅输出一个 JSON 对象。
- 不要使用 ```json 代码块包裹。
- 不要输出任何解释、道歉或除 JSON 之外的任何文字。
- 字段名必须与下方 Schema 完全一致，不得增删或改名。

期望输出的 JSON 格式必须为：
{
  "semantic_consistency_score": 0.0到1.0的浮点数,
  "readability_structure_score": 0.0到1.0的浮点数,
  "flaws": [
    {
      "anchor": "定位，如 [Before X] -> [After Y]",
      "type": "仅限四种之一：过度清洗 | 语义误改 | 可读性下降 | 结构破坏",
      "severity": "仅限三种之一：High | Medium | Low",
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


# ===========================================================================
# 核心评估函数
# ===========================================================================
def run_evaluation(
    before_text: str,
    after_text: str,
    deepseek_api_key: str,
    langsmith_api_key: str | None = None,
) -> dict[str, Any]:
    """调用 DeepSeek 裁判模型，对治理前后文本进行评估。

    Returns:
        成功时返回 LLM JSON（含 semantic_consistency_score 等字段）。
        失败时携带 parse_error 和 raw_output。
    """
    # ---- 1. LangSmith 追踪（可选） ----
    if langsmith_api_key and langsmith_api_key.strip():
        os.environ["LANGSMITH_API_KEY"] = langsmith_api_key.strip()
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ.setdefault("LANGSMITH_PROJECT", "shixun-judge")

    # ---- 2. 锚点预处理 ----
    before_anchored = build_anchored_text(before_text, "Before")
    after_anchored = build_anchored_text(after_text, "After")

    # ---- 3. 构建消息 ----
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_api_key,
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

    # ---- 4. 调用模型 ----
    try:
        raw_output = llm.invoke(messages)
        raw_text = str(raw_output.content) if hasattr(raw_output, "content") else str(raw_output)
    except Exception as exc:
        return {
            "semantic_consistency_score": 0.0,
            "readability_structure_score": 0.0,
            "flaws": [],
            "overall_explanation": f"模型调用失败: {exc}",
            "raw_output": str(exc),
            "parse_error": True,
        }

    # ---- 5. 清洗输出 ----
    clean_text = re.sub(
        r"<think[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE
    ).strip()

    # ---- 6. 提取 JSON（先尝试纯文本，再尝试代码块） ----
    json_str = clean_text
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
    if json_match:
        json_str = json_match.group(1).strip()

    # 去除可能的 BOM 和首尾非 JSON 字符
    json_str = json_str.strip()
    # 尝试找到第一个 { 和最后一个 }
    brace_start = json_str.find("{")
    brace_end = json_str.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        json_str = json_str[brace_start:brace_end + 1]

    try:
        parsed = json.loads(json_str)
        # 补全缺失字段
        parsed.setdefault("semantic_consistency_score", 0.0)
        parsed.setdefault("readability_structure_score", 0.0)
        parsed.setdefault("flaws", [])
        parsed.setdefault("overall_explanation", "")
        return parsed
    except json.JSONDecodeError:
        return {
            "semantic_consistency_score": 0.0,
            "readability_structure_score": 0.0,
            "flaws": [],
            "overall_explanation": "裁判模型返回格式异常，无法解析为 JSON。",
            "raw_output": clean_text,
            "parse_error": True,
        }


# ===========================================================================
# UI 组件
# ===========================================================================

def render_sidebar() -> None:
    """渲染侧边栏 —— API Key 输入区。"""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/shield.png",
            width=72,
        )
        st.markdown(
            '<p style="text-align:center;font-weight:700;font-size:1.2rem;color:#495057;">'
            "🔑 API 密钥配置"
            "</p>",
            unsafe_allow_html=True,
        )

        # DeepSeek API Key
        st.markdown(
            '<p class="sidebar-title">🤖 DeepSeek API Key</p>',
            unsafe_allow_html=True,
        )
        st.text_input(
            "DeepSeek API Key",
            type="password",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            label_visibility="collapsed",
            key="sidebar_deepseek_key",
        )
        st.caption("用于调用 DeepSeek 裁判模型进行评估。")

        st.divider()

        # LangSmith API Key（可选）
        st.markdown(
            '<p class="sidebar-title">📊 LangSmith API Key（可选）</p>',
            unsafe_allow_html=True,
        )
        st.text_input(
            "LangSmith API Key",
            type="password",
            value=os.getenv("LANGSMITH_API_KEY", ""),
            placeholder="lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            label_visibility="collapsed",
            key="sidebar_langsmith_key",
        )
        st.caption("用于追踪 LLM 调用链（LangSmith），可留空。")

        st.divider()

        with st.expander("ℹ️ 关于本系统", expanded=False):
            st.markdown(
                """
                **内容保真度与治理质量评估智能体**

                采用 LLM-as-Judge 架构，自动比对数据治理前后的文本对，
                从以下维度进行审计：

                1. **语义一致性** — 核心事实/实体是否被误改或丢失
                2. **可读性与结构质量** — 排版、语句通顺度、表格对齐

                瑕疵自动锚点定位至行级别。

                *Powered by DeepSeek + LangChain + Streamlit*
                """
            )


def render_input_area():
    """渲染左侧输入区域 + 右侧占位列。"""
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📝 测试输入")

        before_text = st.text_area(
            "治理前原始文本 (Original Text)",
            height=200,
            placeholder=(
                "请输入治理前的原始文本，作为评估基准 (Ground Truth)…\n\n"
                "例如：\n国家电网2023年收入增长10%。\n业务联系人张伟，电话13800138000。"
            ),
            key="input_before",
        )

        after_text = st.text_area(
            "治理后重写文本 (Governed Text)",
            height=200,
            placeholder=(
                "请输入经过数据治理/脱敏后的重写文本…\n\n"
                "例如：\n国家电网2023年收入录得增长。\n联系人张伟电话已脱敏。"
            ),
            key="input_after",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        evaluate_clicked = st.button(
            "🚀 开始评估",
            use_container_width=True,
            type="primary",
            key="btn_evaluate",
        )

    return col_left, col_right, before_text, after_text, evaluate_clicked


def render_results_area(col_right, parsed: dict[str, Any] | None) -> None:
    """在右侧列渲染评估结果。"""
    with col_right:
        st.markdown("### 📊 评估结果")

        if parsed is None:
            st.info("👈 请在左侧输入治理前后文本，点击「开始评估」按钮查看结果。")
            return

        # ---- 解析错误 ----
        if parsed.get("parse_error"):
            st.error("⚠️ 裁判模型返回格式异常，请查看原始输出：")
            st.code(parsed.get("raw_output", "无输出"), language="text")
            return

        # ---- 提取数据 ----
        sem_score: float = float(parsed.get("semantic_consistency_score", 0.0))
        read_score: float = float(parsed.get("readability_structure_score", 0.0))
        flaws: list[dict[str, Any]] = parsed.get("flaws", [])
        overall_explanation: str = parsed.get("overall_explanation", "")

        sem_pct = round(sem_score * 100)
        read_pct = round(read_score * 100)
        overall_pass = (sem_pct >= 80 and read_pct >= 80)

        # ---- 整体通过/不通过徽章 ----
        badge_class = "badge-pass" if overall_pass else "badge-fail"
        badge_text = "✅ 整体通过" if overall_pass else "❌ 整体不通过"
        st.markdown(
            f'<div style="text-align:center;margin-bottom:20px;">'
            f'<span class="{badge_class}">{badge_text}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # ---- Metric 卡片 ----
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            delta_color = "normal" if sem_score >= 0.8 else "inverse"
            st.metric(
                label="📋 语义保真度得分",
                value=f"{sem_pct}%",
                delta="达标 ✓" if sem_pct >= 80 else "未达标 ✗",
                delta_color=delta_color,
            )
        with col_m2:
            delta_color = "normal" if read_score >= 0.8 else "inverse"
            st.metric(
                label="📐 可读性与结构得分",
                value=f"{read_pct}%",
                delta="达标 ✓" if read_pct >= 80 else "未达标 ✗",
                delta_color=delta_color,
            )

        # ---- 瑕疵列表表格 ----
        st.markdown("---")
        st.markdown("#### 🔍 锚点级瑕疵清单")

        if not flaws:
            st.success(
                "恭喜！本次数据治理未检测到任何瑕疵，"
                "内容保真度与结构质量完好！"
            )
        else:
            # 转为 DataFrame
            df_rows = []
            for f in flaws:
                df_rows.append({
                    "关联锚点": f.get("anchor", ""),
                    "瑕疵类型": f.get("type", ""),
                    "严重程度": f.get("severity", ""),
                    "瑕疵详情": f.get("detail", ""),
                })
            df = pd.DataFrame(df_rows)
            # 按严重程度着色
            def severity_color(val: str) -> str:
                if val == "High":
                    return "background-color: #f5576c; color: #fff; font-weight: 700;"
                elif val == "Medium":
                    return "background-color: #ffc107; color: #343a40; font-weight: 600;"
                elif val == "Low":
                    return "background-color: #43e97b; color: #fff;"
                return ""

            styled_df = df.style.map(severity_color, subset=["严重程度"])
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "关联锚点": st.column_config.TextColumn(width="medium"),
                    "瑕疵类型": st.column_config.TextColumn(width="small"),
                    "严重程度": st.column_config.TextColumn(width="small"),
                    "瑕疵详情": st.column_config.TextColumn(width="large"),
                },
            )

        # ---- 判定理由 ----
        if overall_explanation:
            st.markdown("---")
            st.markdown("#### 💬 裁判总体解释")
            st.info(overall_explanation)

        # ---- 原始 JSON（折叠） ----
        with st.expander("🔍 查看原始评估 JSON"):
            # 移除 raw_output 再展示，避免过长
            display_parsed = {k: v for k, v in parsed.items() if k != "raw_output"}
            st.json(display_parsed)


# ===========================================================================
# 主入口
# ===========================================================================
def main() -> None:
    """Streamlit 应用主函数。"""

    # ---- 标题 ----
    st.markdown(
        '<p class="main-title">🛡️ 内容保真度与治理质量评估智能体</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center;color:#6c757d;margin-bottom:2rem;">'
        "LLM-as-Judge · 治理前后文本比对 · 语义一致性 & 可读性结构双维度评测"
        "</p>",
        unsafe_allow_html=True,
    )

    # ---- 侧边栏 ----
    render_sidebar()

    # ---- API Keys ----
    deepseek_key: str = st.session_state.get("sidebar_deepseek_key", "")
    langsmith_key: str = st.session_state.get("sidebar_langsmith_key", "")

    # ---- 输入区 ----
    col_left, col_right, before_text, after_text, evaluate_clicked = render_input_area()

    # ---- 评估逻辑 ----
    if evaluate_clicked:
        if not deepseek_key or not deepseek_key.strip():
            with col_right:
                st.error("⚠️ 请先在侧边栏输入 **DeepSeek API Key**！")
        elif not before_text.strip() or not after_text.strip():
            with col_right:
                st.error("⚠️ 请填写「治理前原始文本」和「治理后重写文本」！")
        else:
            with col_right:
                with st.spinner("🤔 裁判模型正在评估中，请稍候…"):
                    try:
                        result = run_evaluation(
                            before_text=before_text.strip(),
                            after_text=after_text.strip(),
                            deepseek_api_key=deepseek_key.strip(),
                            langsmith_api_key=(
                                langsmith_key.strip() if langsmith_key else None
                            ),
                        )
                        st.session_state["eval_result"] = result
                    except Exception as exc:
                        st.session_state["eval_result"] = {
                            "semantic_consistency_score": 0.0,
                            "readability_structure_score": 0.0,
                            "flaws": [],
                            "overall_explanation": f"系统异常: {exc}",
                            "raw_output": str(exc),
                            "parse_error": True,
                        }

    # ---- 渲染结果 ----
    render_results_area(col_right, st.session_state.get("eval_result"))

    # ---- Footer ----
    st.markdown(
        '<p class="footer">'
        "内容保真度与治理质量评估智能体 · Powered by DeepSeek & Streamlit · 2025"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()