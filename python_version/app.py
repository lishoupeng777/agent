"""
Streamlit Web UI — 内容保真度与治理质量评估智能体（LLM-as-Judge）
统一调用 app/engine.py，4维度评分，支持 CSV 批量评估
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import jwt  # PyJWT — 智谱 API 认证需要

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv
from app.profiles import PROFILE_OPTIONS

load_dotenv()

# ── 页面配置 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="内容保真度与治理质量评估智能体",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, .stApp { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; }
.main-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2rem; font-weight: 800; text-align: center; margin-bottom: 0.3rem;
}
.badge-pass {
    display:inline-block; background:linear-gradient(135deg,#43e97b,#38f9d7);
    color:#fff; font-weight:700; font-size:1.1rem; padding:8px 24px;
    border-radius:50px; box-shadow:0 4px 15px rgba(67,233,123,.4);
}
.badge-review {
    display:inline-block; background:linear-gradient(135deg,#f7971e,#ffd200);
    color:#343a40; font-weight:700; font-size:1.1rem; padding:8px 24px;
    border-radius:50px;
}
.badge-fail {
    display:inline-block; background:linear-gradient(135deg,#f093fb,#f5576c);
    color:#fff; font-weight:700; font-size:1.1rem; padding:8px 24px;
    border-radius:50px; box-shadow:0 4px 15px rgba(245,87,108,.4);
}
[data-testid="stMetric"] {
    background:#f8f9fa; border-radius:16px; padding:16px;
    box-shadow:0 2px 8px rgba(0,0,0,.06); border:1px solid #e9ecef;
}
div.stButton > button {
    background:linear-gradient(135deg,#667eea,#764ba2) !important;
    color:#fff !important; font-weight:700 !important;
    border:none !important; border-radius:50px !important;
    padding:12px 48px !important;
}
</style>
""", unsafe_allow_html=True)


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def _generate_zhipu_token(api_key: str) -> str:
    """将智谱 API Key（格式：{id}.{secret}）转换为 JWT Token"""
    try:
        key_id, secret = api_key.split(".", 1)
    except ValueError:
        raise ValueError("智谱 API Key 格式应为 {id}.{secret}")
    payload = {
        "api_key": key_id,
        "exp": int(time.time()) + 3600,
        "timestamp": int(time.time()),
    }
    headers = {
        "alg": "HS256",
        "sign_type": "SIGN",
    }
    return jwt.encode(payload, secret, algorithm="HS256", headers=headers)


def _setup_env(api_key: str, base_url: str, model: str) -> None:
    """将侧边栏配置写入环境变量，供 engine.py 读取"""
    if api_key and api_key.strip():
        # 智谱 API 需要将 {id}.{secret} 转换为 JWT Token
        if base_url and "bigmodel.cn" in base_url:
            os.environ["DEEPSEEK_API_KEY"] = _generate_zhipu_token(api_key.strip())
        else:
            os.environ["DEEPSEEK_API_KEY"] = api_key.strip()
    if base_url and base_url.strip():
        os.environ["DEEPSEEK_BASE_URL"] = base_url.strip()
    if model and model.strip():
        os.environ["DEEPSEEK_MODEL"] = model.strip()
    # 重置 engine 单例，让新配置生效
    try:
        import app.engine as _eng
        _eng._llm = None
    except Exception:
        pass


def _run_single(before: str, after: str, evaluation_profile: str = "general") -> dict[str, Any]:
    """调用统一的 engine.evaluate()，返回序列化后的结果"""
    from app.engine import evaluate
    from app.models import EvalRequest

    req = EvalRequest(
        request_id=f"ui_{uuid.uuid4().hex[:8]}",
        before_text=before,
        after_text=after,
        evaluation_profile=evaluation_profile,
    )
    resp = evaluate(req, temperature=0.0)

    # 持久化
    try:
        from app.storage import save_evaluation
        save_evaluation(resp)
    except Exception:
        pass

    return resp.model_dump()


def _verdict_badge(verdict: str) -> str:
    labels = {"pass": ("✅ 整体通过", "badge-pass"),
              "review": ("⚠️ 需人工审核", "badge-review"),
              "fail": ("❌ 整体不通过", "badge-fail")}
    text, cls = labels.get(verdict, ("未知", "badge-review"))
    return f'<div style="text-align:center;margin:12px 0"><span class="{cls}">{text}</span></div>'


def _render_dimension_chart(dimensions: list[dict]) -> None:
    """用 st.bar_chart 渲染4维度得分条形图"""
    df = pd.DataFrame({
        "维度": [d["dimension"] for d in dimensions],
        "得分": [round(d["score"] * 100, 1) for d in dimensions],
    }).set_index("维度")
    st.bar_chart(df, height=200)


def _render_flaws_table(flaws: list[dict]) -> None:
    if not flaws:
        st.success("未检测到任何瑕疵，内容保真度与结构质量完好！")
        return

    rows = []
    for f in flaws:
        loc = f.get("location") or {}
        rows.append({
            "严重程度": f.get("severity", ""),
            "类别": f.get("category", ""),
            "描述": f.get("description", ""),
            "锚点 snippet": loc.get("snippet", ""),
            "start_char": loc.get("start_char", ""),
            "修复建议": f.get("suggestion") or "",
        })
    df = pd.DataFrame(rows)

    def _color_severity(val: str) -> str:
        m = {"critical": "background-color:#f5576c;color:#fff;font-weight:700",
             "major": "background-color:#ffc107;color:#343a40;font-weight:600",
             "minor": "background-color:#43e97b;color:#fff"}
        return m.get(val, "")

    styled = df.style.map(_color_severity, subset=["严重程度"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ── 侧边栏 ────────────────────────────────────────────────────────────────

PRESET_MODELS = {
    "DeepSeek Chat（默认）": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "DeepSeek Reasoner": ("https://api.deepseek.com/v1", "deepseek-reasoner"),
    "智谱 GLM-4-Flash（免费）": ("https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
    "智谱 GLM-4-Plus": ("https://open.bigmodel.cn/api/paas/v4", "glm-4-plus"),
    "OpenAI GPT-4o": ("https://api.openai.com/v1", "gpt-4o"),
    "OpenAI GPT-4o mini": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "自定义（手动填写）": ("", ""),
}


def render_sidebar() -> tuple[str, str, str, str]:
    """返回 (api_key, base_url, model_name, evaluation_profile)"""
    with st.sidebar:
        st.markdown("### 🔑 API 配置")

        # ── 模型预设选择 ──
        preset = st.selectbox(
            "选择模型",
            list(PRESET_MODELS.keys()),
            key="sidebar_preset",
        )
        preset_url, preset_model = PRESET_MODELS[preset]

        # ── API Key ──
        api_key = st.text_input(
            "API Key",
            type="password",
            value="",
            placeholder="sk-xxxxxxxx 或 智谱 {id}.{secret}",
            key="sidebar_api_key",
        )

        # ── Base URL（自定义时才展开编辑） ──
        if preset == "自定义（手动填写）":
            base_url = st.text_input(
                "Base URL",
                value=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                placeholder="https://api.xxx.com/v1",
                key="sidebar_base_url",
            )
            model_name = st.text_input(
                "Model Name",
                value=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                placeholder="model-name",
                key="sidebar_model_name",
            )
        else:
            base_url = preset_url
            model_name = preset_model
            st.caption(f"Base URL：`{base_url}`")
            st.caption(f"Model：`{model_name}`")

        st.divider()

        # ── 评估模式选择 ──
        st.markdown("### 📋 评估模式")
        profile_labels = {item["key"]: item["label"] for item in PROFILE_OPTIONS}
        profile_descriptions = {item["key"]: item["description"] for item in PROFILE_OPTIONS}
        selected_profile_label = st.selectbox(
            "选择评估模式",
            list(profile_labels.values()),
            key="sidebar_profile",
        )
        # 反查 key
        evaluation_profile = next(
            k for k, v in profile_labels.items() if v == selected_profile_label
        )
        st.caption(profile_descriptions[evaluation_profile])

        st.divider()

        with st.expander("ℹ️ 关于本系统", expanded=False):
            st.markdown("""
**课题12：内容保真度与治理质量评估智能体**

LLM-as-Judge 架构，4维度评估：
1. **语义一致性**（权重0.4）
2. **过度清洗/误改识别**（权重0.3）
3. **可读性**（权重0.15）
4. **结构质量**（权重0.15）

支持 DeepSeek / OpenAI 及任何兼容 OpenAI API 格式的模型。
""")
    return api_key, base_url, model_name, evaluation_profile


# ── 单条评估页 ────────────────────────────────────────────────────────────

def page_single(api_key: str, base_url: str, model_name: str, evaluation_profile: str = "general") -> None:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("### 📝 输入")
        before = st.text_area("治理前原始文本", height=200, key="single_before",
                              placeholder="输入治理前的原始文本…")
        after = st.text_area("治理后文本", height=200, key="single_after",
                             placeholder="输入经过治理/脱敏后的文本…")

        run_stab = st.checkbox("启用评分稳定性分析（额外调用3次 API）", key="single_stab")
        clicked = st.button("🚀 开始评估", use_container_width=True, key="single_btn")

    with col_r:
        st.markdown("### 📊 评估结果")

        if clicked:
            if not api_key:
                st.error("请先在侧边栏输入 API Key")
                return
            if not before.strip() or not after.strip():
                st.error("请填写治理前和治理后文本")
                return

            _setup_env(api_key, base_url, model_name)

            with st.spinner("裁判模型评估中…"):
                try:
                    result = _run_single(before.strip(), after.strip(), evaluation_profile)

                    if run_stab:
                        from app.stability import run_stability
                        from app.models import EvalRequest
                        req = EvalRequest(
                            request_id="stab_check",
                            before_text=before.strip(),
                            after_text=after.strip(),
                            sample_count=3,
                            evaluation_profile=evaluation_profile,
                        )
                        stab = run_stability(req, sample_count=3)
                        result["_stability"] = {
                            "mean": stab.mean_score,
                            "variance": stab.variance,
                            "std_dev": stab.std_dev,
                            "is_stable": stab.is_stable,
                        }

                    st.session_state["single_result"] = result
                except Exception as e:
                    st.error(f"评估失败：{e}")
                    return

        result = st.session_state.get("single_result")
        if result is None:
            st.info("👈 在左侧输入文本后点击「开始评估」")
            return

        # 判定徽章
        st.markdown(_verdict_badge(result.get("verdict", "")), unsafe_allow_html=True)

        # 本次模式
        profile_used = result.get("evaluation_profile", "general")
        profile_label = next(
            (item["label"] for item in PROFILE_OPTIONS if item["key"] == profile_used),
            profile_used,
        )
        st.caption(f"📋 评估模式：{profile_label}")

        # 总分
        score = result.get("overall_score", 0)
        st.metric("综合得分", f"{score*100:.1f}%",
                  delta="达标 ✓" if score >= 0.8 else ("需审核" if score >= 0.5 else "不通过 ✗"),
                  delta_color="normal" if score >= 0.8 else "inverse")

        # 4维度条形图
        dims = result.get("dimensions", [])
        if dims:
            st.markdown("#### 📊 四维度得分")
            _render_dimension_chart(dims)

            # 维度详情
            with st.expander("查看各维度详情"):
                for d in dims:
                    st.markdown(f"**{d['dimension']}**（权重{d['weight']}）：{d['score']:.2f}")
                    st.caption(d.get("reason", ""))

        # 瑕疵清单
        st.markdown("#### 🔍 锚点级瑕疵清单")
        _render_flaws_table(result.get("flaws", []))

        # 稳定性
        if "_stability" in result:
            stab = result["_stability"]
            st.markdown("#### 📈 评分稳定性")
            c1, c2, c3 = st.columns(3)
            c1.metric("均值", f"{stab['mean']:.4f}")
            c2.metric("方差", f"{stab['variance']:.6f}")
            c3.metric("稳定", "✓" if stab["is_stable"] else "✗")

        # 可复现令牌
        token = result.get("reproducibility_token", "")
        model_v = result.get("model_version", "")
        prompt_v = result.get("prompt_version", "")
        with st.expander("🔑 可复现性信息"):
            st.code(f"token: {token}\nmodel: {model_v}\nprompt_version: {prompt_v}")

        # 原始 JSON
        with st.expander("🔍 原始评估 JSON"):
            display = {k: v for k, v in result.items()
                       if k not in ("raw_llm_output", "_stability")}
            st.json(display)

        # 继续评估按钮
        st.divider()
        if st.button("🔄 继续评估下一条", use_container_width=True, key="single_next"):
            del st.session_state["single_result"]
            st.rerun()


# ── 批量评估页 ────────────────────────────────────────────────────────────

def page_batch(api_key: str, base_url: str, model_name: str, evaluation_profile: str = "general") -> None:
    st.markdown("### 📂 CSV 批量评估")
    st.markdown("""
上传 CSV 文件，要求包含以下列（列名不区分大小写）：
- `before` 或 `before_text`：治理前原始文本
- `after` 或 `after_text`：治理后文本
- `id`（可选）：请求标识

最多支持 **50 条**记录。
""")

    uploaded = st.file_uploader("选择 CSV 文件", type=["csv"], key="batch_csv")

    if uploaded is None:
        st.info("请上传 CSV 文件开始批量评估")
        return

    try:
        df_in = pd.read_csv(uploaded, encoding="utf-8-sig")
    except Exception as e:
        st.error(f"CSV 读取失败：{e}")
        return

    # 标准化列名
    df_in.columns = [c.lower().strip() for c in df_in.columns]
    before_col = "before_text" if "before_text" in df_in.columns else "before"
    after_col = "after_text" if "after_text" in df_in.columns else "after"

    if before_col not in df_in.columns or after_col not in df_in.columns:
        st.error(f"未找到必要列，当前列名：{list(df_in.columns)}")
        return

    df_in = df_in.head(50)
    st.info(f"已加载 {len(df_in)} 条记录")

    if not st.button("🚀 开始批量评估", key="batch_btn"):
        return

    if not api_key:
        st.error("请先在侧边栏输入 DeepSeek API Key")
        return

    _setup_env(api_key, base_url, model_name)

    from app.models import EvalRequest
    from app.batch import batch_evaluate

    requests = []
    for i, row in df_in.iterrows():
        rid = str(row.get("id", f"row_{i+1}"))
        requests.append(EvalRequest(
            request_id=rid,
            before_text=str(row[before_col]),
            after_text=str(row[after_col]),
            evaluation_profile=evaluation_profile,
        ))

    progress_bar = st.progress(0)
    status_text = st.empty()

    def on_progress(done: int, total: int) -> None:
        progress_bar.progress(done / total)
        status_text.text(f"进度：{done}/{total}")

    with st.spinner("批量评估中，请稍候…"):
        results = batch_evaluate(
            requests,
            max_concurrency=3,
            use_cache=True,
            persist=True,
            on_progress=on_progress,
        )

    progress_bar.progress(1.0)
    status_text.text("评估完成！")

    # 汇总表格
    rows = []
    for r in results:
        if r.get("status") == "ok":
            res = r["result"]
            dims = {d["dimension"]: round(d["score"], 3) for d in res.get("dimensions", [])}
            rows.append({
                "request_id": r["request_id"],
                "综合得分": res.get("overall_score", 0),
                "判定": res.get("verdict", ""),
                "瑕疵数": len(res.get("flaws", [])),
                "语义一致性": dims.get("语义一致性", ""),
                "过度清洗/误改": dims.get("过度清洗/误改识别", ""),
                "可读性": dims.get("可读性", ""),
                "结构质量": dims.get("结构质量", ""),
                "来自缓存": "是" if r.get("from_cache") else "否",
            })
        else:
            rows.append({
                "request_id": r["request_id"],
                "综合得分": "",
                "判定": "error",
                "瑕疵数": "",
                "语义一致性": "", "过度清洗/误改": "", "可读性": "", "结构质量": "",
                "来自缓存": "",
            })

    df_out = pd.DataFrame(rows)

    # 统计摘要
    ok_rows = [r for r in results if r.get("status") == "ok"]
    if ok_rows:
        verdicts = [r["result"]["verdict"] for r in ok_rows]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总数", len(results))
        c2.metric("✅ pass", verdicts.count("pass"))
        c3.metric("⚠️ review", verdicts.count("review"))
        c4.metric("❌ fail", verdicts.count("fail"))

    st.markdown("#### 批量评估结果")
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    # 下载 JSON 报告
    json_bytes = json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ 下载完整 JSON 报告",
        data=json_bytes,
        file_name="batch_eval_report.json",
        mime="application/json",
    )

    # 下载结果 CSV
    csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 下载结果 CSV",
        data=csv_bytes,
        file_name="batch_eval_result.csv",
        mime="text/csv",
    )


# ── 主入口 ────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown('<p class="main-title">🛡️ 内容保真度与治理质量评估智能体</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#6c757d;margin-bottom:1.5rem;">'
        'LLM-as-Judge · 四维度评测 · 治理前后文本比对 · 课题12</p>',
        unsafe_allow_html=True,
    )

    api_key, base_url, model_name, evaluation_profile = render_sidebar()

    tab_single, tab_batch = st.tabs(["🔍 单条评估", "📂 批量评估（CSV）"])

    with tab_single:
        page_single(api_key, base_url, model_name, evaluation_profile)

    with tab_batch:
        page_batch(api_key, base_url, model_name, evaluation_profile)

    st.markdown(
        '<p style="text-align:center;color:#adb5bd;font-size:.8rem;margin-top:2rem;">'
        "内容保真度与治理质量评估智能体 · Powered by DeepSeek & Streamlit · 2025</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
