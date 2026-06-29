# 大模型内容安全与质量评估智能体 · 研发日报

---

|          |                                          |
|----------|------------------------------------------|
| **日期** | 2026年7月3日（周四）                      |
| **课题** | 内容保真度与治理质量评估智能体（LLM-as-Judge） |
| **阶段** | Phase 5 — Web 应用与可视化 + 架构优化与多模型适配 |
| **日报编号** | Day 6 / 10                           |
| **组长** | 李首澎                                   |
| **组员** | 阿思晗、由靖喆                            |

---

## 一、组员进度汇报

### 1.1 组员：阿思晗

#### 今日任务：Phase 5 后端与可视化 + LCEL Chain 架构重构 + 多模型适配体系

今日按照研发计划书 Phase 5 展开 Web 应用开发，同时利用进度超前的窗口推进架构优化和多模型适配。

**上午 — Phase 5：FastAPI 后端 + 结果可视化**

完善了 **P5-1 FastAPI 后端**：在 `app/routes.py` 中补全 6 个 API 端点（`/api/v1/evaluate` POST 单条评估、`/api/v1/batch/evaluate` POST 批量评估、`/api/v1/calibrate` POST 校准、`/api/v1/stability` POST 稳定性、`/api/v1/history` GET 历史查询、`/api/v1/health` GET 健康检查），在 `main.py` 中配置 CORS 中间件和静态文件挂载。

完成了 **P5-4 结果可视化**：在 `app.py` 中实现 Metric 卡片（`st.metric()` 展示 overall_score 和各维度评分）、DataFrame 瑕疵表（`st.dataframe()` 展示瑕疵列表，按 severity 着色）、Pass-Fail 徽章（HTML span 标签，pass 绿色 / review 橙色 / fail 红色）。

**下午 — 架构重构与多模型适配**

1. **LCEL Chain 架构重构**：将 `app/chain.py` 重构为 LangChain Expression Language 声明式 Chain（`chain = prompt | llm | parser`），引入 `PydanticOutputParser` 替代自定义三层 JSON 解析，启用 `response_format={"type": "json_object"}` 的 JSON mode，解析失败率从 ~5% 降至接近 0%。

2. **DSPy 集成**：在 `app/dspy_eval.py` 中实现 `TextEvalSignature` 和 `StrictEvalSignature`（评估签名）、`TextEvaluator` / `StrictEvaluator` 模块（使用 `ChainOfThought`）、输出断言（分数范围、权重和约束）。包含 `BootstrapFewShot` Teleprompter 优化器和评估 metric 函数，支持自动 Prompt 优化。

3. **统一评估协议**：在 `app/protocol.py` 中定义 `EvaluationProtocol` 抽象基类（`evaluate()` / `get_model_name()` / `get_model_version()` / `batch_evaluate()`）和 `ProtocolResult`（包装 EvalResponse + 执行元数据）。

4. **模型注册表**：在 `app/model_registry.py` 中实现 `ModelRegistry` 单例，支持注册/注销/单模型评估/全模型对比（`evaluate_all`），基于环境变量自动注册。

5. **模型适配器**：在 `app/adapters.py` 中实现三个适配器——`DeepSeekAdapter`（复用现有逻辑）、`GLMAdapter`（检测 `{id}.{secret}` 格式 API Key，自动用 HS256 转换为 JWT）、`GPTAdapter`（GPT 系列模型适配），均继承 `BaseAdapter` 实现 `EvaluationProtocol` 接口。

6. **评分校准器**：在 `app/calibrator.py` 中实现 `ScoreCalibrator`（线性回归 `scipy.stats.linregress`，`slope * raw + intercept` 映射）和 `MultiModelCalibrator`（管理多个模型各自的校准器，支持 JSON 持久化）。包含 `auto_calibrate()` 函数自动在金标准数据集上训练校准器。

7. **离线校准 Agent**：在 `app/calibration_agent.py` 中实现自动化校准 Agent——加载金标准数据集、运行模型评估、计算一致性指标（Pearson/Spearman/Kappa/Kendall's W）、训练校准器、分类别分析、检测异常（偏差 > 0.3）、生成格式化校准报告。

| 任务 | 完成状态 |
|------|----------|
| P5-1 FastAPI 后端（6 个 API 端点） | ✅ 已完成 |
| P5-4 结果可视化（Metric / DataFrame / 徽章） | ✅ 已完成 |
| LCEL Chain + PydanticOutputParser + JSON mode | ✅ 已完成 |
| DSPy ChainOfThought + BootstrapFewShot Teleprompter | ✅ 已完成 |
| 统一评估协议（`app/protocol.py`） | ✅ 已完成 |
| 模型注册表（`app/model_registry.py`） | ✅ 已完成 |
| 模型适配器（DeepSeek + GLM(JWT) + GPT） | ✅ 已完成 |
| 评分校准器 + MultiModelCalibrator | ✅ 已完成 |
| 离线校准 Agent | ✅ 已完成 |

**计划工时：4h　　实际工时：约 5.5h　　偏差：超时 1.5h（多模型适配任务量大）**

---

### 1.2 组员：由靖喆

#### 今日任务：Phase 5 前端与 CSS + LLM 缓存 + 速率限制 + 流式输出 + GLM 前端适配

**上午 — Phase 5 前端**

完成了 **P5-2 中文前端页面**（`static/index.html` 布局/字体/配色优化）、**P5-3 Streamlit 演示版**（侧边栏 API Key + 双文本域 + 实时评估）、**P5-5 加载动画**（`st.spinner` + `st.error` 友好异常提示）和 **P5-6 自定义 CSS**（渐变标题、圆角卡片 `border-radius: 12px`、按钮 hover 动效 `transition: 0.3s`）。

**下午 — 基础设施**

1. **LLM 缓存**：在 `app/chain.py` 中集成 LangChain `InMemoryCache`（会话级），在 `app/chain.py` 中集成 `SQLiteCache`（持久化级）；
2. **速率限制**：`InMemoryRateLimiter` 控制 API 调用频率；
3. **流式输出**：SSE 协议实时推送评估结果；
4. **Callback 追踪**：`EvalCallbackHandler` 记录每次 LLM 调用的 Token 数和延迟；
5. **GLM 前端适配**：Streamlit 侧边栏增加 GLM API Key 输入，自动识别 `{id}.{secret}` 格式。

| 任务 | 完成状态 |
|------|----------|
| P5-2 中文前端页面优化 | ✅ 已完成 |
| P5-3 Streamlit 演示版完善 | ✅ 已完成 |
| P5-5 加载动画 + 异常友好提示 | ✅ 已完成 |
| P5-6 自定义 CSS（渐变 / 圆角 / hover） | ✅ 已完成 |
| LLM 缓存（InMemoryCache + SQLiteCache） | ✅ 已完成 |
| 速率限制 + 流式输出 + Callback 追踪 | ✅ 已完成 |
| GLM 前端适配（API Key 输入） | ✅ 已完成 |

**计划工时：2h　　实际工时：约 3h　　偏差：超时 1h（基础设施任务较多）**

---

## 二、组长汇总

### 2.1 今日整体进度说明

根据研发计划书安排，Day 6 对应 **Phase 5（Web 应用与可视化）全部 6 项任务**。今日完成了 P5-1 至 P5-6，Phase 5 全部完成。同时利用进度超前的时间窗口，完成了 LCEL Chain 架构重构、DSPy 集成（含 ChainOfThought 和 BootstrapFewShot Teleprompter）、统一评估协议、模型注册表、三模型适配器（DeepSeek/GLM/GPT）、MultiModelCalibrator、离线校准 Agent 和 LLM 缓存/速率限制/流式输出等基础设施。

系统当前已具备完整的 Web 前后端、多模型适配能力和架构级优化，进度超前约 2 天。

### 2.2 组员任务完成情况汇总

| 姓名 | 今日任务 | 完成状态 | 备注 |
|------|----------|----------|------|
| 阿思晗 | Phase 5 后端/可视化 + 架构重构 + 多模型适配 | ✅ 全部完成 | 超时 1.5h |
| 由靖喆 | Phase 5 前端/CSS/动画 + 缓存/速率限制/流式 | ✅ 全部完成 | 超时 1h |
| 李首澎（组长） | UI 评审 + 架构评审 + 适配器验证 | ✅ 全部完成 | 见下节 |

### 2.3 组长今日工作内容

1. **Phase 5 交付物逐项审查**：确认 P5-1 至 P5-6 的完成质量。
2. **架构重构风险评估**：审查 LCEL Chain 重构是否引入回归问题，PydanticOutputParser 与现有模型的兼容性。
3. **多模型适配器验证**：审查 GLM JWT 转换逻辑和 GPT 适配器的接口一致性。
4. **校准器验证**：确认 MultiModelCalibrator 的 JSON 持久化和 auto_calibrate 功能。

**计划工时：6h　　实际工时：约 7h　　偏差：超时 1h**

---

## 三、今日研发过程记录

### 3.1 Phase 5 交付物

| 计划任务 | 交付物 | 说明 |
|----------|--------|------|
| P5-1 FastAPI 后端 | `app/routes.py` | 6 个端点：evaluate/batch/calibrate/stability/history/health |
| P5-2 中文前端 | `static/index.html` | 布局/字体/配色优化 |
| P5-3 Streamlit 演示版 | `app.py` | 侧边栏 Key + 双文本域 + 实时评估 |
| P5-4 结果可视化 | `app.py` | Metric 卡片 + DataFrame 着色 + Pass-Fail 徽章 |
| P5-5 加载动画 | `app.py` | `st.spinner` + `st.error` 友好提示 |
| P5-6 自定义 CSS | `app.py` | 渐变标题 / 圆角卡片 / hover 动效 |

### 3.2 LCEL Chain 架构

重构后评估链路：`ChatPromptTemplate → ChatOpenAI(temperature=0.0) → PydanticOutputParser`。配合 JSON mode，LLM 输出直接映射到 Pydantic 模型。Chain 中还集成了 veto rules（critical 结构缺陷/过度清洗时 cap 分数）和 Profile 感知的扣分逻辑。

### 3.3 多模型适配体系

```
EvaluationProtocol (协议层)
    ├── ModelRegistry (注册层)
    │     ├── DeepSeekAdapter
    │     ├── GLMAdapter (JWT 自动转换)
    │     └── GPTAdapter
    ├── ScoreCalibrator / MultiModelCalibrator (校准层)
    └── CalibrationAgent (离线校准辅助)
```

---

## 四、关键产出

| 产出内容 | 对应任务 | 说明 |
|----------|---------|------|
| `app/routes.py` 6 端点 | P5-1 | 评估/批量/校准/稳定性/历史/健康 |
| `static/index.html` | P5-2 | 中文前端优化 |
| `app.py` Streamlit | P5-3/P5-4/P5-5/P5-6 | 演示版 + 可视化 + 动画 + CSS |
| `app/chain.py` LCEL Chain | 架构优化 | PydanticOutputParser + JSON mode + 缓存 + 限流 |
| `app/dspy_eval.py` DSPy | 架构优化 | ChainOfThought + BootstrapFewShot Teleprompter |
| `app/protocol.py` 协议层 | 多模型适配 | EvaluationProtocol 抽象基类 |
| `app/model_registry.py` 注册表 | 多模型适配 | 单例注册 + evaluate_all |
| `app/adapters.py` 三适配器 | 多模型适配 | DeepSeek + GLM(JWT) + GPT |
| `app/calibrator.py` 校准器 | 多模型适配 | MultiModelCalibrator + auto_calibrate |
| `app/calibration_agent.py` | 多模型适配 | 离线校准 Agent |
| `EvalCallbackHandler` | 基础设施 | Token/延迟追踪 |
| SSE 流式输出 | 基础设施 | 实时结果推送 |

---

## 五、当前进度评估

### 5.1 里程碑进度看板

| 里程碑 | 完成标志 | 目标日 | 当前状态 | 风险等级 |
|--------|----------|--------|----------|----------|
| M1: 校准达标 | Pearson r ≥ 0.8 | Day 2 | ✅ 已达标 | 🟢 低 |
| M2: 指标全部达标 | F1 / 锚点 / 偏置 / 稳定性 通过 | Day 5 | ✅ 已达标 | 🟢 低 |
| M3: 集成测试通过 | 全流程无 Bug | Day 6 | ✅ 已通过 | 🟢 低 |
| M4: 演示材料完备 | PPT + 脚本 + Demo 就绪 | Day 9 | 🔲 未开始 | 🟢 低 |
| M5: 最终提交 | 全部材料打包提交 | Day 10 | 🔲 未开始 | 🟢 低 |

### 5.2 总体进度状态

> **当前状态：进度超前 2 天，Phase 1-5 全部完成，多模型适配和架构优化已到位。**
>
> 截至 Day 6，项目已完成 Phase 1-5 全部计划任务，并提前完成了架构优化（LCEL/DSPy）和多模型适配（3 个适配器 + 校准器 + Agent）。后续将专注于答辩材料准备和最终验收。

---

## 六、后续安排

后续工作将转入答辩材料准备阶段，重点完成答辩 PPT（项目背景 / 架构 / 核心技术 / 指标数据 / Demo 截图）、演示脚本（3 组典型用例）和最终验收材料打包。

---

## 七、总结

今天按照研发计划书 Phase 5 全部任务完成了 Web 应用与可视化工作，同时完成了 LCEL Chain 架构重构、DSPy 集成（含 Teleprompter 自动优化）、多模型适配体系（协议/注册表/3 个适配器/校准器/Agent）和缓存/限流/流式等基础设施。系统已具备最终演示和验收的技术基础。
