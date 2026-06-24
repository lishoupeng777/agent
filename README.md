# 🛡️ 内容保真度与治理质量评估智能体 (LLM-as-Judge)

> 基于 DeepSeek + LangChain + Streamlit 构建的文本治理质量自动评估系统，采用 LLM-as-Judge 架构，对数据治理前后的文本进行语义一致性、可读性与结构质量的双维度自动审计。

---

## 📋 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [技术架构](#技术架构)
- [环境要求](#环境要求)
- [快速开始（组员必读）](#快速开始组员必读)
  - [1. 克隆项目](#1-克隆项目)
  - [2. 安装依赖](#2-安装依赖)
  - [3. 配置 API Key](#3-配置-api-key)
  - [4. 启动应用](#4-启动应用)
- [项目结构](#项目结构)
- [两种运行方式](#两种运行方式)
- [评估维度说明](#评估维度说明)
- [常见问题](#常见问题)

---

## 项目简介

本系统是一个 **LLM-as-Judge 智能体**，能够自动比对"数据治理前"与"数据治理后"的文本对，从以下两个维度进行严格质量审计：

| 评估维度 | 权重 | 说明 |
|----------|------|------|
| 📋 语义一致性 | 50% | 核心事实、实体、量化指标是否被误改或丢失 |
| 📐 可读性与结构质量 | 50% | 排版、语句通顺度、表格对齐、格式规范 |

### 适用场景

- 数据脱敏/清洗后的内容保真度验证
- 文本重写/摘要生成后的语义一致性检查
- LLM 输出质量审计
- 合规文本治理评估

---

## 核心功能

- ✅ **锚点预处理**：自动为文本按行标记 `[Before N]` / `[After N]` 标尺
- ✅ **双维度自动评分**：语义一致性 + 可读性结构，输出 0-100% 得分
- ✅ **瑕疵精准定位**：自动识别"过度清洗""语义误改""可读性下降""结构破坏"四类瑕疵
- ✅ **行级锚点溯源**：每个瑕疵精确定位到原文/后文的行号
- ✅ **可视化仪表盘**：Metric 卡片 + 着色瑕疵表格 + 通过/不通过徽章
- ✅ **评估结果缓存**：相同输入自动缓存，避免重复调用 API
- ✅ **FastAPI 后端**：提供 REST API，支持批量评估和 JSON 报告导出
- ✅ **中文前端页面**：`static/index.html` 提供开箱即用的 Web 操作界面

---

## 技术架构

```
用户输入 (治理前/后文本)
       │
       ▼
┌─────────────────┐
│  锚点预处理模块   │  ← 按行标记 [Before N] / [After N]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prompt 构建模块  │  ← System Prompt（含评分细则 + JSON Schema）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DeepSeek API    │  ← ChatOpenAI (temperature=0.0, max_retries=3)
│  裁判模型调用      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JSON 解析与清洗  │  ← 去除 <think> 标签 + JSON 提取 + 兜底
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  评估结果渲染     │  ← Streamlit / FastAPI 前端可视化
└─────────────────┘
```

**技术栈**：
- **裁判模型**：DeepSeek Chat API
- **LLM 框架**：LangChain + langchain-openai
- **前端**：Streamlit（演示版）+ FastAPI + 中文 HTML 静态页面
- **数据处理**：Pandas + NumPy
- **可观测性**：LangSmith（可选）
- **评估指标**：Pearson r / Spearman ρ / F1 / MAE / RMSE

---

## 环境要求

| 组件 | 要求 |
|------|------|
| Python | **3.11+**（实测兼容 3.13） |
| pip | 最新版 |
| DeepSeek API Key | 需要注册获取：[platform.deepseek.com](https://platform.deepseek.com) |
| 操作系统 | Windows / macOS / Linux |
| 网络 | 需要能够访问 `api.deepseek.com` |

> ⚠️ **重要**：本项目依赖 DeepSeek API，必须拥有有效的 API Key 才能运行。

---

## 快速开始（组员必读）

### 1. 克隆项目

```bash
# 克隆仓库
git clone https://github.com/lishoupeng777/agent.git

# 进入项目目录
cd agent/python_version
```

### 2. 安装依赖

#### 方式 A：使用 pip 安装（推荐）

```bash
# 创建虚拟环境（可选但强烈推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 方式 B：使用一键启动脚本（Windows）

```bash
# 双击运行，自动创建虚拟环境 + 安装依赖 + 启动服务
run.bat
```

> **注意**：`run.bat` 中硬编码了 Python 路径 `D:\python\project\python.exe`，如果你的 Python 路径不同，请修改该脚本或用方式 A 手动安装。

### 3. 配置 API Key

在 `python_version` 目录下创建 `.env` 文件：

```bash
# .env 文件内容
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # 可选，用于追踪 LLM 调用链
```

> 🔐 `.env` 文件已在 `.gitignore` 中排除，不会被提交到 GitHub。
>
> 也可以不创建 `.env` 文件，在 Streamlit 页面侧边栏直接输入 API Key（仅当次有效）。

### 4. 启动应用

#### 方式 A：Streamlit 演示版（推荐用于演示/调试）

```bash
# 在 python_version 目录下运行
streamlit run app.py
```

然后在浏览器打开 **http://localhost:8501**

> 这个版本提供完整的中文可视化界面，包括侧边栏配置、双文本域输入、Metric 卡片、瑕疵表格等。

#### 方式 B：FastAPI 后端 + 中文前端页面

```bash
# 在 python_version 目录下运行
python main.py
```

然后在浏览器打开：
- 中文操作页面：**http://localhost:8081/**
- API 文档（Swagger UI）：**http://localhost:8081/docs**

---

## 项目结构

```
shixun/
├── README.md                          # ← 本文件
├── 智能体研究课题任务书（终稿).pdf      # 课题任务书
│
└── python_version/                    # 主项目目录
    ├── .gitignore                     # Git 忽略规则（排除 venv/.env/__pycache__ 等）
    ├── requirements.txt               # Python 依赖清单（含精确版本号）
    ├── .env                           # 环境变量（需自行创建，不提交到 Git）
    │
    ├── app.py                         # 🎨 Streamlit 可视化演示应用（主 UI）
    ├── main.py                        # 🚀 FastAPI 服务启动入口
    ├── run_calibration.py             # 📊 校准测试脚本（评估 LLM vs 人工一致性）
    ├── run.bat                        # 🪟 Windows 一键启动脚本
    │
    ├── 研发计划书.md                   # 研发计划文档
    ├── 研发计划与操作指导.md            # 详细研发计划与操作指导
    │
    ├── app/                           # 核心模块包
    │   ├── __init__.py
    │   ├── engine.py                  # DeepSeek LLM 调用封装 + JSON 解析
    │   ├── models.py                  # Pydantic 数据模型定义
    │   ├── prompts.py                 # System Prompt 模板
    │   ├── calibration.py             # 一致性校准（Pearson / Spearman / MAE）
    │   ├── metrics.py                 # 瑕疵检出指标（Precision / Recall / F1）
    │   ├── debias.py                  # 抗偏置检测（长度偏置 / 位置偏置）
    │   ├── stability.py               # 评分稳定性验证（多次采样 + 方差）
    │   ├── reporter.py                # 综合评估报告生成器
    │   └── routes.py                  # FastAPI 路由
    │
    ├── static/                        # 中文前端静态文件
    │   └── index.html                 # 中文 Web 操作页面
    │
    ├── data/                          # 数据文件
    │   ├── __init__.py
    │   ├── eval_dataset.json          # 评估数据集（含人工标注 Ground Truth）
    │   └── calibration_result.json    # 校准测试结果输出
    │
    └── tests/                         # 测试用例
        └── test_evaluate.py           # 评估功能测试
```

---

## 两种运行方式

### 🎨 Streamlit 演示版 (`app.py`)

```bash
streamlit run app.py
```

- **端口**：8501
- **特点**：完整可视化界面、侧边栏 API Key 配置、实时评估结果卡片、瑕疵着色表格
- **适用**：演示 / 答辩 / 单次文本对评估

**使用流程**：
1. 在左侧边栏输入 DeepSeek API Key
2. 左侧输入"治理前原始文本"和"治理后重写文本"
3. 点击"🚀 开始评估"
4. 右侧显示得分、瑕疵清单、裁判解释

### 🚀 FastAPI 后端版 (`main.py`)

```bash
python main.py
```

- **端口**：8081
- **特点**：REST API + 中文 Web 页面 + Swagger 文档
- **适用**：集成到其他系统 / 批量评估 / 生产环境

**API 端点**：
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 中文操作页面 |
| POST | `/evaluate` | 单次文本对评估 |
| POST | `/batch-evaluate` | 批量评估 |
| GET | `/docs` | Swagger API 文档 |

---

## 评估维度说明

### 语义一致性评分 (0.0~1.0)

| 瑕疵等级 | 扣分 | 示例 |
|----------|------|------|
| 🔴 严重 | 0.5~0.6 分/次 | 核心事实、实体、量化指标误删/误改 |
| 🟡 中等 | 0.25~0.35 分/次 | 语义偏离，修饰词或专有名词误改 |
| 🟢 轻微 | 0.1~0.15 分/次 | 非必要同义词替换（过度清洗） |

> **累计扣分规则**：2 个以上严重瑕疵 → 总分 ≤0.3；关键数据误改 → 总分 ≤0.4；大幅删除 → 总分 ≤0.2

### 可读性与结构质量评分 (0.0~1.0)

| 瑕疵等级 | 扣分 | 示例 |
|----------|------|------|
| 🔴 严重 | 0.4~0.5 分/次 | 表格完全散架，排版结构被破坏 |
| 🟡 中等 | 0.2~0.25 分/次 | 语句不通顺，表格对齐混乱 |
| ➕ 加分 | +0.05 分 | 格式比原文更规范（加分后不超过 1.0） |

### 整体判定

- ✅ **整体通过**：语义一致性 ≥ 80% **且** 可读性结构 ≥ 80%
- ❌ **整体不通过**：任一维度未达标

---

## 常见问题

### Q1: 运行时报错 `DEEPSEEK_API_KEY not found`

**A**: 请先创建 `.env` 文件并填写 API Key（参考上方 [配置 API Key](#3-配置-api-key)），或在 Streamlit 侧边栏直接输入。

### Q2: DeepSeek API 调用超时

**A**: 
1. 检查网络是否可访问 `api.deepseek.com`
2. 检查 API Key 是否有效、是否有余额
3. 大文本（>5000 字）可能导致超时，请适当缩短文本

### Q3: Streamlit 端口 8501 被占用

**A**: 使用自定义端口启动：
```bash
streamlit run app.py --server.port 8502
```

### Q4: FastAPI 端口 8081 被占用

**A**: 修改 `main.py` 中的 `port=8081` 为其他端口。

### Q5: 安装依赖时报错

**A**: 
1. 确认 Python 版本 ≥ 3.11：`python --version`
2. 升级 pip：`pip install --upgrade pip`
3. 如果某个包安装失败，尝试单独安装：`pip install <包名>`
4. Windows 用户如遇到 `scipy` 安装失败，可尝试：`pip install scipy --only-binary ":all:"`

### Q6: 评估结果不稳定/打分偏低

**A**: 
- 裁判模型使用 `temperature=0.0` 以保证可复现性
- 评分标准设在 System Prompt 中，可通过调整 Prompt 进行校准
- 运行 `run_calibration.py` 验证 Pearson r ≥ 0.8

### Q7: 如何获取 DeepSeek API Key？

**A**: 
1. 访问 [platform.deepseek.com](https://platform.deepseek.com)
2. 注册/登录账号
3. 在"API Keys"页面创建新的 API Key
4. 复制 Key 并配置到 `.env` 文件

---

## 📞 联系方式

- **GitHub**：[https://github.com/lishoupeng777/agent](https://github.com/lishoupeng777/agent)
- **课题**：大模型内容安全与质量评估 / 数据治理前后文本比对
- **技术栈**：Python · DeepSeek · LangChain · Streamlit · FastAPI

---

> 💡 **提示**：首次运行建议先用 Streamlit 版本 (`streamlit run app.py`) 体验完整功能，熟悉后再使用 FastAPI 版本进行集成或批量评估。