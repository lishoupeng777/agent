# 🛡️ 内容保真度与治理质量评估智能体 (LLM-as-Judge)

> 基于 DeepSeek + LangChain + FastAPI 构建的文本治理质量自动评估系统

---

## 项目简介

LLM-as-Judge 智能体，自动比对"数据治理前"与"数据治理后"的文本对，从语义一致性和可读性结构两个维度进行质量审计。

## 快速开始

```bash
cd python_version
# 安装依赖
pip install -r requirements.txt
# 配置 API Key（创建 .env 文件，写入 DEEPSEEK_API_KEY=sk-xxx）
# 启动 FastAPI 后端
python main.py
```

## 项目结构

```
shixun/
├── README.md
├── python_version/
│   ├── main.py                 # FastAPI 服务入口
│   ├── run_calibration.py      # 校准测试脚本
│   ├── run.bat                 # Windows 一键启动脚本
│   ├── requirements.txt
│   ├── app/                    # 核心模块包
│   │   ├── engine.py           # DeepSeek LLM 调用 + JSON 解析
│   │   ├── models.py           # Pydantic 数据模型
│   │   ├── prompts.py          # System Prompt 模板
│   │   ├── calibration.py      # 一致性校准
│   │   ├── metrics.py          # 瑕疵检出指标
│   │   ├── debias.py           # 抗偏置检测
│   │   ├── stability.py        # 评分稳定性验证
│   │   ├── reporter.py         # 综合报告生成
│   │   └── routes.py           # FastAPI 路由
│   ├── static/index.html       # 中文前端页面
│   ├── data/                   # 评估数据集
│   ├── docs/                   # 文档（研发计划书等）
│   ├── scripts/                # 工具脚本
│   └── tests/                  # 测试用例
```

## 运行方式

```bash
cd python_version
python main.py    # 启动 FastAPI 后端，端口 8081
```

前端页面：`static/index.html`，启动后端后访问 `http://localhost:8081`

## 技术栈

Python · DeepSeek · LangChain · FastAPI · Pydantic · LiteLLM

## 文档

- [研发计划书](python_version/docs/研发计划书.md)
- [研发计划与操作指导](python_version/docs/研发计划与操作指导.md)
- [课题任务书](python_version/docs/智能体研究课题任务书（终稿).pdf)

## 📞 联系方式

- **GitHub**：[https://github.com/lishoupeng777/agent](https://github.com/lishoupeng777/agent)