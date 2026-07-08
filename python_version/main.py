"""服务启动入口"""
from __future__ import annotations

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# 自动加载 .env 文件中的环境变量
load_dotenv()

from app.routes import router
from app.task_manager import load_tasks_from_disk
from app.chain import enable_rate_limiter

app = FastAPI(
    title="内容保真度与治理质量评估智能体",
    description="LLM-as-Judge: 抽样比对治理前后文本，判定语义一致性、识别过度清洗与误改、评估可读性与结构质量",
    version="1.0.0",
)

# CORS 中间件（开发环境允许所有来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# 启动时恢复任务状态
load_tasks_from_disk()

# 启用 LLM API 限流（每秒最多 3 次请求，防止触发 429）
enable_rate_limiter(requests_per_second=3.0)

# 挂载静态文件目录（中文前端页面）
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def root():
        """默认跳转到中文操作页面"""
        return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8081,
        reload=False,
    )