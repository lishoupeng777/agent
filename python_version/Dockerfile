# ============================================================
# 多阶段构建：fidelity-eval
# Stage 1: builder — 安装依赖
# Stage 2: runtime — 最小镜像运行 FastAPI
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt pyproject.toml ./
COPY app/ app/
COPY fidelity_eval/ fidelity_eval/
COPY static/ static/
COPY data/ data/
COPY scripts/ scripts/
COPY main.py ./

# 安装依赖到虚拟环境（隔离系统包）
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制应用代码
COPY --from=builder /build /app

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# 环境变量
ENV DEEPSEEK_API_KEY="" \
    DEEPSEEK_BASE_URL="https://api.deepseek.com/v1" \
    DEEPSEEK_MODEL="deepseek-v4-flash"

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:8081/api/v1/health').raise_for_status()"

EXPOSE 8081

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
