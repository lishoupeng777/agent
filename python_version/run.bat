@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ============================================
echo LLM-as-Judge 项目环境安装和启动
echo ============================================
echo.

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境...
    D:\python\project\python.exe -m venv venv --clear
    if errorlevel 1 (
        echo 虚拟环境创建失败！
        pause
        exit /b 1
    )
    echo [1/3] 虚拟环境创建完成
) else (
    echo [1/3] 虚拟环境已存在，跳过
)

echo [2/3] 安装依赖（首次运行可能需要几分钟）...
call venv\Scripts\activate.bat
pip install --only-binary ":all:" fastapi "uvicorn[standard]" langchain langchain-core langchain-openai pydantic httpx python-dotenv jinja2 numpy scipy scikit-learn
if errorlevel 1 (
    echo 依赖安装失败！
    pause
    exit /b 1
)
echo [2/3] 依赖安装完成

echo [3/3] 启动服务（端口 8081）...
echo.
echo 中文操作页面：http://localhost:8081/
echo API 文档：http://localhost:8081/docs
echo 按 Ctrl+C 停止服务
echo.
python main.py

pause