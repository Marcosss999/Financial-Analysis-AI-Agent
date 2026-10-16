"""
系统启动主入口模块 (Application Main Entrypoint)
职责：
1. 初始化 FastAPI 异步应用，配置 CORS 跨域、生命周期管理与健康检查探针。
2. 注册金融投研多智能体路由管道 (/api/v1/finance)。
3. 挂载 Claude / Antigravity 暖白风格前端静态控制台 (StaticFiles)。
4. 规范日志输出与服务端口绑定 (默认 8001 端口，与 RAG 平台 8000 端口互不冲突)。
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.api.financial_router import router as financial_router
from app.api.portfolio_router import router as portfolio_router

# 静态前端资源目录
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("\n" + "=" * 65)
    print(f"[START] {settings.PROJECT_NAME} (v{settings.VERSION}) 正在启动...")
    print(f"[API] 基础前缀: {settings.API_PREFIX}")
    print(f"[LLM] 网关配置: {'已连接云端模型' if llm_provider.is_configured else '已启用本地金融高保真模拟引擎'}")
    print(f"[QUANT] 确定性量化沙箱: 杜邦分析 / 2阶段DCF / MACD / RSI / 10,000次蒙特卡洛 VaR 已就绪")
    print(f"[RISK] 合规风控红线: 资产负债率上限 {settings.risk.MAX_DEBT_TO_ASSET_RATIO*100}% | 一票否决权已激活")
    print(f"[WEB] 交互式前端看板: http://127.0.0.1:8001")
    print(f"[DOCS] OpenAPI 交互文档: http://127.0.0.1:8001/docs")
    print("=" * 65 + "\n")
    yield
    print("\n[STOP] 金融分析多 Agent 系统正在优雅关闭...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="基于 LLM + LangGraph 的华尔街投委会多智能体协同与确定性量化风控投研平台",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置全局跨域资源共享 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册金融投研业务与模拟盘 API 路由
app.include_router(financial_router, prefix=settings.API_PREFIX)
app.include_router(financial_router)  # 兼顾根路径直接访问
app.include_router(portfolio_router, prefix=settings.API_PREFIX)
app.include_router(portfolio_router)


@app.get("/health", tags=["System Health"])
async def health_check():
    """系统健康检查探针"""
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "llm_online": llm_provider.is_configured,
        "quant_engine": "operational",
        "risk_guardrail": "active",
    }


# 挂载前端静态控制台
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
