"""
全局配置管理模块 (Core Configuration)
职责：
1. 统一管理金融多 Agent 系统所需的全局超参数、大模型调用网关与风控硬阈值。
2. 封装确定性量化模型参数（DCF 永续增长率、WACC、蒙特卡洛模拟次数、技术指标周期）。
3. 设定合规风控硬拦截沙箱红线（负债率上限、利息保障倍数、单周 VaR 损失容忍上限）。
4. 保证在无外部 API Key 时可无缝降级到本地离线保真金融模拟器，绝不发生运行时阻塞或崩溃。
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

# 定位项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"


def _load_env_file(filepath: Path) -> None:
    """轻量级 .env 环境变量解析器，无须强依赖外部三方扩展"""
    if not filepath.exists():
        return
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key not in os.environ:
                os.environ[key] = val


_load_env_file(ENV_FILE)


def save_env_llm_settings(api_key: str, base_url: str, model: str, temperature: float = 0.2) -> None:
    """持久化保存大模型配置到 .env 文件，并同步更新当前运行时环境变量"""
    os.environ["LLM_API_KEY"] = api_key.strip()
    os.environ["LLM_BASE_URL"] = base_url.strip()
    os.environ["LLM_DEFAULT_MODEL"] = model.strip()
    os.environ["LLM_TEMPERATURE"] = str(temperature)

    env_lines = []
    keys_written = set()
    target_keys = {
        "LLM_API_KEY": api_key.strip(),
        "LLM_BASE_URL": base_url.strip(),
        "LLM_DEFAULT_MODEL": model.strip(),
        "LLM_TEMPERATURE": str(temperature),
    }

    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    k, _ = stripped.split("=", 1)
                    k = k.strip()
                    if k in target_keys:
                        env_lines.append(f"{k}={target_keys[k]}\n")
                        keys_written.add(k)
                        continue
                env_lines.append(line)

    for k, v in target_keys.items():
        if k not in keys_written:
            env_lines.append(f"{k}={v}\n")

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(env_lines)


class LLMSettings(BaseModel):
    """大语言模型调用规范 (OpenAI 兼容协议)"""
    API_KEY: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    BASE_URL: str = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    DEFAULT_MODEL: str = Field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", "qwen-plus"))
    TEMPERATURE: float = Field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2")))
    REQUEST_TIMEOUT: int = Field(default=60, description="LLM HTTP 请求超时时间(秒)")


class MultiAgentDebateSettings(BaseModel):
    """投委会多智能体协同与辩论控制超参数"""
    MAX_DEBATE_ROUNDS: int = Field(
        default=2,
        description="多空辩论（Bull vs Bear）的最大交互轮次，作为防止状态图无限循环的递归熔断器"
    )
    CONVERGENCE_DELTA: float = Field(
        default=0.15,
        description="多空分歧度收敛容差阈值：当 |多头得分 - 空头得分| <= 阈值时提前达成共识"
    )
    BULL_INITIAL_WEIGHT: float = Field(default=0.5, description="多头证据初始可信度权重")
    BEAR_INITIAL_WEIGHT: float = Field(default=0.5, description="空头证据初始可信度权重")


class QuantQuantEngineSettings(BaseModel):
    """确定性量化与金融建模计算超参数"""
    # DCF 估值超参数
    DEFAULT_RISK_FREE_RATE: float = Field(default=0.025, description="无风险利率 Rf (默认中国十年期国债收益率 2.5%)")
    DEFAULT_MARKET_RISK_PREMIUM: float = Field(default=0.060, description="市场风险溢价 ERP (默认 6.0%)")
    DEFAULT_TERMINAL_GROWTH_RATE: float = Field(default=0.025, description="永续增长率 g (默认 2.5%)")
    DEFAULT_WACC: float = Field(default=0.085, description="默认加权平均资本成本 WACC (8.5%)")

    # 蒙特卡洛 VaR (Value at Risk) 风险价值超参数
    MONTE_CARLO_SIMULATIONS: int = Field(default=10000, description="蒙特卡洛模拟随机价格路径次数")
    VAR_CONFIDENCE_LEVEL: float = Field(default=0.95, description="VaR 风险价值置信水平 (95%)")
    VAR_HORIZON_DAYS: int = Field(default=5, description="VaR 预测时间窗口 (交易日)")

    # 经典技术指标超参数
    RSI_PERIOD: int = Field(default=14, description="相对强弱指标 RSI 周期")
    MACD_FAST: int = Field(default=12, description="MACD 快线 EMA 周期")
    MACD_SLOW: int = Field(default=26, description="MACD 慢线 EMA 周期")
    MACD_SIGNAL: int = Field(default=9, description="MACD 信号平滑周期")
    BOLL_PERIOD: int = Field(default=20, description="布林带移动平均周期")
    BOLL_STD_DEV: float = Field(default=2.0, description="布林带上下轨标准差倍数")


class RiskGuardrailSettings(BaseModel):
    """合规风控硬拦截沙箱红线阈值 (一票否决权)"""
    ENABLE_HARD_GUARDRAILS: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_HARD_GUARDRAILS", "true").lower() in ("true", "1", "yes"),
        description="是否启用确定性风控硬拦截沙箱"
    )
    MAX_DEBT_TO_ASSET_RATIO: float = Field(
        default=0.75,
        description="资产负债率硬性警戒红线 (高于 75% 触发极度杠杆警示)"
    )
    MIN_INTEREST_COVERAGE: float = Field(
        default=1.5,
        description="利息保障倍数硬性红线 (低于 1.5 判定为偿债高危风险)"
    )
    MAX_ACCEPTABLE_VAR_95: float = Field(
        default=0.18,
        description="单周 95% VaR 最大容忍潜在下行回撤 (大于 18% 直接触发下行风险拦截)"
    )


class Settings(BaseModel):
    """系统全局配置汇聚单例容器"""
    PROJECT_NAME: str = "基于LLM的金融分析多Agent协同系统 (Financial Multi-Agent Analysis Platform)"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1", "yes"))

    # 各子域配置组合
    llm: LLMSettings = Field(default_factory=LLMSettings)
    debate: MultiAgentDebateSettings = Field(default_factory=MultiAgentDebateSettings)
    quant: QuantQuantEngineSettings = Field(default_factory=QuantQuantEngineSettings)
    risk: RiskGuardrailSettings = Field(default_factory=RiskGuardrailSettings)


# 导出全局单例配置
settings = Settings()
