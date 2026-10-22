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


def _env_float(name: str, default: float) -> float:
    """读取浮点型环境变量，非法值回退默认，避免因配置笔误导致启动失败"""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("true", "1", "yes")


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
    REQUEST_TIMEOUT: float = Field(
        default_factory=lambda: _env_float("LLM_REQUEST_TIMEOUT", 60.0), description="单次 LLM 请求超时(秒)"
    )
    MAX_RETRIES: int = Field(
        default_factory=lambda: _env_int("LLM_MAX_RETRIES", 2), description="应用层最大重试次数(指数退避)"
    )
    RETRY_BASE_DELAY: float = Field(default=0.5, description="重试退避基数(秒)")
    BREAKER_FAILURE_THRESHOLD: int = Field(
        default_factory=lambda: _env_int("LLM_BREAKER_THRESHOLD", 3), description="连续失败多少次后熔断"
    )
    BREAKER_RECOVERY_SECONDS: float = Field(
        default_factory=lambda: _env_float("LLM_BREAKER_RECOVERY", 30.0), description="熔断后多久进入半开探测"
    )


class MultiAgentDebateSettings(BaseModel):
    """投委会多智能体协同与辩论控制超参数"""
    MAX_DEBATE_ROUNDS: int = Field(
        default_factory=lambda: _env_int("MAX_DEBATE_ROUNDS", 2),
        description="多空辩论（Bull vs Bear）的最大交互轮次，作为防止状态图无限循环的递归熔断器"
    )
    CONVERGENCE_DELTA: float = Field(
        default_factory=lambda: _env_float("CONVERGENCE_DELTA", 0.15),
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
    VAR_SEED: int = Field(default=42, description="蒙特卡洛随机种子(使用局部 Generator，不污染全局随机状态)")

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
        default_factory=lambda: _env_float("MAX_DEBT_TO_ASSET_RATIO", 0.75),
        description="资产负债率硬性警戒红线 (高于 75% 触发极度杠杆警示)"
    )
    MIN_INTEREST_COVERAGE: float = Field(
        default_factory=lambda: _env_float("MIN_INTEREST_COVERAGE_RATIO", 1.5),
        description="利息保障倍数硬性红线 (低于 1.5 判定为偿债高危风险)"
    )
    FINANCIAL_INSTITUTION_MAX_DEBT_PCT: float = Field(
        default=94.0, description="持牌金融机构资产负债率红线(%)，其负债含存款，口径不同于实体企业"
    )
    WARN_DEBT_PCT: float = Field(default=65.0, description="资产负债率预警线(%)，仅降级为 WARNING 不否决")
    WARN_VAR_PCT: float = Field(default=12.0, description="VaR 预警线(%)，仅降级为 WARNING 不否决")
    MAX_ACCEPTABLE_VAR_95: float = Field(
        default_factory=lambda: _env_float("MAX_VAR_TOLERANCE", 0.18),
        description="单周 95% VaR 最大容忍潜在下行回撤 (大于 18% 直接触发下行风险拦截)"
    )


class DecisionSettings(BaseModel):
    """确定性决策参数：评分 → 档位 → 仓位的映射，以及辩论调整幅度"""
    BAND_STRONG_BUY: float = Field(default_factory=lambda: _env_float("BAND_STRONG_BUY", 7.5), description="≥ 该分：强烈买入")
    BAND_OVERWEIGHT: float = Field(default_factory=lambda: _env_float("BAND_OVERWEIGHT", 6.3), description="≥ 该分：逢低增配")
    BAND_NEUTRAL: float = Field(default_factory=lambda: _env_float("BAND_NEUTRAL", 4.8), description="≥ 该分：中性持有，低于则规避")
    DEBATE_MAX_ADJUST: float = Field(
        default_factory=lambda: _env_float("DEBATE_MAX_ADJUST", 0.8),
        description="多空辩论对最终得分的最大调整幅度(±)。有界，保证用户设定的权重始终主导结论",
    )
    ATR_STOP_MULT: float = Field(default=2.0, description="止损距离 = ATR_STOP_MULT × ATR")
    RISK_PER_TRADE_PCT: float = Field(default=1.0, description="单笔交易愿意承担的账户风险(%)，用于反推仓位上限")
    LLM_DIVERGENCE_THRESHOLD: float = Field(default=2.5, description="LLM 评分与量化评分相差超过该值时标记为分歧")


class EngineeringSettings(BaseModel):
    """工程化稳定性与存储配置"""
    ANALYSIS_TIMEOUT_SECONDS: float = Field(
        default_factory=lambda: _env_float("ANALYSIS_TIMEOUT_SECONDS", 120.0), description="单次全流程分析总超时"
    )
    PARALLEL_EXPERTS: bool = Field(
        default_factory=lambda: _env_bool("PARALLEL_EXPERTS", True), description="三位专家是否并行执行(fan-out/fan-in)"
    )
    STRICT_REAL_DATA: bool = Field(
        default_factory=lambda: _env_bool("STRICT_REAL_DATA", False),
        description="为 true 时，行情/财报抓取失败直接报错，而不是降级为模拟数据",
    )
    RATE_LIMIT_PER_MINUTE: int = Field(
        default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 30), description="每个客户端每分钟请求上限"
    )
    IDEMPOTENCY_TTL_SECONDS: float = Field(default=600.0, description="幂等键缓存有效期")
    DB_PATH: str = Field(
        default_factory=lambda: os.getenv("MEMORY_DB_PATH", str(ROOT_DIR / "data" / "memory.db")),
        description="SQLite 记忆库路径",
    )
    CHAT_KEEP_RECENT_TURNS: int = Field(default=6, description="对话记忆保留原文的最近轮数，更早的压缩为摘要")
    CHAT_CONTEXT_CHAR_BUDGET: int = Field(default=6000, description="拼入 prompt 的对话上下文字符预算")
    CORS_ORIGINS: str = Field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "http://127.0.0.1:8001,http://localhost:8001"),
        description="允许的前端来源，逗号分隔",
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
    eng: EngineeringSettings = Field(default_factory=EngineeringSettings)
    decision: DecisionSettings = Field(default_factory=DecisionSettings)


# 导出全局单例配置
settings = Settings()
