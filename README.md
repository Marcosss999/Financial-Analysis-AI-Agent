# 基于 LLM 的金融分析多 Agent 投研协同系统
### (Financial Multi-Agent Investment Research & Trading Platform)

> 一个高保真、工业级架构的金融多智能体（Multi-Agent）投研协同决策平台。结合 **LangGraph 状态图工作流编排**、**多源真实行情与情报抓取**、**确定性量化估值/风控沙箱**、**多空博弈对抗辩论机制** 与 **投研模拟盘交易闭环**。

---

## 🌟 核心亮点与系统特性

1. **多源异构真实金融数据与情报感知**：
   - **资金面**：直连东方财富分时主力资金流接口，实时提取主力大单与超大单净买卖动向。
   - **信息面**：直连东方财富/巨潮官方上市公司正式披露公告库，提取近期重大公告（减持、质押、定增、业绩快报等）。
   - **舆情面**：对接新浪财经个股滚动要闻流，捕获盘口即时动态。
   - **行情与财报**：实时拉取日K线历史与财务三表，支持离线沙箱弹性降级。

2. **专业多智能体分工协同 (LangGraph StateGraph)**：
   - 🏛️ **宏观策略专家 (Macro Agent)**：研判宏观流动性、货币与产业政策、地缘政治与周期位置。
   - 📊 **基本面专家 (Fundamental Agent)**：运行**杜邦分析法 (DuPont Analysis)** 穿透 ROE 质量，结合 **三阶段 DCF 自由现金流折现模型** 测算内在公允价值与安全边际。
   - 📈 **技术形态专家 (Technical Agent)**：纯 Python/NumPy 确定性量化引擎，测算 14日 RSI、MACD、布林带轨道及支撑/阻力位。

3. **双轮动态多空对抗辩论与收敛仲裁 (Debate & Consensus Engine)**：
   - 🐂 **多头首席辩手 (Bull Advocate)**：挖掘预期差、催化剂（Catalysts）与估值修复上行空间；次轮对空方质疑进行针对性反驳与证据防御。
   - 🐻 **空头首席辩手 (Bear Critic)**：紧盯财务杠杆风险、预期透支与估值泡沫，指出致命漏洞（Vulnerabilities）；次轮进行穿透式精准质询。
   - ⚖️ **投委会主席 (Arbiter Node)**：调和多空分歧，提炼核心矛盾与赔率不对称比（Risk/Reward），出具终局投资评级、建议仓位与梯次建仓节奏。
   - 🛡️ **防无限死循环熔断器**：状态图内置收敛容差判定与最大辩论轮次硬熔断控制。

4. **确定性合规风控硬拦截沙箱 (Risk Guardrail)**：
   - **10,000 次蒙特卡洛 VaR 模拟**：基于历史几何布朗运动随机走势，测算 95% 置信度下单周最大潜在在险损失。
   - **一票否决权**：对资产负债率、利息保障倍数、极限下行回撤等硬红线实行确定性规则穿透拦截，防范大模型“幻觉”导致合规失误。

5. **模拟盘交易撮合引擎 (Paper Trading System)**：
   - 提供 100 万元初始虚拟资金账户管理，本地 JSON 持久化存储。
   - 支持市价委托撮合、真实交易印花税/佣金费率核算、持仓浮动盈亏与已实现盈亏统计。
   - **投研一键建仓联动**：支持将投委会决议建议仓位自动换算为 100 股（1手）整倍数并一键直达模拟盘下单。

6. **沉浸式现代 Web 交互看板**：
   - 原生 HTML5/CSS3/JavaScript 编写，无需复杂前端打包环境。
   - 支持多策略风格一键切换（动量主升、机构均衡、深度价值、短线游资）及 4 维指标权重自定义微调。
   - 集成大模型网关可视化配置弹窗（可在线切换 DeepSeek / Qwen 等 OpenAI 协议模型并进行实时联通测试）。
   - 全流程 Server-Sent Events (SSE) 实时打字机流式事件推送。


---

## 🧱 v2：记忆库与工程化稳定性

| 模块 | 做了什么 | 代码位置 |
|---|---|---|
| **专家并行** | 宏观 / 基本面 / 技术面三位专家 fan-out 并发，汇合节点 fan-in 后再进入辩论；`PARALLEL_EXPERTS=false` 可回退串行 | `app/graph/financial_graph.py` |
| **长期记忆** | 每次分析落 SQLite（结论、评分、风控、数据质量、研报、trace）；再次分析同一标的自动召回最近 3 次，研报末尾输出“与上次结论对比” | `app/memory/store.py` |
| **会话记忆 + 上下文压缩** | 追问对话按 `session_id` 存服务端；最近 6 轮保留原文，更早的滚动压缩为摘要（LLM 摘要，不可用时退化为抽取式），再按字符预算兜底裁剪；研报背景直接取记忆库里的最近结论 | `app/memory/chat_memory.py` |
| **重试 / 超时 / 熔断** | LLM 调用：单次超时 + 指数退避重试（带抖动）+ 三态熔断器；整次分析另有总超时 | `app/core/resilience.py`、`llm_provider.py` |
| **限流 / 幂等** | 令牌桶按客户端限流（429 + Retry-After）；`Idempotency-Key` 请求头保证重复提交只执行一次，并发重复请求共享结果 | `app/core/middleware.py`、`resilience.py` |
| **Trace 可观测** | 每次分析一个 `run_id`，每个节点一个 span（耗时 / 状态 / LLM 调用 / 重试 / 降级 / token）；`GET /finance/runs/{run_id}` 可复盘 | `app/core/observability.py` |
| **降级可见** | 行情 / 财报是否模拟、哪些字段是估算、LLM 降级了几次、哪些节点失败降级，全部写进响应的 `data_quality` 和研报顶部横幅；`STRICT_REAL_DATA=true` 时直接报错而不降级 | `app/services/analysis_service.py` |
| **测试** | 84 个用例：风控阈值、熔断/重试/限流/幂等、图拓扑与并行、辩论熔断、记忆与压缩、服务层回归、量化打分单调性与消融实验自检 | `tests/` |

运行测试：`pytest -q`（测试强制离线，全部通过，耗时约 7~10s）。

新增接口：`GET /finance/runs`、`GET /finance/runs/{run_id}`、`GET /finance/status`；追问接口新增可选 `session_id`。

## 🎛️ v3：权重驱动的确定性决策（与 LLM 是否在线无关）

**核心原则：确定性工具算事实和分数 → 用户权重决定结论 → LLM 只负责解读与论证。**

```
各维度量化评分(0~10) ──┐
 基本面: 估值/盈利/成长/杠杆      │   基础分 = Σ 用户权重 × 维度评分
 技术面: 趋势/动量-反转/量能/波动率 ├──►  (缺数据的维度剔除，权重按比例分给其余维度)
 宏观: 沪深300 趋势与波动状态      │            │
 资金面: 主力净流入/市值 + 量能确认 ┘            ▼
                                   最终分 = 基础分 + 多空辩论调整(有界 ±0.8)
                                            │
                      档位 / 仓位 ◄──────────┘──► ATR 止损 + 单笔风险预算仓位上限
```
- 改权重 → 基础分必然改变；LLM 给出任何分数（含离谱值）都不进入决策，只记为 `llm_score` 用于“分歧检测”。（测试：`tests/test_decision.py`）
- **离线模式**：专家叙述由数据模板生成（明确标注“非大模型推理”），分数与在线时完全一致；在线时 LLM 的文字替换模板。
- **新增技术指标**：ATR(14, Wilder)、20/5 日实现波动率及波动比、量比、OBV 斜率、MFI(14)、5/20 日涨跌幅、布林 %B/带宽、MA20/MA60。
- **诚实降级**：资金流接口不可用时不再编造“净流入 1280 万”，该维度直接剔除；公告/新闻不可用时如实写“数据源暂不可用”。
- **实证消融实验套件 (`experiments/`)**：
  - **实验一：技术指标真实回测 (`indicator_backtest.py`)**：基于 2019~2025 年 28 只沪深大盘股真实日 K，验证复合因子与单因子的 1d/3d/5d 秩相关系数 (IC)、IC_IR 与净换手价差。详见 `experiments/results_indicator.md`。
  - **实验二：多空辩论开/关消融 (`debate_ablation.py`)**：在严格晚于大模型训练截止期的时间窗口（2026 年）进行跨截面样本外评估，实测多空辩论对纯量化指标预测力无统计显著提升（95%CI 包含 0），论证辩论核心价值在于多模态冲突仲裁（如业绩暴增 vs 技术破位）。详见 `experiments/results_debate.md`。
  - **实验三：技术面分项消融与时间段切分 (`component_ablation.py`)**：遵循严苛量化纪律切分 train (到2022) / valid (2023) / test (2024起) 防止过拟合，实测动量/反转分项具有稳定增量贡献。详见 `experiments/results_component.md`。
- 新增环境变量：`BAND_STRONG_BUY / BAND_OVERWEIGHT / BAND_NEUTRAL / DEBATE_MAX_ADJUST`。

### 已知局限（面试可直接讲）
- **风控输入的可靠度**：真实财报接口只提供营收、净利润、资产负债率、ROE 等；总资产、经营现金流、利息费用、EBIT 是按经验系数推算的，已标记为 `estimated_fields`，**利息保障倍数检查因此被跳过而不是据此否决**。资产负债率与 VaR 两条硬红线不受影响。
- **监管舆情检查未接入数据源**，在审计清单里如实标为“未检查”。
- **离线模式**（无 API Key / 无外网）下专家与辩手的文字来自数据模板，不是真正的 Agent 推理；决策分数不受影响，所有降级的 LLM 调用都计入 `llm_fallback_calls`。离线时宏观维度使用的是标注为 simulated 的模拟指数。
- **并行只降延迟，不改变分析结论**：三位专家互不读取对方产出，串行与并行的评分在测试中一致。端到端收益受限于辩论环本身是串行的（见基准）。
- 记忆库是单机 SQLite；多实例部署需换成 MySQL/Redis。限流与幂等缓存是进程内的，同理。
- 未使用 LangGraph checkpointer：单次分析是一次性短流程，持久化放在业务层（SQLite）；如需“断点续跑”再接入。
- 模拟盘 JSON 持久化无并发写保护，仅适合单用户演示。

---

## 🏗️ 系统架构图

```mermaid
flowchart TD
    Start([用户输入: 标的代码 & 策略权重]) --> DataNode[数据准备与情报抽取节点\n行情/财报/主力资金/官方公告/滚动新闻]
    
    subgraph 专家投研层 (Fan-Out 并发执行)
        DataNode --> Macro[宏观策略专家 Macro Agent]
        DataNode --> Fund[基本面与估值专家 Fundamental Agent\n杜邦拆解 + DCF现金流模型]
        DataNode --> Tech[量化技术专家 Technical Agent\nMACD / RSI / 布林带 / ATR / 波动率]
    end
    
    Macro --> Merge[状态汇合与确定性量化评分节点\n四维量化打分 + 用户权重驱动聚合]
    Fund --> Merge
    Tech --> Merge
    
    subgraph 多空辩论博弈与仲裁层
        Merge --> Bull[多头首席辩手 Bull Advocate\n首轮立论: 催化剂 / 次轮反驳]
        Bull --> Bear[空头首席辩手 Bear Critic\n首轮批判: 风险警示 / 次轮质询]
        Bear --> Arbiter[投委会主席仲裁官 Arbiter\n核心矛盾判定 / 有界微调 ±0.8]
        Arbiter -->|未收敛且轮次<2| Bull
    end
    
    subgraph 合规风控与交付层
        Arbiter -->|已收敛或达到最大轮次| Risk[合规风控沙箱 Risk Guardrail\n10,000次蒙特卡洛VaR / 财务硬指标一票否决]
        Risk --> Reporter[研报排版总编 Reporter Agent\n生成机构级结构化深度研报 & 决策贡献表]
        Reporter --> Output([展示终稿研报 / 联动模拟盘下单])
    end
```

---

## 📂 项目工程目录结构

```text
01-基于LLM的金融分析AI_Agent/
├── app/
│   ├── agents/                     # 智能体专长模块
│   │   ├── base_agent.py           # Agent 抽象基类与安全 JSON 解析
│   │   ├── macro_agent.py          # 宏观与产业政策专家
│   │   ├── fundamental_agent.py    # 基本面财务与估值专家
│   │   ├── technical_agent.py      # 技术形态与指标专家
│   │   ├── debate_engine.py        # 多空辩论对决与仲裁引擎 (Bull / Bear / Arbiter)
│   │   ├── risk_guardrail.py       # 合规风控硬拦截沙箱
│   │   └── reporter_agent.py       # 机构研报排版生成 Agent
│   ├── api/                        # FastAPI 路由层
│   │   ├── financial_router.py     # 投研分析、Runs 追溯、SSE 流式接口、模型配置
│   │   └── portfolio_router.py     # 模拟盘账户资产、委托撮合接口
│   ├── core/                       # 工业级高可用韧性基础设施
│   │   ├── config.py               # 全局超参数与环境配置加载器
│   │   ├── llm_provider.py         # OpenAI 协议兼容的大模型统一调度器
│   │   ├── resilience.py           # 重试+指数退避+抖动、三态熔断器、令牌桶限流
│   │   ├── observability.py        # 全链路 Trace/Span 调用追踪与性能统计
│   │   └── middleware.py           # 幂等键拦截器 (Idempotency-Key)
│   ├── graph/                      # 状态图编排层
│   │   ├── financial_graph.py      # LangGraph 投委会多智能体状态图拓扑 (Fan-Out/Fan-In)
│   │   └── state_utils.py          # 状态合并与并发写隔离安全工具
│   ├── memory/                     # 记忆管理系统
│   │   ├── store.py                # SQLite 长期投研结论沉淀库 (支持历史对比)
│   │   └── chat_memory.py          # 会话记忆与滑动窗口 LLM 摘要压缩
│   ├── models/                     # Pydantic 数据规范与协议契约
│   │   ├── agent_state.py          # 全局 LangGraph 状态机字典类型 (FinancialAgentState)
│   │   ├── schemas.py              # 研报、行情、情报、多空辩论、量化贡献表结构模型
│   │   └── portfolio_schemas.py    # 模拟盘持仓、委托订单数据契约
│   ├── services/                   # 业务逻辑与协调服务层
│   │   ├── analysis_service.py     # 投研编排服务 (支持同步与 SSE 流式双通道)
│   │   └── runs_service.py         # 历史 Run 记录检索与复盘服务
│   └── tools/                      # 确定性量化分析与外部数据工具
│       ├── scoring.py              # 【v3核心】四维量化打分引擎与权重聚合器
│       ├── market_data.py          # 行情历史与财报接口 (集成东财/新浪/腾讯及离线沙箱)
│       ├── market_intel.py         # 多源资金流、官方披露公告、滚动舆情抓取器
│       ├── financial_metrics.py    # 杜邦拆解与三阶段 DCF 估值引擎
│       ├── quant_indicators.py     # MACD、RSI、布林带、ATR、波动率量化引擎
│       ├── risk_calculator.py      # 10,000 次蒙特卡洛 VaR 模拟风险价值引擎
│       └── portfolio_manager.py    # 模拟账户资产持久化与撮合管理
├── experiments/                    # 【v3实证研究与消融实验套件】
│   ├── data_loader.py              # 历史日 K 数据加载器 (支持断点缓存与双数据源兜底)
│   ├── indicator_backtest.py       # 实验一：技术指标回测 (IC, IC_IR, 分位净价差)
│   ├── debate_ablation.py          # 实验二：多空辩论 ON/OFF 样本外消融研究
│   ├── component_ablation.py       # 实验三：技术面四分项消融与时间段切分 (防过拟合)
│   ├── metrics.py                  # 块自助法 CI (Block Bootstrap)、Spearman 秩相关计算
│   ├── results_indicator.md        # 实验一量化结果报告 (真实数据 2019~2025)
│   ├── results_debate.md           # 实验二多空辩论消融评估报告 (未来样本外)
│   └── results_component.md        # 实验三技术面子项消融报告 (Train/Valid/Test)
├── tests/                          # 自动化测试套件 (83 个用例全部通过)
│   ├── test_decision.py            # 确定性打分、权重驱动、档位与止损测试
│   ├── test_experiments.py         # 消融实验自检测试 (合成数据零信号/植入信号)
│   ├── test_graph.py               # 图拓扑结构、辩论环收敛与轮次熔断测试
│   ├── test_handlers.py            # API 路由与输入校验测试
│   ├── test_llm_provider.py        # LLM 客户端重试、容灾降级测试
│   ├── test_memory.py              # SQLite 长期记忆与会话摘要压缩测试
│   ├── test_resilience.py          # 熔断器、令牌桶限流、幂等键测试
│   ├── test_risk_and_data.py       # 风控阈值一票否决、数据质量透传测试
│   ├── test_schemas.py             # 数据模型契约测试
│   └── test_service.py             # 业务服务层端到端集成测试
├── data/                           # 本地持久化数据目录 (模拟盘账本等)
│   └── paper_portfolio.json        # 模拟盘账户数据
├── static/                         # 前端 Web 静态资源
│   └── index.html                  # 现代化响应式投研交互看板
├── .env.example                    # 环境变量配置模板
├── CHANGELOG.md                    # 详细版本升级与修改事实记录
├── main.py                         # 系统启动入口文件
├── requirements.txt                # Python 依赖清单
└── README.md                       # 项目工程文档
```

---

## 🚀 快速上手与部署运行

### 1. 环境准备
推荐使用 **Python 3.10 ~ 3.12** 运行环境。建议在虚拟环境中运行：

```bash
# 创建虚拟环境 (Windows PowerShell)
python -m venv .venv

# 激活虚拟环境 (Windows)
.venv\Scripts\Activate.ps1

# 激活虚拟环境 (Linux/macOS)
source .venv/bin/activate
```

### 2. 安装项目依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量 (可选但推荐)
复制项目根目录下的 `.env.example` 为 `.env`：
```bash
cp .env.example .env
```
用文本编辑器打开 `.env` 并填入您的大模型 API 密钥（推荐使用性价比极高的 **DeepSeek**）：
```ini
LLM_API_KEY=sk-your-deepseek-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_DEFAULT_MODEL=deepseek-chat
```
> 💡 **提示**：即使未在 `.env` 中预先配置 API Key：
> 1. 您也可以在启动后的网页右上角点击 **⚙️ 大模型配置** 在线录入并测试连通性；
> 2. 若完全不配置 API Key，系统具备完善的高保真降级机制，会自动切换到离线金融模拟器运行，绝不抛出异常阻断流程！

### 4. 启动服务
```bash
python main.py
```
终端输出如下内容即表示启动成功：
```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 5. 访问系统
在浏览器打开：
👉 **http://localhost:8000**

即可开始体验：
- 🔍 输入任意 A 股代码（如 `600519` 贵州茅台、`300750` 宁德时代、`002594` 比亚迪）
- 🎛️ 自由选择策略流派或自定义微调指标权重
- ⚡ 实时观看多智能体执行流、资金舆情感知与多空激烈辩论
- 📝 查阅高规格投研深度研报与一键模拟盘下单

---

## 🛠️ 常见问题排查 (FAQ)

1. **Q: 启动提示 `ModuleNotFoundError: No module named 'requests'`？**  
   **A:** 请确认已执行 `pip install -r requirements.txt`，最新版已包含 `requests`。

2. **Q: 网页端提示“大模型 API 连接失败”？**  
   **A:** 
   - 检查 API Key 是否有效，是否有额度；
   - 检查 Base URL 是否正确（DeepSeek 官方是 `https://api.deepseek.com`）；
   - 如果网络环境有代理，可检查终端是否正常访问该网关域名。

3. **Q: 模拟盘的数据保存在哪里？如何重置？**  
   **A:** 模拟盘数据保存在 `data/paper_portfolio.json`。如需重置为初始 100 万元全新账户，直接删除该文件或在模拟盘页面点击重置即可，系统会在下次读写时自动重新创建。

---

## 📄 授权与免责声明
本项目仅供学术研究、技术交流与个人投研学习使用。系统生成的量化指标、AI 辩论观点及投资评级均不构成任何实质性投资建议或收益承诺。证券市场有风险，入市决策需谨慎。
