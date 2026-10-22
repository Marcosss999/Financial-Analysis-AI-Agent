"""pytest 全局配置：保证测试离线、确定、且绝不会调用真实 LLM / 真实行情接口，也不会写入真实记忆库。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 必须在 import app.* 之前设置：config.py 只在环境变量里没有该 key 时才读取 .env，
# 所以这里把 LLM_API_KEY 设为空串，即使开发机上的 .env 里有真实 Key，测试也走离线模拟引擎。
os.environ["LLM_API_KEY"] = ""
os.environ["MEMORY_DB_PATH"] = ":memory:"
os.environ["RATE_LIMIT_PER_MINUTE"] = "100000"

import requests  # noqa: E402


def _offline_get(*args, **kwargs):
    raise requests.exceptions.ConnectionError("tests run offline")


requests.get = _offline_get  # 行情 / 财报 / 情报抓取一律失败 → 走确定性的模拟数据路径
