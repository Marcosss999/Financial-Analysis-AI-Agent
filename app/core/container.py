"""依赖容器：进程级单例(记忆库 / 分析服务 / 对话记忆 / 限流器)。测试中可直接构造各类的独立实例，不依赖这里。"""

from app.core.config import settings
from app.core.resilience import TokenBucket
from app.memory.chat_memory import ChatMemory
from app.memory.store import MemoryStore
from app.services.analysis_service import AnalysisService

memory_store = MemoryStore(settings.eng.DB_PATH)
analysis_service = AnalysisService(memory_store)
chat_memory = ChatMemory(memory_store)
rate_limiter = TokenBucket(rate_per_minute=settings.eng.RATE_LIMIT_PER_MINUTE)
