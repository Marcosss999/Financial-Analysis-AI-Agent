"""纯 ASGI 限流中间件。

不用 BaseHTTPMiddleware：它会缓冲/包装响应体，与 SSE 长连接配合时有已知的取消与缓冲问题。
只对会触发 LLM 调用的 POST 接口限流；按客户端 IP 做令牌桶；超限返回 429 + Retry-After。
(部署在反向代理后面时，应改为读取可信的 X-Forwarded-For，这里不信任该头以免被伪造绕过限流。)
"""

import json
import math
from typing import Callable, Iterable

from app.core.resilience import TokenBucket


class RateLimitMiddleware:
    def __init__(self, app, limiter: TokenBucket,
                 protected_suffixes: Iterable[str] = ("/analyze", "/analyze/stream", "/chat", "/chat/stream")):
        self.app = app
        self.limiter = limiter
        self.suffixes = tuple(protected_suffixes)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("method") == "POST" and scope.get("path", "").endswith(self.suffixes):
            client = (scope.get("client") or ("unknown", 0))[0]
            allowed, wait = self.limiter.allow(client)
            if not allowed:
                body = json.dumps({"detail": "请求过于频繁，请稍后重试", "retry_after_seconds": math.ceil(wait)},
                                  ensure_ascii=False).encode("utf-8")
                await send({"type": "http.response.start", "status": 429, "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"retry-after", str(math.ceil(wait)).encode()),
                    (b"content-length", str(len(body)).encode()),
                ]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)
