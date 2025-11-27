import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware

from src import log
from src.utils import generate_correlation_id


class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        correlation_id = request.headers.get("X-Request-Id") or generate_correlation_id()
        ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (
            request.client.host if request.client else None
        )

        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
            ip_address=ip,
        )

        start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        except Exception:
            log.exception("Unhandled exception during request")
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            status = getattr(response, "status_code", None)
            log.info("Request completed", status_code=status, duration_ms=duration_ms)

            try:
                if response is not None:
                    response.headers.setdefault("X-Request-Id", correlation_id)
            except Exception:
                pass

            structlog.contextvars.unbind_contextvars(
                "correlation_id", "method", "path", "ip_address"
            )
