from datetime import datetime, timezone

from advanced_alchemy.exceptions import NotFoundError
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src import log


def setup_exception_handlers(app: FastAPI):
    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError):
        log.error(
            "NotFoundError",
            request_method=request.method,
            request_url=str(request.url),
            detail=exc.detail,
        )

        return JSONResponse(
            status_code=404,
            content={
                "detail": "Resource not found",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        log.error(
            "HTTPException",
            request_method=request.method,
            request_url=str(request.url),
            detail=exc.detail,
        )

        return JSONResponse(
            status_code=exc.status_code,
            headers=exc.headers or {},
            content={
                "detail": exc.detail,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        log.error(
            "RequestValidationError",
            request_method=request.method,
            request_url=str(request.url),
            detail=exc.errors(),
        )

        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.errors(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.error(
            "Unhandled Exception",
            request_method=request.method,
            request_url=str(request.url),
            detail=str(exc),
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": "An unexpected error occurred",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
