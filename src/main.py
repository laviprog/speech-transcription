from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.auth.routes import router as auth_router
from src.config import settings
from src.exceptions.handlers import setup_exception_handlers
from src.exceptions.responses import error_responses
from src.lifecycle import lifespan
from src.middlewares import LogMiddleware
from src.routes import router
from src.transcription.routes import router as transcription_router
from src.users.routes import router as user_router

app = FastAPI(
    title="Speech Transcription API",
    version="0.0.2",
    docs_url="/docs/swagger",
    openapi_url="/openapi.json",
    root_path=settings.ROOT_PATH,
    lifespan=lifespan,
    responses=error_responses,
)

setup_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LogMiddleware)

app.include_router(router=router)
app.include_router(router=user_router)
app.include_router(router=auth_router)
app.include_router(router=transcription_router)
