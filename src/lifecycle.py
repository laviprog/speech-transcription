from contextlib import asynccontextmanager

from fastapi import FastAPI

from src import log
from src.config import settings
from src.database.config import sqlalchemy_config
from src.transcription.enums import Model
from src.transcription.services import SpeechTranscriptionService
from src.transcription.speech_transcription import SpeechTranscription
from src.users.services import UserService


async def create_default_admin():
    log.info("Creating default admin...")
    async with UserService.new(config=sqlalchemy_config) as service:
        existing_admin = await service.get_one_or_none(username=settings.ADMIN_USERNAME_DEFAULT)
        if existing_admin:
            log.info(f"Admin with username {settings.ADMIN_USERNAME_DEFAULT} already exists")
        else:
            await service.create_admin(
                username=settings.ADMIN_USERNAME_DEFAULT,
                password=settings.ADMIN_PASSWORD_DEFAULT,
            )
            log.info(f"Admin with username {settings.ADMIN_USERNAME_DEFAULT} has been created")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting application...")
    await create_default_admin()
    transcriber = SpeechTranscription(
        device=settings.DEVICE,
        compute_type=settings.COMPUTE_TYPE,
        download_root=settings.DOWNLOAD_ROOT,
        batch_size=settings.BATCH_SIZE,
        chunk_size=settings.CHUNK_SIZE,
        init_asr_models=[Model.SMALL],
    )

    transcription_service = SpeechTranscriptionService(transcriber=transcriber)
    app.state.transcription_service = transcription_service

    yield

    transcription_service.clean()
    log.info("Application shut down")
