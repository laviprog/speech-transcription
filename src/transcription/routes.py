from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from src.auth.security.dependencies import CurrentUserDep
from src.transcription.dependencies import SpeechTranscriptionServiceDep
from src.transcription.enums import Language, ResultFormat
from src.transcription.enums import Model as Model
from src.transcription.schemas import (
    LanguageList,
    ModelList,
    TranscriptionFullResult,
    TranscriptionSrtResult,
    TranscriptionTextResult,
)

router = APIRouter(prefix="/transcription", tags=["Transcription"])


@router.get(
    "/models",
    summary="Get available models",
    description="Returns a list of available speech transcription models.",
    responses={
        200: {
            "description": "List of available models",
        },
    },
)
async def get_models() -> ModelList:
    """
    Get a list of supported transcription models.

    :return: A list of model names.
    """

    return ModelList(models=Model.values())


@router.get(
    "/languages",
    summary="Get available languages",
    description="Returns a list of supported languages for transcription.",
    responses={
        200: {
            "description": "List of supported languages",
        },
    },
)
async def get_languages() -> LanguageList:
    """
    Get a list of supported transcription languages.

    :return: A list of language codes.
    """

    return LanguageList(languages=Language.values())


@router.post(
    "/transcribe",
    summary="Transcribe speech from audio",
    description="""
        Uploads an audio file and returns the transcribed speech
        either as plain text or in SRT subtitle format.
    """,
    responses={
        200: {
            "description": "Transcription result",
        },
    },
)
async def transcribe(
    transcription_service: SpeechTranscriptionServiceDep,
    user: CurrentUserDep,
    file: Annotated[UploadFile, File(..., description="Upload file (.mp3, .wav, .ts, .mp4)")],
    language: Annotated[
        Language | None, Form(description="Optional language hint for transcription")
    ] = None,
    model: Annotated[Model, Form(description="Transcription model to use")] = Model.SMALL,
    result_format: Annotated[
        ResultFormat, Form(description="Desired format of the result")
    ] = ResultFormat.FULL,
    align_mode: Annotated[bool, Form(description="Enable word-level timestamp alignment")] = True,
    audio_preprocessing: Annotated[bool, Form(description="Enable audio preprocessing")] = True,
) -> TranscriptionSrtResult | TranscriptionTextResult | TranscriptionFullResult:
    """
    Transcribe speech from uploaded audio file.

    :param file: Audio file in supported format (e.g., .mp3, .wav).
    :param language: Optional language hint for transcription.
    :param model: Transcription model to use.
    :param result_format: Desired format of the result.
    :param transcription_service: Injected transcription service.
    :param user: Authenticated user (injected).
    :param align_mode: Whether to enable alignment mode.
    :param audio_preprocessing: Whether to apply audio preprocessing.

    :return: Transcription result as plain text or SRT.
    """

    return transcription_service.transcribe(
        file, model, language, result_format, align_mode, audio_preprocessing
    )
