from fastapi import UploadFile
from whisperx.types import SingleSegment, SingleWordSegment

from src.transcription.enums import Language, Model, ResultFormat
from src.transcription.schemas import (
    Segment,
    TranscriptionFullResult,
    TranscriptionSrtResult,
    TranscriptionTextResult,
    WordSegment,
)
from src.transcription.speech_transcription import SpeechTranscription
from src.transcription.utils import temporary_audio_file


class SpeechTranscriptionService:
    def __init__(self, transcriber: SpeechTranscription):
        self._transcriber = transcriber

    def transcribe(
        self,
        file: UploadFile,
        model: Model = Model.SMALL,
        language: Language | None = None,
        format_result: ResultFormat = ResultFormat.TEXT,
        align_mode: bool = True,
        audio_preprocessing: bool = True,
    ) -> TranscriptionTextResult | TranscriptionSrtResult | TranscriptionFullResult:
        """
        Transcribes speech from an uploaded audio file and returns the result
        in the specified format.

        :param file: Uploaded audio file to be processed.
        :param model: Transcription model to use (e.g., Model.SMALL, Model.MEDIUM).
        :param language: Optional language enum value (e.g., Language.EN, Language.RU).
        :param format_result: Output format for the transcription result.
        :param align_mode: Whether to enable alignment mode for better timestamps.
        :param audio_preprocessing: Whether to apply audio preprocessing before transcription.

        :return: A transcription result in the selected format (text or subtitle).
        """
        words: list[SingleWordSegment] | None = None
        if align_mode:
            segments, words = self._transcribe(
                file=file,
                model=model,
                language=language,
                align_mode=align_mode,
                audio_preprocessing=audio_preprocessing,
            )
        else:
            segments = self._transcribe(
                file=file,
                model=model,
                language=language,
                align_mode=align_mode,
                audio_preprocessing=audio_preprocessing,
            )

        match format_result:
            case ResultFormat.TEXT:
                text = self._to_text(segments)
                return TranscriptionTextResult(text=text)
            case ResultFormat.SRT:
                srt = self._to_srt(segments)
                return TranscriptionSrtResult(srt=srt)
            case ResultFormat.FULL:
                segments = self._to_srt(segments)
                return TranscriptionFullResult(
                    segments=segments,
                    words=[WordSegment(**word) for word in words] if words else None,
                )
            case _:
                raise ValueError(f"Unsupported result format: {format_result}")

    @staticmethod
    def _to_text(segments: list[SingleSegment]) -> str:
        """
        Converts transcription segments into plain text.

        :param segments: List of transcription segments.

        :return: Concatenated transcription text.
        """
        return " ".join(segment["text"].strip() for segment in segments).strip()

    @staticmethod
    def _to_srt(segments: list[SingleSegment]) -> list[Segment]:
        """
        Converts transcription segments into SRT-like structured segments.

        :param segments: List of transcription segments.

        :return: List of Segment objects suitable for SRT serialization.
        """
        return [
            Segment(
                number=index,
                text=segment["text"].strip(),
                start=segment["start"],
                end=segment["end"],
            )
            for index, segment in enumerate(segments, start=1)
        ]

    def _transcribe(
        self,
        file: UploadFile,
        model: Model = Model.SMALL,
        language: Language | None = None,
        align_mode: bool = True,
        audio_preprocessing: bool = True,
    ) -> list[SingleSegment] | tuple[list[SingleSegment], list[SingleWordSegment]]:
        """
        Transcribes from the uploaded audio file using the specified model and language.

        The file is temporarily saved to disk and passed to the underlying transcriber.

        :param file: Uploaded audio file to be processed.
        :param model: Transcription model to use (e.g., Model.SMALL, Model.MEDIUM).
        :param language: Optional language enum value (e.g., Language.EN, Language.RU).
        :param align_mode: Whether to enable alignment mode for better timestamps.
        :param audio_preprocessing: Whether to apply audio preprocessing before transcription.

        :return: List of transcribed segments containing transcribed text and timestamps.
        """

        with temporary_audio_file(file) as path:
            return self._transcriber.transcribe(
                audio_file=path,
                model=model,
                language=language,
                align_mode=align_mode,
                audio_preprocessing=audio_preprocessing,
            )

    def clean(self):
        """
        Clean up resources held by the transcriber (e.g., cached models).
        Should be called on application shutdown.
        """

        self._transcriber.clean()
