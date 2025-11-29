import gc
import os
import uuid

import torch
from audio_separator.separator import Separator
from numpy import ndarray
from whisperx.alignment import align, load_align_model
from whisperx.asr import FasterWhisperPipeline, load_model
from whisperx.audio import load_audio
from whisperx.types import (
    AlignedTranscriptionResult,
    SingleSegment,
    SingleWordSegment,
    TranscriptionResult,
)

from src import log
from src.transcription.enums import Language, Model


class SpeechTranscription:
    """
    Handles speech transcription and alignment using WhisperX.
    """

    def __init__(
        self,
        device: str = "cpu",
        compute_type: str = "float32",
        download_root: str = "models",
        batch_size: int = 4,
        chunk_size: int = 10,
        init_asr_models: list[Model] | None = None,
    ):
        """
        Initializes the SpeechTranscription with device configuration
        and optional models to preload.

        :param device: Device to use for inference ("cpu" or "cuda").
        :param compute_type: Compute type for inference (e.g., "float32", "int8").
        :param download_root: Directory for downloading and caching models.
        :param init_asr_models: Optional list of asr models to preload at startup.
        :param batch_size: Batch size for inference.
        :param chunk_size: Chunk size (in seconds) for audio splitting.
        """

        self.__asr_cache: dict[str, FasterWhisperPipeline] = {}
        self.__align_cache: dict[str, tuple] = {}
        self._audio_separator_model: Separator = Separator(model_file_dir=download_root)

        self._device = device
        self._compute_type = compute_type
        self._download_root = download_root
        self._batch_size = batch_size
        self._chunk_size = chunk_size

        self._load_models(init_asr_models)

    def _load_models(self, asr_models: list[Model] | None) -> None:
        """
        Preloads specified ASR models into cache.
        """
        self._audio_separator_model.load_model("UVR-MDX-NET-Voc_FT.onnx")
        for lang in Language.values():
            self._load_align(lang_code=lang)
        for model in asr_models or [Model.SMALL]:
            self._load_asr(model)

    def _load_asr(self, model_name: Model) -> None:
        """
        Loads an ASR model and stores it in the cache.
        """
        model = model_name.value
        log.debug("Loading ASR model", model_name=model)
        try:
            self.__asr_cache[model] = load_model(
                whisper_arch=model,
                device=self._device,
                compute_type=self._compute_type,
                download_root=self._download_root,
            )
            log.debug("Loaded ASR model", model_name=model)
        except Exception as e:
            log.error("Failed to load ASR model", model_name=model, error=str(e))
            raise e

    def _load_align(self, lang_code: str) -> None:
        """
        Loads an alignment model and stores it in the cache.
        """
        log.debug("Loading align model", lang_code=lang_code)
        try:
            align_model, metadata = load_align_model(
                language_code=lang_code,
                device=self._device,
                model_dir=self._download_root,
            )
            self.__align_cache[lang_code] = (align_model, metadata)
            log.debug("Align model loaded", lang_code=lang_code)
        except Exception as e:
            log.error("Failed to load align model", lang_code=lang_code, error=str(e))
            raise e

    def _get_asr(self, model: Model) -> FasterWhisperPipeline:
        """
        Retrieves the ASR model from cache or loads it if not present.
        """
        if model.value not in self.__asr_cache:
            self._load_asr(model)
        return self.__asr_cache[model.value]

    def _get_align(self, lang_code: str):
        """
        Retrieves the alignment model for the specified language code from cache or loads it
        if not present.
        """
        if lang_code not in self.__align_cache:
            self._load_align(lang_code=lang_code)
        return self.__align_cache[lang_code]

    @staticmethod
    def _load_audio(audio_file: str) -> ndarray:
        """
        Loads audio file into a numpy array.
        """
        log.debug("Loading audio file", audio_file=audio_file)
        try:
            audio = load_audio(file=audio_file)
            log.debug("Loaded audio file", audio_file=audio_file)
        except RuntimeError as e:
            log.error("Failed to load audio file", audio_file=audio_file, error=str(e))
            raise e
        return audio

    def _transcribe(
        self,
        audio: ndarray,
        model: Model,
        language: Language | None,
    ) -> TranscriptionResult:
        """
        Transcribes the given audio using the specified ASR model and language.
        """
        asr = self._get_asr(model)

        log.debug(
            "Transcribing...",
            model=model.value,
            language=language.value,
            batch_size=self._batch_size,
            chuck_size=self._chunk_size,
        )
        try:
            result = asr.transcribe(
                audio=audio,
                language=language.value if language else None,
                batch_size=self._batch_size,
                chunk_size=self._chunk_size,
            )
            log.debug("Transcribed audio file")
        except RuntimeError as e:
            log.error(
                "Transcription runtime error",
                error=str(e),
            )
            self.clean()
            raise e
        except Exception as e:
            log.error("Transcribing failed", error=str(e))
            raise e

        return result

    def _align(
        self, segments: list[SingleSegment], audio: ndarray, language: str
    ) -> AlignedTranscriptionResult | None:
        """
        Aligns the transcription segments with the audio using the alignment model.
        """
        try:
            align_model, metadata = self._get_align(language)
            return align(
                segments,
                align_model,
                metadata,
                audio,
                device=self._device,
            )
        except RuntimeError as e:
            log.warning("Alignment failed", error=str(e))
            self.clean()
            raise e
        except Exception as e:
            log.warning("Alignment failed (fallback to raw segments)", error=str(e))
            return None

    def transcribe(
        self,
        audio_file: str,
        model: Model,
        language: Language | None,
        align_mode: bool,
        audio_preprocessing: bool,
    ) -> list[SingleSegment] | tuple[list[SingleSegment], list[SingleWordSegment]]:
        """
        Transcribes the given audio file, optionally performing speaker diarization.
        """
        if audio_preprocessing:
            output_names = {
                "Vocals": str(uuid.uuid4()),
                "Instrumental": str(uuid.uuid4()),
            }
            output_files = self._audio_separator_model.separate(
                audio_file_path=audio_file, custom_output_names=output_names
            )
            log.debug("Audio separation completed", output_files=output_files)
            audio = self._load_audio(output_files[1])

            # Clean up temporary separated files
            for file in output_files:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass
        else:
            audio = self._load_audio(audio_file)

        transcription_result = self._transcribe(
            audio=audio,
            model=model,
            language=language,
        )

        if align_mode:
            align_result = self._align(
                segments=transcription_result["segments"],
                audio=audio,
                language=transcription_result["language"],
            )

            if align_result:
                transcription_result["segments"] = [
                    SingleSegment(
                        start=seg["start"],
                        end=seg["end"],
                        text=seg["text"].strip(),
                    )
                    for seg in align_result["segments"]
                ]
                return transcription_result["segments"], align_result["word_segments"]

        return transcription_result["segments"]

    def _clean_cuda(self) -> None:
        """
        Cleans up CUDA memory if using GPU.
        """
        if self._device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clean(self) -> None:
        """
        Cleans up cached models and frees memory.
        """
        log.debug("Cleaning up resources...")
        self.__asr_cache.clear()
        self.__align_cache.clear()
        gc.collect()
        self._clean_cuda()
        log.debug("Cleanup complete")
