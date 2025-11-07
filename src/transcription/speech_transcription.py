import gc

import torch
import whisperx
from whisperx.asr import FasterWhisperPipeline
from whisperx.types import SingleSegment

from src.transcription import log
from src.transcription.enums import Language, Model

class SpeechTranscription:
    """
    Handles audio transcription using WhisperX models.

    Supports loading and caching multiple models, transcription audio files,
    and cleaning up memory (including CUDA cache).
    """

    def __init__(
        self,
        device: str = "cpu",
        compute_type: str = "float32",
        download_root: str = "models",
        init_models: list[Model] | None = None,
    ):
        """
        Initializes the SpeechTranscription with device configuration and optional models to preload.

        :param device: Device to use for inference ("cpu" or "cuda").
        :param compute_type: Compute type for inference (e.g., "float32", "int8").
        :param download_root: Directory for downloading and caching models.
        :param init_models: Optional list of models to preload at startup.
        """

        self.__cache: dict[str, FasterWhisperPipeline] = {}
        self._device = device
        self._compute_type = compute_type
        self._download_root = download_root

        self._load_models(init_models)

    def _load_models(self, models: list[Model] | None) -> None:
        """
        Loads and caches the specified models.

        :param models: List of models to load.
        """

        if not models:
            return

        for model in models:
            self._load_model(model.value)

    def _get_model(self, model: Model) -> FasterWhisperPipeline:
        """
        Returns a cached model instance. Loads it if not already cached.

        :param model: Model enum to retrieve.

        :return: Loaded FasterWhisperPipeline instance.
        """

        model_key = model.value
        if model_key not in self.__cache:
            self._load_model(model_key)
        return self.__cache[model_key]

    def _load_model(self, model_name: str) -> None:
        """
        Loads a WhisperX model and caches it.

        :param model_name: Name of the model (e.g., "small").
        """

        log.debug("Loading model %s...", model_name)
        try:
            asr_options = {
                "temperatures": [0.0],
            }

            vad_options = {
                "chunk_size": 10,
                "vad_onset": 0.500,
                "vad_offset": 0.363,
                "min_silence_duration_ms": 500,
            }
            self.__cache[model_name] = whisperx.load_model(
                whisper_arch=model_name,
                device=self._device,
                compute_type=self._compute_type,
                download_root=self._download_root,
                asr_options=asr_options,
                vad_options=vad_options,
            )
            log.debug("Loaded model %s", model_name)
        except Exception as e:
            log.error("Failed to load model %s: %s", model_name, e)
            raise e

    def transcribe(
        self,
        audio_file: str,
        model: Model,
        batch_size: int = 4,
        chunk_size: int = 10,
        language: Language | None = None,
    ) -> list[SingleSegment]:
        """
        Transcribes the given audio file using the specified model and language.

        :param audio_file: Path to the audio file.
        :param model: Transcription model to use.
        :param batch_size: Batch size for inference.
        :param chunk_size: Chunk size (in seconds) for audio splitting.
        :param language: Optional language to guide transcription.

        :return: List of transcribed segments with text and timestamps.
        """

        log.debug("Loading audio file %s...", audio_file)
        try:
            audio = whisperx.load_audio(file=audio_file)
            log.debug("Loaded audio file %s", audio_file)
        except RuntimeError as e:
            log.error("Failed to load audio file %s: %s", audio_file, e)
            raise e

        pipeline = self._get_model(model)

        log.debug("Transcribing audio file %s...", audio_file)
        try:
            result = pipeline.transcribe(
                audio=audio,
                batch_size=batch_size,
                chunk_size=chunk_size,
                language=language.value if language else None,
            )
            log.debug("Transcribed audio file %s", audio_file)
        except Exception as e:
            log.error("Failed to transcribe audio file %s: %s", audio_file, e)
            raise e

        segments = result.get("segments", [])
        return segments

    def clean(self) -> None:
        """
        Releases all cached models and clears memory. If using CUDA, clears GPU memory too.
        """

        log.debug("Cleaning up resources...")
        self.__cache.clear()
        log.debug("Cleared model cache")

        gc.collect()
        if self._device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.debug("Cleared CUDA cache")

        log.debug("Cleanup complete")
