import logging
import logging.config
from typing import Any, Generic, TypeVar

import structlog

from src.config import settings

RendererType = TypeVar("RendererType")

Logger = structlog.stdlib.BoundLogger


def get_level() -> str:
    return settings.LOG_LEVEL


class Logging(Generic[RendererType]):
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.StackInfoRenderer(),
    ]

    @classmethod
    def get_processors(cls) -> list[Any]:
        cls.shared_processors.append(structlog.processors.format_exc_info)

        return cls.shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter]

    @classmethod
    def get_renderer(cls) -> RendererType:
        raise NotImplementedError()

    @classmethod
    def configure_stdlib(
        cls,
    ) -> None:
        level = get_level()

        cls.shared_processors.append(structlog.processors.format_exc_info)

        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": True,
                "formatters": {
                    "myLogger": {
                        "()": structlog.stdlib.ProcessorFormatter,
                        "processors": [
                            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                            cls.get_renderer(),
                        ],
                        "foreign_pre_chain": cls.shared_processors,
                    },
                },
                "handlers": {
                    "default": {
                        "level": level,
                        "class": "logging.StreamHandler",
                        "formatter": "myLogger",
                    },
                },
                "loggers": {
                    "": {"handlers": ["default"], "level": level, "propagate": False},
                    "uvicorn": {"handlers": [], "level": "INFO", "propagate": True},
                    "uvicorn.error": {"handlers": [], "level": "INFO", "propagate": True},
                    "uvicorn.access": {"handlers": [], "level": "NOTSET", "propagate": False},
                    "sqlalchemy": {"handlers": [], "level": "WARNING", "propagate": True},
                    "python_multipart.multipart": {
                        "handlers": [],
                        "level": "WARNING",
                        "propagate": True,
                    },
                },
            }
        )

    @classmethod
    def configure_structlog(cls) -> None:
        structlog.configure_once(
            processors=cls.get_processors(),
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    @classmethod
    def configure(cls) -> None:
        cls.configure_stdlib()
        cls.configure_structlog()


class Production(Logging[structlog.processors.JSONRenderer]):
    @classmethod
    def get_renderer(cls) -> structlog.processors.JSONRenderer:
        return structlog.processors.JSONRenderer(ensure_ascii=False)


class Development(Logging[structlog.dev.ConsoleRenderer]):
    @classmethod
    def get_renderer(cls) -> structlog.dev.ConsoleRenderer:
        return structlog.dev.ConsoleRenderer()


def configure() -> None:
    if settings.ENV == "prod":
        Production.configure()
    else:
        Development.configure()
    structlog.contextvars.bind_contextvars(service="speech-transcription-service")
