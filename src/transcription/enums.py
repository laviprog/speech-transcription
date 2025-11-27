from src.enums import BaseEnum


class Language(BaseEnum):
    RUSSIAN = "ru"
    ENGLISH = "en"


class Model(BaseEnum):
    SMALL = "small"
    MEDIUM = "medium"
    TURBO = "turbo"


class ResultFormat(BaseEnum):
    TEXT = "text"
    SRT = "srt"
    FULL = "full"
