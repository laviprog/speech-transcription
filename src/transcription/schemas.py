from pydantic import BaseModel, Field


class ModelList(BaseModel):
    models: list[str]


class LanguageList(BaseModel):
    languages: list[str]


class Segment(BaseModel):
    number: int
    text: str
    start: float = Field(..., description="Start of the segment in seconds")
    end: float = Field(..., description="End of the segment in seconds")


class TranscriptionSrtResult(BaseModel):
    srt: list[Segment]


class TranscriptionTextResult(BaseModel):
    text: str


class WordSegment(BaseModel):
    word: str
    start: float
    end: float
    score: float


class TranscriptionFullResult(BaseModel):
    segments: list[Segment]
    words: list[WordSegment] | None
