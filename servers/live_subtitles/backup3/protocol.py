from typing import Literal, TypedDict


class AudioChunk(TypedDict):
    type: Literal["audio"]
    data: bytes


class TextResult(TypedDict):
    type: Literal["text"]
    text: str
