from typing import Literal, TypedDict


class AudioBegin(TypedDict):
    type: Literal["start"]


class AudioChunk(TypedDict):
    type: Literal["audio"]
    data: bytes


class AudioEnd(TypedDict):
    type: Literal["end"]


class TextResult(TypedDict):
    type: Literal["text"]
    text: str
