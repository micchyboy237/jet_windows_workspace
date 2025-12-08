# Cleaned up – only re-export the types we actually need elsewhere
__all__ = ["Device", "BatchType"]

from typing import Literal
from typing_extensions import TypeAlias

Device: TypeAlias = Literal["cpu", "cuda", "auto"]
BatchType: TypeAlias = Literal["examples", "tokens"]