from abc import ABC, abstractmethod

from ..speech_events import (
    SpeechSegmentEndEvent,
    SpeechSegmentStartEvent,
)


class SpeechSegmentHandler(ABC):
    """
    Interface that all speech segment handlers must implement.
    """

    @abstractmethod
    def on_segment_start(self, event: SpeechSegmentStartEvent) -> None:
        """Called when a speech segment begins."""
        pass

    @abstractmethod
    def on_segment_end(self, event: SpeechSegmentEndEvent) -> None:
        """Called when a speech segment ends — usually where saving/uploading happens."""
        pass
