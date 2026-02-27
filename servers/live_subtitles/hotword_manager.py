from typing import Dict, List


class HotwordManager:
    """
    Session-level hotword manager.
    Maintains frequency-weighted noun list with decay.
    Designed to be generic and reusable.
    """

    def __init__(
        self,
        *,
        max_hotwords: int = 20,
        decay: float = 0.85,
    ) -> None:
        self._counts: Dict[str, float] = {}
        self._max_hotwords = max_hotwords
        self._decay = decay

    def update_from_tokens(self, tokens: List[str]) -> None:
        """
        Update frequency counts using already-extracted tokens.
        Keeps this manager NLP-agnostic.
        """
        for token in tokens:
            word = token.strip()

            if len(word) <= 1:
                continue

            if word.isdigit():
                continue

            self._counts[word] = self._counts.get(word, 0.0) + 1.0

    def decay(self) -> None:
        """
        Apply decay to all stored hotwords.
        Removes very low-weight entries.
        """
        to_delete: List[str] = []

        for key in list(self._counts):
            self._counts[key] *= self._decay
            if self._counts[key] < 0.5:
                to_delete.append(key)

        for key in to_delete:
            del self._counts[key]

    def get_hotwords(self) -> List[str]:
        """
        Return ranked hotwords capped to max_hotwords.
        """
        if not self._counts:
            return []

        ranked = sorted(
            self._counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [word for word, _ in ranked[: self._max_hotwords]]

    def clear(self) -> None:
        """
        Reset all hotwords.
        """
        self._counts.clear()
