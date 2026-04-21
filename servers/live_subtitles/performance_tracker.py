from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceMetrics:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    ttft: Optional[float]

    prompt_eval_speed: Optional[float]  # tokens/sec (approx)
    decode_speed: Optional[float]  # tokens/sec (generation)

    total_latency: float
    end_to_end_throughput: Optional[float]  # tokens/sec


class PerformanceTracker:
    """
    Llama.cpp-aligned performance tracker.

    Metrics:
    - TTFT
    - Decode speed (eval tokens/sec)
    - Approx prompt eval speed
    - Total latency
    - Optional end-to-end throughput (tokens/sec)
    """

    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.first_token_time: Optional[float] = None
        self.last_token_time: Optional[float] = None

    def mark_token(self) -> None:
        now = time.perf_counter()

        if self.first_token_time is None:
            self.first_token_time = now

        self.last_token_time = now

    def finalize(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> PerformanceMetrics:
        end_time = time.perf_counter()
        total_latency = end_time - self.start_time

        # --- TTFT ---
        ttft: Optional[float] = None
        if self.first_token_time is not None:
            ttft = self.first_token_time - self.start_time

        # --- Decode speed (TRUE llama.cpp "eval") ---
        decode_speed: Optional[float] = None
        if (
            self.first_token_time is not None
            and self.last_token_time is not None
            and completion_tokens > 0
        ):
            generation_duration = self.last_token_time - self.first_token_time
            if generation_duration > 0:
                decode_speed = completion_tokens / generation_duration

        # --- Prompt eval speed (approximation) ---
        # NOTE: We approximate using TTFT
        prompt_eval_speed: Optional[float] = None
        if ttft is not None and prompt_tokens > 0 and ttft > 0:
            prompt_eval_speed = prompt_tokens / ttft

        # --- Non-standard overall throughput ---
        end_to_end_throughput: Optional[float] = None
        if completion_tokens > 0 and total_latency > 0:
            end_to_end_throughput = completion_tokens / total_latency

        return PerformanceMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            ttft=ttft,
            prompt_eval_speed=prompt_eval_speed,
            decode_speed=decode_speed,
            total_latency=total_latency,
            end_to_end_throughput=end_to_end_throughput,
        )
