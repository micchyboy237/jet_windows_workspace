from __future__ import annotations
import argparse
import os


from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk


import time
from dataclasses import dataclass
from typing import Optional

from rich.console import Console

console = Console()


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



def log_metrics(metrics: PerformanceMetrics) -> None:
    console.print("\n\n=== Completion Details (llama.cpp aligned) ===", style="bold green")

    console.print(f"Prompt tokens     : {metrics.prompt_tokens}")
    console.print(f"Completion tokens : {metrics.completion_tokens}")
    console.print(f"Total tokens      : {metrics.total_tokens}")

    if metrics.ttft is not None:
        console.print(f"TTFT              : {metrics.ttft:.3f}s")

    if metrics.prompt_eval_speed is not None:
        console.print(
            f"Prompt eval speed : {metrics.prompt_eval_speed:.2f} tokens/s (approx)"
        )

    if metrics.decode_speed is not None:
        console.print(f"Decode speed      : {metrics.decode_speed:.2f} tokens/s (eval)")

    console.print(f"Total latency     : {metrics.total_latency:.3f}s")

    # Optional: keep but clearly marked as non-standard
    if metrics.end_to_end_throughput is not None:
        console.print(
            f"End-to-end throughput : {metrics.end_to_end_throughput:.2f} tokens/s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream chat completion from llama.cpp OpenAI API-compatible endpoint"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="Write a 2 sentence short story about a curious robot.",
        help="User input prompt for the chat model (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--system",
        type=str,
        default=None,
        help="Optional system prompt for the chat model",
    )
    args = parser.parse_args()

    user_prompt = args.prompt
    system_prompt = args.system

    base_url = os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:1234/v1")
    console.print(f"Using llama.cpp base_url: {base_url}", style="bold blue")
    client = OpenAI(
        base_url=base_url,
        api_key="sk-1234",
    )

    messages = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )
    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    if system_prompt:
        console.print(f"System prompt: {system_prompt}", style="bold magenta")
    console.print(f"User prompt: {user_prompt}", style="dim")

    tracker = PerformanceTracker()

    stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
        model="Qwen/Qwen3.5-2B",
        messages=messages,
        max_tokens=100,
        temperature=1.0,
        top_p=1.0,
        presence_penalty=2.0,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
        stream=True,
    )

    for part in stream:
        if part.choices and part.choices[0].delta:
            delta = part.choices[0].delta

            # Check for reasoning_content first
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                tracker.mark_token()
                console.print(delta.reasoning_content, style="orange1", end="", highlight=False)
            # Then check for regular content
            elif hasattr(delta, "content") and delta.content:
                tracker.mark_token()
                console.print(delta.content, style="cyan", end="", highlight=False)

        usage = getattr(part, "usage", None)
        if usage is not None:
            metrics = tracker.finalize(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )

            log_metrics(metrics)
