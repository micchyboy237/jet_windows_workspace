"""
llama_cpp_wrapper.py
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

import llama_cpp.llama_chat_format as llama_chat_format
import llama_cpp.llama_cpp as llama_cpp
from llama_cpp import LogitsProcessorList
from llama_cpp import Llama as BaseLlama
from llama_cpp.llama_grammar import LlamaGrammar
from llama_cpp.llama_speculative import LlamaDraftModel
from llama_cpp.llama_tokenizer import BaseLlamaTokenizer
from llama_cpp.llama_types import *
from log_utils import get_entry_file_dir, get_entry_file_name
from llama_log_utils import (
    console,
    get_file_logger,
    make_call_dir,
    print_content_chunk,
    print_reasoning_chunk,
    print_request_panel,
    print_response_panel,
    print_stream_end_panel,
    save_json,
    save_markdown,
)

MAX_LOG_LLM_DIRS = 20


class Llama(BaseLlama):
    def __init__(
        self,
        model_path: str,
        *,
        # Model Params
        n_gpu_layers: int = 0,
        split_mode: int = llama_cpp.LLAMA_SPLIT_MODE_LAYER,
        main_gpu: int = 0,
        tensor_split: Optional[List[float]] = None,
        vocab_only: bool = False,
        use_mmap: bool = True,
        use_mlock: bool = False,
        kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]] = None,
        # Context Params
        seed: int = llama_cpp.LLAMA_DEFAULT_SEED,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_ubatch: int = 512,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        rope_scaling_type: Optional[
            int
        ] = llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        pooling_type: int = llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,
        rope_freq_base: float = 0.0,
        rope_freq_scale: float = 0.0,
        yarn_ext_factor: float = -1.0,
        yarn_attn_factor: float = 1.0,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_orig_ctx: int = 0,
        logits_all: bool = False,
        embedding: bool = False,
        offload_kqv: bool = True,
        flash_attn: bool = False,
        op_offload: Optional[bool] = None,
        swa_full: Optional[bool] = None,
        # Sampling Params
        no_perf: bool = False,
        last_n_tokens_size: int = 64,
        # LoRA Params
        lora_base: Optional[str] = None,
        lora_scale: float = 1.0,
        lora_path: Optional[str] = None,
        # Backend Params
        numa: Union[bool, int] = False,
        # Chat Format Params
        chat_format: Optional[str] = None,
        chat_handler: Optional[llama_chat_format.LlamaChatCompletionHandler] = None,
        # Speculative Decoding
        draft_model: Optional[LlamaDraftModel] = None,
        # Tokenizer Override
        tokenizer: Optional[BaseLlamaTokenizer] = None,
        # KV cache quantization
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
        # Misc
        spm_infill: bool = False,
        verbose: bool = True,
        # Extra Params
        logs_dir: str | None = None,
        **kwargs,  # type: ignore
    ):
        _caller_base_dir = (
            Path(get_entry_file_dir())
            / "generated"
            / Path(get_entry_file_name()).stem
            / "llm_calls"
        )
        self.logs_dir = Path(logs_dir).resolve() if logs_dir else _caller_base_dir
        super().__init__(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            split_mode=split_mode,
            main_gpu=main_gpu,
            tensor_split=tensor_split,
            vocab_only=vocab_only,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            kv_overrides=kv_overrides,
            seed=seed,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            rope_scaling_type=rope_scaling_type,
            pooling_type=pooling_type,
            rope_freq_base=rope_freq_base,
            rope_freq_scale=rope_freq_scale,
            yarn_ext_factor=yarn_ext_factor,
            yarn_attn_factor=yarn_attn_factor,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
            yarn_orig_ctx=yarn_orig_ctx,
            logits_all=logits_all,
            embedding=embedding,
            offload_kqv=offload_kqv,
            flash_attn=flash_attn,
            op_offload=op_offload,
            swa_full=swa_full,
            no_perf=no_perf,
            last_n_tokens_size=last_n_tokens_size,
            lora_base=lora_base,
            lora_scale=lora_scale,
            lora_path=lora_path,
            numa=numa,
            chat_format=chat_format,
            chat_handler=chat_handler,
            draft_model=draft_model,
            tokenizer=tokenizer,
            type_k=type_k,
            type_v=type_v,
            spm_infill=spm_infill,
            verbose=verbose,
            **kwargs,
        )

    def create_chat_completion(
        self,
        messages: List[ChatCompletionRequestMessage],
        functions: Optional[List[ChatCompletionFunction]] = None,
        function_call: Optional[ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[ChatCompletionTool]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[ChatCompletionRequestResponseFormat] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> Union[
        CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]
    ]:
        # ── 1. Collect all call params for logging ──────────────────────────
        call_params = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            stop=stop,
            seed=seed,
            response_format=response_format,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )

        # ── 2. Create per-call log folder + file logger ──────────────────────
        call_dir = make_call_dir(self.logs_dir)
        logger = get_file_logger(call_dir)
        self._prune_call_dirs()
        logger.info("create_chat_completion called  stream=%s", stream)

        # ── 3. Print + save request ──────────────────────────────────────────
        print_request_panel(messages, call_params)
        save_json(
            call_dir / "request.json",
            {"messages": messages, "params": call_params},
        )
        logger.info("request.json saved  messages=%d", len(messages))

        # ── 4. Fire the actual llama_cpp call ────────────────────────────────
        t_start = time.perf_counter()

        raw = super().create_chat_completion(
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            stop=stop,
            seed=seed,
            response_format=response_format,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )

        # ── 5a. NON-STREAM path ──────────────────────────────────────────────
        if not stream:
            elapsed = time.perf_counter() - t_start
            response: CreateChatCompletionResponse = raw

            choice = response["choices"][0] if response.get("choices") else {}
            message = choice.get("message", {})
            content: str = message.get("content") or ""
            reasoning: str = message.get("reasoning_content") or ""
            usage = response.get("usage")

            print_response_panel(content, reasoning or None, usage)

            save_json(call_dir / "response.json", response)
            save_markdown(call_dir / "response.md", content, reasoning or None)
            response_md = call_dir / "response.md"
            file_uri = response_md.resolve().as_uri()
            logger.info("Response saved  path=%s  uri=%s", response_md, file_uri)
            console.print(f"[dim]Response → [link={file_uri}]{response_md}[/link][/dim]")
            save_json(
                call_dir / "metadata.json",
                {
                    "elapsed_seconds": round(elapsed, 4),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": usage,
                    "model": response.get("model"),
                    "id": response.get("id"),
                    "created": response.get("created"),
                    "call_dir": str(call_dir),
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.info(
                "Non-stream done  elapsed=%.3fs  tokens=%s",
                elapsed,
                usage,
            )
            return response

        # ── 5b. STREAM path — wrap the iterator ─────────────────────────────
        def _stream_wrapper(
            iterator: Iterator[CreateChatCompletionStreamResponse],
        ) -> Iterator[CreateChatCompletionStreamResponse]:
            accumulated_content = []
            accumulated_reasoning = []
            last_chunk = None
            finish_reason = None

            console.print(
                "[bold blue]⟶ Streaming…[/bold blue]", highlight=False
            )

            for chunk in iterator:
                last_chunk = chunk
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason") or finish_reason

                    # Regular content token
                    token: str = delta.get("content") or ""
                    if token:
                        accumulated_content.append(token)
                        print_content_chunk(token)
                        logger.debug("content_chunk: %r", token)

                    # Reasoning / thinking token (DeepSeek-R1 style)
                    r_token: str = delta.get("reasoning_content") or ""
                    if r_token:
                        accumulated_reasoning.append(r_token)
                        print_reasoning_chunk(r_token)
                        logger.debug("reasoning_chunk: %r", r_token)

                yield chunk

            elapsed = time.perf_counter() - t_start
            full_content = "".join(accumulated_content)
            full_reasoning = "".join(accumulated_reasoning)

            print_stream_end_panel(full_content, full_reasoning or None, elapsed)

            stream_response_summary = {
                "id": last_chunk.get("id") if last_chunk else None,
                "model": last_chunk.get("model") if last_chunk else None,
                "object": "chat.completion.chunk",
                "created": last_chunk.get("created") if last_chunk else None,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content,
                            "reasoning_content": full_reasoning or None,
                        },
                        "finish_reason": finish_reason,
                    }
                ],
            }

            save_json(call_dir / "response.json", stream_response_summary)
            save_markdown(call_dir / "response.md", full_content, full_reasoning or None)
            response_md = call_dir / "response.md"
            file_uri = response_md.resolve().as_uri()
            logger.info("Response saved  path=%s  uri=%s", response_md, file_uri)
            console.print(f"[dim]Response → [link={file_uri}]{response_md}[/link][/dim]")
            save_json(
                call_dir / "metadata.json",
                {
                    "elapsed_seconds": round(elapsed, 4),
                    "finish_reason": finish_reason,
                    "usage": None,
                    "model": last_chunk.get("model") if last_chunk else None,
                    "id": last_chunk.get("id") if last_chunk else None,
                    "created": last_chunk.get("created") if last_chunk else None,
                    "call_dir": str(call_dir),
                    "timestamp": datetime.now().isoformat(),
                    "streamed": True,
                },
            )
            logger.info(
                "Stream done  elapsed=%.3fs  content_chars=%d  reasoning_chars=%d",
                elapsed,
                len(full_content),
                len(full_reasoning),
            )

        return _stream_wrapper(raw)

    def _prune_call_dirs(self) -> None:
        """Remove oldest call dirs under self.logs_dir, keeping at most MAX_LOG_LLM_DIRS."""
        logs_dir = Path(self.logs_dir)
        if not logs_dir.is_dir():
            return
        call_dirs = sorted(
            (d for d in logs_dir.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
        )
        excess = len(call_dirs) - MAX_LOG_LLM_DIRS
        if excess <= 0:
            return
        import shutil
        for old_dir in call_dirs[:excess]:
            shutil.rmtree(old_dir, ignore_errors=True)
