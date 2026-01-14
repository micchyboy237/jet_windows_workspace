from __future__ import annotations
from typing import Literal, overload, Union, Optional
import re
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# Common special tokens that many tokenizers treat specially
# (you can extend this set per model family if needed)
COMMON_SPECIAL_PATTERNS = [
    r"<\|[a-zA-Z0-9_]+?\|>",     # <|im_start|>, <|endoftext|>, etc
    r"<\|endoftext\|>",
    r"<\|fim_prefix\|>",
    r"<\|fim_suffix\|>",
    r"<\|fim_middle\|>",
    r"<\|repo_name\|>",
    r"<\|file_sep\|>",
    r"<\|\d+\|>",                 # some models use <|"number"|> style
]

class TokenCounter:
    """
    Reusable token counter for llama.cpp python bindings (GGUF models)
    
    Main advantages:
    - Works even when you don't want to load whole model just for counting
    - Caches tokenizer when possible
    - Provides multiple ways to get same information
    - Handles common special tokens reasonably well
    """
    
    def __init__(
        self,
        model_path: str | Path | None = None,
        llama_instance: Llama | None = None,
        tokenizer_only: bool = True,
        n_ctx: int = 8192,
        verbose: bool = False
    ):
        self._llama = llama_instance
        self._tokenizer = None
        self._model_path = Path(model_path) if model_path else None
        self.verbose = verbose
        
        if self._llama is not None:
            # Use existing llama instance (recommended when you already loaded model)
            self._tokenizer = self._llama.tokenizer()
            if self.verbose:
                print("Using tokenizer from existing Llama instance")
                
        elif self._model_path is not None and Llama is not None:
            # Try to create tokenizer only (much lighter than full model)
            try:
                if tokenizer_only:
                    from llama_cpp.llama_tokenizer import LlamaHFTokenizer
                    self._tokenizer = LlamaHFTokenizer.from_pretrained(str(self._model_path))
                    if self.verbose:
                        print("Created lightweight tokenizer only instance")
                else:
                    # Full model load (heavier but sometimes more accurate)
                    self._llama = Llama(
                        model_path=str(self._model_path),
                        n_ctx=n_ctx,
                        n_batch=512,
                        verbose=verbose,
                        # We don't really need these for counting
                        embedding=False,
                        logits_all=False,
                        use_mlock=False
                    )
                    self._tokenizer = self._llama.tokenizer()
            except Exception as e:
                print(f"Warning: Could not initialize tokenizer: {e}")
                self._tokenizer = None

    @overload
    def count_tokens(self, text: str, return_tokens: Literal[False] = False) -> int: ...
    
    @overload
    def count_tokens(self, text: str, return_tokens: Literal[True] = True) -> tuple[int, list[int]]: ...
    
    def count_tokens(
        self,
        text: str,
        return_tokens: bool = False,
        add_special_tokens: bool = True,
        ignore_special_patterns: bool = False
    ) -> Union[int, tuple[int, list[int]]]:
        """
        Count tokens in text using the model's tokenizer
        
        Args:
            text:                   input text
            return_tokens:          whether to also return the token ids
            add_special_tokens:     usually True for chat/prompt templates
            ignore_special_patterns: when doing rough estimation without real tokenizer
            
        Returns:
            int or (int, list[int]) depending on return_tokens parameter
        """
        if self._tokenizer is not None:
            try:
                tokens = self._tokenizer.tokenize(
                    text.encode("utf-8", "ignore"),
                    add_special_tokens=add_special_tokens,
                    special=False  # llama.cpp special handling
                )
                count = len(tokens)
                
                if return_tokens:
                    return count, tokens
                return count
                
            except Exception as e:
                print(f"Tokenizer failed: {e}. Falling back to estimation...")
        
        # Fallback - very rough estimation (used when tokenizer not available)
        if ignore_special_patterns:
            # Very naive fallback
            count = len(text) // 4 + text.count(" ") + 2
        else:
            # Try to detect common special tokens
            special_count = sum(len(re.findall(pat, text)) for pat in COMMON_SPECIAL_PATTERNS)
            clean_text = re.sub("|".join(COMMON_SPECIAL_PATTERNS), " ", text)
            count = (len(clean_text) // 4) + clean_text.count(" ") + special_count + 4
        
        if return_tokens:
            # We can't return real tokens in fallback mode
            return count, []
        return count


# ──────────────────────────────────────────────────────────────────────────────
# Convenience functions / factory
# ──────────────────────────────────────────────────────────────────────────────

_token_counter_cache: dict[str, TokenCounter] = {}

def get_token_counter(
    model_path: str | Path | None = None,
    llama: Llama | None = None,
    tokenizer_only: bool = True,
    cache: bool = True,
    **kwargs
) -> TokenCounter:
    """
    Factory function with simple caching by model path
    """
    if llama is not None:
        # When passing existing instance → no caching needed
        return TokenCounter(llama_instance=llama, **kwargs)
        
    if cache and model_path:
        key = str(Path(model_path).resolve())
        if key in _token_counter_cache:
            return _token_counter_cache[key]
            
    counter = TokenCounter(
        model_path=model_path,
        tokenizer_only=tokenizer_only,
        **kwargs
    )
    
    if cache and model_path:
        _token_counter_cache[str(Path(model_path).resolve())] = counter
        
    return counter


# Quick usage example:
if __name__ == "__main__":
    model_path = (
        r"C:\Users\druiv\.cache\llama.cpp\translators\shisa-v2.1-llama3.2-3b.Q4_K_M.gguf"
    )

    # Option 1: When you already have loaded model
    # llm = Llama("models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf", n_ctx=8192)
    # counter = get_token_counter(llama=llm)
    
    # Option 2: Just tokenizer (recommended for counting only)
    counter = get_token_counter(
        model_path=model_path,
        tokenizer_only=True,
        verbose=True
    )

    
    text = """\
    <|begin_of_text|><|start_header_id|>system<|end_header_id>
    
    You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id>
    
    Hello! Tell me a short joke about programming.<|eot_id|><|start_header_id|>assistant<|end_header_id>"""
    
    count, tokens = counter.count_tokens(text, return_tokens=True)
    print(f"Token count: {count}")
    print(f"First 10 tokens: {tokens[:10]}")
    
    # Option 3: When you already loaded the model anyway
    # counter = get_token_counter(llama=your_llama_instance)
