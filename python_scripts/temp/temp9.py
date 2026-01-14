from llama_cpp.llama_tokenizer import LlamaHFTokenizer
from llama_cpp import Llama
from rich import print
from rich.pretty import pprint

model_path = r"C:\Users\druiv\.cache\llama.cpp\translators\shisa-v2.1-llama3.2-3b.Q4_K_M.gguf"
add_special_tokens = True

text = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id>

You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id>

Hello! Tell me a short joke about programming.<|eot_id|><|start_header_id|>assistant<|end_header_id>"""

MODEL_SETTINGS = {
    "n_ctx": 128,
    "n_gpu_layers": 0,
    "flash_attn": True,
    "logits_all": True,
    "type_k": 8,
    "type_v": 8,
    "tokenizer_kwargs": {"add_bos_token": add_special_tokens},
    "n_batch": 128,
    "n_threads": 6,
    "n_threads_batch": 6,
    "use_mlock": True,
    "use_mmap": True,
    "verbose": False,
}

llm = Llama(model_path=model_path, **MODEL_SETTINGS)
tokenizer = llm.tokenizer()

tokens = tokenizer.tokenize(
    text.encode("utf-8", "ignore"),
    special=add_special_tokens  # llama.cpp special handling
)
count = len(tokens)
small_tokens = tokenizer.tokenize("print('Hello world!')".encode("utf-8", "ignore"), special=True)

print(f"Detokenized tokens: {tokenizer.detokenize(tokens, special=add_special_tokens)}")
print(f"Decoded text: {tokenizer.decode(tokens)}")
print(f"Token ids:\n{tokens}")
print(f"Count: {count}")
