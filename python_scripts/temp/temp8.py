from llama_cpp.llama_tokenizer import LlamaHFTokenizer
from rich import print
from rich.pretty import pprint

# model_path = r"C:\Users\druiv\.cache\llama.cpp\translators\shisa-v2.1-llama3.2-3b.Q4_K_M.gguf"
model_path = "shisa-ai/shisa-v2.1-llama3.2-3b"
add_special_tokens = False

text = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id>

You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id>

Hello! Tell me a short joke about programming.<|eot_id|><|start_header_id|>assistant<|end_header_id>"""

tokenizer = LlamaHFTokenizer.from_pretrained(model_path)

tokens = tokenizer.tokenize(
    text.encode("utf-8", "ignore"),
    special=add_special_tokens  # llama.cpp special handling
)
count = len(tokens)

print(f"Detokenized tokens: {tokenizer.detokenize(tokens, special=add_special_tokens)}")
print(f"Decoded text: {tokenizer.decode(tokens)}")
print(f"Token ids:\n{tokens}")
print(f"Count: {count}")
