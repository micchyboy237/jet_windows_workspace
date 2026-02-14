from llama_cpp import Llama
from typing import List, Dict

llm = Llama(
    model_path=r"C:\Users\druiv\.cache\llama.cpp\DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf",
    n_gpu_layers=-1,           # adjust for your GTX 1660
    n_ctx=8192,
    verbose=False
)

def create_thinking_messages(user_question: str) -> List[Dict[str, str]]:
    """Helper to force thinking according to DeepSeek recommendation"""
    return [
        {
            "role": "user",
            "content": user_question
        },
        {
            "role": "assistant",
            "content": "<think>\n"          # ← official recommended force-start
        }
    ]

# Example usage
messages = create_thinking_messages("What is 17 × 23? Please reason step by step and box the answer.")

stream = llm.create_chat_completion(
    messages=messages,
    temperature=0.6,           # DeepSeek recommended range
    max_tokens=400,
    stream=True,
    # stop=["</think>"]          # optional: stop after thinking if you want only reasoning
)

# Stream and print only the content deltas
response_text = ""
for chunk in stream:
    content = chunk["choices"][0]["delta"].get("content", "")
    if content:
        print(content, end="", flush=True)
    response_text += content
