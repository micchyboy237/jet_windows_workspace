model_path = r"C:\Users\druiv\.cache\llama.cpp\translators\LFM2-350M-ENJP-MT.Q4_K_M.gguf"
prompt = "What is the capital of Japan?"

SYSTEM_MESSAGE = (
    "You are a professional, natural-sounding Japanese-to-English translator. "
    "Translate accurately while making the English sound fluent and idiomatic "
    "as if written by a native English speaker."
)

from llama_cpp import Llama

# Critical: logits_all=True is usually required for reliable logprobs
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,           # or whatever you normally use
    n_ctx=1024,                # or your context size
    logits_all=True,           # ← almost always needed for logprobs
    verbose=True,
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system",    "content": SYSTEM_MESSAGE},
        {"role": "user",      "content": prompt},
    ],
    max_tokens=30,
    temperature=0.7,
    logprobs=True,                
    top_logprobs=3,            # ← request top 3 alternatives per position
    stream=True,
)

response = list(response)
# Access top logprobs
choice = response["choices"][0]
print(choice["message"]["content"])               # the generated text

if choice.get("logprobs"):
    top = choice["logprobs"]["top_logprobs"]
    tokens = choice["logprobs"]["tokens"]
    
    print("\nTop logprobs per position:")
    for pos, (tok, alts) in enumerate(zip(tokens, top)):
        print(f"  Pos {pos:2d} | chosen: {tok:12}")
        for alt_token, logp in sorted(alts.items(), key=lambda x: x[1], reverse=True):
            print(f"          {alt_token:12} {logp:.4f}")