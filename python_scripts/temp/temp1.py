from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # bfloat16 recommended; use float16 on older GPUs or float32 on CPU
    device_map="auto",            # automatically uses GPU if available
)

# Example prompt for Japanese → English translation
prompt = """Translate the following text from Japanese to English:

Japanese: こんにちは、今日はお元気ですか？ 私は最近、仕事がとても忙しくて、週末はゆっくり本を読んだり、散歩したりしてリラックスしています。あなたはどう過ごしていますか？

English:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.2,      # Low temperature for precise, faithful translation
    top_p=0.9,            # Helps reduce token leakage in bilingual tasks
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the full output (it will include the prompt + generated English translation)
print(f"EN: {translated_text}")

# Optional: Extract just the translation part (after "English:")
if "English:" in translated_text:
    english_translation = translated_text.split("English:")[-1].strip()
    print("\n--- Clean English Translation ---")
    print(english_translation)