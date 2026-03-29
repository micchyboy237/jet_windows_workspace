from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "shisa-ai/shisa-v2.1c-lfm2-350m-sft3-tlonly"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

ja_text = """
こんにちは、今日はお元気ですか？ 
私は最近、仕事がとても忙しくて、週末はゆっくり本を読んだり、散歩したりしてリラックスしています。
あなたはどう過ごしていますか？
"""

# Strong system prompt for reliable translation
messages = [
    {
        "role": "system",
        "content": "You are a precise and accurate translator. Always translate Japanese text to natural, fluent English. Output ONLY the English translation. Do not add any explanations, Japanese text, or extra comments."
    },
    {
        "role": "user",
        "content": f"Translate the following text from Japanese to English:\nJapanese: {ja_text.strip()}"
    }
]

# Apply official chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.1,            # Very low → more deterministic & faithful
    top_p=0.95,
    repetition_penalty=1.18,    # Keeps loops away
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Full generated output:")
print(generated_text)

# Improved cleaning: Extract ONLY the assistant's response
if "<|im_start|>assistant" in generated_text:
    # Take everything after the last "assistant" marker
    translation_part = generated_text.split("<|im_start|>assistant")[-1].strip()
    # Remove any trailing <|im_end|> 
    translation_part = translation_part.replace("<|im_end|>", "").strip()
    
    # Extra safety: remove any echoed system/user if they appear after assistant
    if "system" in translation_part.lower() or "user" in translation_part.lower():
        translation_part = translation_part.split("assistant")[-1].strip()
    
    print("\n--- Clean English Translation ---")
    print(translation_part)
else:
    print("\n--- Clean English Translation ---")
    print(generated_text)