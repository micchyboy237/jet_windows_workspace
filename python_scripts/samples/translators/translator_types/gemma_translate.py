import json
import requests
from transformers import AutoTokenizer

# Updated system prompt focused on Japanese → English translation
system_prompt = (
    "You are a highly skilled professional Japanese-to-English translator. "
    "Translate the given Japanese text into natural, accurate, and fluent English. "
    "Preserve the original meaning, tone, and cultural nuances. "
    "Only add an explicit subject in English when it is clearly specified in the Japanese sentence. "
    "For technical terms and proper nouns, use standard English equivalents when they exist, "
    "or keep them in romanized form if no common translation is available. "
    "After translating, review your output to ensure it is grammatically correct and reads naturally. "
    "Take a deep breath and produce the best possible translation.\n\n"
)

# Instruction with optional hints (example uses casual style)
instruct = (
    "Translate the following Japanese text to English.\n"
    "When translating, please use the following hints:\n"
    "[writing_style: casual]"
)

initial_messages = [
    {"role": "user", "content": system_prompt + instruct},
    {"role": "assistant", "content": "OK"}
]

# Japanese sentences demonstrating real-world examples (from a famous Jimmy Carter anecdote told in Japanese contexts)
message_list = [
    "世界各国が水面架で知列な情報戦を繰り広げる時代に、にらみ合う2つの国、東のオスタニア、西のウェスタリス、戦",
    "争を加わだてるオスタニア政府要順の動向をさせ、",
]

tokenizer = AutoTokenizer.from_pretrained("webbigdata/gemma-2-2b-jpn-it-translate")

if __name__ == "__main__":
    messages = initial_messages.copy()
    
    for i in range(len(message_list)):
        messages.append({"role": "user", "content": message_list[i]})
        print("user: " + message_list[i])
        
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        payload = {
            "prompt": prompt,
            "n_predict": 1200
        }
        
        url = "http://localhost:8080/completion"
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code != 200:
            print(f"Error: {response.text}")
            continue
            
        response_data = response.json()
        response_content = response_data.get('content', '').strip()
        
        print("assistant: " + response_content)
        
        messages.append({"role": "assistant", "content": response_content})
        
        # Keep context manageable (initial 2 + max 6 recent exchanges)
        if len(messages) > 8:
            messages = initial_messages + messages[-6:]