# service.py
from typing import AsyncGenerator, List, Dict, Any
from pydantic import BaseModel
import json
import httpx
from transformers import AutoTokenizer

# Shared tokenizer (loaded once at import time)
_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("webbigdata/gemma-2-2b-jpn-it-translate")

# Constant prompts
SYSTEM_PROMPT = (
    "You are a highly skilled professional Japanese-to-English translator. "
    "Translate the given Japanese text into natural, accurate, and fluent English. "
    "Preserve the original meaning, tone, and cultural nuances. "
    "Only add an explicit subject in English when it is clearly specified in the Japanese sentence. "
    "For technical terms and proper nouns, use standard English equivalents when they exist, "
    "or keep them in romanized form if no common translation is available. "
    "After translating, review your output to ensure it is grammatically correct and reads naturally. "
    "Take a deep breath and produce the best possible translation.\n\n"
)

INSTRUCT = (
    "Translate the following Japanese text to English.\n"
    "When translating, please use the following hints:\n"
    "[writing_style: casual]"
)

INITIAL_MESSAGES: List[Dict[str, str]] = [
    {"role": "user", "content": SYSTEM_PROMPT + INSTRUCT},
    {"role": "assistant", "content": "OK"},
]

LLM_URL = "http://localhost:8080/completion"


class TranslationRequest(BaseModel):
    sentences: List[str]
    """List of Japanese sentences to translate sequentially (maintaining conversation context)"""


async def stream_batch_translation(
    request: TranslationRequest,
) -> AsyncGenerator[str, None]:
    """
    Streams translation results token-by-token for a batch of Japanese sentences.
    Maintains conversation history with the local LLM to preserve context across sentences.
    Yields SSE-formatted lines for immediate frontend feedback.
    """
    messages: List[Dict[str, str]] = INITIAL_MESSAGES.copy()

    async with httpx.AsyncClient(timeout=None) as client:
        for sentence in request.sentences:
            # Append new user message
            messages.append({"role": "user", "content": sentence})

            # Build prompt using the chat template
            prompt = _tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            payload = {
                "prompt": prompt,
                "n_predict": 1200,
                "stream": True,  # Important: enable streaming from the local LLM
                # "temperature": 0.0,
                # "top_p": 1.0,
            }

            # Stream response from local LLM
            async with client.stream("POST", LLM_URL, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"data: {json.dumps({'error': error_text.decode()})}\n\n"
                    return

                full_content = ""
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            content = chunk.get("content", "")
                            full_content += content
                            # Stream partial content to client
                            yield f"data: {json.dumps({'partial': content, 'sentence': sentence})}\n\n"
                        except json.JSONDecodeError:
                            continue

            # After completion of one sentence, append full assistant response to history
            messages.append({"role": "assistant", "content": full_content})

            # Final complete translation for this sentence
            yield f"data: {json.dumps({'done': full_content.strip(), 'sentence': sentence})}\n\n"

            # Prune history to keep context manageable (initial 2 + last 6 exchanges)
            if len(messages) > 8:
                messages = INITIAL_MESSAGES + messages[-6:]


async def translate_text(text: str) -> str:
    """
    Translates a single Japanese text to English using the local LLM.
    Maintains minimal conversation context for consistent translation style.
    
    Returns the complete translated string.
    """
    request = TranslationRequest(sentences=[text])
    full_translation = ""
    
    async for line in stream_batch_translation(request):
        if line.startswith("data: "):
            data_str = line[len("data: "):].strip()
            if not data_str:
                continue
            try:
                event = json.loads(data_str)
                if "done" in event:
                    full_translation = event["done"].strip()
            except json.JSONDecodeError:
                pass
    
    return full_translation


async def translate_batch_texts(texts: List[str]) -> List[str]:
    """
    Translates a list of Japanese texts sequentially to English,
    preserving conversation context across sentences for better coherence.
    
    Returns a list of complete translated strings in the same order.
    """
    request = TranslationRequest(sentences=texts)
    results: List[str] = []
    current_translation = ""
    current_sentence = ""
    
    async for line in stream_batch_translation(request):
        if line.startswith("data: "):
            data_str = line[len("data: "):].strip()
            if not data_str:
                continue
            try:
                event = json.loads(data_str)
                if "partial" in event:
                    current_translation += event["partial"]
                    current_sentence = event["sentence"]
                elif "done" in event:
                    results.append(event["done"].strip())
                    current_translation = ""
            except json.JSONDecodeError:
                pass
    
    return results


if __name__ == "__main__":
    import asyncio
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    async def demo_single_sentence() -> None:
        console.rule("[bold green]Demo 1: Single Sentence Translation[/bold green]")
        request = TranslationRequest(sentences=["こんにちは、世界！今日は良い天気ですね。"])
        
        console.print(f"[bold cyan]Japanese:[/bold cyan] {request.sentences[0]}")
        console.print("[dim]Streaming response...[/dim]\n")
        
        async for line in stream_batch_translation(request):
            if line.startswith("data: "):
                data_str = line[len("data: "):].strip()
                if not data_str:
                    continue
                try:
                    event = json.loads(data_str)
                    if "partial" in event:
                        console.print(event["partial"], end="")
                    elif "done" in event:
                        console.print("\n\n[bold green]Complete translation:[/bold green]")
                        console.print(Panel(event["done"], border_style="bright_blue"))
                except json.JSONDecodeError:
                    pass


    async def demo_batch_story() -> None:
        console.rule("[bold green]Demo 2: Sequential Batch (Jimmy Carter Anecdote)[/bold green]")
        sentences = [
            "私も、日本人の聴衆に本当に初めて話すので、少し緊張していました。ホワイトハウスを去ってから間違いなく初めてです。",
            "そして、とても優秀な通訳がいました。",
            "そこで、私は自分が知っている一番短いジョークを話して、場を和ませようと思いました。",
            "それで私はジョークを話し、通訳がそのジョークを伝えました。すると聴衆は大笑いしました。",
            "「私のジョークをどうやって伝えたんですか？」",
            "「『カーター大統領が面白い話をしてくれました。みんな、笑ってください』と伝えました。」"
        ]
        request = TranslationRequest(sentences=sentences)
        
        console.print(f"[bold cyan]Translating {len(sentences)} sentences sequentially with context preservation...[/bold cyan]\n")
        
        sentence_idx = 0
        japanese_printed = False  # Flag to ensure Japanese text is printed only once per sentence
        
        async for line in stream_batch_translation(request):
            if line.startswith("data: "):
                data_str = line[len("data: "):].strip()
                if not data_str:
                    continue
                try:
                    event = json.loads(data_str)
                    if "partial" in event:
                        # Print the original Japanese text exactly once, at the start of each new sentence
                        if not japanese_printed:
                            sentence_idx += 1
                            console.print(f"\n[bold magenta]Japanese ({sentence_idx}):[/bold magenta] {event['sentence']}")
                            japanese_printed = True
                        console.print(event["partial"], end="")
                    elif "done" in event:
                        console.print(f"\n\n[bold green]Sentence {sentence_idx} complete:[/bold green]")
                        console.print(Panel(event["done"], title=f"Sentence {sentence_idx}", border_style="bright_blue"))
                        console.print()
                        japanese_printed = False  # Reset for the next sentence
                except json.JSONDecodeError:
                    pass

    async def main() -> None:
        single = await translate_text("こんにちは、世界！")
        print("Single:", single)
        
        batch = await translate_batch_texts([
            "昨日は楽しかったですね。",
            "今日はどんな予定ですか？"
        ])
        print("Batch:", batch)

        await demo_single_sentence()
        console.print("\n" + "="*80 + "\n")
        await demo_batch_story()

    asyncio.run(main())
