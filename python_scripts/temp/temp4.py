from rich.console import Console
from rich.progress import track
import re
from fast_bunkai import FastBunkai
import jaconv
import fugashi
# from kudasai import Kudasai       # ← 削除したいならここをコメントアウト

console = Console()

def split_sentences_ja(text: str) -> list[str]:
    # Preprocess: treat isolated spaces (common in informal text) as potential sentence breaks
    # Only replace spaces that are between Japanese chars (hiragana, katakana, kanji, some punctuation)
    text = re.sub(
        r'([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3000-\u303F])[ ]+([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3000-\u303F])',
        r'\1。\2',
        text,
    )

    splitter = FastBunkai()
    sentences = list(splitter(text))
    return [s.strip() for s in sentences if s.strip()]

def clean_asr_text(text: str) -> str:
    """ASR日本語テキストを翻訳用にきれいにする"""
    # 1. 極端な空白・改行の正規化
    text = jaconv.normalize(text)
    # 2. フィラー・言い直し系をざっくり削除
    text = re.sub(r'(えっと|あのー|あのね|えー|まぁ|そのー|じゃなくて|っていうか)\s*', '', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    # 3. 文境界復元
    sentences = split_sentences_ja(text)

    # 4. 形態素レベルの正規化（表記ゆれ対策）
    tagger = fugashi.Tagger('-Owakati')  # Much more reasonable token grouping for translation

    cleaned_sentences = []
    for sent in track(sentences, description="Cleaning..."):
        # Parse → get surface forms only
        parsed_lines = tagger.parse(sent).splitlines()
        surfaces = []
        for line in parsed_lines:
            if line == 'EOS':
                continue
            fields = line.split('\t')
            if len(fields) >= 1:
                surfaces.append(fields[0])  # surface form

        # Optional: join with space + post-process
        cleaned = jaconv.h2z(' '.join(surfaces))
        cleaned = jaconv.normalize(cleaned)
        cleaned_sentences.append(cleaned)

    # 5. 最終結合
    final_text = ' '.join(cleaned_sentences)

    console.print("[bold green]Before:[/]", text[:200] + "...")
    console.print("[bold cyan]After :[/]", final_text[:200] + "...")
    return final_text


# 使用例
raw_asr = "あのーえっと今日はですねえー晴れてますあのー昨日じゃなくて一昨日は雨だったんですけど"
cleaned = clean_asr_text(raw_asr)