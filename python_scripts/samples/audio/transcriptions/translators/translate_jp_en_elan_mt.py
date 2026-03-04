import argparse
from transformers import pipeline

DEFAULT_JA_TEXT = """
恥ずかしい…見ないでください…
んっ…そこ、弱いんです…
はぁ…はぁ…気持ちいい…
もう…ダメかも…頭おかしくなりそう…
お願い…もっと激しくして…壊して…！
あぁんっ！すごい…奥まで届いてる…♡
出さないで…まだ中にいてて…
"""

def main():
    parser = argparse.ArgumentParser(description="Translate Japanese text to English.")
    parser.add_argument(
        "ja_text",
        type=str,
        nargs="?",
        default=DEFAULT_JA_TEXT,
        help="Japanese text to translate (default: sample text)"
    )
    args = parser.parse_args()

    translator = pipeline('translation', model='Mitsua/elan-mt-bt-ja-en')
    result = translator(args.ja_text)
    print(result)

if __name__ == "__main__":
    main()
