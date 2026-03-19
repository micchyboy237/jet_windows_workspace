import re
from typing import List
from fast_bunkai import FastBunkai


def split_sentences(text):
    sentences = []
    last = 0
    # 句点で文を分割
    for match in re.finditer(r'[。！？…]', text):
        end = match.end()
        # 句点の直後に続く改行を含める
        while end < len(text) and text[end] == '\n':
            end += 1
        sentence = text[last:end]
        sentences.append(sentence)
        last = end
    # 残りのテキストを追加
    if last < len(text):
        remaining = text[last:]
        sentences.append(remaining)
    # 各文内の改行を適切に分割
    final_sentences = []
    for s in sentences:
      if '\n' in s:
          parts = s.split('\n')
          for i, part in enumerate(parts):
              if part:
                  # 最後の部分でなければ改行を追加
                  if i < len(parts) - 1:
                      final_sentences.append(part + '\n')
                  else:
                      final_sentences.append(part)
              # 改行自体を保持
              if i < len(parts) - 1:
                  final_sentences.append('\n')
      else:
          final_sentences.append(s)
    return final_sentences


def split_sentences_ja(text: str) -> List[str]:
    """
    Split Japanese text into sentences.
    """
    splitter = FastBunkai()
    sentences = list(splitter(text))

    return sentences
