from transformers import pipeline
from split_sentences_ja import split_sentences_ja

translator = pipeline('translation', model='Mitsua/elan-mt-bt-ja-en')
translator('こんにちは。私はAIです。')

ja_text = """
世界各国が水面下で熾烈な情報戦を繰り広げる時代にらみ合う2つの国東のオスタニア西の西のウェスタリス戦争を企てるオスタニア
政の動向を探るべくウェスタリスはオペレーションを担うディエンとたそがれ100の顔を使い分正体ロイドフォージャー
コードネームたそがれ母ヨルフォージャー市役所職員正体殺し屋コードネーム茨原姫 母ヨルフォージャー正体
仕事職員正体、コロシアコードネームイバラヒメ娘。妻に正体、正体、心を読むことができるエスパー犬、女ボンドフォージャー、正
体、未来を予知できる超能力家族を作り物狩りのため疑似家族を作り互いに正体を隠した彼らのミッションは続く
"""

print(translator(split_sentences_ja(ja_text)))
