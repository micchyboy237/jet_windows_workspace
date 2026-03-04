from transformers import pipeline

translator = pipeline('translation', model='Mitsua/elan-mt-bt-ja-en')

ja_text = """
こんにちは。私はAIです。
"""

print(translator(ja_text))
