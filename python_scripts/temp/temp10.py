from transformers import pipeline

ner = pipeline(
    "ner",
    model="knosing/japanese_ner_model",
    tokenizer="tohoku-nlp/bert-base-japanese-v3",
    aggregation_strategy="simple"   # merges subwords nicely
)

text = "山田太郎は2025年1月に東京都渋谷区で株式会社xAIジャパンと面談を行い、イーロン・マスクと握手した。"

results = ner(text)

for entity in results:
    print(f"{entity['word']:12}  {entity['entity_group']:6}  {entity['score']:.4f}")