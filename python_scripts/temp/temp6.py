from sudachipy import tokenizer, dictionary
from rich import print
from rich.pretty import pprint

sent = "あのーえっと今日はですねえー晴れてますあのー昨日じゃなくて一昨日は雨だったんですけど"

sudachi = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C  # C = most natural grouping

# inside the loop:
tokens = sudachi.tokenize(sent, mode)
surfaces = [token.surface() for token in tokens]

print("\nSudachi Tokens:")
pprint(tokens)

print("\nSurfaces:")
pprint(surfaces)




from fugashi import Tagger

tagger = Tagger('-Owakati')
tagger.parse(sent)
# => '麩 菓子 は 、 麩 を 主材 料 と し た 日本 の 菓子 。'

print("\n\nFugashi Tokens:")
for word in tagger(sent):
    print(word, word.feature.lemma, word.pos, sep='\t')
    # "feature" is the Unidic feature data as a named tuple