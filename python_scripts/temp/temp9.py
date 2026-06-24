import shutil
from pathlib import Path

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

speaker_1_path1 = r"C:\Users\druiv\.cache\files\audio\speakers\spyx_narrator\sound1.wav"
speaker_1_path2 = r"C:\Users\druiv\.cache\files\audio\speakers\spyx_narrator\sound2.wav"
speaker_2_path1 = r"C:\Users\druiv\.cache\files\audio\speakers\spyx_lloyd\sound1.wav"

sv_pipeline = pipeline(
    task=Tasks.speaker_verification,
    model='iic/speech_eres2netv2_sv_zh-cn_16k-common',
    # model_revision='v1.0.2'
)

# Same speaker
result = sv_pipeline([speaker_1_path1, speaker_1_path2])
print(result)
assert result['text'] == 'yes'
assert result['score'] > 0.5

# Different speakers
result = sv_pipeline([speaker_1_path1, speaker_2_path1])
print(result)
assert result['text'] == 'no'
assert result['score'] < 0.5

# Get embeddings + score
result = sv_pipeline(
    [speaker_1_path1, speaker_2_path1],
    output_emb=True, # Return embeddings too
    save_dir=str(OUTPUT_DIR / "embeddings"), # Save .npy files
)
print(result['outputs'])
print(type(result['embs'])) # <class 'numpy.ndarray'>
print(result['embs'].shape) # (2, 192)
print(result['embs'].dtype) # float32
assert result['outputs']['text'] == 'no'
assert result['outputs']['score'] < 0.5

# Custom threshold
result3 = sv_pipeline([speaker_1_path1, speaker_2_path1], thr=0.25)
print(result3)
assert result3['text'] == 'yes'
assert result3['score'] > 0.25