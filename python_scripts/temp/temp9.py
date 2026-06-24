from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

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

# Custom threshold
result = sv_pipeline([speaker_1_path1, speaker_2_path1], thr=0.30)
print(result)
assert result['text'] == 'yes'
assert result['score'] > 0.30