from funasr import AutoModel
from funasr.models.fun_asr_nano.model import FunASRNano

m = AutoModel(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    trust_remote_code=True,
    device="cuda:0",
)

print("Model loaded successfully!")