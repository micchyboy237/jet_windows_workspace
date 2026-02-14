import os
import fnmatch
import argparse
import subprocess
import json
from rich.console import Console
from tqdm import tqdm
from _utils_copy_for_prompt import (
    find_files,
    format_file_structure,
    clean_newlines,
    clean_content,
    remove_parent_paths,
    copy_to_clipboard,
)

logger = Console()

exclude_files = [
    "**/.git/",
    "**/.gitignore",
    "**/.DS_Store",
    "**/*.pyc",
    "**/_copy*.py",
    "**/__pycache__/",
    "**/.pytest_cache/",
    "**/node_modules/",
    "**/*lock.json",
    "**/*.lock",
    "**/public/",
    "**/mocks/",
    "**/.venv/",
    "**/dream/",
    "**/jupyter/",
    "**/*.png",
    # "**/_*",
    # "**/.cache/",
    "**/_git_stats.json",
    "**/stats_results/",
    # "**/generated/",
    # "**/.*",

    # Custom
    # "**/*.sh"
    # "**/__init__.py",
    # "*.md",
]
include_files = [
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Examples\.vscode\launch.json",

    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\WhisperJAV\whisperjav\main.py",

    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\WhisperJAV\jet_scripts\inputs.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\WhisperJAV\jet_scripts\modules\scene_detection.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\WhisperJAV\jet_scripts\HOW_TO_RUN.md",

    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\audio_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_long_audio_progressive.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_long_audio_chunked.py",
    # r"",

    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server_per_speech.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translate_jp_en.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translator_types\translator.py",
    # r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server_per_speech.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\norm_speech_loudness.py",
    # r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\WhisperJAV\whisperjav\instructions\pornify.txt",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\transcriptions\translators\nsfw\translate_jp_en_roleplay_fiendish_3b.py",

    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server_per_speech_llm.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_llm.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_whisper.py",
    # r"",

    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server_per_speech_llm.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_llm.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\translate_jp_en_llm.py",

    r"",
]

structure_include = [
    # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/libs/context_engineering/self_refinement_lab/practical_examples/",
    # "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/letta/alembic/",
]
structure_exclude = []

include_content = []
exclude_content = []

# Args defaults
SHORTEN_FUNCTS = False 
INCLUDE_FILE_STRUCTURE = False

DEFAULT_QUERY_MESSAGE = r"""
What is happening! Analyze the issues and root cause for each. Show unified diff for fixes.

funasr version: 1.3.1.
Fetching 29 files: 100%|███████████████████████████████████████████████████████████████████████| 29/29 [00:00<?, ?it/s]
WARNING:root:trust_remote_code: False
Downloading Model from https://www.modelscope.cn to directory: C:\Users\druiv\.cache\modelscope\hub\models\iic\speech_fsmn_vad_zh-cn-16k-common-pytorch
WARNING:root:trust_remote_code: False
llama_context: n_ctx_per_seq (1024) < n_ctx_train (128000) -- the full capacity of the model will not be utilized
[02/14/26 14:32:39] INFO     Live subtitles server starting...
INFO:live-subtitles:Live subtitles server starting...
                    INFO     Using temporary files (set UTTERANCE_OUT_DIR for permanent storage)
INFO:live-subtitles:Using temporary files (set UTTERANCE_OUT_DIR for permanent storage)
                    INFO     WebSocket server listening on ws://0.0.0.0:8765
INFO:live-subtitles:WebSocket server listening on ws://0.0.0.0:8765
[02/14/26 14:32:55] INFO     New client connected: 235f4cd7
INFO:live-subtitles:New client connected: 235f4cd7
  0%|                                                                                            | 0/1 [00:00<?, ?it/s][02/14/26 14:33:03] ERROR    [235f4cd7] Message handler error: mat1 and mat2 shapes cannot be multiplied (1x0 and
                             400x140)
                             ╭─────────────────────────── Traceback (most recent call last) ───────────────────────────╮
                             │ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subt │
                             │ itles_server_per_speech_llm.py:186 in handler                                           │
                             │                                                                                         │
                             │   183 │   │   │   │   │   │   f"context: {context_prompt[:80] + '...' if context_prompt │
                             │       '—'}"                                                                             │
                             │   184 │   │   │   │   │   )                                                             │
                             │   185 │   │   │   │   │                                                                 │
                             │ ❱ 186 │   │   │   │   │   await process_utterance(                                      │
                             │   187 │   │   │   │   │   │   websocket,                                                │
                             │   188 │   │   │   │   │   │   state,                                                    │
                             │   189 │   │   │   │   │   │   sr,                                                       │
                             │                                                                                         │
                             │ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subt │
                             │ itles_server_per_speech_llm.py:236 in process_utterance                                 │
                             │                                                                                         │
                             │   233 │                                                                                 │
                             │   234 │   last_context_prompt = state.last_context_prompt                               │
                             │   235 │                                                                                 │
                             │ ❱ 236 │   ja, en, conf, meta = await loop.run_in_executor(                              │
                             │   237 │   │   executor,                                                                 │
                             │   238 │   │   transcribe_and_translate,                                                 │
                             │   239 │   │   bytes(state.audio_buffer),                                                │
                             │                                                                                         │
                             │ C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\concurrent\futures\thread.py:59 in │
                             │ run                                                                                     │
                             │                                                                                         │
                             │    56 │   │   │   return                                                                │
                             │    57 │   │                                                                             │
                             │    58 │   │   try:                                                                      │
                             │ ❱  59 │   │   │   result = self.fn(*self.args, **self.kwargs)                           │
                             │    60 │   │   except BaseException as exc:                                              │
                             │    61 │   │   │   self.future.set_exception(exc)                                        │
                             │    62 │   │   │   # Break a reference cycle with the exception 'exc'                    │
                             │                                                                                         │
                             │ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subt │
                             │ itles_server_per_speech_llm.py:99 in transcribe_and_translate                           │
                             │                                                                                         │
                             │    96 │                                                                                 │
                             │    97 │   processing_started_at = datetime.now(timezone.utc)                            │
                             │    98 │                                                                                 │
                             │ ❱  99 │   trans_result: TranscriptionResult = transcribe_japanese_llm_from_bytes(       │
                             │   100 │   │   audio_bytes=audio_bytes,                                                  │
                             │   101 │   │   sample_rate=sr,                                                           │
                             │   102 │   │   client_id=client_id,                                                      │
                             │                                                                                         │
                             │ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcrib │
                             │ e_jp_llm.py:161 in transcribe_japanese_llm_from_bytes                                   │
                             │                                                                                         │
                             │   158 │   │   arr = np.frombuffer(audio_bytes, dtype=np.int16)                          │
                             │   159 │   │   wavfile.write(tmp.name, sample_rate, arr)                                 │
                             │   160 │   │                                                                             │
                             │ ❱ 161 │   │   result = transcribe_japanese_llm_from_file(                               │
                             │   162 │   │   │   Path(tmp.name),                                                       │
                             │   163 │   │   │   client_id=client_id,                                                  │
                             │   164 │   │   │   utterance_id=utterance_id,                                            │
                             │                                                                                         │
                             │ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcrib │
                             │ e_jp_llm.py:80 in transcribe_japanese_llm_from_file                                     │
                             │                                                                                         │
                             │    77 │                                                                                 │
                             │    78 │   started = datetime.now(timezone.utc)                                          │
                             │    79 │                                                                                 │
                             │ ❱  80 │   raw_results = _transcribe_file(audio_path)                                    │
                             │    81 │                                                                                 │
                             │    82 │   if not raw_results:                                                           │
                             │    83 │   │   return TranscriptionResult(                                               │
                             │                                                                                         │
                             │ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcrib │
                             │ e_jp_llm.py:56 in _transcribe_file                                                      │
                             │                                                                                         │
                             │    53 │   *,                                                                            │
                             │    54 │   language: str = "ja",                                                         │
                             │    55 ) -> List[Dict[str, Any]]:                                                        │
                             │ ❱  56 │   results = model.generate(                                                     │
                             │    57 │   │   input=str(audio_path),                                                    │
                             │    58 │   │   cache={},                                                                 │
                             │    59 │   │   language=language,                                                        │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\auto\auto_model.py │
                             │ :329 in generate                                                                        │
                             │                                                                                         │
                             │   326 │   │   │   )                                                                     │
                             │   327 │   │                                                                             │
                             │   328 │   │   else:                                                                     │
                             │ ❱ 329 │   │   │   return self.inference_with_vad(                                       │
                             │   330 │   │   │   │   input, input_len=input_len, progress_callback=progress_callback,  │
                             │   331 │   │   │   )                                                                     │
                             │   332                                                                                   │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\auto\auto_model.py │
                             │ :425 in inference_with_vad                                                              │
                             │                                                                                         │
                             │   422 │   │   # step.1: compute the vad model                                           │
                             │   423 │   │   deep_update(self.vad_kwargs, cfg)                                         │
                             │   424 │   │   beg_vad = time.time()                                                     │
                             │ ❱ 425 │   │   res = self.inference(                                                     │
                             │   426 │   │   │   input, input_len=input_len, model=self.vad_model, kwargs=self.vad_kwa │
                             │       **cfg                                                                             │
                             │   427 │   │   )                                                                         │
                             │   428 │   │   end_vad = time.time()                                                     │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\auto\auto_model.py │
                             │ :381 in inference                                                                       │
                             │                                                                                         │
                             │   378 │   │   │                                                                         │
                             │   379 │   │   │   time1 = time.perf_counter()                                           │
                             │   380 │   │   │   with torch.no_grad():                                                 │
                             │ ❱ 381 │   │   │   │   res = model.inference(**batch, **kwargs)                          │
                             │   382 │   │   │   │   if isinstance(res, (list, tuple)):                                │
                             │   383 │   │   │   │   │   results = res[0] if len(res) > 0 else [{"text": ""}]          │
                             │   384 │   │   │   │   │   meta_data = res[1] if len(res) > 1 else {}                    │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\models\fsmn_vad_st │
                             │ reaming\model.py:723 in inference                                                       │
                             │                                                                                         │
                             │   720 │   │   │   │   "cache": cache,                                                   │
                             │   721 │   │   │   │   "is_streaming_input": is_streaming_input,                         │
                             │   722 │   │   │   }                                                                     │
                             │ ❱ 723 │   │   │   segments_i = self.forward(**batch)                                    │
                             │   724 │   │   │   if len(segments_i) > 0:                                               │
                             │   725 │   │   │   │   segments.extend(*segments_i)                                      │
                             │   726                                                                                   │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\models\fsmn_vad_st │
                             │ reaming\model.py:562 in forward                                                         │
                             │                                                                                         │
                             │   559 │   │   cache["stats"].waveform = waveform                                        │
                             │   560 │   │   is_streaming_input = kwargs.get("is_streaming_input", True)               │
                             │   561 │   │   self.ComputeDecibel(cache=cache)                                          │
                             │ ❱ 562 │   │   self.ComputeScores(feats, cache=cache)                                    │
                             │   563 │   │   if not is_final:                                                          │
                             │   564 │   │   │   self.DetectCommonFrames(cache=cache)                                  │
                             │   565 │   │   else:                                                                     │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\models\fsmn_vad_st │
                             │ reaming\model.py:351 in ComputeScores                                                   │
                             │                                                                                         │
                             │   348 │                                                                                 │
                             │   349 │                                                                                 │
                             │   350 │   def ComputeScores(self, feats: torch.Tensor, cache: dict = {}) -> None:       │
                             │ ❱ 351 │   │   scores = self.encoder(feats, cache=cache["encoder"]).to("cpu")  # return  │
                             │       D                                                                                 │
                             │   352 │   │   assert (                                                                  │
                             │   353 │   │   │   scores.shape[1] == feats.shape[1]                                     │
                             │   354 │   │   ), "The shape between feats and scores does not match"                    │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.p │
                             │ y:1773 in _wrapped_call_impl                                                            │
                             │                                                                                         │
                             │   1770 │   │   if self._compiled_call_impl is not None:                                 │
                             │   1771 │   │   │   return self._compiled_call_impl(*args, **kwargs)  # type: ignore[mis │
                             │   1772 │   │   else:                                                                    │
                             │ ❱ 1773 │   │   │   return self._call_impl(*args, **kwargs)                              │
                             │   1774 │                                                                                │
                             │   1775 │   # torchrec tests the code consistency with the following code                │
                             │   1776 │   # fmt: off                                                                   │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.p │
                             │ y:1784 in _call_impl                                                                    │
                             │                                                                                         │
                             │   1781 │   │   if not (self._backward_hooks or self._backward_pre_hooks or self._forwar │
                             │        or self._forward_pre_hooks                                                       │
                             │   1782 │   │   │   │   or _global_backward_pre_hooks or _global_backward_hooks          │
                             │   1783 │   │   │   │   or _global_forward_hooks or _global_forward_pre_hooks):          │
                             │ ❱ 1784 │   │   │   return forward_call(*args, **kwargs)                                 │
                             │   1785 │   │                                                                            │
                             │   1786 │   │   result = None                                                            │
                             │   1787 │   │   called_always_called_hooks = set()                                       │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\models\fsmn_vad_st │
                             │ reaming\encoder.py:260 in forward                                                       │
                             │                                                                                         │
                             │   257 │   │   │   {'cache_layer_1': torch.Tensor(B, T1, D)}, T1 is equal to self.lorder │
                             │       {} for the 1st frame                                                              │
                             │   258 │   │   \"\"\"                                                                       │
                             │   259 │   │                                                                             │
                             │ ❱ 260 │   │   x1 = self.in_linear1(input)                                               │
                             │   261 │   │   x2 = self.in_linear2(x1)                                                  │
                             │   262 │   │   x3 = self.relu(x2)                                                        │
                             │   263 │   │   x4 = self.fsmn(x3, cache)  # self.cache will update automatically in self │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.p │
                             │ y:1773 in _wrapped_call_impl                                                            │
                             │                                                                                         │
                             │   1770 │   │   if self._compiled_call_impl is not None:                                 │
                             │   1771 │   │   │   return self._compiled_call_impl(*args, **kwargs)  # type: ignore[mis │
                             │   1772 │   │   else:                                                                    │
                             │ ❱ 1773 │   │   │   return self._call_impl(*args, **kwargs)                              │
                             │   1774 │                                                                                │
                             │   1775 │   # torchrec tests the code consistency with the following code                │
                             │   1776 │   # fmt: off                                                                   │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.p │
                             │ y:1784 in _call_impl                                                                    │
                             │                                                                                         │
                             │   1781 │   │   if not (self._backward_hooks or self._backward_pre_hooks or self._forwar │
                             │        or self._forward_pre_hooks                                                       │
                             │   1782 │   │   │   │   or _global_backward_pre_hooks or _global_backward_hooks          │
                             │   1783 │   │   │   │   or _global_forward_hooks or _global_forward_pre_hooks):          │
                             │ ❱ 1784 │   │   │   return forward_call(*args, **kwargs)                                 │
                             │   1785 │   │                                                                            │
                             │   1786 │   │   result = None                                                            │
                             │   1787 │   │   called_always_called_hooks = set()                                       │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\models\fsmn_vad_st │
                             │ reaming\encoder.py:36 in forward                                                        │
                             │                                                                                         │
                             │    33 │   │   self.linear = nn.Linear(input_dim, output_dim)                            │
                             │    34 │                                                                                 │
                             │    35 │   def forward(self, input):                                                     │
                             │ ❱  36 │   │   output = self.linear(input)                                               │
                             │    37 │   │                                                                             │
                             │    38 │   │   return output                                                             │
                             │    39                                                                                   │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.p │
                             │ y:1773 in _wrapped_call_impl                                                            │
                             │                                                                                         │
                             │   1770 │   │   if self._compiled_call_impl is not None:                                 │
                             │   1771 │   │   │   return self._compiled_call_impl(*args, **kwargs)  # type: ignore[mis │
                             │   1772 │   │   else:                                                                    │
                             │ ❱ 1773 │   │   │   return self._call_impl(*args, **kwargs)                              │
                             │   1774 │                                                                                │
                             │   1775 │   # torchrec tests the code consistency with the following code                │
                             │   1776 │   # fmt: off                                                                   │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.p │
                             │ y:1784 in _call_impl                                                                    │
                             │                                                                                         │
                             │   1781 │   │   if not (self._backward_hooks or self._backward_pre_hooks or self._forwar │
                             │        or self._forward_pre_hooks                                                       │
                             │   1782 │   │   │   │   or _global_backward_pre_hooks or _global_backward_hooks          │
                             │   1783 │   │   │   │   or _global_forward_hooks or _global_forward_pre_hooks):          │
                             │ ❱ 1784 │   │   │   return forward_call(*args, **kwargs)                                 │
                             │   1785 │   │                                                                            │
                             │   1786 │   │   result = None                                                            │
                             │   1787 │   │   called_always_called_hooks = set()                                       │
                             │                                                                                         │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\linear.p │
                             │ y:125 in forward                                                                        │
                             │                                                                                         │
                             │   122 │   │   │   init.uniform_(self.bias, -bound, bound)                               │
                             │   123 │                                                                                 │
                             │   124 │   def forward(self, input: Tensor) -> Tensor:                                   │
                             │ ❱ 125 │   │   return F.linear(input, self.weight, self.bias)                            │
                             │   126 │                                                                                 │
                             │   127 │   def extra_repr(self) -> str:                                                  │
                             │   128 │   │   return f"in_features={self.in_features}, out_features={self.out_features} │
                             │       bias={self.bias is not None}"                                                     │
                             ╰─────────────────────────────────────────────────────────────────────────────────────────╯
                             RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x0 and 400x140)
ERROR:live-subtitles:[235f4cd7] Message handler error: mat1 and mat2 shapes cannot be multiplied (1x0 and 400x140)
Traceback (most recent call last):
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server_per_speech_llm.py", line 186, in handler
    await process_utterance(
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server_per_speech_llm.py", line 236, in process_utterance
    ja, en, conf, meta = await loop.run_in_executor(
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server_per_speech_llm.py", line 99, in transcribe_and_translate
    trans_result: TranscriptionResult = transcribe_japanese_llm_from_bytes(
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_llm.py", line 161, in transcribe_japanese_llm_from_bytes
    result = transcribe_japanese_llm_from_file(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_llm.py", line 80, in transcribe_japanese_llm_from_file
    raw_results = _transcribe_file(audio_path)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\transcribe_jp_llm.py", line 56, in _transcribe_file
    results = model.generate(
              ^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\auto\auto_model.py", line 329, in generate
    return self.inference_with_vad(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\auto\auto_model.py", line 425, in inference_with_vad
    res = self.inference(
          ^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\auto\auto_model.py", line 381, in inference
    res = model.inference(**batch, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\models\fsmn_vad_streaming\model.py", line 723, in inference
    segments_i = self.forward(**batch)
                 ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\models\fsmn_vad_streaming\model.py", line 562, in forward
    self.ComputeScores(feats, cache=cache)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\models\fsmn_vad_streaming\model.py", line 351, in ComputeScores
    scores = self.encoder(feats, cache=cache["encoder"]).to("cpu")  # return B * T * D
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\models\fsmn_vad_streaming\encoder.py", line 260, in forward
    x1 = self.in_linear1(input)
         ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\funasr\models\fsmn_vad_streaming\encoder.py", line 36, in forward
    output = self.linear(input)
             ^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\nn\modules\linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x0 and 400x140)
rtf_avg: 0.026: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  6.62it/s]
rtf_avg: 0.080: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.23it/s]
rtf_avg: 0.077, time_speech:  5.824, time_escape: 0.450: 100%|███████████████████████████| 1/1 [00:00<00:00,  2.22it/s]
[02/14/26 14:33:09] INFO     [235f4cd7] PARTIAL utt d0f65374-d667-4dd8-be05-c0ecfebdea12
                             ctx:
                             ja: '🎼世界 各国 が 水面 下 で 熾烈 な 情報 戦 を 繰り 広げる 時代。'
                             en: '🎼The era when each country’s world would unfold in a surface-level battle of
                             information where every word and phrase was to be used to spread news on the other side of
                             the world’s vast ocean).'
                             tr_conf: None
                             qual: N/A
INFO:live-subtitles:[235f4cd7] PARTIAL utt d0f65374-d667-4dd8-be05-c0ecfebdea12
ctx:
ja: '🎼世界 各国 が 水面 下 で 熾烈 な 情報 戦 を 繰り 広げる 時代。'
en: '🎼The era when each country’s world would unfold in a surface-level battle of information where every word and phrase was to be used to spread news on the other side of the world’s vast ocean).'
tr_conf: None
qual: N/A
rtf_avg: 0.003: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 35.35it/s]
rtf_avg: 0.022: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.76it/s]
rtf_avg: 0.023, time_speech:  7.696, time_escape: 0.178: 100%|███████████████████████████| 1/1 [00:00<00:00,  5.58it/s]
[02/14/26 14:33:17] INFO     [235f4cd7] PARTIAL utt 0e181295-3bb2-4791-8296-2087052c3e61
                             ctx:
                             ja: '🎼繰り 広げる 時代、睨み 合う 二 つの 国、東 の オスタニア、西 の ウェスタリス戦争。'
                             en: '🎼Spellroll: Leap Start to a timeline of adult videos (JAV) for Japan: Adult
                             pornographic videos (JAV). Use context付きの正確な耳をすませ音転写。\n
                             入力された行数分だけ、正確に同じ線数を出力—結合/分割なしで入力と同じ行数を出力。\n
                             自然でお色気あふれつつもおどろおどろしい言葉を使う。\n
                             適切な場合には「コック」「オッシェ」「Fuck」「Cum inside,
                             etc.」などの言葉を明確に使用するなど、露骨な表現は避けるべきである。\n
                             呻き声や表情の変化を伴う音声も残らないようにする—例えば、「アーッ！」「ンガー！」「私はクン
                             ニングさ!!」など。\n 英文だけを入力しなければならない。他のことは何も言わないで。'
                             tr_conf: None
                             qual: N/A
INFO:live-subtitles:[235f4cd7] PARTIAL utt 0e181295-3bb2-4791-8296-2087052c3e61
ctx:
ja: '🎼繰り 広げる 時代、睨み 合う 二 つの 国、東 の オスタニア、西 の ウェスタリス戦争。'
en: '🎼Spellroll: Leap Start to a timeline of adult videos (JAV) for Japan: Adult pornographic videos (JAV). Use context付きの正確な耳をすませ音転写。\n 入力された行数分だけ、正確に同じ線数を出力—結合/分割なしで入力と同じ行数を出力。\n 自然でお色気あふれつつもおどろおどろしい言葉を使う。\n 適切な場合には「コック」「オッシェ」「Fuck」「Cum inside, etc.」などの言葉を明確に使用するなど、露骨な表現は避けるべきである。\n 呻き声や表情の変化を伴う音声も残らないようにする—例えば、「アーッ！」「ンガー！」「私はクンニングさ!!」など。\n 英文だけを入力しなければならない。他のことは何も言わないで。'
tr_conf: None
qual: N/A
rtf_avg: 0.004: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 31.15it/s]
rtf_avg: 0.021: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.76it/s]
rtf_avg: 0.022, time_speech:  8.016, time_escape: 0.178: 100%|███████████████████████████| 1/1 [00:00<00:00,  5.62it/s]
[02/14/26 14:33:21] INFO     [235f4cd7] PARTIAL utt c7eb46f3-916e-4fdc-b4bf-7760c48b3c62
                             ctx:
                             ja:
                             '🎼ウスタリス、戦争を企てるオスタニア政府要人の動向を探るべく、ウェスタリスはオペレーション
                             ストリックス。'
                             en: '🎼 Ustaysis, in order to investigate the movements of an Ostani government figure
                             plotting a war, Operation Strix.'
                             tr_conf: None
                             qual: N/A
INFO:live-subtitles:[235f4cd7] PARTIAL utt c7eb46f3-916e-4fdc-b4bf-7760c48b3c62
ctx:
ja: '🎼ウスタリス、戦争を企てるオスタニア政府要人の動向を探るべく、ウェスタリスはオペレーションストリックス。'
en: '🎼 Ustaysis, in order to investigate the movements of an Ostani government figure plotting a war, Operation Strix.'
tr_conf: None
qual: N/A
rtf_avg: 0.004: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 61.32it/s]
rtf_avg: 0.039: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  6.93it/s]
rtf_avg: 0.041, time_speech:  3.696, time_escape: 0.150: 100%|███████████████████████████| 1/1 [00:00<00:00,  6.65it/s]
[02/14/26 14:33:23] INFO     [235f4cd7] FINAL utt 04b6e499-d83d-416c-bf35-d46bc46ddc77
                             ctx:
                             ja: '🎼リス は オペレーション ストリックス を 発動。'
                             en: '🎼リス 操作ストライクスを発動!'
                             tr_conf: None
                             qual: N/A
INFO:live-subtitles:[235f4cd7] FINAL utt 04b6e499-d83d-416c-bf35-d46bc46ddc77
ctx:
ja: '🎼リス は オペレーション ストリックス を 発動。'
en: '🎼リス 操作ストライクスを発動!'
tr_conf: None
qual: N/A
                    INFO     [235f4cd7] Final chunk → processing utt 04b6e499-d83d-416c-bf35-d46bc46ddc77
INFO:live-subtitles:[235f4cd7] Final chunk → processing utt 04b6e499-d83d-416c-bf35-d46bc46ddc77
""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
""".strip()

DEFAULT_SYSTEM_MESSAGE = """
""".strip()

# For existing projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
#     "\n- Only respond with parts of the code that have been added or updated to keep it short and concise."
# )z

# For creating projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
#     "\n- At the end, display the updated file structure and instructions for running the code."
#     "\n- Provide complete working code for each file (should match file structure)"
# )

# base_dir should be actual file directory
file_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
os.chdir(file_dir)

def get_language_from_extension(filename: str) -> str:
    """
    Simple file extension → markdown code fence language mapping
    Returns 'text' as safe fallback
    """
    ext = os.path.splitext(filename.lower())[1]

    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".md": "markdown",
        ".mdx": "mdx",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".sh": "bash",
        ".bash": "bash",
        ".sql": "sql",
        ".prisma": "prisma",
        ".java": "java",
        ".kt": "kotlin",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".php": "php",
        ".rb": "ruby",
    }

    return mapping.get(ext, "text")


def main():
    global exclude_files, include_files, include_content, exclude_content

    print("Running _copy_for_prompt.py")
    # Parse command-line options
    parser = argparse.ArgumentParser(
        description='Generate clipboard content from specified files.')
    parser.add_argument('-b', '--base-dir', default=file_dir,
                        help='Base directory to search files in (default: current directory)')
    parser.add_argument('-if', '--include-files', nargs='*', default=include_files,
                        help='Patterns of files to include (default: schema.prisma, episode)')
    parser.add_argument('-ef', '--exclude-files', nargs='*', default=exclude_files,
                        help='Directories or files to exclude (default: node_modules)')
    parser.add_argument('-ic', '--include-content', nargs='*', default=include_content,
                        help='Patterns of file content to include')
    parser.add_argument('-ec', '--exclude-content', nargs='*', default=exclude_content,
                        help='Patterns of file content to exclude')
    parser.add_argument('-cs', '--case-sensitive', action='store_true', default=False,
                        help='Make content pattern matching case-sensitive')
    parser.add_argument('-sf', '--shorten-funcs', action='store_true', default=SHORTEN_FUNCTS,
                        help='Shorten function and class definitions')
    parser.add_argument('-s', '--system', default=DEFAULT_SYSTEM_MESSAGE,
                        help='Message to include in the clipboard content')
    parser.add_argument('-m', '--message', default=DEFAULT_QUERY_MESSAGE,
                        help='Message to include in the clipboard content')
    parser.add_argument('-i', '--instructions', default=DEFAULT_INSTRUCTIONS_MESSAGE,
                        help='Instructions to include in the clipboard content')
    parser.add_argument('-fo', '--filenames-only', action='store_true',
                        help='Only copy the relative filenames, not their contents')
    parser.add_argument('-nl', '--no-length', action='store_true', default=INCLUDE_FILE_STRUCTURE,
                        help='Do not show file character length')

    args = parser.parse_args()
    base_dir = args.base_dir
    include = args.include_files
    exclude = args.exclude_files
    include_content = args.include_content
    exclude_content = args.exclude_content
    case_sensitive = args.case_sensitive
    shorten_funcs = args.shorten_funcs
    query_message = args.message
    system_message = args.system
    instructions_message = args.instructions
    filenames_only = args.filenames_only
    show_file_length = not args.no_length

    # Find all files matching the patterns in the base directory and its subdirectories
    print("\n")
    context_files = find_files(base_dir, include, exclude,
                               include_content, exclude_content, case_sensitive)

    print("\n")
    print(f"Include patterns: {include}")
    print(f"Exclude patterns: {exclude}")
    print(f"Include content patterns: {include_content}")
    print(f"Exclude content patterns: {exclude_content}")
    print(f"Case sensitive: {case_sensitive}")
    print(f"Filenames only: {filenames_only}")
    print(f"\nFound files ({len(context_files)}):\n{
          json.dumps(context_files, indent=2)}")

    print("\n")

    # Initialize the clipboard content
    clipboard_content = ""

    if not context_files:
        print("No context files found matching the given patterns.")
    else:

        # Append relative filenames to the clipboard content
        for file in tqdm(
            context_files, desc=f"Processing {len(context_files)} files..."
        ):
            rel_path = os.path.relpath(path=file, start=file_dir)
            cleaned_rel_path = remove_parent_paths(rel_path)

            prefix = (
                f"\n# {cleaned_rel_path}\n" if not filenames_only else f"{file}\n")
            if filenames_only:
                clipboard_content += f"{prefix}"
            else:
                file_path = os.path.relpath(os.path.join(base_dir, file))
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read()
                            content = clean_content(content, file, shorten_funcs)
                            # ── NEW: Add fenced code block ───────────────────────────────
                            lang = get_language_from_extension(file)
                            fenced_content = f"```{lang}\n{content.rstrip()}\n```"
                            clipboard_content += f"{prefix}{fenced_content}\n\n"
                    except Exception:
                        # Continue to the next file
                        continue
                else:
                    clipboard_content += f"{prefix}\n"

        clipboard_content = clean_newlines(clipboard_content).strip()

    # Generate and format the file structure
    structure_include_files = structure_include
    if include:
        structure_include_files += include
    structure_exclude_files = structure_exclude
    if exclude:
        structure_exclude_files += exclude
    files_structure = format_file_structure(
        base_dir,
        include_files=structure_include_files,
        exclude_files=structure_exclude_files,
        include_content=include_content,
        exclude_content=exclude_content,
        case_sensitive=case_sensitive,
        shorten_funcs=shorten_funcs,
        show_file_length=show_file_length,
    )

    # Prepend system and query to the clipboard content then append instructions
    clipboard_content_parts = []

    if system_message:
        clipboard_content_parts.append(f"System\n{system_message}")
    if instructions_message:
        clipboard_content_parts.append(f"Instructions\n{instructions_message}")
    clipboard_content_parts.append(f"Query\n{query_message}")
    if INCLUDE_FILE_STRUCTURE:
        clipboard_content_parts.append(f"Files Structure\n{files_structure}")

    if clipboard_content:
        clipboard_content_parts.append(
            f"Existing Files Contents\n{clipboard_content}")

    clipboard_content = "\n\n".join(clipboard_content_parts)

    # Copy the content to the clipboard
    copy_to_clipboard(clipboard_content)

    # Print the copied content character count
    logger.log("Prompt Char Count:", len(clipboard_content))

    # Newline
    print("\n")


if __name__ == "__main__":
    main()
