import os
import fnmatch
import argparse
import subprocess
import json
import tiktoken
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
from headroom import compress

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
    "**/*.svg",
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
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\embedding_model_factory.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\segment_speaker_labeler.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speaker_labeler_utils\speaker_reference.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speaker_labeler_utils\segment_types.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speaker_labeler_utils\outlier_orchestrator.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speaker_labeler_utils\outlier_pool.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speaker_labeler_utils\speaker_labeler_serializer.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\embedding_model_factory.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\evaluate_speaker_embeddings.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_evaluate_speaker_embeddings.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\generated\_main_evaluate_speaker_embeddings\eval_results",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\processing.py",
    r"",
]

structure_include = [
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers",
]
structure_exclude = []

include_content = [
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\tagger",
]
exclude_content = []

# Args defaults
SHORTEN_FUNCTS = False 
INCLUDE_FILE_STRUCTURE = False

COMPRESSION_MODEL = "gpt-4o"
TOKEN_BUDGET = 8000

DEFAULT_QUERY_MESSAGE = r"""
(jet_venv) PS C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace> python C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\evaluate_speaker_embeddings.py
[11:37:20] INFO     Speaker Embedding Evaluation                                      evaluate_speaker_embeddings.py:748
           INFO     Dataset: C:\Users\druiv\.cache\files\audio\speakers               evaluate_speaker_embeddings.py:749
           INFO     Models to evaluate: ['pyannote', 'speechbrain_ecapa',             evaluate_speaker_embeddings.py:753
                    'nemo_titanet', 'modelscope_eres2netv2']
           INFO     Device: cuda                                                      evaluate_speaker_embeddings.py:764
           INFO     Scanning dataset at: C:\Users\druiv\.cache\files\audio\speakers   evaluate_speaker_embeddings.py:138
           WARNING  Skipping speaker 'spyx_yor': only 1 file(s), need >= 2            evaluate_speaker_embeddings.py:151
           INFO     Found 4 speakers, 23 total utterances                             evaluate_speaker_embeddings.py:156
           INFO     Generated 32 positive trials                                      evaluate_speaker_embeddings.py:199
           INFO     Generated 32 negative trials                                      evaluate_speaker_embeddings.py:213
           INFO     Total trials: 64                                                  evaluate_speaker_embeddings.py:217
           INFO                                                                       evaluate_speaker_embeddings.py:508
                    ────────────────────────────────────────────────────────────
           INFO     Evaluating model: pyannote                                        evaluate_speaker_embeddings.py:509
[11:37:20] EmbeddingFactory Creating PyannoteEmbeddingModel (device=cuda)                 embedding_model_factory.py:763
[11:37:26] INFO     NumExpr defaulting to 12 threads.                                                       utils.py:164
C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\core\io.py:135: UserWarning: torchcodec is not installed (Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version (2.11.0+cu130) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:

[start of libtorchcodec loading traceback]
FFmpeg version 8:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1503, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core8.dll' (or one of its dependencies). Try using the full path with constructor syntax.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1505, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core8.dll

FFmpeg version 7:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1503, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: [WinError 127] The specified procedure could not be found

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1505, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core7.dll

FFmpeg version 6:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1503, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core6.dll' (or one of its dependencies). Try using the full path with constructor syntax.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1505, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core6.dll

FFmpeg version 5:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1503, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core5.dll' (or one of its dependencies). Try using the full path with constructor syntax.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1505, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core5.dll

FFmpeg version 4:
Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1503, in load_library
    ctypes.CDLL(path)
  File "C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\ctypes\__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: Could not find module 'C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core4.dll' (or one of its dependencies). Try using the full path with constructor syntax.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\_core\ops.py", line 57, in load_torchcodec_shared_libraries
    torch.ops.load_library(core_library_path)
  File "C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torch\_ops.py", line 1505, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\torchcodec\libtorchcodec_core4.dll
[end of libtorchcodec loading traceback].). Using soundfile backend instead.
For better performance and format support, consider installing torchcodec.
  warnings.warn(
[11:37:27] EmbeddingFactory Loading pyannote model 'pyannote/embedding'...                embedding_model_factory.py:311
[11:37:31] INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/pyannote/embedding/resolve/main/pytorch_model.bin "HTTP/1.1
                    302 Found"
W0625 11:37:31.409000 41604 Lib\site-packages\torch\utils\flop_counter.py:29] triton not found; flop counting will not work for triton kernels
C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\lightning\pytorch\utilities\migration\utils.py:197: Redirecting import of pytorch_lightning.callbacks.early_stopping.EarlyStopping to lightning.pytorch.callbacks.early_stopping.EarlyStopping
C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\lightning\pytorch\utilities\migration\utils.py:197: Redirecting import of pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint to lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint
C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\lightning\pytorch\utilities\migration\migration.py:208: You have multiple `ModelCheckpoint` callback states in this checkpoint, but we found state keys that would end up colliding with each other after an upgrade, which means we can't differentiate which of your checkpoint callbacks needs which states. At least one of your `ModelCheckpoint` callbacks will not be able to reload the state.
INFO: Lightning automatically upgraded your loaded checkpoint from v1.2.7 to v2.4.0. To apply the upgrade to your files permanently, run `python -m lightning.pytorch.utilities.upgrade_checkpoint C:\Users\druiv\.cache\huggingface\hub\models--pyannote--embedding\snapshots\4db4899737a38b2d618bbd74350915aa10293cb2\pytorch_model.bin`
           INFO     Lightning automatically upgraded your loaded checkpoint from v1.2.7 to v2.4.0. To apply utils.py:154
                    the upgrade to your files permanently, run `python -m
                    lightning.pytorch.utilities.upgrade_checkpoint
                    C:\Users\druiv\.cache\huggingface\hub\models--pyannote--embedding\snapshots\4db4899737a
                    38b2d618bbd74350915aa10293cb2\pytorch_model.bin`
C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\core\model.py:639: UserWarning: Model has been trained with a task-dependent loss function. Set 'strict' to False to load the model without its loss function and prevent this warning from appearing.
  warnings.warn(msg)
INFO: Lightning automatically upgraded your loaded checkpoint from v1.2.7 to v2.4.0. To apply the upgrade to your files permanently, run `python -m lightning.pytorch.utilities.upgrade_checkpoint C:\Users\druiv\.cache\huggingface\hub\models--pyannote--embedding\snapshots\4db4899737a38b2d618bbd74350915aa10293cb2\pytorch_model.bin`
[11:37:32] INFO     Lightning automatically upgraded your loaded checkpoint from v1.2.7 to v2.4.0. To apply utils.py:154
                    the upgrade to your files permanently, run `python -m
                    lightning.pytorch.utilities.upgrade_checkpoint
                    C:\Users\druiv\.cache\huggingface\hub\models--pyannote--embedding\snapshots\4db4899737a
                    38b2d618bbd74350915aa10293cb2\pytorch_model.bin`
C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\lightning\pytorch\core\saving.py:195: Found keys that are not in the model state dict but in the checkpoint: ['loss_func.W']
[11:37:32] EmbeddingFactory Pyannote model ready on cuda                                  embedding_model_factory.py:319
           INFO     Model loaded: PyannoteEmbeddingModel(type=pyannote, dim=512)      evaluate_speaker_embeddings.py:516
           EmbeddingFactory Thresholds for pyannote: same=0.2, possible=0.13,             embedding_model_factory.py:162
           new_speaker=0.14
           INFO      Configured thresholds — same=0.2, possible=0.13,                 evaluate_speaker_embeddings.py:522
                    new_speaker=0.14
           INFO      Extracting embeddings for 23 unique files...                     evaluate_speaker_embeddings.py:252
[11:37:33] INFO      Done. Cache hits: 0, Computed: 23, Avg time: 75.0 ms             evaluate_speaker_embeddings.py:298
           INFO      Scored 64 trials                                                 evaluate_speaker_embeddings.py:537
           INFO      EER=31.25% @ threshold=0.1533                                    evaluate_speaker_embeddings.py:542
           INFO      minDCF=0.8750                                                    evaluate_speaker_embeddings.py:543
           INFO      Threshold check OK: configured same=0.2 vs EER threshold=0.1533  evaluate_speaker_embeddings.py:559
                    (ratio=1.30)
           INFO      Intra=0.2743 | Inter=0.1080 | Sep=0.1664                         evaluate_speaker_embeddings.py:571
           INFO      EER=31.25% | minDCF=0.8750 | Intra=0.2743 | Inter=0.1080 |       evaluate_speaker_embeddings.py:793
                    Sep=0.1664 | Latency=75.0ms | Thresh(same=0.2, possible=0.13,
                    new_spk=0.14)
           INFO                                                                       evaluate_speaker_embeddings.py:508
                    ────────────────────────────────────────────────────────────
           INFO     Evaluating model: speechbrain_ecapa                               evaluate_speaker_embeddings.py:509
[11:37:33] EmbeddingFactory Creating SpeechBrainECAPAEmbeddingModel (device=cuda)         embedding_model_factory.py:763
[11:37:34] INFO     Applied quirks (see `speechbrain.utils.quirks`):                                       quirks.py:115
           INFO     Excluded quirks specified by the `SB_DISABLE_QUIRKS` environment (comma-separated      quirks.py:120
                    list): []
[11:37:34] EmbeddingFactory Loading SpeechBrain ECAPA from                                embedding_model_factory.py:393
           'speechbrain/spkrec-ecapa-voxceleb'...
           INFO     Fetch hyperparams.yaml: Fetching from HuggingFace Hub                                fetching.py:403
                    'speechbrain/spkrec-ecapa-voxceleb' if not cached
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/hyperparams.ya
                    ml "HTTP/1.1 307 Temporary Redirect"
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/api/resolve-cache/models/speechbrain/spkrec-ecapa-voxceleb/0f
                    99f2d0ebe89ac095bcc5903c4dd8f72b367286/hyperparams.yaml "HTTP/1.1 200 OK"
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/hyperparams.ya
                    ml "HTTP/1.1 307 Temporary Redirect"
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/api/resolve-cache/models/speechbrain/spkrec-ecapa-voxceleb/0f
                    99f2d0ebe89ac095bcc5903c4dd8f72b367286/hyperparams.yaml "HTTP/1.1 200 OK"
           INFO     Fetch embedding_model.ckpt: Fetching from HuggingFace Hub                            fetching.py:403
                    'speechbrain/spkrec-ecapa-voxceleb' if not cached
[11:37:35] INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_mode
                    l.ckpt "HTTP/1.1 302 Found"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
           WARNING  Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN  _http.py:904
                    to enable higher rate limits and faster downloads.
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_mode
                    l.ckpt "HTTP/1.1 302 Found"
           INFO     Fetch mean_var_norm_emb.ckpt: Fetching from HuggingFace Hub                          fetching.py:403
                    'speechbrain/spkrec-ecapa-voxceleb' if not cached
[11:37:36] INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/mean_var_norm_
                    emb.ckpt "HTTP/1.1 307 Temporary Redirect"
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/api/resolve-cache/models/speechbrain/spkrec-ecapa-voxceleb/0f
                    99f2d0ebe89ac095bcc5903c4dd8f72b367286/mean_var_norm_emb.ckpt "HTTP/1.1 200 OK"
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/mean_var_norm_
                    emb.ckpt "HTTP/1.1 307 Temporary Redirect"
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/api/resolve-cache/models/speechbrain/spkrec-ecapa-voxceleb/0f
                    99f2d0ebe89ac095bcc5903c4dd8f72b367286/mean_var_norm_emb.ckpt "HTTP/1.1 200 OK"
           INFO     Fetch classifier.ckpt: Fetching from HuggingFace Hub                                 fetching.py:403
                    'speechbrain/spkrec-ecapa-voxceleb' if not cached
[11:37:37] INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/classifier.ckp
                    t "HTTP/1.1 302 Found"
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/classifier.ckp
                    t "HTTP/1.1 302 Found"
           INFO     Fetch label_encoder.txt: Fetching from HuggingFace Hub                               fetching.py:403
                    'speechbrain/spkrec-ecapa-voxceleb' if not cached
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/label_encoder.
                    txt "HTTP/1.1 307 Temporary Redirect"
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/api/resolve-cache/models/speechbrain/spkrec-ecapa-voxceleb/0f
                    99f2d0ebe89ac095bcc5903c4dd8f72b367286/label_encoder.txt "HTTP/1.1 200 OK"
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/label_encoder.
                    txt "HTTP/1.1 307 Temporary Redirect"
           INFO     HTTP Request: HEAD                                                                   _client.py:1025
                    https://huggingface.co/api/resolve-cache/models/speechbrain/spkrec-ecapa-voxceleb/0f
                    99f2d0ebe89ac095bcc5903c4dd8f72b367286/label_encoder.txt "HTTP/1.1 200 OK"
           INFO     Loading pretrained files for: embedding_model, mean_var_norm_emb,          parameter_transfer.py:301
                    classifier, label_encoder
[11:37:38] WARNING  Could not parse CUDA device string 'cuda': not enough values to unpack (expected   interfaces.py:290
                    2, got 1). Falling back to device 0.
[11:37:38] EmbeddingFactory SpeechBrain ECAPA ready on cuda                               embedding_model_factory.py:400
           INFO     Model loaded:                                                     evaluate_speaker_embeddings.py:516
                    SpeechBrainECAPAEmbeddingModel(type=speechbrain_ecapa, dim=192)
           EmbeddingFactory Thresholds for speechbrain_ecapa: same=0.23, possible=0.17,   embedding_model_factory.py:162
           new_speaker=0.17
           INFO      Configured thresholds — same=0.23, possible=0.17,                evaluate_speaker_embeddings.py:522
                    new_speaker=0.17
           INFO      Extracting embeddings for 23 unique files...                     evaluate_speaker_embeddings.py:252
[11:37:39] INFO      Done. Cache hits: 0, Computed: 23, Avg time: 40.2 ms             evaluate_speaker_embeddings.py:298
           INFO      Scored 64 trials                                                 evaluate_speaker_embeddings.py:537
           INFO      EER=32.81% @ threshold=0.1051                                    evaluate_speaker_embeddings.py:542
           INFO      minDCF=0.7188                                                    evaluate_speaker_embeddings.py:543
           INFO      Threshold check OK: configured same=0.23 vs EER threshold=0.1051 evaluate_speaker_embeddings.py:559
                    (ratio=2.19)
           INFO      Intra=0.3200 | Inter=0.1417 | Sep=0.1783                         evaluate_speaker_embeddings.py:571
           INFO      EER=32.81% | minDCF=0.7188 | Intra=0.3200 | Inter=0.1417 |       evaluate_speaker_embeddings.py:793
                    Sep=0.1783 | Latency=40.2ms | Thresh(same=0.23, possible=0.17,
                    new_spk=0.17)
           INFO                                                                       evaluate_speaker_embeddings.py:508
                    ────────────────────────────────────────────────────────────
           INFO     Evaluating model: nemo_titanet                                    evaluate_speaker_embeddings.py:509
[11:37:39] EmbeddingFactory Creating NeMoTitaNetEmbeddingModel (device=cuda)              embedding_model_factory.py:763
[NeMo W 2026-06-25 11:37:39 megatron_init:62] Megatron num_microbatches_calculator not found, using Apex version.
W0625 11:37:39.780000 41604 Lib\site-packages\torch\distributed\elastic\multiprocessing\redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
[11:37:40] INFO     Numba verbose is deactivated. To enable it, set NUMBA_VERBOSE to 1.            transducer_loss.py:27
[11:37:41] WARNING  OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR  config.py:193
                    for rank (rank=0) with OneLogger disabled. To override: explicitly set
                    error_handling_strategy parameter.
           INFO     Final configuration contains 0 exporter(s)                              export_config_manager.py:108
           WARNING  No exporters were provided. This means that no telemetry data     training_telemetry_provider.py:309
                    will be collected.
[11:37:41] EmbeddingFactory Loading NeMo 'titanet_large'...                               embedding_model_factory.py:447
[NeMo I 2026-06-25 11:37:41 cloud:58] Found existing object C:\Users\druiv\.cache\torch\NeMo\NeMo_2.7.2\titanet-l\11ba0924fdf87c049e339adbf6899d48\titanet-l.nemo.
[NeMo I 2026-06-25 11:37:41 cloud:64] Re-using file from: C:\Users\druiv\.cache\torch\NeMo\NeMo_2.7.2\titanet-l\11ba0924fdf87c049e339adbf6899d48\titanet-l.nemo
[NeMo I 2026-06-25 11:37:41 common:939] Instantiating model from pre-trained checkpoint
[NeMo W 2026-06-25 11:37:42 modelPT:188] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config :
    manifest_filepath: /manifests/combined_fisher_swbd_voxceleb12_librispeech/train.json
    sample_rate: 16000
    labels: null
    batch_size: 64
    shuffle: true
    is_tarred: false
    tarred_audio_filepaths: null
    tarred_shard_strategy: scatter
    augmentor:
      noise:
        manifest_path: /manifests/noise/rir_noise_manifest.json
        prob: 0.5
        min_snr_db: 0
        max_snr_db: 15
      speed:
        prob: 0.5
        sr: 16000
        resample_type: kaiser_fast
        min_speed_rate: 0.95
        max_speed_rate: 1.05
    num_workers: 15
    pin_memory: true

[NeMo W 2026-06-25 11:37:42 modelPT:195] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s).
    Validation config :
    manifest_filepath: /manifests/combined_fisher_swbd_voxceleb12_librispeech/dev.json
    sample_rate: 16000
    labels: null
    batch_size: 128
    shuffle: false
    num_workers: 15
    pin_memory: true

[NeMo I 2026-06-25 11:37:42 save_restore_connector:285] Model EncDecSpeakerLabelModel was successfully restored from C:\Users\druiv\.cache\torch\NeMo\NeMo_2.7.2\titanet-l\11ba0924fdf87c049e339adbf6899d48\titanet-l.nemo.
[11:37:42] EmbeddingFactory NeMo TitaNet ready on cuda                                    embedding_model_factory.py:453
[11:37:42] INFO     Model loaded: NeMoTitaNetEmbeddingModel(type=nemo_titanet,        evaluate_speaker_embeddings.py:516
                    dim=192)
           EmbeddingFactory Thresholds for nemo_titanet: same=0.5, possible=0.35,         embedding_model_factory.py:162
           new_speaker=0.23
           INFO      Configured thresholds — same=0.5, possible=0.35,                 evaluate_speaker_embeddings.py:522
                    new_speaker=0.23
           INFO      Extracting embeddings for 23 unique files...                     evaluate_speaker_embeddings.py:252
[11:37:43] INFO      Done. Cache hits: 0, Computed: 23, Avg time: 29.3 ms             evaluate_speaker_embeddings.py:298
           INFO      Scored 64 trials                                                 evaluate_speaker_embeddings.py:537
           INFO      EER=18.75% @ threshold=0.3568                                    evaluate_speaker_embeddings.py:542
           INFO      minDCF=0.4062                                                    evaluate_speaker_embeddings.py:543
           INFO      Threshold check OK: configured same=0.5 vs EER threshold=0.3568  evaluate_speaker_embeddings.py:559
                    (ratio=1.40)
           INFO      Intra=0.6197 | Inter=0.2041 | Sep=0.4156                         evaluate_speaker_embeddings.py:571
           INFO      EER=18.75% | minDCF=0.4062 | Intra=0.6197 | Inter=0.2041 |       evaluate_speaker_embeddings.py:793
                    Sep=0.4156 | Latency=29.3ms | Thresh(same=0.5, possible=0.35,
                    new_spk=0.23)
           INFO                                                                       evaluate_speaker_embeddings.py:508
                    ────────────────────────────────────────────────────────────
           INFO     Evaluating model: modelscope_eres2netv2                           evaluate_speaker_embeddings.py:509
[11:37:43] EmbeddingFactory Creating ModelScopeEres2Netv2EmbeddingModel (device=cuda)     embedding_model_factory.py:763
           EmbeddingFactory Loading ModelScope ERes2NetV2                                 embedding_model_factory.py:528
           'iic/speech_eres2netv2_sv_zh-cn_16k-common'...
Downloading Model from https://www.modelscope.cn to directory: C:\Users\druiv\.cache\modelscope\hub\models\iic\speech_eres2netv2_sv_zh-cn_16k-common
2026-06-25 11:38:01,934 - modelscope - INFO - initiate model from C:\Users\druiv\.cache\modelscope\hub\models\iic\speech_eres2netv2_sv_zh-cn_16k-common
2026-06-25 11:38:01,934 - modelscope - INFO - initiate model from location C:\Users\druiv\.cache\modelscope\hub\models\iic\speech_eres2netv2_sv_zh-cn_16k-common.
2026-06-25 11:38:01,938 - modelscope - INFO - initialize model from C:\Users\druiv\.cache\modelscope\hub\models\iic\speech_eres2netv2_sv_zh-cn_16k-common
2026-06-25 11:38:01,938 - modelscope - WARNING - Use allow_remote=True. Will invoke codes from C:\Users\druiv\.cache\modelscope\hub\models\iic\speech_eres2netv2_sv_zh-cn_16k-common. Please make sure that you can trust the external codes.
2026-06-25 11:38:02,293 - modelscope - WARNING - No preprocessor field found in cfg.
2026-06-25 11:38:02,293 - modelscope - WARNING - No val key and type key found in preprocessor domain of configuration.json file.
2026-06-25 11:38:02,293 - modelscope - WARNING - Cannot find available config to build preprocessor at mode inference, current config: {'model_dir': 'C:\\Users\\druiv\\.cache\\modelscope\\hub\\models\\iic\\speech_eres2netv2_sv_zh-cn_16k-common'}. trying to build by task and model information.
2026-06-25 11:38:02,293 - modelscope - INFO - No preprocessor key ('eres2netv2-sv', 'speaker-verification') found in PREPROCESSOR_MAP, skip building preprocessor. If the pipeline runs normally, please ignore this log.
[11:38:02] EmbeddingFactory ModelScope ERes2NetV2 ready (pipeline device is managed by    embedding_model_factory.py:536
           ModelScope)
[11:38:02] INFO     Model loaded:                                                     evaluate_speaker_embeddings.py:516
                    ModelScopeEres2Netv2EmbeddingModel(type=modelscope_eres2netv2,
                    dim=512)
           EmbeddingFactory Thresholds for modelscope_eres2netv2: same=0.55,              embedding_model_factory.py:162
           possible=0.4, new_speaker=0.3
           INFO      Configured thresholds — same=0.55, possible=0.4, new_speaker=0.3 evaluate_speaker_embeddings.py:522
           INFO      Extracting embeddings for 23 unique files...                     evaluate_speaker_embeddings.py:252
⠋ Embedding  0:00:00           EmbeddingFactory Updating embedding_dim from 512 to 192                        embedding_model_factory.py:611
⠹ Embedding  0:00:00           EmbeddingFactory __call__: 44100 Hz input detected — routing through encode()  embedding_model_factory.py:688
           for pre-resampling (avoids torchaudio.sox_effects)
           EmbeddingFactory Pre-resampling 44100 Hz → 16000 Hz (avoids                    embedding_model_factory.py:572
           torchaudio.sox_effects)
⠼ Embedding  0:00:00           EmbeddingFactory __call__: 44100 Hz input detected — routing through encode()  embedding_model_factory.py:688
           for pre-resampling (avoids torchaudio.sox_effects)
           EmbeddingFactory Pre-resampling 44100 Hz → 16000 Hz (avoids                    embedding_model_factory.py:572
           torchaudio.sox_effects)
⠇ Embedding  0:00:00           EmbeddingFactory __call__: 44100 Hz input detected — routing through encode()  embedding_model_factory.py:688
           for pre-resampling (avoids torchaudio.sox_effects)
           EmbeddingFactory Pre-resampling 44100 Hz → 16000 Hz (avoids                    embedding_model_factory.py:572
           torchaudio.sox_effects)
⠏ Embedding  0:00:00[11:38:03] EmbeddingFactory __call__: 44100 Hz input detected — routing through encode()  embedding_model_factory.py:688
           for pre-resampling (avoids torchaudio.sox_effects)
           EmbeddingFactory Pre-resampling 44100 Hz → 16000 Hz (avoids                    embedding_model_factory.py:572
           torchaudio.sox_effects)
⠴ Embedding  0:00:01           EmbeddingFactory __call__: 44100 Hz input detected — routing through encode()  embedding_model_factory.py:688
           for pre-resampling (avoids torchaudio.sox_effects)
           EmbeddingFactory Pre-resampling 44100 Hz → 16000 Hz (avoids                    embedding_model_factory.py:572
           torchaudio.sox_effects)
⠦ Embedding  0:00:01           EmbeddingFactory __call__: 44100 Hz input detected — routing through encode()  embedding_model_factory.py:688
           for pre-resampling (avoids torchaudio.sox_effects)
           EmbeddingFactory Pre-resampling 44100 Hz → 16000 Hz (avoids                    embedding_model_factory.py:572
           torchaudio.sox_effects)
[11:38:04] INFO      Done. Cache hits: 0, Computed: 23, Avg time: 77.9 ms             evaluate_speaker_embeddings.py:298
           INFO      Scored 64 trials                                                 evaluate_speaker_embeddings.py:537
           INFO      EER=15.62% @ threshold=0.4838                                    evaluate_speaker_embeddings.py:542
           INFO      minDCF=0.5000                                                    evaluate_speaker_embeddings.py:543
           INFO      Threshold check OK: configured same=0.55 vs EER threshold=0.4838 evaluate_speaker_embeddings.py:559
                    (ratio=1.14)
           INFO      Intra=0.6943 | Inter=0.3355 | Sep=0.3588                         evaluate_speaker_embeddings.py:571
           INFO      EER=15.62% | minDCF=0.5000 | Intra=0.6943 | Inter=0.3355 |       evaluate_speaker_embeddings.py:793
                    Sep=0.3588 | Latency=77.9ms | Thresh(same=0.55, possible=0.4,
                    new_spk=0.3)

                                           Speaker Embedding Model Comparison
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━
┃ #   ┃ Model                ┃ Dim ┃  EER ↓ ┃ EER Thresh ┃ minDCF ↓ ┃ Intra ↑ ┃ Inter ↓ ┃  Sep ↑ ┃ Cfg Same ┃ ms/file ↓
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━
│ 1   │ modelscope_eres2net… │ 192 │ 15.62% │     0.4838 │   0.5000 │  0.6943 │  0.3355 │ 0.3588 │    0.550 │      77.9
│ 2   │ nemo_titanet         │ 192 │ 18.75% │     0.3568 │   0.4062 │  0.6197 │  0.2041 │ 0.4156 │    0.500 │      29.3
│ 3   │ pyannote             │ 512 │ 31.25% │     0.1533 │   0.8750 │  0.2743 │  0.1080 │ 0.1664 │    0.200 │      75.0
│ 4   │ speechbrain_ecapa    │ 192 │ 32.81% │     0.1051 │   0.7188 │  0.3200 │  0.1417 │ 0.1783 │    0.230 │      40.2
└─────┴──────────────────────┴─────┴────────┴────────────┴──────────┴─────────┴─────────┴────────┴──────────┴───────────

↓ = lower is better   ↑ = higher is better   Sep = Intra − Inter   Cfg Same = configured same-speaker threshold
""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
General:
- Browse when beneficial or requested.
- Keep explanations simple and clear.

When coding:
- Provide step-by-step analysis and explain the flow.
- Use visuals, diagrams, or tables when helpful.
- For additions, show full code for new files, classes, methods, or functions.
- For changes, show full code for updated functions or methods; otherwise, show only the changed lines with surrounding context.
- Write smart, flexible, reusable, maintainable, optimal, robust, and minimal code.
- Always add logs so we can trace and know if all features work correctly.
""".strip()

DEFAULT_SYSTEM_MESSAGE = """
""".strip()

# For existing projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
# "\n- Only respond with parts of the code that have been added or updated to keep it short and concise."
# )z
# For creating projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
# "\n- At the end, display the updated file structure and instructions for running the code."
# "\n- Provide complete working code for each file (should match file structure)"
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
        description="Generate clipboard content from specified files."
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        default=file_dir,
        help="Base directory to search files in (default: current directory)",
    )
    parser.add_argument(
        "-if",
        "--include-files",
        nargs="*",
        default=include_files,
        help="Patterns of files to include (default: schema.prisma, episode)",
    )
    parser.add_argument(
        "-ef",
        "--exclude-files",
        nargs="*",
        default=exclude_files,
        help="Directories or files to exclude (default: node_modules)",
    )
    parser.add_argument(
        "-ic",
        "--include-content",
        nargs="*",
        default=include_content,
        help="Patterns of file content to include",
    )
    parser.add_argument(
        "-ec",
        "--exclude-content",
        nargs="*",
        default=exclude_content,
        help="Patterns of file content to exclude",
    )
    parser.add_argument(
        "-cs",
        "--case-sensitive",
        action="store_true",
        default=False,
        help="Make content pattern matching case-sensitive",
    )
    parser.add_argument(
        "-sf",
        "--shorten-funcs",
        action="store_true",
        default=SHORTEN_FUNCTS,
        help="Shorten function and class definitions",
    )
    parser.add_argument(
        "-s",
        "--system",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="Message to include in the clipboard content",
    )
    parser.add_argument(
        "-m",
        "--message",
        default=DEFAULT_QUERY_MESSAGE,
        help="Message to include in the clipboard content",
    )
    parser.add_argument(
        "-i",
        "--instructions",
        default=DEFAULT_INSTRUCTIONS_MESSAGE,
        help="Instructions to include in the clipboard content",
    )
    parser.add_argument(
        "-fo",
        "--filenames-only",
        action="store_true",
        help="Only copy the relative filenames, not their contents",
    )
    parser.add_argument(
        "-nl",
        "--no-length",
        action="store_true",
        default=INCLUDE_FILE_STRUCTURE,
        help="Do not show file character length",
    )
    parser.add_argument(
        "-c",
        "--compress",
        action="store_true",
        default=False,
        help="Enable compression of the clipboard content before copying (default: False)",
    )
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
    compress_enabled = args.compress
    # Find all files matching the patterns in the base directory and its subdirectories
    print("\n")
    context_files = find_files(
        base_dir, include, exclude, include_content, exclude_content, case_sensitive
    )
    print("\n")
    print(f"Include patterns: {include}")
    print(f"Exclude patterns: {exclude}")
    print(f"Include content patterns: {include_content}")
    print(f"Exclude content patterns: {exclude_content}")
    print(f"Case sensitive: {case_sensitive}")
    print(f"Filenames only: {filenames_only}")
    print(f"Compress enabled: {compress_enabled}")
    print(
        f"\nFound files ({len(context_files)}):\n{json.dumps(context_files, indent=2)}"
    )
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
            prefix = f"\n# {cleaned_rel_path}\n" if not filenames_only else f"{file}\n"
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
        clipboard_content_parts.append(f"<system>\n{system_message}\n</system>")
    # Query should come before instructions
    clipboard_content_parts.append(f"<query>\n{query_message}\n</query>")
    if instructions_message:
        clipboard_content_parts.append(f"<instructions>\n{instructions_message}\n</instructions>")
    if INCLUDE_FILE_STRUCTURE:
        clipboard_content_parts.append(f"Files Structure\n{files_structure}\n")
    if clipboard_content:
        clipboard_content_parts.append(
            f"Existing Files Contents\n{clipboard_content}\n"
        )
    clipboard_content = "\n\n".join(clipboard_content_parts)
    # Compress to reduce tokens (optional)
    if compress_enabled:
        messages = [{"role": "user", "content": clipboard_content}]
        result = compress(
            messages,
            model=COMPRESSION_MODEL,  # headroom uses this for strategy selection only
            token_budget=TOKEN_BUDGET,  # enforce fit within llama-server context
            ccr_enabled=True,  # reversible compression (default)
            compress_user_messages=True,
            target_ratio=0.5,  # keep 50% — safe for mixed prose + code
            protect_recent=0,  # only 1 message, nothing to protect
            protect_analysis_context=False,  # do not protect code from compression
            # kompress_model="disabled",
        )
        # Log compression stats using logger.log for each result.*
        logger.log("Tokens before:", f"{result.tokens_before:,}")
        logger.log("Tokens after:", f"{result.tokens_after:,}")
        logger.log(
            "Tokens saved:",
            f"{result.tokens_saved:,} ({result.compression_ratio:.1%})",
        )
        logger.log(
            "Transforms applied:",
            str(result.transforms_applied),
        )
    else:
        logger.log("Compression skipped (use -c or --compress to enable)")
    # Copy the content to the clipboard
    copy_to_clipboard(clipboard_content)
    # Print the copied content character count
    logger.log("Prompt Char Count:", len(clipboard_content))
    logger.log("Tokens Count (gpt-4o):", count_tokens(clipboard_content))
    # Newline
    print("\n")


def count_tokens(
    text: str,
    model: str = "gpt-4o",  # Best default
    encoding_name: str | None = None,
) -> int:
    """
    Count the number of tokens in a string using tiktoken.
    Args:
        text: The input string to tokenize.
        model: OpenAI model name to determine the encoding
               (default: "gpt-4o" — recommended).
        encoding_name: Optional direct encoding name
                       (e.g., "o200k_base", "cl100k_base").
                       Takes precedence over model.
    Returns:
        Number of tokens.
    """
    if encoding_name:
        encoding = tiktoken.get_encoding(encoding_name)
    else:
        encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


if __name__ == "__main__":
    main()
