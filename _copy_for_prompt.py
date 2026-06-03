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

    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\core\inference.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\pyannote-audio\src\pyannote\audio\pipelines\clustering.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\cluster_speakers.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\compare_speakers.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\segment_speaker_labeler.py",
    # r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\segment_speaker_labeler_example.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speaker_html_visualizer.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speaker_visualizer.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\processing.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speech_waves.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\segment_speaker_labeler.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_segment_speaker_labeler.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\sherpa-onnx\python-api-examples\audio-tagging-from-a-file.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\sherpa-onnx\python-api-examples\audio-tagging-from-a-file-ced.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\audio_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\audio_tagger_base.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\audio_tagger_core.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\audio_tagger_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\audio_tagger_zipformer.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\audio_tagger_ced.py",
    # r"C:\Users\druiv\.cache\pretrained_models\sherpa-onnx\sherpa-onnx-zipformer-audio-tagging-2024-04-09\class_labels_indices.csv",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\vad_firered.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\_demo_speech_checker.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\speech_checker.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\audio_tagger_zipformer.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\generated\speech_checker\chunk_results.json",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\generated\speech_checker\metadata.json",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\generated\speech_checker\results.json",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\generated\speech_checker\speech_check_results.json",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\features\generated\speech_checker\speech_insights.json",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_tagger_chunk_plots.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_audio_tagger.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_tagger.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\processing.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\state.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\__init__.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\models\schemas.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\routes\websocket.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\save_utils.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\live_subtitles_server_utils.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\main.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\routes\speakers.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\routes\tagger.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\plots.html",
    r"",
]

structure_include = [
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en",
    # r"C:\Users\druiv\.cache\huggingface\hub\models--pyannote--speech-separation-ami-1.0\snapshots\9486b106945ae0cc0784041a08bfcdba5edadfb9",
]
structure_exclude = []

include_content = []
exclude_content = []

# Args defaults
SHORTEN_FUNCTS = False 
INCLUDE_FILE_STRUCTURE = False

DEFAULT_QUERY_MESSAGE = r"""
[06/03/26 15:38:21] INFO     192.168.68.100:53852 - "GET /tag/plots HTTP/1.1" 500                                         h11_impl.py:473
                    ERROR    Exception in ASGI application                                                                h11_impl.py:408

                             ╭─────────────────────────── Traceback (most recent call last) ────────────────────────────╮
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\uvicorn\protocols\http\h11 │
                             │ _impl.py:403 in run_asgi                                                                 │
                             │                                                                                          │
                             │   400 │   # ASGI exception wrapper                                                       │
                             │   401 │   async def run_asgi(self, app: ASGI3Application) -> None:                       │
                             │   402 │   │   try:                                                                       │
                             │ ❱ 403 │   │   │   result = await app(  # type: ignore[func-returns-value]                │
                             │   404 │   │   │   │   self.scope, self.receive, self.send                                │
                             │   405 │   │   │   )                                                                      │
                             │   406 │   │   except BaseException as exc:                                               │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\uvicorn\middleware\proxy_h │
                             │ eaders.py:60 in __call__                                                                 │
                             │                                                                                          │
                             │    57 │   │   │   │   │   port = 0                                                       │
                             │    58 │   │   │   │   │   scope["client"] = (host, port)                                 │
                             │    59 │   │                                                                              │
                             │ ❱  60 │   │   return await self.app(scope, receive, send)                                │
                             │    61                                                                                    │
                             │    62                                                                                    │
                             │    63 def _parse_raw_hosts(value: str) -> list[str]:                                     │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\fastapi\applications.py:11 │
                             │ 34 in __call__                                                                           │
                             │                                                                                          │
                             │   1131 │   async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None: │
                             │   1132 │   │   if self.root_path:                                                        │
                             │   1133 │   │   │   scope["root_path"] = self.root_path                                   │
                             │ ❱ 1134 │   │   await super().__call__(scope, receive, send)                              │
                             │   1135 │                                                                                 │
                             │   1136 │   def add_api_route(                                                            │
                             │   1137 │   │   self,                                                                     │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\starlette\applications.py: │
                             │ 113 in __call__                                                                          │
                             │                                                                                          │
                             │   110 │   │   scope["app"] = self                                                        │
                             │   111 │   │   if self.middleware_stack is None:                                          │
                             │   112 │   │   │   self.middleware_stack = self.build_middleware_stack()                  │
                             │ ❱ 113 │   │   await self.middleware_stack(scope, receive, send)                          │
                             │   114 │                                                                                  │
                             │   115 │   def on_event(self, event_type: str) -> Callable:  # type: ignore[type-arg]     │
                             │   116 │   │   return self.router.on_event(event_type)  # pragma: no cover                │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\starlette\middleware\error │
                             │ s.py:186 in __call__                                                                     │
                             │                                                                                          │
                             │   183 │   │   │   # We always continue to raise the exception.                           │
                             │   184 │   │   │   # This allows servers to log the error, or allows test clients         │
                             │   185 │   │   │   # to optionally raise the error within the test case.                  │
                             │ ❱ 186 │   │   │   raise exc                                                              │
                             │   187 │                                                                                  │
                             │   188 │   def format_line(self, index: int, line: str, frame_lineno: int, frame_index: i │
                             │       str:                                                                               │
                             │   189 │   │   values = {                                                                 │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\starlette\middleware\error │
                             │ s.py:164 in __call__                                                                     │
                             │                                                                                          │
                             │   161 │   │   │   await send(message)                                                    │
                             │   162 │   │                                                                              │
                             │   163 │   │   try:                                                                       │
                             │ ❱ 164 │   │   │   await self.app(scope, receive, _send)                                  │
                             │   165 │   │   except Exception as exc:                                                   │
                             │   166 │   │   │   request = Request(scope)                                               │
                             │   167 │   │   │   if self.debug:                                                         │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\starlette\middleware\excep │
                             │ tions.py:63 in __call__                                                                  │
                             │                                                                                          │
                             │   60 │   │   else:                                                                       │
                             │   61 │   │   │   conn = WebSocket(scope, receive, send)                                  │
                             │   62 │   │                                                                               │
                             │ ❱ 63 │   │   await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)    │
                             │   64 │                                                                                   │
                             │   65 │   async def http_exception(self, request: Request, exc: Exception) -> Response:   │
                             │   66 │   │   assert isinstance(exc, HTTPException)                                       │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\starlette\_exception_handl │
                             │ er.py:53 in wrapped_app                                                                  │
                             │                                                                                          │
                             │   50 │   │   │   │   handler = _lookup_exception_handler(exception_handlers, exc)        │
                             │   51 │   │   │                                                                           │
                             │   52 │   │   │   if handler is None:                                                     │
                             │ ❱ 53 │   │   │   │   raise exc                                                           │
                             │   54 │   │   │                                                                           │
                             │   55 │   │   │   if response_started:                                                    │
                             │   56 │   │   │   │   raise RuntimeError("Caught handled exception, but response already  │
                             │      started.") from exc                                                                 │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\starlette\_exception_handl │
                             │ er.py:42 in wrapped_app                                                                  │
                             │                                                                                          │
                             │   39 │   │   │   await send(message)                                                     │
                             │   40 │   │                                                                               │
                             │   41 │   │   try:                                                                        │
                             │ ❱ 42 │   │   │   await app(scope, receive, sender)                                       │
                             │   43 │   │   except Exception as exc:                                                    │
                             │   44 │   │   │   handler = None                                                          │
                             │   45                                                                                     │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\fastapi\middleware\asyncex │
                             │ itstack.py:18 in __call__                                                                │
                             │                                                                                          │
                             │   15 │   async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:   │
                             │   16 │   │   async with AsyncExitStack() as stack:                                       │
                             │   17 │   │   │   scope[self.context_name] = stack                                        │
                             │ ❱ 18 │   │   │   await self.app(scope, receive, send)                                    │
                             │   19                                                                                     │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\starlette\routing.py:716   │
                             │ in __call__                                                                              │
                             │                                                                                          │
                             │   713 │   │   \"\"\"                                                                        │
                             │   714 │   │   The main entry point to the Router class.                                  │
                             │   715 │   │   \"\"\"                                                                        │
                             │ ❱ 716 │   │   await self.middleware_stack(scope, receive, send)                          │
                             │   717 │                                                                                  │
                             │   718 │   async def app(self, scope: Scope, receive: Receive, send: Send) -> None:       │
                             │   719 │   │   assert scope["type"] in ("http", "websocket", "lifespan")                  │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\starlette\routing.py:736   │
                             │ in app                                                                                   │
                             │                                                                                          │
                             │   733 │   │   │   match, child_scope = route.matches(scope)                              │
                             │   734 │   │   │   if match == Match.FULL:                                                │
                             │   735 │   │   │   │   scope.update(child_scope)                                          │
                             │ ❱ 736 │   │   │   │   await route.handle(scope, receive, send)                           │
                             │   737 │   │   │   │   return                                                             │
                             │   738 │   │   │   elif match == Match.PARTIAL and partial is None:                       │
                             │   739 │   │   │   │   partial = route                                                    │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\starlette\routing.py:290   │
                             │ in handle                                                                                │
                             │                                                                                          │
                             │   287 │   │   │   │   response = PlainTextResponse("Method Not Allowed", status_code=405 │
                             │       headers=headers)                                                                   │
                             │   288 │   │   │   await response(scope, receive, send)                                   │
                             │   289 │   │   else:                                                                      │
                             │ ❱ 290 │   │   │   await self.app(scope, receive, send)                                   │
                             │   291 │                                                                                  │
                             │   292 │   def __eq__(self, other: Any) -> bool:                                          │
                             │   293 │   │   return (                                                                   │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\fastapi\routing.py:124 in  │
                             │ app                                                                                      │
                             │                                                                                          │
                             │    121 │   │   │   │   )                                                                 │
                             │    122 │   │                                                                             │
                             │    123 │   │   # Same as in Starlette                                                    │
                             │ ❱  124 │   │   await wrap_app_handling_exceptions(app, request)(scope, receive, send)    │
                             │    125 │                                                                                 │
                             │    126 │   return app                                                                    │
                             │    127                                                                                   │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\starlette\_exception_handl │
                             │ er.py:53 in wrapped_app                                                                  │
                             │                                                                                          │
                             │   50 │   │   │   │   handler = _lookup_exception_handler(exception_handlers, exc)        │
                             │   51 │   │   │                                                                           │
                             │   52 │   │   │   if handler is None:                                                     │
                             │ ❱ 53 │   │   │   │   raise exc                                                           │
                             │   54 │   │   │                                                                           │
                             │   55 │   │   │   if response_started:                                                    │
                             │   56 │   │   │   │   raise RuntimeError("Caught handled exception, but response already  │
                             │      started.") from exc                                                                 │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\starlette\_exception_handl │
                             │ er.py:42 in wrapped_app                                                                  │
                             │                                                                                          │
                             │   39 │   │   │   await send(message)                                                     │
                             │   40 │   │                                                                               │
                             │   41 │   │   try:                                                                        │
                             │ ❱ 42 │   │   │   await app(scope, receive, sender)                                       │
                             │   43 │   │   except Exception as exc:                                                    │
                             │   44 │   │   │   handler = None                                                          │
                             │   45                                                                                     │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\fastapi\routing.py:110 in  │
                             │ app                                                                                      │
                             │                                                                                          │
                             │    107 │   │   │   async with AsyncExitStack() as stack:                                 │
                             │    108 │   │   │   │   scope["fastapi_inner_astack"] = stack                             │
                             │    109 │   │   │   │   # Same as in Starlette                                            │
                             │ ❱  110 │   │   │   │   response = await f(request)                                       │
                             │    111 │   │   │   │   await response(scope, receive, send)                              │
                             │    112 │   │   │   │   # Continues customization                                         │
                             │    113 │   │   │   │   response_awaited = True                                           │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\fastapi\routing.py:390 in  │
                             │ app                                                                                      │
                             │                                                                                          │
                             │    387 │   │   )                                                                         │
                             │    388 │   │   errors = solved_result.errors                                             │
                             │    389 │   │   if not errors:                                                            │
                             │ ❱  390 │   │   │   raw_response = await run_endpoint_function(                           │
                             │    391 │   │   │   │   dependant=dependant,                                              │
                             │    392 │   │   │   │   values=solved_result.values,                                      │
                             │    393 │   │   │   │   is_coroutine=is_coroutine,                                        │
                             │                                                                                          │
                             │ C:\Users\druiv\.cache\venv\servers\jet_venv\Lib\site-packages\fastapi\routing.py:289 in  │
                             │ run_endpoint_function                                                                    │
                             │                                                                                          │
                             │    286 │   assert dependant.call is not None, "dependant.call must be a function"        │
                             │    287 │                                                                                 │
                             │    288 │   if is_coroutine:                                                              │
                             │ ❱  289 │   │   return await dependant.call(**values)                                     │
                             │    290 │   else:                                                                         │
                             │    291 │   │   return await run_in_threadpool(dependant.call, **values)                  │
                             │    292                                                                                   │
                             │                                                                                          │
                             │ C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subti │
                             │ tles_server2_with_en\routes\tagger.py:959 in get_tagger_plots                            │
                             │                                                                                          │
                             │   956 │   - Speech probability trend line                                                │
                             │   957 │   \"\"\"                                                                            │
                             │   958 │   template_path = TEMPLATES_DIR / "plots.html"                                   │
                             │ ❱ 959 │   template = Template(template_path.read_text())                                 │
                             │   960 │   return HTMLResponse(content=template.render())                                 │
                             │   961                                                                                    │
                             │                                                                                          │
                             │ C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\pathlib.py:1028 in read_text        │
                             │                                                                                          │
                             │   1025 │   │   \"\"\"                                                                       │
                             │   1026 │   │   encoding = io.text_encoding(encoding)                                     │
                             │   1027 │   │   with self.open(mode='r', encoding=encoding, errors=errors) as f:          │
                             │ ❱ 1028 │   │   │   return f.read()                                                       │
                             │   1029 │                                                                                 │
                             │   1030 │   def write_bytes(self, data):                                                  │
                             │   1031 │   │   \"\"\"                                                                       │
                             │                                                                                          │
                             │ C:\Users\druiv\.pyenv\pyenv-win\versions\3.12.10\Lib\encodings\cp1252.py:23 in decode    │
                             │                                                                                          │
                             │    20                                                                                    │
                             │    21 class IncrementalDecoder(codecs.IncrementalDecoder):                               │
                             │    22 │   def decode(self, input, final=False):                                          │
                             │ ❱  23 │   │   return codecs.charmap_decode(input,self.errors,decoding_table)[0]          │
                             │    24                                                                                    │
                             │    25 class StreamWriter(Codec,codecs.StreamWriter):                                     │
                             │    26 │   pass                                                                           │
                             ╰──────────────────────────────────────────────────────────────────────────────────────────╯
                             UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 13494: character maps
                             to <undefined>
                    INFO     192.168.68.100:53853 - "GET /favicon.ico HTTP/1.1" 404    
""".strip()

DEFAULT_INSTRUCTIONS_MESSAGE = """
Provide step-by-step analysis and explain the flow first.
Use visuals, diagrams, or tables when helpful.

Show full code for new files, then show full function code for new or updated functions.
Keep explanations simple and clear.

Write smart, flexible, reusable, maintainable, optimal and robust code.
If issues are encountered, immediately include debug logs to trace all relevant steps.
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
        clipboard_content_parts.append(f"System\n{system_message}\n")
    # Query should come before instructions
    clipboard_content_parts.append(f"Query\n{query_message}\n")
    if instructions_message:
        clipboard_content_parts.append(f"Instructions\n{instructions_message}\n")
    if INCLUDE_FILE_STRUCTURE:
        clipboard_content_parts.append(f"Files Structure\n{files_structure}\n")

    if clipboard_content:
        clipboard_content_parts.append(
            f"Existing Files Contents\n{clipboard_content}\n"
        )

    clipboard_content = "\n\n".join(clipboard_content_parts)

    # Copy the content to the clipboard
    copy_to_clipboard(clipboard_content)

    # Print the copied content character count
    logger.log("Prompt Char Count:", len(clipboard_content))

    # Newline
    print("\n")


if __name__ == "__main__":
    main()
