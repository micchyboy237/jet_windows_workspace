# live_subtitles/tests/__init__.py

import os
from pathlib import Path

# Change working directory to the project root (parent of 'servers') when tests are imported.
# This enables running `pytest` from the root while keeping relative imports simple.
project_root = Path(__file__).parent.parent.parent.resolve()
original_cwd = os.getcwd()
if original_cwd != str(project_root):
    os.chdir(project_root)
    print(f"[tests/__init__.py] Changed cwd from {original_cwd} to {project_root}")
