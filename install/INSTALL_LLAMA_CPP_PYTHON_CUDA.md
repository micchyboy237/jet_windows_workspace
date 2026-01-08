# Activate venv first!
$env:FORCE_CMAKE     = "1"
$env:CMAKE_ARGS      = "-DGGML_CUDA=ON"

pip install llama-cpp-python --no-cache-dir --verbose --force-reinstall --upgrade
