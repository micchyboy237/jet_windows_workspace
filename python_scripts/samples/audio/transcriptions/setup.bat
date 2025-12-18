@REM === FAST REBUILD USING CACHED WHEELS (recommended now) ===

@REM 1. Keep cache — just uninstall the broken packages
pip uninstall -y torch torchaudio torchvision pyannote-audio pyannote.audio numpy whisperx pytorch-lightning lightning speechbrain

@REM 2. Reinstall from cache + PyTorch index (will instantly reuse your 2.9.1+cu128 wheels)
pip install --force-reinstall torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128

@REM 3. Reinstall compatible numpy from cache
pip install --force-reinstall "numpy>=2.0.2,<2.1.0"

@REM 4. Reinstall the correct pyannote.audio 3.4.0 (cached)
pip install --force-reinstall "pyannote-audio==3.4.0"

@REM 5. Reinstall latest whisperx from git (small download, but very fast)
pip install --force-reinstall git+https://github.com/m-bain/whisperx.git

@REM === QUICK VERIFICATION ===
Write-Host "`nFAST REBUILD COMPLETE – CHECK:" -ForegroundColor Green
python -c "import torch, whisperx; print('CUDA:', torch.cuda.is_available()); print('Torch:', torch.__version__); print('Diarization import OK:', bool(__import__('whisperx.diarize').DiarizationPipeline))"
pip show torch torchaudio pyannote-audio whisperx numpy | Select-String "Name|Version"
