@echo off
setlocal

set OUTPUT_DIR=C:\Users\druiv\.cache\hf_ctranslate2_models\anime-whisper-ct2

ct2-transformers-converter ^
  --model litagin/anime-whisper ^
  --output_dir "%OUTPUT_DIR%" ^
  --copy_files preprocessor_config.json ^
  --quantization float16 ^
  --force

echo.
echo Conversion complete.
echo.
echo Model ready at: %OUTPUT_DIR%
echo You can now load it with:
echo WhisperModel(r"%OUTPUT_DIR%", device="cuda", compute_type="float16")
