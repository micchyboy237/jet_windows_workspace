@echo off
setlocal

@REM ct2-transformers-converter ^
@REM   --model openai/whisper-large-v2 ^
@REM   --output_dir "C:\Users\druiv\.cache\hf_ctranslate2_models\whisper-large-v2-ct2-int8_float16" ^
@REM   --quantization int8_float16 ^
@REM   --copy_files tokenizer.json preprocessor_config.json ^
@REM   --force

ct2-transformers-converter --model openai/whisper-large-v2 --output_dir C:\Users\druiv\.cache\hf_ctranslate2_models\whisper-large-v2-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization int8

echo.
echo Conversion complete: whisper-large-v2 int8_float16