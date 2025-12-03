@echo off
setlocal

@REM ct2-transformers-converter ^
@REM   --model openai/whisper-medium ^
@REM   --output_dir "C:\Users\druiv\.cache\hf_ctranslate2_models\whisper-medium-ct2-int8_float16" ^
@REM   --quantization int8_float16 ^
@REM   --copy_files tokenizer.json preprocessor_config.json ^
@REM   --force

ct2-transformers-converter --model openai/whisper-medium --output_dir C:\Users\druiv\.cache\hf_ctranslate2_models\whisper-medium-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization int8_float16

echo.
echo Conversion complete: whisper-medium int8_float16