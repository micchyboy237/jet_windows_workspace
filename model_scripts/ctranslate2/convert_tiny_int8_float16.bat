@echo off
setlocal

@REM ct2-transformers-converter ^
@REM   --model openai/whisper-tiny ^
@REM   --output_dir "C:\Users\druiv\.cache\hf_ctranslate2_models\whisper-tiny-ct2-int8_float16" ^
@REM   --quantization int8_float16 ^
@REM   --copy_files tokenizer.json preprocessor_config.json ^
@REM   --force

ct2-transformers-converter --model openai/whisper-tiny --output_dir C:\Users\druiv\.cache\hf_ctranslate2_models\whisper-tiny-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization int8_float16

echo.
echo Conversion complete: whisper-tiny int8_float16