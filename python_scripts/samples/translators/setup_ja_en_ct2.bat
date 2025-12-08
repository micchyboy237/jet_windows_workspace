@echo off
curl -L -o opus-2019-12-18.zip https://object.pouta.csc.fi/OPUS-MT-models/ja-en/opus-2019-12-18.zip

:: Extract the zip (tar works with zip files on recent Windows)
tar -xf opus-2019-12-18.zip

:: Convert
ct2-opus-mt-converter --model_dir . --output_dir ja_en_ct2