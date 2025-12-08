# Download the model zip
Invoke-WebRequest -Uri "https://object.pouta.csc.fi/OPUS-MT-models/ja-en/opus-2019-12-18.zip" -OutFile "opus-2019-12-18.zip"

# Extract (built-in since Windows 10 / Server 2016)
Expand-Archive -Path "opus-2019-12-18.zip" -DestinationPath "." -Force

# Convert to CTranslate2 format
ct2-opus-mt-converter --model_dir . --output_dir ja_en_ct2
