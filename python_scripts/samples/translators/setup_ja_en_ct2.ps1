# Download the model zip
Invoke-WebRequest -Uri "https://object.pouta.csc.fi/OPUS-MT-models/ja-en/opus-2019-12-18.zip" -OutFile "opus-2019-12-18.zip"

# Extract
Expand-Archive -Path "opus-2019-12-18.zip" -DestinationPath "." -Force

# Convert to CTranslate2 format
ct2-opus-mt-converter --model_dir . --output_dir ja_en_ct2

# ←←← This line deletes the zip when you're done
Remove-Item -Path "opus-2019-12-18.zip" -Force
