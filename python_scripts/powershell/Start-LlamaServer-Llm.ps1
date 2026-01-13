# Start-LlamaServer-Llm.ps1
# Optimized for: Ryzen 5 3600 + GTX 1660 6GB + 16GB RAM
# Cleaned version - fixed encoding & ampersand issues

Write-Host "`n  Llama.cpp Server Launcher  " -BackgroundColor DarkCyan -ForegroundColor Black
Write-Host "  Ryzen 5 3600 • GTX 1660 • 16 GB`n" -ForegroundColor DarkGray

function Show-Menu {
    param (
        [string]$Title,
        [string[]]$Options,
        [string]$Prompt = "Select option"
    )

    Write-Host "`n$Title" -ForegroundColor Cyan
    Write-Host ("-" * ($Title.Length + 4)) -ForegroundColor DarkGray   # ← changed to safe ASCII -

    for ($i = 0; $i -lt $Options.Count; $i++) {
        Write-Host ("{0,2}. " -f ($i+1)) -NoNewline -ForegroundColor White
        Write-Host $Options[$i]
    }

    Write-Host ""
    $choice = Read-Host "$Prompt (1-$($Options.Count)) or 0 to go back"
    return $choice
}

# =============================================================================
#                                 MODEL DATABASE
# =============================================================================

$categories = @(
    @{
        Name = "General Purpose / Chat"
        Items = @(
            @{ Num=1; Name="Llama-3.2-3B-Instruct (Small)";              File="bartowski_Llama-3.2-3B-Instruct-GGUF_Llama-3.2-3B-Instruct-Q4_K_M.gguf";          Ctx=8192;  Gpu=999; Jinja=$true;  Desc="Fast & capable - most popular 3B" }
            @{ Num=2; Name="Mistral-Nemo-Instruct-2407 (Medium)";       File="bartowski_Mistral-Nemo-Instruct-2407-GGUF_Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"; Ctx=8192;  Gpu=35;  Jinja=$true;  Desc="Very good quality - 12B feeling" }
            @{ Num=3; Name="Hermes-3-Llama-3.1-8B (Large)";             File="bartowski_Hermes-3-Llama-3.1-8B-GGUF_Hermes-3-Llama-3.1-8B-Q4_K_M.gguf";          Ctx=8192;  Gpu=35;  Jinja=$true;  Desc="Strong reasoning & chat - good balance" }
        )
    },
    @{
        Name = "Japanese <-> English Translators"
        Items = @(
            @{ Num=1; Name="LFM2-350M-ENJP-MT (Tiny)";                  File="translators\LFM2-350M-ENJP-MT.Q4_K_M.gguf";                                   Ctx=2048;  Gpu=999; Jinja=$false; Desc="Very fast - acceptable quality" }
            @{ Num=2; Name="gemma-2-2b-jpn-it-translate (Small)";       File="translators\gemma-2-2b-jpn-it-translate-Q4_K_M.gguf";                        Ctx=2048;  Gpu=999; Jinja=$false; Desc="Fast - good for short/medium text"; Extra="--override-kv tokenizer.ggml.add_bos_token=bool:false" }
            @{ Num=3; Name="shisa-v2.1-llama3.2-3b Q4 (Small)";         File="translators\shisa-v2.1-llama3.2-3b.Q4_K_M.gguf";                             Ctx=8192;  Gpu=999; Jinja=$true;  Desc="Modern 3B - very good speed/quality" }
            @{ Num=4; Name="shisa-v2.1-llama3.2-3b IQ4_XS (Small)";     File="translators\shisa-v2.1-llama3.2-3b.IQ4_XS.gguf";                             Ctx=8192;  Gpu=999; Jinja=$true;  Desc="Slightly better than Q4 - still fast" }
            @{ Num=5; Name="Llama-3-ELYZA-JP-8B IQ2XXS (Large)";        File="translators\Llama-3-ELYZA-JP-8B.i1-IQ2_XXS.gguf";                             Ctx=4096;  Gpu=25;  Jinja=$true;  Desc="Heavy quantization - still usable" }
            @{ Num=6; Name="ALMA-7B-Ja-V2 (Large)";                     File="translators\ALMA-7B-Ja-V2.Q4_K_M.gguf";                                       Ctx=4096;  Gpu=30;  Jinja=$false; Desc="Strongest classic - bit slower" }
        )
    },
    @{
        Name = "Uncensored / Spicy Models"
        Items = @(
            @{ Num=1; Name="nano_imp_1b-q8_0 (Tiny)";                   File="nsfw\nano_imp_1b-q8_0.gguf";                       Ctx=4096; Gpu=999; Jinja=$false; Desc="Extremely fast - minimal quality" }
            @{ Num=2; Name="dolphin-2_6-phi-2 (Small)";                 File="nsfw\dolphin-2_6-phi-2.Q4_K_M.gguf";               Ctx=4096; Gpu=999; Jinja=$false; Desc="Small & surprisingly capable" }
            @{ Num=3; Name="Fiendish_LLAMA_3B (Small)";                 File="nsfw\Fiendish_LLAMA_3B.Q4_K_M.gguf";                Ctx=8192; Gpu=999; Jinja=$true;  Desc="Very direct - 3B" }
            @{ Num=4; Name="Llama-3.2-3B-Instruct uncensored (Small)";  File="nsfw\Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf"; Ctx=8192; Gpu=999; Jinja=$true;  Desc="Popular base - loosened" }
            @{ Num=5; Name="Impish_LLAMA_4B (Small)";                   File="nsfw\SicariusSicariiStuff_Impish_LLAMA_4B-Q4_K_M.gguf"; Ctx=8192; Gpu=999; Jinja=$true; Desc="4B spicy variant" }
            @{ Num=6; Name="WizardLM-7B uncensored (Large)";            File="nsfw\WizardLM-7B-uncensored.Q4_K_M.gguf";           Ctx=8192; Gpu=40;  Jinja=$true;  Desc="Classic uncensored" }
        )
    }
)

# =============================================================================
#                                 MAIN LOOP
# =============================================================================

while ($true) {
    $catOptions = $categories | ForEach-Object { $_.Name }
    $catOptions += "Exit"

    $catChoice = Show-Menu -Title "MAIN CATEGORIES" -Options $catOptions -Prompt "Choose category"

    if ($catChoice -eq "0" -or $catChoice -eq "Exit") {
        Write-Host "`nGoodbye!`n" -ForegroundColor Cyan
        exit 0
    }

    if (![int]::TryParse($catChoice, [ref]$null) -or $catChoice -lt 1 -or $catChoice -gt $categories.Count) {
        Write-Host "Invalid category selection." -ForegroundColor Red
        continue
    }

    $selectedCategory = $categories[$catChoice - 1]

    $modelOptions = $selectedCategory.Items | ForEach-Object {
        "{0,-38} {1}" -f $_.Name, $_.Desc
    }

    $modelChoice = Show-Menu -Title $selectedCategory.Name -Options $modelOptions -Prompt "Select model"

    if ($modelChoice -eq "0") { continue }  # back to categories

    $model = $selectedCategory.Items | Where-Object { $_.Num -eq [int]$modelChoice } | Select-Object -First 1

    if (-not $model) {
        Write-Host "Invalid model selection." -ForegroundColor Red
        continue
    }

    $modelPath = Join-Path "C:\Users\druiv\.cache\llama.cpp" $model.File

    if (-Not (Test-Path $modelPath)) {
        Write-Host "`nModel not found!" -ForegroundColor Red
        Write-Host "  $modelPath`n" -ForegroundColor DarkGray
        pause
        continue
    }

    $cmd = "llama-server.exe " +
           "-m `"$modelPath`" " +
           "--host 0.0.0.0 --port 8080 " +
           "--ctx-size $($model.Ctx) " +
           "--flash-attn on " +
           "--cache-type-k q8_0 --cache-type-v q8_0 "

    $gpuLayers = if ($model.Gpu -lt 999) { $model.Gpu } else { 999 }
    $cmd += "--n-gpu-layers $gpuLayers "

    if ($model.Jinja)         { $cmd += "--jinja " }
    if ($model.Extra)         { $cmd += "$($model.Extra) " }

    Write-Host "`n" -NoNewline
    Write-Host "Starting model: " -NoNewline -ForegroundColor Green
    Write-Host $model.Name -ForegroundColor White

    Write-Host "  File      : " -NoNewline -ForegroundColor DarkGray
    Write-Host $model.File -ForegroundColor DarkCyan

    Write-Host "  Context   : " -NoNewline -ForegroundColor DarkGray
    Write-Host "$($model.Ctx) tokens" -ForegroundColor Yellow

    Write-Host "  GPU layers: " -NoNewline -ForegroundColor DarkGray
    Write-Host $gpuLayers -ForegroundColor Magenta

    if ($model.Extra) {
        Write-Host "  Extra     : " -NoNewline -ForegroundColor DarkGray
        Write-Host $model.Extra -ForegroundColor Magenta
    }

    Write-Host "`n$cmd`n" -ForegroundColor DarkGray

    $confirm = Read-Host "Launch now? [Y/n]"
    if ($confirm -notin '', 'y', 'Y') { continue }

    Invoke-Expression $cmd

    Write-Host "`nServer session ended. Press Enter to return to menu..."
    Read-Host
}
