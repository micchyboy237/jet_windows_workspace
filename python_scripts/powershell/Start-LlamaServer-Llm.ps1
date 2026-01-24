# Jet_Windows_Workspace/python_scripts/powershell/Start-LlamaServer-Llm.ps1

[CmdletBinding()]
param(
    [Alias("p")]
    [Parameter(HelpMessage = "Port for llama-server to listen on (default: 8080)")]
    [ValidateRange(1, 65535)]
    [int]$Port = 8080
)

Write-Host "`n  Llama.cpp Server Launcher  " -BackgroundColor DarkCyan -ForegroundColor Black
Write-Host "  Ryzen 5 3600 • GTX 1660 • 16 GB`n" -ForegroundColor DarkGray

function Show-Menu {
    param (
        [string]$Title,
        [string[]]$Options,
        [string]$Prompt = "Select option"
    )
    Write-Host "`n$Title" -ForegroundColor Cyan
    Write-Host ("-" * ($Title.Length + 4)) -ForegroundColor DarkGray   
    for ($i = 0; $i -lt $Options.Count; $i++) {
        Write-Host ("{0,2}. " -f ($i+1)) -NoNewline -ForegroundColor White
        Write-Host $Options[$i]
    }
    Write-Host ""
    $choice = Read-Host "$Prompt (1-$($Options.Count)) or 0 to go back"
    return $choice
}

$categories = @(
    @{
        Name = "General Purpose / Chat"
        Items = @(
            @{ Num=1;  Size="Small";  Name="Llama-3.2-3B-Instruct";              File="bartowski_Llama-3.2-3B-Instruct-GGUF_Llama-3.2-3B-Instruct-Q4_K_M.gguf";                   Ctx=8192; Gpu=999; Jinja=$true;  Desc="Fast & very popular 3B model" }
            @{ Num=2;  Size="Small";  Name="Gemma-3-4B-it";                      File="ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf";                                     Ctx=8192; Gpu=999; Jinja=$true;  Desc="Fresh Gemma 2025 release" }
            @{ Num=3;  Size="Small";  Name="Qwen3-4B-Instruct-2507";             File="bartowski_Qwen_Qwen3-4B-Instruct-2507-GGUF_Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf";         Ctx=8192; Gpu=999; Jinja=$true;  Desc="Very recent Qwen3 series" }
            @{ Num=4;  Size="Medium"; Name="Mistral-Nemo-Instruct-2407";         File="bartowski_Mistral-Nemo-Instruct-2407-GGUF_Mistral-Nemo-Instruct-2407-Q4_K_M.gguf";         Ctx=8192; Gpu=35;  Jinja=$true;  Desc="Excellent quality - feels like 12B" }
            @{ Num=5;  Size="Medium"; Name="DeepSeek-R1-Distill-Qwen-7B";        File="bartowski_DeepSeek-R1-Distill-Qwen-7B-GGUF_DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf";         Ctx=8192; Gpu=35;  Jinja=$true;  Desc="Strong reasoning focused distill" }
            @{ Num=6;  Size="Medium"; Name="Meta-Llama-3.1-8B-Instruct";         File="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";                                                       Ctx=8192; Gpu=35;  Jinja=$true;  Desc="Still very capable classic 8B" }
            @{ Num=7;  Size="Medium"; Name="Hermes-3-Llama-3.1-8B";              File="bartowski_Hermes-3-Llama-3.1-8B-GGUF_Hermes-3-Llama-3.1-8B-Q4_K_M.gguf";                    Ctx=8192; Gpu=35;  Jinja=$true;  Desc="Excellent reasoning & chat" }
        )
    },
    @{
        Name = "Coding & Technical"
        Items = @(
            @{ Num=1; Size="Medium"; Name="DeepSeek-Coder-V2-Lite-Instruct";     File="bartowski_DeepSeek-Coder-V2-Lite-Instruct-GGUF_DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf";  Ctx=32768; Gpu=30; Jinja=$true; Desc="One of the best small coders 2025" }
        )
    },
    @{
        Name = "Vision / Multimodal"
        Items = @(
            @{ Num=1; Size="Medium"; Name="Qwen2.5-VL-7B-Instruct";              File="ggml-org_Qwen2.5-VL-7B-Instruct-GGUF_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf";                  Ctx=8192; Gpu=28; Jinja=$true; 
               Desc="Solid vision-language model"; 
               Extra="--mmproj `"ggml-org_Qwen2.5-VL-7B-Instruct-GGUF_mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf`"" }
        )
    },
    @{
        Name = "Embeddings / RAG"
        Items = @(
            @{ Num=1; Size="Small";  Name="nomic-embed-text-v1.5";               File="embed_models\nomic-embed-text-v1.5.Q4_K_M.gguf";               Ctx=8192; Gpu=999; Jinja=$false; Desc="Very strong & popular embedding" }
            @{ Num=2; Size="Small";  Name="nomic-embed-text-v2-moe";             File="embed_models\nomic-embed-text-v2-moe.Q4_K_M.gguf";             Ctx=8192; Gpu=999; Jinja=$false; Desc="Latest & most capable Nomic" }
            @{ Num=3; Size="Small";  Name="all-MiniLM-L12-v2 (q4)";              File="embed_models\all-MiniLM-L12-v2-q4_0.gguf";                      Ctx=8192; Gpu=999; Jinja=$false; Desc="Classic fast & compact" }
            @{ Num=4; Size="Tiny";   Name="embedding-gemma-300M";                File="embed_models\embeddinggemma-300M-Q8_0.gguf";                    Ctx=8192; Gpu=999; Jinja=$false; Desc="Gemma-based embedding model" }
        )
    },
    @{
        Name = "Japanese <-> English Translators"
        Items = @(
            @{ Num=1; Size="Tiny";   Name="LFM2-350M-ENJP-MT";                   File="translators\LFM2-350M-ENJP-MT.Q4_K_M.gguf";                                                  Ctx=2048; Gpu=999; Jinja=$false; Desc="Very fast - acceptable quality" }
            @{ Num=2; Size="Small";  Name="gemma-2-2b-jpn-it-translate";         File="translators\gemma-2-2b-jpn-it-translate-Q4_K_M.gguf";                                         Ctx=2048; Gpu=999; Jinja=$false; Desc="Good for short/medium text"; Extra="--override-kv tokenizer.ggml.add_bos_token=bool:false" }
            @{ Num=3; Size="Small";  Name="shisa-v2.1-llama3.2-3b Q4";           File="translators\shisa-v2.1-llama3.2-3b.Q4_K_M.gguf";                                              Ctx=8192; Gpu=999; Jinja=$true;  Desc="Modern 3B - excellent speed/quality" }
            @{ Num=4; Size="Small";  Name="shisa-v2.1-llama3.2-3b IQ4_XS";       File="translators\shisa-v2.1-llama3.2-3b.IQ4_XS.gguf";                                              Ctx=8192; Gpu=999; Jinja=$true;  Desc="Slightly better quality than Q4" }
            @{ Num=5; Size="Medium"; Name="Llama-3-ELYZA-JP-8B IQ2XXS";          File="translators\Llama-3-ELYZA-JP-8B.i1-IQ2_XXS.gguf";                                             Ctx=4096; Gpu=25;  Jinja=$true;  Desc="Very heavy quantization" }
            @{ Num=6; Size="Medium"; Name="ALMA-7B-Ja-V2";                       File="translators\ALMA-7B-Ja-V2.Q4_K_M.gguf";                                                       Ctx=4096; Gpu=30;  Jinja=$false; Desc="Strong classic Japanese model" }
        )
    },
    @{
        Name = "Uncensored / Spicy Models"
        Items = @(
            @{ Num=1; Size="Tiny";   Name="nano_imp_1b-q8_0";                    File="nsfw\nano_imp_1b-q8_0.gguf";                                                                  Ctx=4096; Gpu=999; Jinja=$false; Desc="Extremely fast - minimal quality" }
            @{ Num=2; Size="Small";  Name="dolphin-2_6-phi-2";                   File="nsfw\dolphin-2_6-phi-2.Q4_K_M.gguf";                                                          Ctx=4096; Gpu=999; Jinja=$false; Desc="Small but surprisingly capable" }
            @{ Num=3; Size="Small";  Name="Fiendish_LLAMA_3B";                   File="nsfw\Fiendish_LLAMA_3B.Q4_K_M.gguf";                                                          Ctx=8192; Gpu=999; Jinja=$true;  Desc="Very direct 3B model" }
            @{ Num=4; Size="Small";  Name="Llama-3.2-3B uncensored";             File="nsfw\Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf";                                           Ctx=8192; Gpu=999; Jinja=$true;  Desc="Popular base loosened" }
            @{ Num=5; Size="Small";  Name="Impish_LLAMA_4B";                     File="nsfw\SicariusSicariiStuff_Impish_LLAMA_4B-Q4_K_M.gguf";                                      Ctx=8192; Gpu=999; Jinja=$true;  Desc="4B spicy variant" }
            @{ Num=6; Size="Medium"; Name="WizardLM-7B uncensored";              File="nsfw\WizardLM-7B-uncensored.Q4_K_M.gguf";                                                     Ctx=8192; Gpu=40;  Jinja=$true;  Desc="Classic uncensored 7B" }
        )
    }
)

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
        "{0,6} {1,-36} {2}" -f $_.Size, $_.Name, $_.Desc
    }

    $modelChoice = Show-Menu -Title $selectedCategory.Name -Options $modelOptions -Prompt "Select model"
    if ($modelChoice -eq "0") { continue }

    $model = $selectedCategory.Items | Where-Object { $_.Num -eq [int]$modelChoice } | Select-Object -First 1
    if (-not $model) {
        Write-Host "Invalid model selection." -ForegroundColor Red
        continue
    }

    $modelPath = Join-Path "C:\Users\druiv\.cache\llama.cpp" $model.File
    if (-Not (Test-Path $modelPath)) {
        Write-Host "`nModel not found!" -ForegroundColor Red
        Write-Host "  Expected path: $modelPath`n" -ForegroundColor DarkGray
        pause
        continue
    }

    $cmd = "llama-server.exe " +
           "-m `"$modelPath`" " +
           "--host 0.0.0.0 --port $Port " +
           "--ctx-size $($model.Ctx) " +
           "--flash-attn on " +
           "--cache-type-k q8_0 --cache-type-v q8_0 "

    $gpuLayers = if ($model.Gpu -lt 999) { $model.Gpu } else { 999 }
    $cmd += "--n-gpu-layers $gpuLayers "

    if ($model.Jinja)         { $cmd += "--jinja " }
    if ($model.Extra)         { $cmd += "$($model.Extra) " }

    Write-Host "`nLaunching:" -ForegroundColor Green
    Write-Host "  Size     : " -NoNewline -ForegroundColor DarkGray
    Write-Host $model.Size -ForegroundColor Yellow
    Write-Host "  Model    : " -NoNewline -ForegroundColor DarkGray
    Write-Host $model.Name -ForegroundColor White
    Write-Host "  File     : " -NoNewline -ForegroundColor DarkGray
    Write-Host $model.File -ForegroundColor DarkCyan
    Write-Host "  Context  : " -NoNewline -ForegroundColor DarkGray
    Write-Host "$($model.Ctx) tokens" -ForegroundColor Yellow
    Write-Host "  GPU      : " -NoNewline -ForegroundColor DarkGray
    Write-Host "$gpuLayers layers" -ForegroundColor Magenta
    Write-Host "  Port     : " -NoNewline -ForegroundColor DarkGray
    Write-Host $Port -ForegroundColor Cyan

    if ($model.Extra) {
        Write-Host "  Extra    : " -NoNewline -ForegroundColor DarkGray
        Write-Host $model.Extra -ForegroundColor Magenta
    }

    Write-Host "`nCommand: " -ForegroundColor DarkGray
    Write-Host $cmd -ForegroundColor DarkGray
    Write-Host ""

    $confirm = Read-Host "Start server? [Y/n]"
    if ($confirm -notin '', 'y', 'Y') { continue }

    Invoke-Expression $cmd
    Write-Host "`nServer session ended. Press Enter to return..." -ForegroundColor Cyan
    Read-Host
}
