# Jet_Windows_Workspace/python_scripts/powershell/Start-LlamaServer-Llm.ps1

[CmdletBinding()]
param(
    [Alias("p")]
    [Parameter(HelpMessage = "Port for llama-server to listen on (default: 8080)")]
    [ValidateRange(1, 65535)]
    [int]$Port = 8080
)

Write-Host "`n  Llama.cpp Server Launcher  " -BackgroundColor DarkCyan -ForegroundColor Black
Write-Host "  Ryzen 5 3600 • GTX 1660 6GB • 16 GB RAM`n" -ForegroundColor DarkGray

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

$BaseModelDir = "C:\Users\druiv\.cache\llama.cpp"

$categories = @(
    @{
        Name = "General Purpose / Chat"
        Items = @(
            @{ Num=1;  Size="Small";  Name="SmolLM3-3B";                          File="SmolLM3-3B-Q4_K_M.gguf";                                         Ctx=8192; Gpu=999; Jinja=$true;  Desc="Very compact & surprisingly capable 3B" }
            @{ Num=2;  Size="Small";  Name="Llama-3.2-3B-Instruct";              File="Llama-3.2-3B-Instruct-Q4_K_M.gguf";                              Ctx=8192; Gpu=999; Jinja=$true;  Desc="Fast & very popular 3B model" }
            @{ Num=3;  Size="Small";  Name="Gemma-3-4B-it";                      File="gemma-3-4b-it-Q4_K_M.gguf";                                      Ctx=8192; Gpu=999; Jinja=$true;  Desc="Fresh Gemma 2025 release" }
            @{ Num=4;  Size="Small";  Name="Qwen3-4B-Instruct-2507";             File="Qwen3-4B-Instruct-2507-Q4_K_M.gguf";                             Ctx=8192; Gpu=999; Jinja=$true;  Desc="Very recent strong 4B model" }
            @{ Num=5;  Size="Medium"; Name="Mistral-Nemo-Instruct-2407";         File="Mistral-Nemo-Instruct-2407-Q4_K_M.gguf";                         Ctx=8192; Gpu=35;  Jinja=$true;  Desc="Excellent quality - feels like 12B" }
            @{ Num=6;  Size="Medium"; Name="DeepSeek-R1-Distill-Qwen-7B";        File="DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf";                        Ctx=8192; Gpu=35;  Jinja=$true;  Desc="Strong reasoning focused distill" }
            @{ Num=7;  Size="Medium"; Name="Meta-Llama-3.1-8B-Instruct";         File="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";                         Ctx=8192; Gpu=35;  Jinja=$true;  Desc="Still very capable classic 8B" }
            @{ Num=8;  Size="Medium"; Name="Hermes-3-Llama-3.1-8B";              File="Hermes-3-Llama-3.1-8B-Q4_K_M.gguf";                              Ctx=8192; Gpu=35;  Jinja=$true;  Desc="Excellent reasoning & chat" }
        )
    },
    @{
        Name = "Coding & Technical"
        Items = @(
            @{ Num=1; Size="Medium"; Name="DeepSeek-Coder-V2-Lite-Instruct";     File="DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf";                    Ctx=32768; Gpu=30; Jinja=$true; Desc="One of the best small coders 2025" }
        )
    },
    @{
        Name = "Vision / Multimodal"
        Items = @(
            @{ Num=1; Size="Medium"; Name="Qwen2.5-VL-7B-Instruct";              
               File="Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf";                  
               Ctx=8192; Gpu=28; Jinja=$true; 
               Desc="Strong vision + text (OCR, charts, objects)"; 
               MmprojFile="mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf" 
            }
            # ... other vision models ...
        )
    }
    # ... keep your other categories (Embeddings, Japanese, Uncensored) ...
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

    # ── Improved path handling ──
    $modelPath = Join-Path $BaseModelDir $model.File

    # Try to find file if not exact match (useful when folder names differ)
    if (-Not (Test-Path $modelPath)) {
        $found = Get-ChildItem -Path $BaseModelDir -Recurse -File -Filter "*.gguf" -ErrorAction SilentlyContinue |
                 Where-Object { $_.Name -like "*$($model.Name)*Q4*K*M*" -or $_.Name -eq $model.File } |
                 Select-Object -First 1 -ExpandProperty FullName
        
        if ($found) {
            $modelPath = $found
            Write-Host "  Found model at: $modelPath" -ForegroundColor Yellow
        }
    }

    if (-Not (Test-Path $modelPath)) {
        Write-Host "`nModel file not found!" -ForegroundColor Red
        Write-Host "  Expected : $modelPath" -ForegroundColor DarkGray
        Write-Host "  Hint     : Check filename case & folder structure" -ForegroundColor DarkGray
        Write-Host "  You may need to rename or move the .gguf file." -ForegroundColor DarkGray
        pause
        continue
    }

    # ── Build command ──
    $cmd = "llama-server.exe " +
           "-m `"$modelPath`" " +
           "--host 0.0.0.0 --port $Port " +
           "--ctx-size $($model.Ctx) " +
           "--n-gpu-layers $($model.Gpu) " +
           "--flash-attn on " +
           "--cache-type-k q8_0 --cache-type-v q8_0 " +
           "--threads 8 --threads-batch 8 " +
           "--mlock --no-mmap " +
           "--cont-batching "

    if ($model.Jinja) { $cmd += "--jinja " }

    # Reasonable sampling defaults (most people prefer this style)
    $cmd += "--temp 0.75 --min-p 0.05 --top-k 40 --top-p 0.92 "

    # Multimodal support
    if ($model.MmprojFile) {
        $mmprojPath = Join-Path $BaseModelDir $model.MmprojFile
        if (Test-Path $mmprojPath) {
            $cmd += "--mmproj `"$mmprojPath`" "
        } else {
            Write-Host "  Warning: mmproj not found → vision disabled" -ForegroundColor Yellow
            Write-Host "  Expected: $mmprojPath" -ForegroundColor DarkGray
        }
    }

    if ($model.Extra) { $cmd += "$($model.Extra) " }

    # ── Show summary ──
    Write-Host "`n  Starting server with:" -ForegroundColor Green
    Write-Host "  Model    : " -NoNewline; Write-Host $model.Name -ForegroundColor White
    Write-Host "  File     : " -NoNewline; Write-Host (Split-Path $modelPath -Leaf) -ForegroundColor DarkCyan
    Write-Host "  Context  : " -NoNewline; Write-Host "$($model.Ctx) tokens" -ForegroundColor Yellow
    Write-Host "  GPU      : " -NoNewline; Write-Host "$($model.Gpu) layers" -ForegroundColor Magenta
    Write-Host "  Port     : " -NoNewline; Write-Host $Port -ForegroundColor Cyan

    Write-Host "`nCommand:" -ForegroundColor DarkGray
    Write-Host $cmd -ForegroundColor DarkGray
    Write-Host ""

    $confirm = Read-Host "Launch now? [Y/n]"
    if ($confirm -notin '', 'y', 'Y') { continue }

    Write-Host "`nStarting llama-server ...`n" -ForegroundColor Cyan
    Invoke-Expression $cmd

    Write-Host "`nServer session ended. Press Enter to return to menu..." -ForegroundColor Cyan
    Read-Host
}
