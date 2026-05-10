<#
──────────────────────────────────────────────────────────────────────────────
sync-torch-cudnn-to-paddle.ps1
──────────────────────────────────────────────────────────────────────────────

Purpose
-------
Fixes Paddle / PaddleOCR runtime crashes on Windows when cuDNN DLLs bundled
with Paddle are incompatible with the installed CUDA or driver environment.

Typical error this resolves:

    OSError: [WinError 127] The specified procedure could not be found.
    Error loading "...site-packages\nvidia\cudnn\bin\cudnn_cnn64_9.dll"

Why this happens
----------------
PyTorch distributes a **complete and working set of CUDA + cuDNN runtime DLLs**
inside:

    site-packages\torch\lib\

However Paddle packages its cuDNN DLLs separately inside:

    site-packages\nvidia\cudnn\bin\

Sometimes these DLL versions are **ABI-incompatible** with the CUDA toolkit
or GPU driver installed on the system, causing the WinError 127 crash when
Paddle attempts to load them.

Solution
--------
Replace Paddle’s cuDNN DLLs with the working versions shipped with PyTorch.

This script:

1. Detects the active Python virtual environment.
2. Locates:
       torch\lib
       nvidia\cudnn\bin
3. Copies all cuDNN DLLs from Torch into Paddle’s directory.
4. Overwrites existing Paddle DLLs.

This allows PaddleOCR to reuse the **same CUDA/cuDNN runtime already working
with PyTorch**, avoiding version mismatches.

Safe to run multiple times.

Usage
-----

Activate your venv first:

    .\sync-torch-cudnn-to-paddle.ps1

or

    powershell -ExecutionPolicy Bypass -File sync-torch-cudnn-to-paddle.ps1

──────────────────────────────────────────────────────────────────────────────
#>

$ErrorActionPreference = "Stop"

# Torch CUDA/cuDNN source directory
$torchDir = Join-Path $env:VIRTUAL_ENV "Lib\site-packages\torch\lib"

# Paddle cuDNN destination directory
$paddleDir = Join-Path $env:VIRTUAL_ENV "Lib\site-packages\nvidia\cudnn\bin"

Write-Host ""
Write-Host "Torch CUDA directory:   $torchDir"
Write-Host "Paddle cuDNN directory: $paddleDir"
Write-Host ""

if (!(Test-Path $torchDir)) {
    Write-Error "Torch lib directory not found"
}

if (!(Test-Path $paddleDir)) {
    Write-Error "Paddle cuDNN directory not found"
}

# cuDNN files to copy
$patterns = @(
    "cudnn64_*.dll",
    "cudnn_adv64_*.dll",
    "cudnn_cnn64_*.dll",
    "cudnn_engines_precompiled64_*.dll",
    "cudnn_engines_runtime_compiled64_*.dll",
    "cudnn_graph64_*.dll",
    "cudnn_heuristic64_*.dll",
    "cudnn_ops64_*.dll"
)

$files = foreach ($pattern in $patterns) {
    Get-ChildItem -Path $torchDir -Filter $pattern -File
}

if ($files.Count -eq 0) {
    Write-Error "No cuDNN DLLs found in Torch directory"
}

Write-Host "Found $($files.Count) cuDNN DLLs in Torch."

foreach ($file in $files) {
    $dest = Join-Path $paddleDir $file.Name
    Write-Host "Copying $($file.Name)"
    Copy-Item $file.FullName $dest -Force
}

Write-Host ""
Write-Host "cuDNN sync complete."
