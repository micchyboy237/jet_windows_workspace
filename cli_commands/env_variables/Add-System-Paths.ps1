# Add-System-Path.ps1
# Run this script as Administrator

$ErrorActionPreference = 'Stop'

# Check if running as Administrator
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator to modify the System PATH."
    Write-Host "Right-click the script → 'Run with PowerShell' or open elevated PowerShell and run it." -ForegroundColor Yellow
    exit 1
}

$pathsToAdd = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\CUPTI\lib64"
    # Uncomment if you really need it (rarely required for runtime):
    # "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include"
    # "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
)

# Get current System PATH (Machine scope)
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

# Split into array for easier checking (handle possible empty/trailing semicolons)
$existingPaths = $currentPath -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ }

$added = @()
$skipped = @()

foreach ($path in $pathsToAdd) {
    $path = $path.Trim().TrimEnd('\')  # Normalize: remove trailing slash, extra spaces

    if ([string]::IsNullOrWhiteSpace($path)) { continue }

    # Check if already present (case-insensitive, with or without trailing slash)
    $normalizedExisting = $existingPaths | ForEach-Object { $_.TrimEnd('\').ToLower() }
    if ($normalizedExisting -contains $path.ToLower()) {
        $skipped += $path
    }
    else {
        $added += $path
        $existingPaths += $path  # Update local list to avoid duplicates in same run
    }
}

if ($added.Count -gt 0) {
    # Build new PATH string
    $newPathString = ($existingPaths -join ';').Trim(';')

    # Safety check: warn if PATH would become unreasonably long
    if ($newPathString.Length -gt 30000) {
        Write-Warning "The resulting PATH would be very long ($($newPathString.Length) characters)."
        Write-Host "Consider cleaning up duplicates or unnecessary entries first." -ForegroundColor Yellow
        # You could add Read-Host "Continue anyway? (Y/N)" logic here if desired
    }

    # Apply to System (Machine) scope
    [Environment]::SetEnvironmentVariable("Path", $newPathString, "Machine")

    Write-Host "`nSuccessfully added $($added.Count) path(s) to System PATH:" -ForegroundColor Green
    $added | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
}
else {
    Write-Host "`nNo new paths to add (all already present or empty)." -ForegroundColor Yellow
}

if ($skipped.Count -gt 0) {
    Write-Host "`nSkipped (already present in PATH):" -ForegroundColor Cyan
    $skipped | ForEach-Object { Write-Host "  $_" }
}

# Refresh current session's PATH (only affects this PowerShell window)
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
$env:Path -split ';'

Write-Host "`nDone. Restart any open terminals, VS Code, etc. for the changes to take effect in new processes."
Write-Host "To verify:"
Write-Host "  Run:  `$env:Path -split ';' | Select-String 'CUDA\\v12.8' " -ForegroundColor Gray
