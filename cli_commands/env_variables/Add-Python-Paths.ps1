# Add-Python-Paths.ps1
# Purpose: Add common Python installation paths to the System PATH
# Run this script as Administrator

$ErrorActionPreference = 'Stop'

# ────────────────────────────────────────────────
# Check for Administrator rights
# ────────────────────────────────────────────────
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator to modify the System PATH."
    Write-Host "Right-click → 'Run with PowerShell' or use an elevated PowerShell window." -ForegroundColor Yellow
    exit 1
}

# ────────────────────────────────────────────────
# Python paths to add (edit these to match your actual installation)
# Most common locations as of 2025–2026
# ────────────────────────────────────────────────
$pathsToAdd = @(
    "C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedVAD\fireredvad\bin"
    # Python 3.13 (example – change version number as needed)
    # "C:\Python313",
    # "C:\Python313\Scripts",

    # Very common modern install locations (user or all-users)
    #"C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python313",
    #"C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python313\Scripts",

    # All-users install (common on corporate / dev machines)
    #"C:\Program Files\Python313",
    #"C:\Program Files\Python313\Scripts",

    # Python Launcher & Tools (strongly recommended)
    # "C:\Python313\Tools\scripts",

    # If you also use pyenv-win, conda base, or miniconda
    #"C:\Users\$env:USERNAME\.pyenv\pyenv-win\shims",
    #"C:\Users\$env:USERNAME\miniconda3\condabin",
    #"C:\ProgramData\miniconda3\condabin"
)

# ────────────────────────────────────────────────
# Get current SYSTEM (Machine) PATH
# ────────────────────────────────────────────────
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

$existingPaths = $currentPath -split ';' |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and $_.Trim() -ne '' }

$added   = @()
$skipped = @()

foreach ($path in $pathsToAdd) {
    $path = $path.Trim().TrimEnd('\')

    if ([string]::IsNullOrWhiteSpace($path)) { continue }

    # Case-insensitive + trailing-slash-insensitive comparison
    $alreadyExists = $existingPaths |
        ForEach-Object { $_.TrimEnd('\').ToLower() } |
        Where-Object { $_ -eq $path.ToLower() }

    if ($alreadyExists) {
        $skipped += $path
    }
    else {
        $added += $path
        $existingPaths += $path   # prevent duplicate additions in same run
    }
}

# ────────────────────────────────────────────────
# Apply changes if anything new was found
# ────────────────────────────────────────────────
if ($added.Count -gt 0) {
    $newPathString = ($existingPaths -join ';').Trim(';')

    # Optional: warn on very long PATH
    if ($newPathString.Length -gt 32000) {
        Write-Warning "New PATH length would be $($newPathString.Length) characters — this is unusually long."
        Write-Host "Consider removing unused entries before continuing." -ForegroundColor Yellow
    }

    # Write to SYSTEM environment
    [Environment]::SetEnvironmentVariable("Path", $newPathString, "Machine")

    Write-Host "`nAdded $($added.Count) path(s) to System PATH:" -ForegroundColor Green
    $added | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
}
else {
    Write-Host "`nNo new Python paths needed — everything is already present." -ForegroundColor Yellow
}

if ($skipped.Count -gt 0) {
    Write-Host "`nAlready in PATH (skipped):" -ForegroundColor Cyan
    $skipped | ForEach-Object { Write-Host "  $_" }
}

# ────────────────────────────────────────────────
# Refresh current session (this window only)
# ────────────────────────────────────────────────
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

Write-Host "`nDone." -ForegroundColor Green
Write-Host "• New terminals / applications will see the updated PATH."
Write-Host "• To check right now in this session:"
Write-Host "    `$env:Path -split ';' | Select-String -Pattern 'Python|Scripts|pyenv|conda' -AllMatches" -ForegroundColor Gray
Write-Host "• Recommended: restart VS Code, terminals, etc."
