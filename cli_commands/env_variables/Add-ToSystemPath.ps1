# =============================================
# Add-ToSystemPath.ps1
# Adds paths to System (Machine) PATH with deduplication
# Run as Administrator
# =============================================

# ================== CONFIGURATION ==================
$PathsToAdd = @(
    "C:\ngrok",                    # Example: ngrok folder
    "C:\Program Files\SomeTool\bin",
    # Add more paths here
)
# ===================================================

Write-Host "=== Updating System PATH ===" -ForegroundColor Magenta

# Get current System PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
if (-not $currentPath) { $currentPath = "" }

$pathList = $currentPath -split ';' | Where-Object { $_ -ne "" } | ForEach-Object { $_.TrimEnd('\') }

$added = 0
foreach ($p in $PathsToAdd) {
    $cleanPath = $p.TrimEnd('\')
    if ($pathList -notcontains $cleanPath) {
        $pathList += $cleanPath
        $added++
        Write-Host "  [+] Added: $cleanPath" -ForegroundColor Green
    } else {
        Write-Host "  [-] Already exists: $cleanPath" -ForegroundColor Yellow
    }
}

# Rebuild and apply
$newPath = ($pathList -join ';') + ';'
[Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")

Write-Host "`nSystem PATH updated. Added $added new path(s)." -ForegroundColor Green
Write-Host "Restart any open terminals for changes to take effect." -ForegroundColor Cyan