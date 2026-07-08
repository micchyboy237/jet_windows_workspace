# =============================================
# Add-ToUserPath.ps1
# Adds paths to User PATH with deduplication
# =============================================

# ================== CONFIGURATION ==================
$PathsToAdd = @(
    "C:\Users\druiv\.cache\ngrok\ngrok-v3-stable-windows-amd64"                   # Example: ngrok folder
    # "C:\Users\$env:USERNAME\tools",
    # Add more paths here
)
# ===================================================

Write-Host "=== Updating User PATH ===" -ForegroundColor Magenta

# Get current User PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
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
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")

Write-Host "`nUser PATH updated. Added $added new path(s)." -ForegroundColor Green
Write-Host "Restart any open terminals for changes to take effect." -ForegroundColor Cyan