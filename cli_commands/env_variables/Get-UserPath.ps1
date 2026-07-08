# =============================================
# Get-UserPath.ps1
# Displays current User PATH with nice formatting
# =============================================

Write-Host "=== Current User PATH ===" -ForegroundColor Cyan

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($userPath) {
    $paths = $userPath -split ';' | Where-Object { $_ -ne "" } | Sort-Object
    $paths | ForEach-Object {
        Write-Host "  $_" -ForegroundColor White
    }
    Write-Host "`nTotal entries: $($paths.Count)" -ForegroundColor Green
} else {
    Write-Host "User PATH is empty." -ForegroundColor Yellow
}