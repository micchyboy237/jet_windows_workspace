# =============================================
# Get-SystemPath.ps1
# Displays current System (Machine) PATH with nice formatting
# =============================================

Write-Host "=== Current System PATH ===" -ForegroundColor Cyan

$systemPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

if ($systemPath) {
    $paths = $systemPath -split ';' | Where-Object { $_ -ne "" } | Sort-Object
    $paths | ForEach-Object {
        Write-Host "  $_" -ForegroundColor White
    }
    Write-Host "`nTotal entries: $($paths.Count)" -ForegroundColor Green
} else {
    Write-Host "No System PATH found." -ForegroundColor Red
}

Write-Host "`nRun this script without Admin rights if you only need to view." -ForegroundColor Gray