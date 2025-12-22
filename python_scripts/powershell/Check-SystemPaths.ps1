<#
.SYNOPSIS
    Displays and validates the Windows PATH environment variable.

.DESCRIPTION
    This script shows:
    - The effective PATH (as seen by the current process)
    - Separate Machine (System) and User PATH contributions
    - Splits the effective PATH into individual directories
    - Checks each directory for existence and highlights missing/invalid ones

    Useful for diagnosing issues with commands not found or bloated/invalid PATH entries.
#>

Write-Host "Effective PATH (current process):" -ForegroundColor Cyan
Write-Host $env:PATH -ForegroundColor Gray

Write-Host "`n--- PATH Entries (checked for existence) ---" -ForegroundColor Yellow

$pathEntries = $env:PATH -split ';' | Where-Object { $_ -ne '' }

foreach ($entry in $pathEntries) {
    $exists = Test-Path -Path $entry -PathType Container
    if ($exists) {
        Write-Host "$entry" -ForegroundColor Green
    } else {
        Write-Host "$entry  <-- MISSING or INVALID" -ForegroundColor Red
    }
}

# Optional: Separate Machine and User PATH (requires no elevation for reading)
Write-Host "`nMachine (System) PATH:" -ForegroundColor Magenta
[Environment]::GetEnvironmentVariable('Path', 'Machine')

Write-Host "`nUser PATH:" -ForegroundColor Magenta
[Environment]::GetEnvironmentVariable('Path', 'User')

Write-Host "`nSummary:" -ForegroundColor Cyan
$missing = $pathEntries | Where-Object { -not (Test-Path -Path $_ -PathType Container) }
if ($missing) {
    Write-Host "Found $($missing.Count) missing/invalid directories in PATH." -ForegroundColor Red
} else {
    Write-Host "All PATH directories exist." -ForegroundColor Green
}
