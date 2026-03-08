# List-Python-Paths.ps1
# Lists Python-related PATH entries from both USER and SYSTEM scopes.

$ErrorActionPreference = "Stop"

function Normalize-PathString {
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    return $Path.Trim().TrimEnd('\')
}

function Get-PathEntries {
    param(
        [Parameter(Mandatory)]
        [string]$Scope
    )

    $pathValue = [Environment]::GetEnvironmentVariable("Path", $Scope)

    if ([string]::IsNullOrWhiteSpace($pathValue)) {
        return @()
    }

    return $pathValue -split ';' |
        ForEach-Object { Normalize-PathString $_ } |
        Where-Object { $_ } |
        ForEach-Object {
            [PSCustomObject]@{
                Scope = $Scope
                Path  = $_
            }
        }
}

Write-Host "`nScanning PATH for Python-related directories..." -ForegroundColor Cyan

# Collect paths
$userPaths = Get-PathEntries -Scope "User"
$systemPaths = Get-PathEntries -Scope "Machine"

$allPaths = $userPaths + $systemPaths

# Python-related filters
$pythonKeywords = @(
    "python",
    "conda",
    "pip",
    "pyenv",
    "scripts",
    "site-packages"
)

function Is-PythonRelated {
    param([string]$Path)

    $lower = $Path.ToLower()

    foreach ($keyword in $pythonKeywords) {
        if ($lower -like "*$keyword*") {
            return $true
        }
    }

    return $false
}

$pythonPaths = $allPaths | Where-Object { Is-PythonRelated $_.Path }

if ($pythonPaths.Count -eq 0) {
    Write-Host "No Python-related PATH entries detected." -ForegroundColor Yellow
    exit 0
}

# Normalize for duplicate detection
$normalized = $pythonPaths.Path | ForEach-Object { $_.ToLower() }

$dupMap = @{}
foreach ($p in $normalized) {
    if ($dupMap.ContainsKey($p)) {
        $dupMap[$p] += 1
    } else {
        $dupMap[$p] = 1
    }
}

Write-Host "`nPython-related PATH entries:`n" -ForegroundColor Green

$index = 0
$missingCount = 0
$duplicateCount = 0

foreach ($entry in $pythonPaths) {

    $exists = Test-Path $entry.Path
    $dup = $dupMap[$entry.Path.ToLower()] -gt 1

    $status = @()

    if (-not $exists) {
        $status += "Missing"
        $missingCount++
    }

    if ($dup) {
        $status += "Duplicate"
        $duplicateCount++
    }

    if ($status.Count -eq 0) {
        $statusText = "OK"
        $color = "White"
    } else {
        $statusText = $status -join ", "
        $color = if ($status -contains "Missing") { "Red" } else { "Yellow" }
    }

    Write-Host ("[{0}] [{1}] {2}  ({3})" -f $index, $entry.Scope, $entry.Path, $statusText) -ForegroundColor $color

    $index++
}

Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "  Python PATH entries : $($pythonPaths.Count)"
Write-Host "  Missing directories : $missingCount"
Write-Host "  Duplicates          : $duplicateCount"

Write-Host "`nLegend:" -ForegroundColor Gray
Write-Host "  White  = OK"
Write-Host "  Yellow = Duplicate"
Write-Host "  Red    = Directory missing"
