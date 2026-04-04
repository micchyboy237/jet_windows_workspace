# List-FilesFoldersSizes.ps1 - Optimized & Fast Version
#
# Now much faster on large directories with smart progress updates.

param(
    [Alias("p")]
    [string]$Path = ".",
    
    [Alias("a")]
    [switch]$Ascending,
    
    [Alias("d")]
    [switch]$Descending
)

# Default: largest first
if (-not $Ascending -and -not $Descending) {
    $Descending = $true
}
if ($Descending) { $Ascending = $false }

$Path = Resolve-Path $Path

Write-Host "Scanning and calculating sizes for: $Path" -ForegroundColor Cyan

$sizeCache = @{}          # directory -> total size
$filesProcessed = 0
$progressId = 1
$lastProgress = [DateTime]::Now

Get-ChildItem -Path $Path -Recurse -Force -File -ErrorAction SilentlyContinue |
    ForEach-Object {
        $filesProcessed++
        $dir = $_.DirectoryName

        if (-not $sizeCache.ContainsKey($dir)) {
            $sizeCache[$dir] = 0
        }
        $sizeCache[$dir] += $_.Length

        # Smart progress: update only every 500 files OR every ~800ms
        if (($filesProcessed % 500 -eq 0) -or 
            (([DateTime]::Now - $lastProgress).TotalMilliseconds -gt 800)) {
            
            $status = "Processed {0:N0} files..." -f $filesProcessed
            $percent = -1   # indeterminate if we don't know total

            Write-Progress -Id $progressId `
                           -Activity "Calculating recursive folder sizes" `
                           -Status $status `
                           -PercentComplete $percent

            $lastProgress = [DateTime]::Now
        }
    }

Write-Progress -Id $progressId -Completed

# Enrich top-level items with sizes
$items = Get-ChildItem -Path $Path -Force |
    ForEach-Object {
        $size = if ($_.PSIsContainer) {
                    if ($sizeCache.ContainsKey($_.FullName)) { $sizeCache[$_.FullName] } else { 0 }
                } else {
                    $_.Length
                }
        $_ | Add-Member -NotePropertyName "Size" -NotePropertyValue $size -PassThru -Force
    }

# Sort (largest first by default)
$sorted = $items | Sort-Object Size -Descending:(!$Ascending)

# Output
foreach ($item in $sorted) {
    $size = $item.Size
    $sizeStr = switch ($size) {
        { $_ -ge 1GB } { "{0:N2} GB" -f ($size / 1GB); break }
        { $_ -ge 100MB } { "{0:N1} MB" -f ($size / 1MB); break }
        { $_ -ge 1MB }   { "{0:N0} MB" -f ($size / 1MB); break }
        { $_ -ge 1KB }   { "{0:N0} KB" -f ($size / 1KB); break }
        default          { "{0} B" -f $size }
    }

    Write-Host ("{0,12}  {1} " -f $sizeStr, $item.LastWriteTime.ToString("yyyy-MM-dd HH:mm")) -NoNewline

    if ($item.PSIsContainer) {
        Write-Host "<DIR> " -ForegroundColor Yellow -NoNewline
    } else {
        Write-Host "      " -NoNewline
    }

    $color = if ($item.PSIsContainer) { "Cyan" } else { "Gray" }
    Write-Host $item.Name -ForegroundColor $color
}

Write-Host "`nTotal items shown: $($sorted.Count) | Files scanned: $filesProcessed" -ForegroundColor DarkGray
