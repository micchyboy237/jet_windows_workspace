Get-ChildItem -Force |
  Sort-Object LastWriteTime -Descending |   # Sort first, then process
  ForEach-Object {
    $size = if ($_.PSIsContainer) {
              (Get-ChildItem $_ -Recurse -Force -File -ErrorAction SilentlyContinue |
               Measure-Object -Property Length -Sum).Sum
            } else {
              $_.Length
            }

    $size = if ($null -eq $size) { 0 } else { $size }

    $sizeStr = if ($size -ge 1GB)     { "{0:N2} GB" -f ($size / 1GB) }
               elseif ($size -ge 1MB) { "{0:N0} MB" -f ($size / 1MB) }
               elseif ($size -ge 1KB) { "{0:N0} KB" -f ($size / 1KB) }
               else                   { "{0} B " -f $size }

    # Size + Date
    Write-Host ("{0,10} {1} " -f $sizeStr, $_.LastWriteTime.ToString("yyyy-MM-dd HH:mm")) -NoNewline

    # <DIR> indicator only for folders
    if ($_.PSIsContainer) {
      Write-Host "<DIR>     " -ForegroundColor Yellow -NoNewline
    } else {
      Write-Host "          " -NoNewline   # 10 spaces to align
    }

    # Name with color
    $color = if ($_.PSIsContainer) { "Cyan" } else { "Gray" }
    Write-Host $_.Name -ForegroundColor $color
  }
