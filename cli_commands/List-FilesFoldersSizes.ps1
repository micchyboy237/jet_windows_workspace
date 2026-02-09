# List-FilesFoldersSizes.ps1
#
# PURPOSE:
#   Displays files and folders with their sizes and last modified dates.
#   Shows recursive size for directories, color coding, human-readable sizes.
#
# FEATURES:
#   - Recursive folder size calculation
#   - Cyan folders / Gray files
#   - Yellow <DIR> marker
#   - Sort by LastWriteTime (oldest first by default)
#
# USAGE EXAMPLES:
#
#   # Current directory, oldest first (default)
#   .\List-FilesFoldersSizes.ps1
#
#   # Current directory, newest first
#   .\List-FilesFoldersSizes.ps1 -Descending
#   .\List-FilesFoldersSizes.ps1 -d
#
#   # Specific folder, oldest first
#   .\List-FilesFoldersSizes.ps1 -Path C:\Projects
#   .\List-FilesFoldersSizes.ps1 -p C:\Projects
#
#   # Specific folder, newest first (short form)
#   .\List-FilesFoldersSizes.ps1 -p .. -d
#
#   # Explicit oldest first (rarely needed)
#   .\List-FilesFoldersSizes.ps1 -Ascending -p .\src
#   .\List-FilesFoldersSizes.ps1 -a -p .\src
#

param(
    [Alias("p")]
    [string]$Path = ".",

    [Alias("a")]
    [switch]$Ascending = $true,

    [Alias("d")]
    [switch]$Descending
)

# If both -Descending and -Ascending provided, Descending takes precedence
if ($Descending) { $Ascending = $false }

Get-ChildItem -Path $Path -Force |
  Sort-Object LastWriteTime -Descending:(!$Ascending) |
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
