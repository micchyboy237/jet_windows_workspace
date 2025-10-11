# Run-DiskInfo.ps1
# Display disk information once in CSV format with color-coded output

# Function to get disk info in CSV format
function Get-DiskInfo {
    $disks = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DriveType=3"  # Filter for fixed drives
    $result = @()
    $index = 0
    foreach ($disk in $disks) {
        $totalSpace = [math]::Round($disk.Size / 1GB, 2)  # Convert bytes to GB
        $freeSpace = [math]::Round($disk.FreeSpace / 1GB, 2)  # Convert bytes to GB
        $usedSpace = [math]::Round($totalSpace - $freeSpace, 2)
        $utilization = if ($totalSpace -eq 0) { 0 } else { [math]::Round(($usedSpace / $totalSpace) * 100, 2) }
        $result += "$index, $($disk.DeviceID), Utilization: $utilization %, Used: ${usedSpace} GB  /  ${totalSpace} GB"
        $index++
    }
    return $result
}

# Function to display output with colors
function Write-ColoredOutput {
    param (
        [string]$line
    )
    $parts = $line -split ", "
    $index = $parts[0]
    $name = $parts[1]
    $utilParts = $parts[2] -split ": "
    $utilLabel = $utilParts[0] + ": "  # e.g., "Utilization: "
    $utilValue = $utilParts[1]         # e.g., "42.50 %"
    $spaceParts = $parts[3] -split "  /  "
    $usedParts = $spaceParts[0] -split ": "
    $usedLabel = $usedParts[0] + ": "  # e.g., "Used: "
    $usedValue = $usedParts[1]         # e.g., "85.20 GB"
    $spaceSeparator = "  /  "          # Separator with spaces
    $totalSpace = $spaceParts[1]       # e.g.,p "200.00 GB"

    Write-Host "$index, " -NoNewline -ForegroundColor White
    Write-Host $name -NoNewline -ForegroundColor White
    Write-Host ", " -NoNewline -ForegroundColor White
    Write-Host $utilLabel -NoNewline -ForegroundColor Gray
    Write-Host $utilValue -NoNewline -ForegroundColor Green
    Write-Host ", " -NoNewline -ForegroundColor White
    Write-Host $usedLabel -NoNewline -ForegroundColor Gray
    Write-Host $usedValue -NoNewline -ForegroundColor Cyan
    Write-Host $spaceSeparator -NoNewline -ForegroundColor Gray
    Write-Host $totalSpace -ForegroundColor Magenta
}

# Clear the console and output CSV header
Clear-Host
Write-Host "index, drive, utilization, space" -ForegroundColor Cyan

# Get disk information
$diskInfo = Get-DiskInfo

# Display disk info with colors
foreach ($line in $diskInfo) {
    Write-ColoredOutput -line $line
}
