# Run-VramRamInfo.ps1
# Continuously display GPU, system RAM, and disk information in aligned CSV format with color-coded output
# Colors are usage-based: Green = low load/high remaining, Yellow = moderate, Red = high/critical/low remaining

# Configuration
$refreshIntervalSeconds = 0.5   # Change this value to adjust refresh rate (e.g. 0.5, 1, 2, 5)
$yellowThreshold = 50           # % at/above which a value turns Yellow
$redThreshold = 80              # % at/above which a value turns Red

# Column width configuration for alignment
$colIndexWidth = 6
$colNameWidth = 25
$colUtilWidth = 18
$colMemWidth = 32
$colRemWidth = 14

# Function to get system RAM info in CSV format
function Get-SystemRamInfo {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $cpu = Get-CimInstance -ClassName Win32_Processor
    $totalRam = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    $freeRam = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $usedRam = [math]::Round($totalRam - $freeRam, 2)
    $cpuUtilization = [math]::Round($cpu.LoadPercentage, 2)
    return "0, System RAM, CPU: $cpuUtilization %, Mem: ${usedRam} GB  /  ${totalRam} GB"
}

# Function to get disk info in CSV format
function Get-DiskInfo {
    $disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='C:'"
    $totalDisk = [math]::Round($disk.Size / 1GB, 2)
    $freeDisk = [math]::Round($disk.FreeSpace / 1GB, 2)
    $usedDisk = [math]::Round($totalDisk - $freeDisk, 2)
    $usagePercent = if ($totalDisk -eq 0) { 0 } else { [math]::Round(($usedDisk / $totalDisk) * 100, 2) }
    return "1, System Disk, Usage: $usagePercent %, Mem: ${usedDisk} GB  /  ${totalDisk} GB"
}

# Function to convert GPU memory from MiB to GB and format output
function Convert-GpuMemoryToGB {
    param ([string]$gpuInfoLine)
    $parts = $gpuInfoLine -split ", "
    $index = $parts[0]
    $name = $parts[1]
    $utilization = $parts[2] -replace "%", "%"
    $usedMiB = [float]($parts[3] -replace " MiB", "")
    $totalMiB = [float]($parts[4] -replace " MiB", "")
    $usedGB = [math]::Round($usedMiB / 1024, 2)
    $totalGB = [math]::Round($totalMiB / 1024, 2)
    return "$index, $name, Cuda: $utilization, Mem: ${usedGB} GB  /  ${totalGB} GB"
}

# Maps a usage percentage to a color
function Get-UsageColor {
    param ([double]$Percent)
    if ($Percent -ge $redThreshold) { return "Red" }
    elseif ($Percent -ge $yellowThreshold) { return "Yellow" }
    else { return "Green" }
}

# Display output with aligned columns and colors including remaining memory
function Write-ColoredOutput {
    param ([string]$line)
    
    $parts = $line -split ", "
    $index = $parts[0].PadRight($colIndexWidth)
    $name = $parts[1].PadRight($colNameWidth)
    
    # Parse utilization
    $cudaParts = $parts[2] -split ": "
    $cudaLabel = $cudaParts[0] + ": "
    $cudaValue = $cudaParts[1]
    $utilPercent = 0
    if ($cudaValue -match '([\d\.]+)\s*%') { $utilPercent = [double]$Matches[1] }
    $utilColor = Get-UsageColor -Percent $utilPercent
    $utilFormatted = "$cudaLabel$cudaValue".PadRight($colUtilWidth)
    
    # Parse memory used/total
    $memoryParts = $parts[3] -split "  /  "
    $memUsedParts = $memoryParts[0] -split ": "
    $memLabel = $memUsedParts[0] + ": "
    $memUsedValue = $memUsedParts[1]
    $memTotal = $memoryParts[1]
    
    $memUsedNum = 0; $memTotalNum = 0
    if ($memUsedValue -match '([\d\.]+)') { $memUsedNum = [double]$Matches[1] }
    if ($memTotal -match '([\d\.]+)') { $memTotalNum = [double]$Matches[1] }
    
    $memPercent = if ($memTotalNum -eq 0) { 0 } else { ($memUsedNum / $memTotalNum) * 100 }
    $memColor = Get-UsageColor -Percent $memPercent
    $memFormatted = "$memLabel$memUsedValue  /  $memTotal".PadRight($colMemWidth)
    
    # Calculate and format remaining memory
    $remainingGB = [math]::Round($memTotalNum - $memUsedNum, 2)
    $remColor = Get-UsageColor -Percent $memPercent  # Same color logic: red when nearly full
    $remFormatted = "Rem: $remainingGB GB".PadRight($colRemWidth)
    
    # Write aligned colored output
    Write-Host $index -NoNewline -ForegroundColor White
    Write-Host $name -NoNewline -ForegroundColor White
    Write-Host $utilFormatted -NoNewline -ForegroundColor $utilColor
    Write-Host $memFormatted -NoNewline -ForegroundColor $memColor
    Write-Host $remFormatted -ForegroundColor $remColor
}

# Clear console and output header + legend once
Clear-Host
$header = "{0}{1}{2}{3}{4}" -f `
    "index".PadRight($colIndexWidth), `
    "name".PadRight($colNameWidth), `
    "utilization".PadRight($colUtilWidth), `
    "memory".PadRight($colMemWidth), `
    "remaining".PadRight($colRemWidth)
Write-Host $header -ForegroundColor Cyan

Write-Host "Legend: " -NoNewline -ForegroundColor Gray
Write-Host "Green" -NoNewline -ForegroundColor Green
Write-Host " < $yellowThreshold%   " -NoNewline -ForegroundColor Gray
Write-Host "Yellow" -NoNewline -ForegroundColor Yellow
Write-Host " $yellowThreshold-$($redThreshold-1)%   " -NoNewline -ForegroundColor Gray
Write-Host "Red" -NoNewline -ForegroundColor Red
Write-Host " >= $redThreshold%" -ForegroundColor Gray
Write-Host ""

# Main loop
while ($true) {
    $gpuInfo = nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    $gpuInfoGB = Convert-GpuMemoryToGB -gpuInfoLine $gpuInfo
    $ramInfo = Get-SystemRamInfo
    $diskInfo = Get-DiskInfo

    Write-ColoredOutput -line $gpuInfoGB
    Write-ColoredOutput -line $ramInfo
    Write-ColoredOutput -line $diskInfo
    Write-Host ""

    Start-Sleep -Seconds $refreshIntervalSeconds
}
