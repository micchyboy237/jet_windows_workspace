# Run-VramRamInfo.ps1
# Continuously display GPU, system RAM, and disk information in CSV format with color-coded output
# Colors are usage-based: Green = low load, Yellow = moderate, Red = high/critical

# Configuration
$refreshIntervalSeconds = 0.5   # Change this value to adjust refresh rate (e.g. 0.5, 1, 2, 5)
$yellowThreshold = 50           # % at/above which a value turns Yellow
$redThreshold = 80              # % at/above which a value turns Red

# Function to get system RAM info in CSV format
function Get-SystemRamInfo {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $cpu = Get-CimInstance -ClassName Win32_Processor
    $totalRam = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)  # Convert KB to GB
    $freeRam = [math]::Round($os.FreePhysicalMemory / 1MB, 2)      # Convert KB to GB
    $usedRam = [math]::Round($totalRam - $freeRam, 2)
    $cpuUtilization = [math]::Round($cpu.LoadPercentage, 2)        # Get CPU utilization percentage
    return "0, System RAM, CPU: $cpuUtilization %, Mem: ${usedRam} GB  /  ${totalRam} GB"
}

# Function to get disk info in CSV format
function Get-DiskInfo {
    $disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='C:'"
    $totalDisk = [math]::Round($disk.Size / 1GB, 2)  # Convert bytes to GB
    $freeDisk = [math]::Round($disk.FreeSpace / 1GB, 2)  # Convert bytes to GB
    $usedDisk = [math]::Round($totalDisk - $freeDisk, 2)
    $usagePercent = if ($totalDisk -eq 0) { 0 } else { [math]::Round(($usedDisk / $totalDisk) * 100, 2) }  # Calculate usage percentage
    return "1, System Disk, Usage: $usagePercent %, Mem: ${usedDisk} GB  /  ${totalDisk} GB"
}

# Function to convert GPU memory from MiB to GB and format output
function Convert-GpuMemoryToGB {
    param (
        [string]$gpuInfoLine
    )
    $parts = $gpuInfoLine -split ", "
    $index = $parts[0]
    $name = $parts[1]
    $utilization = $parts[2] -replace "%", "%"  # Ensure % is preserved
    $usedMiB = [float]($parts[3] -replace " MiB", "")
    $totalMiB = [float]($parts[4] -replace " MiB", "")
    $usedGB = [math]::Round($usedMiB / 1024, 2)  # Convert MiB to GB
    $totalGB = [math]::Round($totalMiB / 1024, 2)  # Convert MiB to GB
    return "$index, $name, Cuda: $utilization, Mem: ${usedGB} GB  /  ${totalGB} GB"
}

# NEW: Maps a usage percentage to a color, so severity is visible at a glance
function Get-UsageColor {
    param (
        [double]$Percent
    )
    if ($Percent -ge $redThreshold) {
        return "Red"
    } elseif ($Percent -ge $yellowThreshold) {
        return "Yellow"
    } else {
        return "Green"
    }
}

# Function to display output with colors (now usage-aware for utilization + memory)
function Write-ColoredOutput {
    param (
        [string]$line
    )
    $parts = $line -split ", "
    $index = $parts[0]
    $name = $parts[1]
    $cudaParts = $parts[2] -split ": "
    $cudaLabel = $cudaParts[0] + ": "  # e.g., "CPU: ", "Cuda: ", or "Usage: "
    $cudaValue = $cudaParts[1]          # e.g., "4 %"
    $memoryParts = $parts[3] -split "  /  "
    $memUsedParts = $memoryParts[0] -split ": "
    $memLabel = $memUsedParts[0] + ": "  # e.g., "Mem: "
    $memUsedValue = $memUsedParts[1]     # e.g., "0.46 GB"
    $memSeparator = "  /  "              # Separator with spaces
    $memTotal = $memoryParts[1]          # e.g., "6 GB"

    # Parse the utilization percentage (CPU/Cuda/Usage) for color decisions
    $utilPercent = 0
    if ($cudaValue -match '([\d\.]+)\s*%') {
        $utilPercent = [double]$Matches[1]
    }
    $utilColor = Get-UsageColor -Percent $utilPercent

    # Parse memory used/total to compute a usage percentage for color decisions
    $memUsedNum = 0
    $memTotalNum = 0
    if ($memUsedValue -match '([\d\.]+)') { $memUsedNum = [double]$Matches[1] }
    if ($memTotal -match '([\d\.]+)') { $memTotalNum = [double]$Matches[1] }
    $memPercent = if ($memTotalNum -eq 0) { 0 } else { ($memUsedNum / $memTotalNum) * 100 }
    $memColor = Get-UsageColor -Percent $memPercent

    Write-Host "$index, " -NoNewline -ForegroundColor White
    Write-Host $name -NoNewline -ForegroundColor White
    Write-Host ", " -NoNewline -ForegroundColor White
    Write-Host $cudaLabel -NoNewline -ForegroundColor Gray
    Write-Host $cudaValue -NoNewline -ForegroundColor $utilColor
    Write-Host ", " -NoNewline -ForegroundColor White
    Write-Host $memLabel -NoNewline -ForegroundColor Gray
    Write-Host $memUsedValue -NoNewline -ForegroundColor $memColor
    Write-Host $memSeparator -NoNewline -ForegroundColor Gray
    Write-Host $memTotal -ForegroundColor White
}

# Clear the console and output CSV header + color legend once
Clear-Host
Write-Host "index, name, utilization, memory" -ForegroundColor Cyan
Write-Host "Legend: " -NoNewline -ForegroundColor Gray
Write-Host "Green" -NoNewline -ForegroundColor Green
Write-Host " < $yellowThreshold%   " -NoNewline -ForegroundColor Gray
Write-Host "Yellow" -NoNewline -ForegroundColor Yellow
Write-Host " $yellowThreshold-$($redThreshold-1)%   " -NoNewline -ForegroundColor Gray
Write-Host "Red" -NoNewline -ForegroundColor Red
Write-Host " >= $redThreshold%" -ForegroundColor Gray
Write-Host ""

# Main loop to display GPU, RAM, and disk info
while ($true) {
    # Get GPU information using nvidia-smi
    $gpuInfo = nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

    # Convert GPU memory units to GB and format output
    $gpuInfoGB = Convert-GpuMemoryToGB -gpuInfoLine $gpuInfo

    # Get RAM information in CSV format
    $ramInfo = Get-SystemRamInfo

    # Get disk information in CSV format
    $diskInfo = Get-DiskInfo

    # Display GPU, RAM, and disk info with usage-based colors
    Write-ColoredOutput -line $gpuInfoGB
    Write-ColoredOutput -line $ramInfo
    Write-ColoredOutput -line $diskInfo
    Write-Host ""  # Add newline for separation between iterations

    # Wait for n seconds before refreshing
    Start-Sleep -Seconds $refreshIntervalSeconds
}
