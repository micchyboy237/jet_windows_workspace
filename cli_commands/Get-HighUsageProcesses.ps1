# Get-HighUsageProcesses.ps1

# Define thresholds for high resource usage
$VRAMThresholdMB = 500  # VRAM usage > 500 MB
$RAMThresholdMB = 1000  # RAM usage > 1 GB
$CPUThresholdPercent = 10  # CPU usage > 10% over 1 minute

# Function to get VRAM usage using nvidia-smi
function Get-VRAMUsage {
    try {
        # Run nvidia-smi to get process-level GPU memory usage
        Write-Debug "Executing nvidia-smi --query-compute-apps..."
        $nvidiaSmiOutput = & "nvidia-smi" --query-compute-apps=pid,used_memory --format=csv,noheader
        Write-Debug "Compute apps output: $nvidiaSmiOutput"
        $vramData = @{}
        if (-not $nvidiaSmiOutput) {
            Write-Debug "No compute apps found. Parsing process table..."
            # Fallback to full nvidia-smi output and parse process table
            $fullOutput = & "nvidia-smi" | Out-String
            $lines = $fullOutput -split "
"
            $inProcessSection = $false
            foreach ($line in $lines) {
                if ($line -match "Processes:") {
                    $inProcessSection = $true
                    continue
                }
                if ($inProcessSection -and $line -match "\|\s*\d+\s+N/A\s+N/A\s+(\d+)\s+.*\s+(\d+)\s*MiB\s*\|") {
                    $processId = $matches[1]
                    $vramValue = $matches[2]
                    Write-Debug "Line: $line, PID: $processId, VRAM: $vramValue"
                    if ($vramValue -match '^\d+$' -and [int]$vramValue -ge $VRAMThresholdMB) {
                        $vramData[$processId] = [int]$vramValue
                        Write-Debug "Added PID: $processId, VRAM: $vramValue MB"
                    } else {
                        Write-Debug "Skipping non-numeric or low VRAM value: $vramValue"
                    }
                }
            }
        } else {
            foreach ($line in $nvidiaSmiOutput) {
                Write-Debug "Parsing compute app line: $line"
                $processId, $vram = $line -split ', '
                $vramMB = [int]($vram -replace ' MiB')
                Write-Debug "PID: $processId, VRAM: $vramMB MB"
                if ($vramMB -ge $VRAMThresholdMB) {
                    $vramData[$processId] = $vramMB
                }
            }
        }
        Write-Debug "Final VRAM Data: $($vramData | ConvertTo-Json)"
        return $vramData
    } catch {
        Write-Warning "Failed to retrieve VRAM data: $($_.Exception.Message)"
        Write-Debug "Error details: $($_.Exception | Format-List -Force | Out-String)"
        return @{}
    }
}

# Function to get CPU usage percentage over the last minute
function Get-CPUUsagePercent {
    param ($Process, $IntervalSeconds = 60)
    # Get initial CPU time
    $initialCPU = $Process.CPU
    Start-Sleep -Seconds 1  # Small sample to calculate rate
    $process.Refresh()
    $finalCPU = $Process.CPU
    # Calculate CPU usage percentage (relative to one core)
    $cpuTimeSeconds = ($finalCPU - $initialCPU) / 1000  # Convert from milliseconds
    $cpuPercent = ($cpuTimeSeconds / $IntervalSeconds) * 100
    return [math]::Round($cpuPercent, 2)
}

# Collect process data
$processes = Get-Process | Where-Object { $_.WorkingSet64 -gt ($RAMThresholdMB * 1MB) }
$vramData = Get-VRAMUsage

# Build result array
$results = @()
foreach ($proc in $processes) {
    $pid = $proc.Id
    $vramMB = $vramData["$pid"] ? $vramData["$pid"] : 0
    $ramMB = [math]::Round($proc.WorkingSet64 / 1MB, 2)
    $cpuPercent = Get-CPUUsagePercent -Process $proc

    # Include processes meeting any high-usage threshold
    if ($vramMB -ge $VRAMThresholdMB -or $ramMB -ge $RAMThresholdMB -or $cpuPercent -ge $CPUThresholdPercent) {
        $results += [PSCustomObject]@{
            ProcessName = $proc.ProcessName
            PID         = $pid
            VRAM_MB     = $vramMB
            RAM_MB      = $ramMB
            CPU_Percent = $cpuPercent
        }
    }
}

# Display results in a sorted, formatted table
$results | Sort-Object -Property VRAM_MB, RAM_MB, CPU_Percent -Descending | Format-Table -AutoSize
