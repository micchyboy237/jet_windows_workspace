# Get-HighVRAMUsageProcesses.ps1
function Show-GpuRamUsage {
    param(
        [int]$IntervalSeconds = 2,
        [ValidateSet("VRAM","RAM")]
        [string]$SortBy = "VRAM",
        [string[]]$ProcessName = $null  # Optional filter for specific process names
    )

    # Get total VRAM and RAM for percentage calculations
    $totalVRAM_MB = (Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.AdapterCompatibility -eq "NVIDIA" } | Select-Object -First 1).AdapterRAM / 1MB
    $totalRAM_MB = (Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory / 1MB

    while ($true) {
        $gpuCounters = Get-Counter -Counter "\GPU Process Memory(*)\Local Usage"
        $valid = $gpuCounters.CounterSamples | Where-Object { $_.CookedValue -gt 0 }

        $map = $valid | ForEach-Object {
            if ($_.Path -match "pid_(\d+)_") {
                [pscustomobject] @{
                    PID = [int]$Matches[1]
                    VRAM_MB = [math]::Round($_.CookedValue / 1MB, 2)
                }
            }
        }

        # Retrieve TCP connections, excluding common system ports
        $systemPorts = @(135, 139, 445, 1001, 1462)  # Common Windows system ports
        $tcpConnections = Get-NetTCPConnection | 
            Where-Object { $_.State -in 'Listen', 'Established' -and $_.LocalPort -notin $systemPorts } | 
            Group-Object OwningProcess | 
            ForEach-Object {
                $ports = ($_.Group | Select-Object -ExpandProperty LocalPort | Sort-Object -Unique) -join ','
                if ($ports) {
                    [pscustomobject] @{
                        PID = [int]$_.Name
                        Ports = $ports
                    }
                }
            }

        $output = $map | ForEach-Object {
            $p = Get-Process -Id $_.PID -ErrorAction SilentlyContinue
            if ($null -ne $p) {
                # Apply process name filter if specified
                if ($ProcessName -and $p.ProcessName -notin $ProcessName) {
                    return
                }
                $port = ($tcpConnections | Where-Object { $_.PID -eq $p.Id } | Select-Object -First 1).Ports
                [pscustomobject] @{
                    ProcessName = $p.ProcessName
                    PID = $_.PID
                    VRAM_MB = $_.VRAM_MB
                    "VRAM_%" = [string]::Format("% {0}", [math]::Round(($_.VRAM_MB / $totalVRAM_MB) * 100, 2))
                    RAM_MB = [math]::Round($p.WorkingSet64 / 1MB, 2)
                    "RAM_%" = [string]::Format("% {0}", [math]::Round(($p.WorkingSet64 / 1MB / $totalRAM_MB) * 100, 2))
                    Port = $port ?? 'N/A'
                }
            }
        }

        switch ($SortBy) {
            "VRAM" { $output = $output | Sort-Object -Property VRAM_MB -Descending }
            "RAM" { $output = $output | Sort-Object -Property RAM_MB -Descending }
        }

        Clear-Host
        Write-Host "GPU + RAM Usage (sorted by $SortBy, updated: $(Get-Date))" -ForegroundColor Cyan
        $output | Format-Table -Property ProcessName,PID,VRAM_MB,"VRAM_%",RAM_MB,"RAM_%",Port -AutoSize

        Start-Sleep -Seconds $IntervalSeconds
    }
}

# Run the monitor (refresh every 2s)
Show-GpuRamUsage -IntervalSeconds 2

# Usage examples
# Show-GpuRamUsage -SortBy VRAM
# Show-GpuRamUsage -SortBy RAM
# Show-GpuRamUsage -ProcessName "llama-server"
# Show-GpuRamUsage -ProcessName "llama-server","msedge"
