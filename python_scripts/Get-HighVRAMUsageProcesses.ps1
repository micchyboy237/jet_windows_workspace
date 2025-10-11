# Get-HighVRAMUsageProcesses.ps1

function Show-GpuRamUsage {
    param(
        [int]$IntervalSeconds = 2,
        [ValidateSet("VRAM","RAM","TOTAL")]
        [string]$SortBy = "VRAM"
    )

    while ($true) {
        $gpuCounters = Get-Counter -Counter "\GPU Process Memory(*)\Local Usage"
        $valid = $gpuCounters.CounterSamples | Where-Object { $_.CookedValue -gt 0 }

        $map = $valid | ForEach-Object {
            if ($_.Path -match "pid_(\d+)_") {
                [pscustomobject] @{
                    PID     = [int]$Matches[1]
                    VRAM_MB = [math]::Round($_.CookedValue / 1MB, 2)
                }
            }
        }

        $output = $map | ForEach-Object {
            $p = Get-Process -Id $_.PID -ErrorAction SilentlyContinue
            if ($null -ne $p) {
                [pscustomobject] @{
                    ProcessName = $p.ProcessName
                    PID         = $_.PID
                    VRAM_MB     = $_.VRAM_MB
                    RAM_MB      = [math]::Round($p.WorkingSet64 / 1MB, 2)
                    TOTAL_MB    = $_.VRAM_MB + ([math]::Round($p.WorkingSet64 / 1MB, 2))
                }
            }
        }

        switch ($SortBy) {
            "VRAM"  { $output = $output | Sort-Object -Property VRAM_MB -Descending }
            "RAM"   { $output = $output | Sort-Object -Property RAM_MB -Descending }
            "TOTAL" { $output = $output | Sort-Object -Property TOTAL_MB -Descending }
        }   

        Clear-Host
        Write-Host "GPU + RAM Usage (sorted by $SortBy, updated: $(Get-Date))
" -ForegroundColor Cyan
        $output | Format-Table -Property ProcessName,PID,VRAM_MB,RAM_MB,TOTAL_MB -AutoSize

        Start-Sleep -Seconds $IntervalSeconds
    }
}

# Run the monitor (refresh every 2s)
Show-GpuRamUsage -IntervalSeconds 2

# Usage examples
# Show-GpuRamUsage -SortBy VRAM
# Show-GpuRamUsage -SortBy RAM
# Show-GpuRamUsage -SortBy TOTAL
