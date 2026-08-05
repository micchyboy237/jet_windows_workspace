# get-system-info.ps1

Write-Host "=== System Information ===" -ForegroundColor Cyan

$cs = Get-CimInstance -ClassName Win32_ComputerSystem
$proc = Get-CimInstance -ClassName Win32_Processor
$os = Get-CimInstance -ClassName Win32_OperatingSystem
$gpu = Get-CimInstance -ClassName Win32_VideoController

# 1. Model Name
$finalModel = if ($cs.SystemFamily) { $cs.SystemFamily } else { $cs.Model }

# 2. RAM (Sum actual physical memory sticks)
$physicalRamBytes = (Get-CimInstance -ClassName Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum
if ($physicalRamBytes -eq 0) { $physicalRamBytes = $cs.TotalPhysicalMemory } 

$totalRamGB = "{0:N2}" -f ($physicalRamBytes / 1GB)
$usableRamGB = "{0:N2}" -f ($os.TotalVisibleMemorySize / 1MB)

Write-Host "PC Model       : $($cs.Manufacturer) $finalModel"

# Processor
$procName = $proc.Name -replace 'CPU @ \d+\.\d+GHz', '' | ForEach-Object { $_.Trim() }
Write-Host "Processor      : $procName @ $([math]::Round($proc.MaxClockSpeed/1000, 2)) GHz ($([math]::Round(($proc.CurrentClockSpeed)/1000, 2)) GHz)"

Write-Host "Installed RAM  : $totalRamGB GB ($usableRamGB GB usable)"

# System Type (e.g. "64-bit operating system, x64-based processor")
$osArch = $os.OSArchitecture
$procArch = $cs.SystemType -replace '-based PC', '-based processor'
Write-Host "System Type    : $osArch operating system, $procArch"

# 3. Graphics (Reads DedicatedSegmentSize from Registry - Requires Admin!)
$displayRegPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\*"
$gpuRegistry = Get-ItemProperty $displayRegPath -ErrorAction SilentlyContinue | Where-Object { $_.DriverDesc }

Write-Host "Graphics Cards :"
$gpuIndex = 0
foreach ($g in $gpu) {
    $gpuIndex++
    $regMatch = $gpuRegistry | Where-Object { $_.DriverDesc -eq $g.Name } | Select-Object -First 1

    if ($gpuIndex -eq 2) {
        # Display 2nd graphics card without size
        Write-Host "  - $($g.Name)"
    } else {
        if ($regMatch -and $regMatch.DedicatedSegmentSize -gt 0) {
            # Use the exact Dedicated VRAM from Registry
            $vramSize = $regMatch.DedicatedSegmentSize
            $vramDisplay = if ($vramSize -ge 1024) { "$([math]::Round($vramSize / 1024, 1)) GB" } else { "$vramSize MB" }
        } else {
            # Fallback: If not Admin or key missing, calculate from WMI
            $vramBytes = $g.AdapterRAM
            if ($vramBytes -gt 0) {
                $vramGB = $vramBytes / 1GB
                $vramDisplay = if ($vramGB -lt 1) { "$([math]::Round($vramBytes / 1MB, 0)) MB" } else { "$([math]::Round($vramGB, 0)) GB" }
            } else {
                $vramDisplay = "Shared"
            }
        }
        Write-Host "  - $($g.Name) ($vramDisplay)"
    }
}

# 4. Storage (C: and D: Drives)
Write-Host "`n=== Storage Information ===" -ForegroundColor Cyan

$targetDrives = @('C:', 'D:')
foreach ($letter in $targetDrives) {
    # DriveType 3 ensures we only look at Local Disks (ignores CD-ROMs/Network drives)
    $disk = Get-CimInstance -ClassName Win32_LogicalDisk | Where-Object { $_.DeviceID -eq $letter -and $_.DriveType -eq 3 }
    
    if ($disk) {
        $totalGB = "{0:N2}" -f ($disk.Size / 1GB)
        $freeGB = "{0:N2}" -f ($disk.FreeSpace / 1GB)
        $usedGB = "{0:N2}" -f (($disk.Size - $disk.FreeSpace) / 1GB)
        
        if ($disk.Size -gt 0) {
            $freePct = "{0:N1}" -f (($disk.FreeSpace / $disk.Size) * 100)
            $usedPct = "{0:N1}" -f ((($disk.Size - $disk.FreeSpace) / $disk.Size) * 100)
        } else {
            $freePct = "0.0"
            $usedPct = "0.0"
        }
        
        $diskInfo = "$totalGB GB Total | $usedGB GB Used ($usedPct%) | $freeGB GB Free ($freePct%)"
        $label = "Drive $letter"
        
        # PadRight(14) ensures the colon aligns perfectly with the rest of the script's output
        Write-Host "$($label.PadRight(14)) : $diskInfo"
    } else {
        $label = "Drive $letter"
        Write-Host "$($label.PadRight(14)) : Not found or not a local disk"
    }
}