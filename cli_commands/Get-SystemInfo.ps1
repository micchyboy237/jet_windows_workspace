# Get-SystemInfo.ps1

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

# 3.5 GPU VRAM Usage (Real-time from nvidia-smi)
Write-Host "`n=== GPU VRAM Information ===" -ForegroundColor Cyan
try {
    $nvidiaSmiOutput = nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $nvidiaSmiOutput) {
        $gpuVramLines = $nvidiaSmiOutput -split "`n" | Where-Object { $_.Trim() -ne "" }
        foreach ($line in $gpuVramLines) {
            $parts = $line -split ',\s*'
            $gpuIndex    = $parts[0].Trim()
            $gpuName     = $parts[1].Trim()
            # Strip non-numeric characters (handles "1032 MiB", "1032", etc.)
            $memUsedMiB  = [int]($parts[2] -replace '[^\d]', '')
            $memTotalMiB = [int]($parts[3] -replace '[^\d]', '')
            $memFreeMiB  = $memTotalMiB - $memUsedMiB
            
            # Calculate percentages
            if ($memTotalMiB -gt 0) {
                $usedPctVal = ($memUsedMiB / $memTotalMiB) * 100
                $freePctVal = ($memFreeMiB / $memTotalMiB) * 100
            } else {
                $usedPctVal = 0
                $freePctVal = 0
            }
            
            # VRAM thresholds: Dynamic memory, needs headroom for workloads
            # ≤50% Green, ≤80% Yellow, >80% Red
            if ($usedPctVal -le 50) {
                $usedColor = "Green"
                $freeColor = "Green"
            } elseif ($usedPctVal -le 80) {
                $usedColor = "Yellow"
                $freeColor = "Yellow"
            } else {
                $usedColor = "Red"
                $freeColor = "Red"
            }
            
            $usedPctStr = "{0:N1}" -f $usedPctVal
            $freePctStr = "{0:N1}" -f $freePctVal
            
            Write-Host "GPU $gpuIndex ($gpuName):" -ForegroundColor White
            Write-Host "  Total : $memTotalMiB MiB"
            Write-Host "  Used  : $memUsedMiB MiB (" -NoNewline
            Write-Host "$usedPctStr%" -ForegroundColor $usedColor -NoNewline
            Write-Host ")"
            Write-Host "  Free  : $memFreeMiB MiB (" -NoNewline
            Write-Host "$freePctStr%" -ForegroundColor $freeColor -NoNewline
            Write-Host ")"
        }
    } else {
        Write-Host "  nvidia-smi not available or no NVIDIA GPU detected" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  Error querying GPU VRAM: $_" -ForegroundColor Red
}

# 4. Storage (C: and D: Drives)
Write-Host "`n=== Storage Information ===" -ForegroundColor Cyan
$targetDrives = @('C:', 'D:')
foreach ($letter in $targetDrives) {
    # DriveType 3 ensures we only look at Local Disks (ignores CD-ROMs/Network drives)
    $disk = Get-CimInstance -ClassName Win32_LogicalDisk | Where-Object { $_.DeviceID -eq $letter -and $_.DriveType -eq 3 }
    if ($disk) {
        # Detect Drive Type (SSD vs HDD)
        $driveLetter = $letter -replace ':', ''
        $mediaType = "Unknown"
        try {
            # Map logical drive to physical disk to check MediaType
            $part = Get-Partition -DriveLetter $driveLetter -ErrorAction Stop | Select-Object -First 1
            $physDisk = Get-PhysicalDisk -DeviceNumber $part.DiskNumber -ErrorAction Stop
            $mediaType = if ($physDisk.MediaType) { $physDisk.MediaType } else { "Unknown" }
        } catch {
            # Fallback to your note if the Storage module fails (e.g., in some VMs or older OS)
            if ($driveLetter -eq 'C') { $mediaType = 'SSD' }
            elseif ($driveLetter -eq 'D') { $mediaType = 'HDD' }
        }
        $totalGB = "{0:N2}" -f ($disk.Size / 1GB)
        $freeGB = "{0:N2}" -f ($disk.FreeSpace / 1GB)
        $usedGB = "{0:N2}" -f (($disk.Size - $disk.FreeSpace) / 1GB)
        if ($disk.Size -gt 0) {
            $freePctVal = ($disk.FreeSpace / $disk.Size) * 100
            $usedPctVal = (($disk.Size - $disk.FreeSpace) / $disk.Size) * 100
        } else {
            $freePctVal = 0.0
            $usedPctVal = 0.0
        }
        
        # Storage thresholds: Gradual accumulation, more forgiving headroom
        # ≤70% Green, ≤90% Yellow, >90% Red
        if ($usedPctVal -le 70) {
            $usedColor = "Green"
            $freeColor = "Green"
        } elseif ($usedPctVal -le 90) {
            $usedColor = "Yellow"
            $freeColor = "Yellow"
        } else {
            $usedColor = "Red"
            $freeColor = "Red"
        }
        
        $freePctStr = "{0:N1}" -f $freePctVal
        $usedPctStr = "{0:N1}" -f $usedPctVal
        
        $label = "Drive $letter ($mediaType)"
        Write-Host "$($label.PadRight(14)) : $totalGB GB Total | $usedGB GB Used (" -NoNewline
        Write-Host "$usedPctStr%" -ForegroundColor $usedColor -NoNewline
        Write-Host ") | $freeGB GB Free (" -NoNewline
        Write-Host "$freePctStr%" -ForegroundColor $freeColor -NoNewline
        Write-Host ")"
    } else {
        $label = "Drive $letter (N/A)"
        Write-Host "$($label.PadRight(14)) : Not found or not a local disk"
    }
}