function Get-GPUVRAMInfo {
    <#
    .SYNOPSIS
        Retrieves total, free, and used GPU VRAM in GB for NVIDIA GPUs.
    .DESCRIPTION
        Queries Win32_VideoController for total VRAM and nvidia-smi for free/used VRAM.
        Returns results in gigabytes. If nvidia-smi is unavailable, only total VRAM is returned.
    .OUTPUTS
        Hashtable with TotalVRAM, FreeVRAM, and UsedVRAM in GB, or null if failed.
    .EXAMPLE
        Get-GPUVRAMInfo
    #>
    [CmdletBinding()]
    param ()

    try {
        # Get GPU information from Win32_VideoController
        $gpu = Get-CimInstance -ClassName Win32_VideoController | Where-Object { $_.AdapterCompatibility -eq "NVIDIA" }
        if (-not $gpu) {
            Write-Warning "No NVIDIA GPU found."
            return $null
        }

        # Calculate total VRAM (convert bytes to GB)
        $totalVRAM = [math]::Round($gpu.AdapterRAM / 1GB, 2)
        $result = @{
            TotalVRAM = $totalVRAM
            FreeVRAM = $null
            UsedVRAM = $null
        }

        # Attempt to get free/used VRAM using nvidia-smi
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            $nvidiaSmiOutput = & nvidia-smi --query-gpu=memory.used,memory.free --format=csv,nounits | Select-Object -Skip 1
            if ($nvidiaSmiOutput) {
                $memoryInfo = $nvidiaSmiOutput -split ","
                $usedVRAM = [math]::Round([int]($memoryInfo[0].Trim()) / 1024, 2)  # Convert MB to GB
                $freeVRAM = [math]::Round([int]($memoryInfo[1].Trim()) / 1024, 2)  # Convert MB to GB
                $result.FreeVRAM = $freeVRAM
                $result.UsedVRAM = $usedVRAM
            }
            else {
                Write-Warning "Failed to parse nvidia-smi output. Free and used VRAM unavailable."
            }
        }
        else {
            Write-Warning "nvidia-smi not found. Free and used VRAM unavailable."
        }

        return $result
    }
    catch {
        Write-Error "Failed to retrieve GPU VRAM information: $_"
        return $null
    }
}

# Execute and display results
$result = Get-GPUVRAMInfo
if ($result) {
    Write-Output "Total GPU VRAM: $($result.TotalVRAM) GB"
    if ($null -ne $result.FreeVRAM -and $null -ne $result.UsedVRAM) {
        Write-Output "Free GPU VRAM: $($result.FreeVRAM) GB"
        Write-Output "Used GPU VRAM: $($result.UsedVRAM) GB"
    }
    else {
        Write-Output "Free and Used GPU VRAM: Unavailable (requires nvidia-smi)"
    }
}
