function Get-RAMInfo {
    <#
    .SYNOPSIS
        Retrieves total, free, and used RAM in GB.
    .DESCRIPTION
        Queries system memory using Win32_OperatingSystem and returns RAM details in gigabytes.
    .OUTPUTS
        Hashtable with Total, Free, and Used RAM in GB.
    .EXAMPLE
        Get-RAMInfo
    #>
    try {
        $memory = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
        $total = [math]::Round($memory.TotalVisibleMemorySize / 1MB, 2)
        $free = [math]::Round($memory.FreePhysicalMemory / 1MB, 2)
        $used = [math]::Round(($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory) / 1MB, 2)
        
        return @{
            TotalRAM = $total
            FreeRAM = $free
            UsedRAM = $used
        }
    }
    catch {
        Write-Error "Failed to retrieve RAM information: $_"
        return $null
    }
}

# Execute and display results
$result = Get-RAMInfo
if ($result) {
    Write-Output "Total RAM: $($result.TotalRAM) GB"
    Write-Output "Free RAM: $($result.FreeRAM) GB"
    Write-Output "Used RAM: $($result.UsedRAM) GB"
}
