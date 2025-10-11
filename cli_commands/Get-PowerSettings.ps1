# Get-PowerSettings.ps1

function Get-PowerSettings {
    <#
    .SYNOPSIS
        Retrieves current AC power settings for sleep, hibernation, and monitor timeouts.
    .DESCRIPTION
        Queries the active power plan using powercfg to get standby-timeout-ac, 
        hibernate-timeout-ac, and monitor-timeout-ac settings in minutes.
    .OUTPUTS
        PSCustomObject with properties: StandbyTimeoutAC, HibernateTimeoutAC, MonitorTimeoutAC
    .EXAMPLE
        Get-PowerSettings
    #>
    try {
        # Get the active power scheme GUID
        $activeScheme = powercfg /getactivescheme | ForEach-Object {
            if ($_ -match 'Power Scheme GUID: ([a-f0-9-]+)') { $matches[1] }
        }
        if (-not $activeScheme) {
            throw "Could not determine active power scheme"
        }

        # Query settings (values are in seconds)
        $standbyRaw = powercfg /query $activeScheme SUB_SLEEP STANDBYIDLE | 
            Where-Object { $_ -match 'AC Setting Index: 0x([0-9a-f]+)' } | 
            ForEach-Object { [Convert]::ToInt32($matches[1], 16) }
        $hibernateRaw = powercfg /query $activeScheme SUB_SLEEP HIBERNATEIDLE | 
            Where-Object { $_ -match 'AC Setting Index: 0x([0-9a-f]+)' } | 
            ForEach-Object { [Convert]::ToInt32($matches[1], 16) }
        $monitorRaw = powercfg /query $activeScheme SUB_VIDEO VIDEOIDLE | 
            Where-Object { $_ -match 'AC Setting Index: 0x([0-9a-f]+)' } | 
            ForEach-Object { [Convert]::ToInt32($matches[1], 16) }

        # Convert seconds to minutes for readability
        $standbyMinutes = $standbyRaw / 60
        $hibernateMinutes = $hibernateRaw / 60
        $monitorMinutes = $monitorRaw / 60

        # Return as a custom object
        [PSCustomObject]@{
            StandbyTimeoutAC  = $standbyMinutes
            HibernateTimeoutAC = $hibernateMinutes
            MonitorTimeoutAC  = $monitorMinutes
        }
    }
    catch {
        Write-Error "Error retrieving power settings: $_"
        return $null
    }
}

# Example usage
Get-PowerSettings
