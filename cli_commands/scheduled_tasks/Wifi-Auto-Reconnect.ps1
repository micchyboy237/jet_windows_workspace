# =========================
# CONFIGURATION
# =========================
$SSID = "estradadeco"
$LogFile = Join-Path $PSScriptRoot "wifi_reconnect.log"
$CheckIntervalSeconds = 10

$ParentDir = Split-Path -Path $PSScriptRoot -Parent
$ResetScriptPath = Join-Path $ParentDir "Reset-Wifi.ps1"

# =========================
# LOGGING
# =========================
function Write-Log {
    param (
        [string]$Message,
        [string]$Level = "INFO"
    )

    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $logLine = "[$timestamp] [$Level] $Message"

    Add-Content -Path $LogFile -Value $logLine
    Write-Output $logLine
}

# Log configuration variables
Write-Log "Script configuration:"
Write-Log "SSID = $SSID"
Write-Log "LogFile = $LogFile"
Write-Log "CheckIntervalSeconds = $CheckIntervalSeconds"
Write-Log "ResetScriptPath = $ResetScriptPath"

# =========================
# WIFI HELPERS
# =========================
function Get-CurrentSSID {
    try {
        $netshOutput = netsh wlan show interfaces

        foreach ($line in $netshOutput) {
            if ($line -match "^\s*SSID\s*:\s*(.+)$") {
                return $Matches[1].Trim()
            }
        }
    } catch {
        Write-Log "Failed to get current SSID: $_" "ERROR"
    }

    return $null
}

function Is-WifiConnected {
    try {
        $netshOutput = netsh wlan show interfaces

        foreach ($line in $netshOutput) {
            if ($line -match "^\s*State\s*:\s*(.+)$") {
                return $Matches[1].Trim() -eq "connected"
            }
        }
    } catch {
        Write-Log "Failed to check Wi-Fi state: $_" "ERROR"
    }

    return $false
}

function Connect-ToSSID {
    param (
        [string]$TargetSSID
    )

    try {
        Write-Log "Attempting to connect to SSID: $TargetSSID"

        netsh wlan connect name="$TargetSSID" | Out-Null

        Start-Sleep -Seconds 5

        if (Is-WifiConnected) {
            $connectedSSID = Get-CurrentSSID
            if ($connectedSSID -eq $TargetSSID) {
                Write-Log "Successfully connected to $TargetSSID"
                return $true
            } else {
                Write-Log "Connected, but SSID mismatch: $connectedSSID" "WARN"
            }
        } else {
            Write-Log "Connection attempt failed" "ERROR"
        }
    } catch {
        Write-Log "Exception during connection: $_" "ERROR"
    }

    return $false
}

# =========================
# WIFI RESET
# =========================
function Reset-WifiAdapter {
    try {
        if (-not (Test-Path $ResetScriptPath)) {
            Write-Log "Reset script not found at: $ResetScriptPath" "ERROR"
            return $false
        }

        Write-Log "Invoking Wi-Fi reset script..."

        & $ResetScriptPath

        Write-Log "Wi-Fi reset script executed"
        Start-Sleep -Seconds 5

        return $true
    } catch {
        Write-Log "Failed to execute reset script: $_" "ERROR"
        return $false
    }
}

# =========================
# INIT
# =========================
if (!(Test-Path $LogFile)) {
    New-Item -Path $LogFile -ItemType File | Out-Null
}

Write-Log "=== Wi-Fi Auto Reconnect Script Started ==="
Write-Log "Target SSID: $SSID"

# =========================
# MAIN LOOP
# =========================
while ($true) {
    $isConnected = Is-WifiConnected
    $currentSSID = Get-CurrentSSID

    if (-not $isConnected -or $currentSSID -ne $SSID) {
        Write-Log "Wi-Fi not connected or wrong SSID (Current: $currentSSID)"

        Reset-WifiAdapter | Out-Null

        Connect-ToSSID -TargetSSID $SSID | Out-Null
    } else {
        Write-Log "Already connected to correct SSID: $SSID"
    }

    Start-Sleep -Seconds $CheckIntervalSeconds
}
