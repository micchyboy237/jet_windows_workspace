# wifi-auto-reconnect.ps1
# Auto reconnect to available known WiFi networks on Windows 11 with logging

# Define log file path
$logFile = "$Env:USERPROFILE\.cache\scheduled_tasks\wifi-auto-reconnect.log"

# Ensure log directory exists
$logDir = Split-Path $logFile -Parent
if (-not (Test-Path $logDir)) {
    New-Item -Path $logDir -ItemType Directory -Force | Out-Null
}

# Function to write to both console and log file
function Write-Log {
    param ($Message)
    $timestamp = Get-Date -Format 'HH:mm:ss'
    Write-Host "$timestamp - $Message"
    Add-Content -Path $logFile -Value "$timestamp - $Message"
}

function Get-CurrentWiFi {
    $wifi = netsh wlan show interfaces | Select-String " SSID" | Select-String -NotMatch "BSSID"
    if ($wifi) {
        return ($wifi -split " : ")[1].Trim()
    }
    return $null
}

function Get-AvailableNetworks {
    $networks = netsh wlan show networks mode=bssid | Select-String "SSID"
    $unique = @()
    foreach ($n in $networks) {
        $ssid = ($n -split " : ")[1].Trim()
        if ($ssid -and -not ($unique -contains $ssid)) {
            $unique += $ssid
        }
    }
    return $unique
}

function Get-KnownProfiles {
    $profiles = netsh wlan show profiles | Select-String "All User Profile"
    $list = @()
    foreach ($p in $profiles) {
        $ssid = ($p -split " : ")[1].Trim()
        if ($ssid) { $list += $ssid }
    }
    return $list
}

function Connect-ToKnownWiFi {
    $available = Get-AvailableNetworks
    $known = Get-KnownProfiles
    $toTry = $available | Where-Object { $known -contains $_ }

    foreach ($ssid in $toTry) {
        Write-Log "Trying to connect to known network: $ssid"
        netsh wlan connect name="$ssid" ssid="$ssid" | Out-Null
        Start-Sleep -Seconds 5

        if (Get-CurrentWiFi) {
            Write-Log "✅ Connected to $ssid"
            return $true
        }
    }
    Write-Log "⚠️ No known WiFi networks available"
    return $false
}

# Initialize log with script start
Write-Log "WiFi Auto-Reconnect script started"

while ($true) {
    $current = Get-CurrentWiFi
    if (-not $current) {
        Write-Log "Not connected. Attempting reconnect..."
        Connect-ToKnownWiFi | Out-Null
    }
    else {
        Write-Log "Connected to: $current"
    }
    Start-Sleep -Seconds 30
}