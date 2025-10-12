# wifi-auto-reconnect.ps1
# Auto reconnect to available known WiFi networks on Windows 11

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
        Write-Host "Trying to connect to known network: $ssid"
        netsh wlan connect name="$ssid" ssid="$ssid" | Out-Null
        Start-Sleep -Seconds 5

        if (Get-CurrentWiFi) {
            Write-Host "✅ Connected to $ssid"
            return $true
        }
    }
    Write-Host "⚠️ No known WiFi networks available"
    return $false
}

while ($true) {
    $current = Get-CurrentWiFi
    if (-not $current) {
        Write-Host "$(Get-Date -Format 'HH:mm:ss') - Not connected. Attempting reconnect..."
        Connect-ToKnownWiFi | Out-Null
    }
    else {
        Write-Host "$(Get-Date -Format 'HH:mm:ss') - Connected to: $current"
    }
    Start-Sleep -Seconds 30
}
