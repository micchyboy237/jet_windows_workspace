[CmdletBinding()]
param (
    [Parameter(Mandatory = $false)]
    [string]$AdapterName = "Wi-Fi",

    [Parameter(Mandatory = $false)]
    [int]$DelaySeconds = 3
)

function Get-WifiAdapter {
    param (
        [string]$Name
    )

    $adapter = Get-NetAdapter -Name $Name -ErrorAction SilentlyContinue

    if (-not $adapter) {
        Write-Host "[ERROR] Adapter '$Name' not found." -ForegroundColor Red
        Write-Host "Available adapters:" -ForegroundColor Yellow
        Get-NetAdapter | Format-Table -AutoSize
        exit 1
    }

    return $adapter
}

function Disable-Wifi {
    param ([string]$Name)

    Write-Host "[INFO] Disabling adapter '$Name'..." -ForegroundColor Cyan
    Disable-NetAdapter -Name $Name -Confirm:$false -ErrorAction Stop
}

function Enable-Wifi {
    param ([string]$Name)

    Write-Host "[INFO] Enabling adapter '$Name'..." -ForegroundColor Cyan
    Enable-NetAdapter -Name $Name -Confirm:$false -ErrorAction Stop
}

function Reset-Wifi {
    param (
        [string]$Name,
        [int]$Delay
    )

    $adapter = Get-WifiAdapter -Name $Name

    Write-Host "[INFO] Current Status: $($adapter.Status)" -ForegroundColor Gray

    Disable-Wifi -Name $Name

    Write-Host "[INFO] Waiting $Delay seconds..." -ForegroundColor DarkGray
    Start-Sleep -Seconds $Delay

    Enable-Wifi -Name $Name

    $updated = Get-NetAdapter -Name $Name
    Write-Host "[SUCCESS] Adapter '$Name' is now $($updated.Status)" -ForegroundColor Green
}

# --- Entry ---
try {
    Reset-Wifi -Name $AdapterName -Delay $DelaySeconds
}
catch {
    Write-Host "[ERROR] $_" -ForegroundColor Red
    exit 1
}
