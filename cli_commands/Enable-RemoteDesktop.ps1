# Enable-RemoteDesktop.ps1

function Enable-RemoteDesktop {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Username
    )

    try {
        # Enable RDP
        Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Terminal Server' -Name 'fDenyTSConnections' -Value 0 -ErrorAction Stop
        Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Terminal Server\WinStations\RDP-Tcp' -Name 'UserAuthentication' -Value 1 -ErrorAction Stop

        # Enable firewall rules
        Enable-NetFirewallRule -DisplayGroup 'Remote Desktop' -ErrorAction Stop

        # Add user to Remote Desktop Users group
        Add-LocalGroupMember -Group 'Remote Desktop Users' -Member $Username -ErrorAction Stop

        # Prevent sleep
        powercfg /change standby-timeout-ac 0
        powercfg /change hibernate-timeout-ac 0

        # Output PC details
        $pcName = $env:COMPUTERNAME
        $ipAddress = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -like '*Wi-Fi*' -or $_.InterfaceAlias -like '*Ethernet*' }).IPAddress
        Write-Output "Remote Desktop enabled for $Username"
        Write-Output "PC Name: $pcName"
        Write-Output "Local IP: $ipAddress"
    }
    catch {
        Write-Error "Error configuring Remote Desktop: $_"
    }
}

# Example usage: Enable-RemoteDesktop -Username 'YourUsername'
