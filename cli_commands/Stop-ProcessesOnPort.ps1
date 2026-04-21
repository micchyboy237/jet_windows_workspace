# Stop processes running on a specific port (e.g., 8080):
$port = 8765
Get-NetTCPConnection -LocalPort $port | ForEach-Object {
    try {
        Stop-Process -Id $_.OwningProcess -Force
        Write-Host "Stopped process $($_.OwningProcess) using port $port"
    } catch {
        Write-Host "Could not stop process $($_.OwningProcess): $($_.Exception.Message)"
    }
}
