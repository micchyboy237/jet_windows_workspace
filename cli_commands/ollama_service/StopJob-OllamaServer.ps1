# StopJob-OllamaServer.ps1
# Stops the Ollama server background job and terminates the process

try {
    Write-Host "Attempting to find Ollama server process on port 11434..."
    $connections = Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue

    if ($connections) {
        $processIds = $connections.OwningProcess | Sort-Object -Unique
        Write-Host "Found $($processIds.Count) process(es) on port 11434: $processIds"

        foreach ($processId in $processIds) {
            Write-Host "Attempting to stop process ID $processId..."
            Stop-Process -Id $processId -Force -ErrorAction Stop
            Write-Host "Process ID $processId stopped successfully."
        }
    } else {
        Write-Host "No process found listening on port 11434."
    }

    # Check for any Ollama-related background jobs
    $jobs = Get-Job | Where-Object { $_.Command -like "*ollama serve*" }
    if ($jobs) {
        Write-Host "Found $($jobs.Count) Ollama-related job(s). Stopping them..."
        $jobs | Stop-Job -ErrorAction Stop
        $jobs | Remove-Job -Force
        Write-Host "Ollama background job(s) stopped and removed."
    } else {
        Write-Host "No Ollama-related background jobs found."
    }

    # Verify port 11434 is no longer in use
    Start-Sleep -Seconds 1
    $check = Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue
    if ($check) {
        Write-Host "Warning: Port 11434 is still in use. Check for other processes or services."
        Write-Host (netstat -aon | findstr :11434)
    } else {
        Write-Host "Ollama server stopped successfully. Port 11434 is no longer in use."
    }
} catch {
    Write-Host "Error stopping Ollama server: $($_.Exception.Message)"
    exit 1
}
