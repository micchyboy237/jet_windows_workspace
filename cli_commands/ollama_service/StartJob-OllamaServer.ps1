# StartJob-OllamaServer.ps1
# Runs Ollama server as a background job

# Path to Ollama executable (update if needed)
$ollamaPath = "ollama"
$logFile = "$PSScriptRoot\ollama-server.log"

try {
    # Check if Ollama is already running on port 11434
    $connection = Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue
    if ($connection) {
        Write-Host "Ollama server is already running on port 11434 (PID: $($connection.OwningProcess))."
        exit 1
    }

    # Verify Ollama executable exists
    if (-not (Get-Command $ollamaPath -ErrorAction SilentlyContinue)) {
        Write-Host "Error: Ollama executable not found at '$ollamaPath'. Please verify installation."
        exit 1
    }

    Write-Host "Starting Ollama server as a background job..."
    # Start Ollama server as a background job, redirecting output to a log file
    $job = Start-Job -ScriptBlock {
        param($path, $log)
        & $path serve >> $log 2>&1
    } -ArgumentList $ollamaPath, $logFile

    # Wait briefly to ensure the job starts
    Start-Sleep -Seconds 2

    # Check if the job is running
    if ($job.State -eq "Running") {
        Write-Host "Ollama server started successfully. Job ID: $($job.Id)"
        Write-Host "Logs are being written to: $logFile"

        # Verify port 11434 is in use
        $connection = Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue
        if ($connection) {
            Write-Host "Ollama server is listening on port 11434 (PID: $($connection.OwningProcess))."
        } else {
            Write-Host "Warning: Ollama server job started, but port 11434 is not in use. Check logs at $logFile."
        }
    } else {
        Write-Host "Error: Ollama server job failed to start. State: $($job.State)"
        Write-Host "Check logs at $logFile for details."
        exit 1
    }
} catch {
    Write-Host "Error starting Ollama server: $($_.Exception.Message)"
    exit 1
}
