# Get-OllamaServer.ps1
# Retrieves information about the Ollama server process and job

try {
    Write-Host "Checking for Ollama server process on port 11434..."
    $connections = Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue

    if ($connections) {
        $processIds = $connections.OwningProcess | Sort-Object -Unique
        Write-Host "Found $($processIds.Count) process(es) listening on port 11434."

        foreach ($processId in $processIds) {
            $process = Get-Process -Id $processId -ErrorAction Stop
            Write-Host "
Process Details for PID ${processId}:"
            Write-Host "-----------------------------"
            Write-Host "Process Name: $($process.ProcessName)"
            Write-Host "PID: $($process.Id)"
            Write-Host "CPU Time: $($process.CPU) seconds"
            Write-Host "Memory Usage: $([math]::Round($process.WorkingSet64 / 1MB, 2)) MB"
            Write-Host "Start Time: $($process.StartTime)"
            Write-Host "Path: $($process.Path)"
        }
    } else {
        Write-Host "No process found listening on port 11434."
    }

    # Check for Ollama-related background jobs
    Write-Host "
Checking for Ollama-related background jobs..."
    $jobs = Get-Job | Where-Object { $_.Command -like "*ollama serve*" }
    if ($jobs) {
        Write-Host "Found $($jobs.Count) Ollama-related job(s)."
        foreach ($job in $jobs) {
            Write-Host "
Job Details for Job ID $($job.Id):"
            Write-Host "-----------------------------"
            Write-Host "Job Name: $($job.Name)"
            Write-Host "Job ID: $($job.Id)"
            Write-Host "State: $($job.State)"
            Write-Host "Start Time: $($job.PSBeginTime)"
            Write-Host "Command: $($job.Command)"
        }
    } else {
        Write-Host "No Ollama-related background jobs found."
    }

    # Verify port status with netstat for additional context
    Write-Host "
Verifying port 11434 status..."
    $netstatOutput = netstat -aon | findstr :11434
    if ($netstatOutput) {
        Write-Host "Port 11434 is in use:"
        Write-Host $netstatOutput
    } else {
        Write-Host "Port 11434 is not in use."
    }
} catch {
    Write-Host "Error retrieving Ollama server information: $($_.Exception.Message)"
    exit 1
}
