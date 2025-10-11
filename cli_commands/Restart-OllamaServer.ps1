# Define log file path
$logDir = "C:\Users\druiv\ollama_logs"
$logFile = Join-Path $logDir "server.log"

# Create log directory if it doesn't exist
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# Stop the Ollama process running on port 11434
$process = Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($process) {
    Stop-Process -Id $process -Force
    Write-Host "Ollama process on port 11434 terminated."
} else {
    Write-Host "No Ollama process found on port 11434."
}

# Restart Ollama serve with log redirection
Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow -RedirectStandardOutput $logFile -RedirectStandardError $logFile
Write-Host "Ollama serve restarted. Logs are being written to $logFile"
