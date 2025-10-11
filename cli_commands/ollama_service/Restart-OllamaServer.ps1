# Define log file paths
$logDir = "C:\Users\druiv\ollama_logs"
$stdoutLogFile = Join-Path $logDir "server.log"
$stderrLogFile = Join-Path $logDir "error.log"

# Create log directory if it doesn't exist
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# Define environment variables for Ollama
$ollamaEnv = @{
    "OLLAMA_CONTEXT_LENGTH" = "4096"
    "OLLAMA_DEBUG" = "1"
    "OLLAMA_FLASH_ATTENTION" = "1"
    "OLLAMA_HOST" = "0.0.0.0:11434"
    "OLLAMA_KEEP_ALIVE" = "300"
    "OLLAMA_KV_CACHE_TYPE" = "q8_0"
    "OLLAMA_MAX_LOADED_MODELS" = "1"
    "OLLAMA_MAX_QUEUE" = "64"
    "OLLAMA_NUM_PARALLEL" = "1"
}

# Stop the Ollama process running on port 11434
$process = Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -First 1
if ($process -and $process -ne 0) {
    try {
        $processInfo = Get-Process -Id $process -ErrorAction Stop
        Stop-Process -Id $process -Force -ErrorAction Stop
        Write-Host "Ollama process (ID: $process) on port 11434 terminated."
    } catch {
        Write-Host "Failed to stop Ollama process (ID: $process): $_"
    }
} else {
    Write-Host "No valid Ollama process found on port 11434."
}

# Store original environment variables to restore later
$originalEnv = @{}
foreach ($key in $ollamaEnv.Keys) {
    $originalEnv[$key] = [System.Environment]::GetEnvironmentVariable($key, [System.EnvironmentVariableTarget]::Process)
}

# Set environment variables for the Ollama process
foreach ($key in $ollamaEnv.Keys) {
    [System.Environment]::SetEnvironmentVariable($key, $ollamaEnv[$key], [System.EnvironmentVariableTarget]::Process)
}

# Restart Ollama serve with log redirection
Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow -RedirectStandardOutput $stdoutLogFile -RedirectStandardError $stderrLogFile

# Restore original environment variables
foreach ($key in $ollamaEnv.Keys) {
    if ($null -eq $originalEnv[$key]) {
        [System.Environment]::SetEnvironmentVariable($key, $null, [System.EnvironmentVariableTarget]::Process)
    } else {
        [System.Environment]::SetEnvironmentVariable($key, $originalEnv[$key], [System.EnvironmentVariableTarget]::Process)
    }
}

Write-Host "Ollama serve restarted. Logs are being written to $stdoutLogFile (stdout) and $stderrLogFile (stderr)."
