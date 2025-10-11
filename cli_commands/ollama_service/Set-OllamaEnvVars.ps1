# Set-OllamaEnvVars.ps1
# Removes all OLLAMA_* environment variables from user scope and sets specified OLLAMA variables

# Function to remove all OLLAMA_* environment variables from user scope
function Remove-OllamaEnvVars {
    Get-ChildItem Env: | Where-Object { $_.Name -like "OLLAMA_*" } | ForEach-Object {
        [System.Environment]::SetEnvironmentVariable($_.Name, $null, "User")
    }
}

# Function to set specified OLLAMA environment variables
function Set-OllamaEnvVars {
    param (
        [hashtable]$variables
    )
    foreach ($key in $variables.Keys) {
        [System.Environment]::SetEnvironmentVariable($key, $variables[$key], "User")
    }
}

# Define the OLLAMA environment variables to set
$ollamaVars = @{
    "OLLAMA_CONTEXT_LENGTH"    = "4096"
    "OLLAMA_FLASH_ATTENTION"   = "1"
    "OLLAMA_HOST"              = "0.0.0.0:11434"
    "OLLAMA_KEEP_ALIVE"        = "300"
    "OLLAMA_KV_CACHE_TYPE"     = "q8_0"
    "OLLAMA_MAX_LOADED_MODELS" = "1"
    "OLLAMA_MAX_QUEUE"         = "64"
    "OLLAMA_NUM_PARALLEL"      = "1"
}

# Main execution
try {
    # Remove existing OLLAMA_* variables
    Remove-OllamaEnvVars

    # Set new OLLAMA variables
    Set-OllamaEnvVars -variables $ollamaVars

    Write-Host "Successfully removed existing OLLAMA_* environment variables and set new values:" -ForegroundColor Green
    Get-ChildItem Env: | Where-Object { $_.Name -like "OLLAMA_*" } | Sort-Object Name | Format-Table Name, Value -AutoSize
}
catch {
    Write-Host "Error occurred: $_" -ForegroundColor Red
}
