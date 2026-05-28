# Add-User-Path.ps1
# Set the user environment variable

# Local
# [Environment]::SetEnvironmentVariable('LLAMA_CPP_LLM_URL', 'http://127.0.0.1:8080', 'User')

# Mac Air M1
[Environment]::SetEnvironmentVariable('LLAMA_CPP_LLM_URL', 'http://192.168.68.40:8080', 'User')
