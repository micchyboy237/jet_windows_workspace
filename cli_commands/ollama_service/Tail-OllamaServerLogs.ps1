# Get-Content -Path "$env:USERPROFILE\AppData\Local\Ollama\server.log" -Tail 50 -Wait
Get-Content -Path "$env:USERPROFILE\ollama_logs\server.log" -Tail 50 -Wait
