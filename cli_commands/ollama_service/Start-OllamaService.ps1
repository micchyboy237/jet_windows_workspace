# Start Ollama serve
Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow
Write-Host "Ollama server started."
