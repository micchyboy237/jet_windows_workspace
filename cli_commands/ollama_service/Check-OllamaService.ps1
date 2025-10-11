Get-Service | Where-Object { $_.Name -like "*ollama*" -or $_.DisplayName -like "*ollama*" }
