Get-ChildItem Env: | Where-Object { $_.Name -like "OLLAMA_*" } | Sort-Object Name
