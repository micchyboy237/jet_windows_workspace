# Just view
# pip list --not-required

# View in freeze format
# pip list --not-required --format=freeze

# Save only direct installs and show confirmation
pip list --not-required --format=freeze | Sort-Object | Out-File -Encoding utf8 requirements.txt; 
if ($?) { Write-Host "requirements.txt saved successfully at $(Resolve-Path requirements.txt)" -ForegroundColor Green } 
else { Write-Host "Failed to save requirements.txt" -ForegroundColor Red }

