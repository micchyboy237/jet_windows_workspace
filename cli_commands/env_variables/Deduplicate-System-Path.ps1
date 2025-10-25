# Deduplicate-System-Path.ps1
# Get the current system PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

# Split the PATH into an array and filter out non-existent paths and duplicates
$pathArray = $currentPath -split ';' | Where-Object {
    $_ -and (Test-Path $_ -PathType Container)
} | Select-Object -Unique

# Reconstruct the PATH
$cleanPath = $pathArray -join ';'

# Update the system PATH
[Environment]::SetEnvironmentVariable("Path", $cleanPath, "Machine")
Write-Host "PATH updated successfully. Non-existent paths and duplicates removed."

# Verify the updated PATH
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine")
$env:Path -split ';'
