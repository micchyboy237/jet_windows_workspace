# Add-System-Path.ps1
# Define the new path to add
$newPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"

# Get the current system PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

# Check if the new path is already in the PATH
if ($currentPath -notlike "*$newPath*") {
    # Append the new path
    $updatedPath = "$currentPath;$newPath"
    [Environment]::SetEnvironmentVariable("Path", $updatedPath, "Machine")
    Write-Host "Path updated successfully."
} else {
    Write-Host "Path already exists in the system PATH."
}

# Verify the updated PATH
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine")
$env:Path -split ';'
