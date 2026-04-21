# Set to $PROFILE 
# Skip heavy profile stuff for SSH / non-interactive / SCP/SFTP sessions
if ($env:SSH_CLIENT -or $env:SSH_TTY -or $env:ssh_session -or [Console]::IsOutputRedirected) {
    # Minimal or no customizations here
    return   # or continue with only essential code
}

# Your normal profile code starts below...
Write-Host "Welcome to PowerShell 7!" -ForegroundColor Green

# Activate virtual env
. C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\powershell\Activate-Venv.ps1
