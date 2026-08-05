# Set-Command-Aliases.ps1

# Display full executable path (like Unix which)
function which {
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [switch]$All
    )

    $params = @{ Name = $Name; ErrorAction = 'SilentlyContinue' }
    if ($All) { $params['All'] = $true }

    $commands = Get-Command @params
    if (-not $commands) {
        Write-Warning "which: no $Name found"
        return
    }

    $commands.Source
}

# Create files and directories (like Unix touch)
Set-Alias -Name touch -Value New-Item

# Create single file
# touch my_new_file.txt

# Create multiple files
# touch file1.txt, file2.txt, file3.txt

# Autocreate subdirectories
# touch -Force -Path ".\deep\nested\folder\myfile.txt"