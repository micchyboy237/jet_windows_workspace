<#
.SYNOPSIS
    Permanently adds a directory to the system or user PATH environment variable.

.DESCRIPTION
    Idempotent script – safe to run multiple times.
    Supports both Machine (all users) and User scope.
    Works with PowerShell 5.1+ and PowerShell 7+.

.PARAMETER PathToAdd
    The folder to add to PATH (must be a full path).

.PARAMETER Scope
    "Machine" (requires admin) or "User" (default, no elevation needed).

.EXAMPLE
    .\Add-SystemPath.ps1 -PathToAdd "C:\redis"
    .\Add-SystemPath.ps1 -PathToAdd "C:\mytools\bin" -Scope Machine
#>

[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory, Position=0)]
    [ValidateNotNullOrEmpty()]
    [string]$PathToAdd,

    [ValidateSet('User', 'Machine')]
    [string]$Scope = 'User'
)

# Resolve full path and normalize trailing slash
$PathToAdd = (Resolve-Path -Path $PathToAdd -ErrorAction Stop).Path.TrimEnd('\')

# Determine target (Machine needs elevation)
$target = if ($Scope -eq 'Machine') { 'Machine' } else { 'User' }

# Get current PATH
$currentPath = [Environment]::GetEnvironmentVariable('PATH', $target) ?? ''

# Split into array, trim whitespace, remove empty entries
$paths = $currentPath -split ';' | Where-Object { $_ } | ForEach-Object { $_.Trim() }

if ($paths -contains $PathToAdd) {
    Write-Host "'$PathToAdd' is already in $Scope PATH." -ForegroundColor Green
    return
}

if ($PSCmdlet.ShouldProcess("$Scope PATH", "Add '$PathToAdd'")) {
    # Append and rebuild PATH string
    $newPath = ($paths + $PathToAdd) -join ';'

    # Persist permanently
    [Environment]::SetEnvironmentVariable('PATH', $newPath, $target)

    # Also update current session
    $env:PATH = "$newPath;$env:PATH"  # prepend for immediate use

    Write-Host "Successfully added '$PathToAdd' to $Scope PATH (permanent)." -ForegroundColor Green
}
