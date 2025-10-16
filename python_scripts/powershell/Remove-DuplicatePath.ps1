function Remove-DuplicatePath {
    <#
    .SYNOPSIS
        Deduplicates entries in the system PATH environment variable and applies changes immediately.
    .DESCRIPTION
        Retrieves the system PATH, removes duplicate entries while preserving order, and updates the PATH.
        Notifies running processes of the environment change.
    .EXAMPLE
        Remove-DuplicatePath
        Deduplicates the system PATH and applies changes.
    #>
    [CmdletBinding()]
    param()

    # Get the current system PATH
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    Write-Debug "Current system PATH: $currentPath"
    
    # Split PATH into an array, remove duplicates while preserving order, and join back
    $pathArray = $currentPath -split ";" | Select-Object -Unique | Where-Object { $_ -ne "" }
    $newPath = $pathArray -join ";"
    Write-Debug "Deduplicated PATH: $newPath"
    Write-Debug "Number of entries before: $($currentPath -split ';').Count, after: $pathArray.Count"

    # Update the system PATH if changes were made
    if ($newPath -ne $currentPath) {
        try {
            [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
            Write-Output "System PATH updated successfully. Duplicates removed."
            Write-Debug "System PATH set to: $newPath"

            # Broadcast WM_SETTINGCHANGE to apply changes immediately
            $hwnd = [IntPtr]::Zero
            $result = [Win32]::SendMessageTimeout(
                $hwnd,
                0x1A, # WM_SETTINGCHANGE
                [IntPtr]::Zero,
                [System.Runtime.InteropServices.Marshal]::StringToCoTaskMemUni("Environment"),
                0x2, # SMTO_ABORTIFHUNG
                5000, # Timeout in milliseconds
                [ref][IntPtr]::Zero
            )
            if ($result -eq 0) {
                Write-Warning "Failed to broadcast PATH change to running processes. LastError: $(([System.ComponentModel.Win32Exception][System.Runtime.InteropServices.Marshal]::GetLastWin32Error()).Message)"
            } else {
                Write-Output "PATH changes broadcasted to running processes."
            }
        }
        catch {
            Write-Error "Failed to update system PATH: $_"
            Write-Debug "Exception details: $($_.Exception.GetType().FullName)"
        }
    } else {
        Write-Output "No duplicates found in system PATH. No changes made."
    }
}
