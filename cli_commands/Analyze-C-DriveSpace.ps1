Write-Host "`n=== C: DRIVE SPACE ANALYZER ===" -ForegroundColor Cyan

# Start the timer
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

# Get C: drive total size for percentage calculation
$cDrive = Get-CimInstance -ClassName Win32_LogicalDisk | Where-Object { $_.DeviceID -eq 'C:' }
$totalBytes = $cDrive.Size

# Function to calculate folder size quickly
function Get-FolderSize {
    param ([string]$Path)
    if (Test-Path $Path) {
        $size = (Get-ChildItem $Path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        return $size
    }
    return 0
}

# ==========================================
# 1. Scan Root of C:
# ==========================================
Write-Host "`n[1] Scanning C:\ ROOT DIRECTORIES..." -ForegroundColor Yellow
$rootFolders = Get-ChildItem -Path "C:\" -Directory -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -notin @('$Recycle.Bin', 'System Volume Information') }
$totalRoot = $rootFolders.Count
$currentIndex = 0
$results = @()

foreach ($folder in $rootFolders) {
    $currentIndex++
    $percent = [math]::Round(($currentIndex / $totalRoot) * 100)
    
    # Update Progress Bar
    Write-Progress -Activity "Phase 1: Scanning C:\ Root" -Status "Analyzing: $($folder.Name) ($currentIndex/$totalRoot)" -PercentComplete $percent -Id 1
    
    $size = Get-FolderSize -Path $folder.FullName
    $sizeGB = [math]::Round($size / 1GB, 2)
    $pct = [math]::Round(($size / $totalBytes) * 100, 1)
    $results += [PSCustomObject]@{ Folder = $folder.Name; SizeGB = $sizeGB; Percent = "$pct%" }
}
Write-Progress -Activity "Phase 1: Scanning C:\ Root" -Status "Completed" -Completed -Id 1

# ==========================================
# 2. Scan User Profile
# ==========================================
Write-Host "[2] Scanning C:\Users\$env:USERNAME DIRECTORIES..." -ForegroundColor Yellow
$userFolders = Get-ChildItem -Path "C:\Users\$env:USERNAME" -Directory -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -notin @('AppData') }
$totalUser = $userFolders.Count + 2 # +2 for Local and Roaming AppData
$currentIndex = 0
$userResults = @()

foreach ($folder in $userFolders) {
    $currentIndex++
    $percent = [math]::Round(($currentIndex / $totalUser) * 100)
    
    # Update Progress Bar
    Write-Progress -Activity "Phase 2: Scanning User Profile" -Status "Analyzing: $($folder.Name) ($currentIndex/$totalUser)" -PercentComplete $percent -Id 2
    
    $size = Get-FolderSize -Path $folder.FullName
    $sizeGB = [math]::Round($size / 1GB, 2)
    $pct = [math]::Round(($size / $totalBytes) * 100, 1)
    $userResults += [PSCustomObject]@{ Folder = $folder.Name; SizeGB = $sizeGB; Percent = "$pct%" }
}

# Scan AppData subfolders
Write-Progress -Activity "Phase 2: Scanning User Profile" -Status "Analyzing: AppData\Local ($($currentIndex+1)/$totalUser)" -PercentComplete ([math]::Round((($currentIndex+1) / $totalUser) * 100)) -Id 2
$appDataLocal = Get-FolderSize -Path "$env:LOCALAPPDATA"
$userResults += [PSCustomObject]@{ Folder = "AppData\Local"; SizeGB = [math]::Round($appDataLocal / 1GB, 2); Percent = "$([math]::Round(($appDataLocal / $totalBytes) * 100, 1))%" }

Write-Progress -Activity "Phase 2: Scanning User Profile" -Status "Analyzing: AppData\Roaming ($($currentIndex+2)/$totalUser)" -PercentComplete 100 -Id 2
$appDataRoaming = Get-FolderSize -Path "$env:APPDATA"
$userResults += [PSCustomObject]@{ Folder = "AppData\Roaming"; SizeGB = [math]::Round($appDataRoaming / 1GB, 2); Percent = "$([math]::Round(($appDataRoaming / $totalBytes) * 100, 1))%" }

Write-Progress -Activity "Phase 2: Scanning User Profile" -Status "Completed" -Completed -Id 2

# Stop the timer
$stopwatch.Stop()
$elapsedTime = $stopwatch.Elapsed.ToString('mm\:ss\.ff')

# ==========================================
# 3. Output Results
# ==========================================
Write-Host "`n=== RESULTS ===" -ForegroundColor Cyan

Write-Host "`n[1] C:\ ROOT DIRECTORIES" -ForegroundColor Yellow
$results | Sort-Object SizeGB -Descending | ForEach-Object {
    $color = if ($_.SizeGB -ge 10) { "Red" } elseif ($_.SizeGB -ge 2) { "Yellow" } else { "Gray" }
    Write-Host ("{0,-25} : {1,8} GB  ({2,5})" -f $_.Folder, $_.SizeGB, $_.Percent) -ForegroundColor $color
}

Write-Host "`n[2] C:\Users\$env:USERNAME DIRECTORIES" -ForegroundColor Yellow
$userResults | Sort-Object SizeGB -Descending | ForEach-Object {
    $color = if ($_.SizeGB -ge 10) { "Red" } elseif ($_.SizeGB -ge 2) { "Yellow" } else { "Gray" }
    Write-Host ("{0,-25} : {1,8} GB  ({2,5})" -f $_.Folder, $_.SizeGB, $_.Percent) -ForegroundColor $color
}

Write-Host "`n=== SCAN COMPLETE ===" -ForegroundColor Cyan
Write-Host "Total Scan Time: $elapsedTime (Minutes:Seconds.Milliseconds)" -ForegroundColor Magenta
Write-Host "Legend: RED = >10GB (Investigate) | YELLOW = >2GB (Check) | GRAY = Normal" -ForegroundColor Magenta