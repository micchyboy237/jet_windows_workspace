Write-Host "`n=== DRILLING DOWN INTO HIDDEN FOLDERS ===" -ForegroundColor Cyan
Write-Host "This will take ~30 seconds to find the top space hogs..." -ForegroundColor Yellow

$targets = @(
    @{ Path = "$env:LOCALAPPDATA"; Name = "AppData\Local" },
    @{ Path = "$env:APPDATA"; Name = "AppData\Roaming" },
    @{ Path = "C:\ProgramData"; Name = "C:\ProgramData" }
)

foreach ($t in $targets) {
    Write-Host "`n--- TOP 5 LARGEST FOLDERS IN $($t.Name) ---" -ForegroundColor Yellow
    Get-ChildItem -Path $t.Path -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        [PSCustomObject]@{ Folder = $_.Name; SizeGB = [math]::Round($size / 1GB, 2) }
    } | Sort-Object SizeGB -Descending | Select-Object -First 5 | Format-Table -AutoSize
}
Write-Host "`n=== DRILL-DOWN COMPLETE ===" -ForegroundColor Cyan