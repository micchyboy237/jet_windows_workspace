Get-ChildItem -Directory |
Sort-Object LastWriteTime -Descending |
ForEach-Object {
  $size = (Get-ChildItem $_ -Recurse -Force -File | Measure-Object -Property Length -Sum).Sum
  $sizeStr = if ($size -ge 1GB) { "{0:N2}G" -f ($size/1GB) }
             elseif ($size -ge 1MB) { "{0:N0}M" -f ($size/1MB) }
             else { "{0:N0}K" -f ($size/1KB) }
  "{0,8} {1} {2}" -f $sizeStr, $_.LastWriteTime.ToString("MMM dd HH:mm"), $_.Name
}
