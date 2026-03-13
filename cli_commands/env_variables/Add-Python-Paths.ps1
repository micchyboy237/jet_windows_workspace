# Add-Python-Paths.ps1

$path_to_add="C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\FireRedASR2S"

[Environment]::SetEnvironmentVariable("PYTHONPATH", $path_to_add, "User")

$env:PYTHONPATH -split ';'
