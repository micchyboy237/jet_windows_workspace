# Get Python processes with their parent process
Get-WmiObject Win32_Process -Filter "Name='python.exe'" | 
  Select-Object ProcessId, CommandLine, ParentProcessId |
  Format-Table -AutoSize

# Kill by specific PID (replace 1234 with the actual PID)
taskkill /F /PID 1234



Get-Process python | Stop-Process -Force -PassThru | 
  ForEach-Object { 
    Get-Process -Id $_.ParentProcessId -ErrorAction SilentlyContinue | Stop-Process -Force 
  }
