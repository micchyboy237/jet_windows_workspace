# Or, for more general display of all used ports:
Get-NetTCPConnection | Select-Object LocalAddress, LocalPort, State, OwningProcess | Sort-Object LocalPort
