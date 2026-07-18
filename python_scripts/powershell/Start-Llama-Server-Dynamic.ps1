param(
    [Alias('p')]
    [int]$Port = 8080
)

& llama-server --models-preset models.ini --models-max 1 --host 0.0.0.0 --port $Port
