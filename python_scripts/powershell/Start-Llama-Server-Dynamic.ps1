param(
    [Alias('p')]
    [int]$Port = 8080,

    [Alias('m')]
    [int]$MaxModels = 3
)
& llama-server --models-preset models.ini --models-max $MaxModels --host 0.0.0.0 --port $Port
