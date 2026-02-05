import uvicorn

from llama_cpp_interceptors import create_app

def main() -> None:
    upstream_url = "http://127.0.0.1:8001"
    log_dir = r"C:\Users\druiv\.cache\logs\llama.cpp\interceptors\embed_logs"
    host = "0.0.0.0"
    port = 8081

    app = create_app(
        upstream_url=upstream_url,
        log_dir=log_dir,
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
