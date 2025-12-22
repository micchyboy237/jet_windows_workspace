from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError


class ModelDownloader:
    """Handles downloading .gguf models from Hugging Face to the default cache."""

    def __init__(self, repo_id: str):
        """
        Initialize the downloader with repository details.

        Args:
            repo_id: Hugging Face repository ID (e.g., 'user/model-name').
        """
        self.repo_id = repo_id

    def download_gguf_model(self, filename: str, cache_dir: Optional[str] = None) -> Path:
        """
        Download a .gguf model file from the specified repository to the HF cache.

        Args:
            filename: Name of the .gguf file to download.
            cache_dir: Optional custom cache directory (defaults to HF cache).

        Returns:
            Path to the downloaded file.

        Raises:
            ValueError: If filename doesn't end with .gguf.
            HfHubHTTPError: If download fails due to network or repo issues.
        """
        if not filename.endswith(".gguf"):
            raise ValueError("Filename must end with .gguf")

        try:
            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                cache_dir=cache_dir,
                local_dir_use_symlinks=False
            )
            return Path(file_path)
        except HfHubHTTPError as e:
            raise HfHubHTTPError(
                f"Failed to download {filename}: {str(e)}") from e


def download_model(
    repo_id: str, filename: str, cache_dir: Optional[str] = None
) -> str:
    """
    Convenience function to download a .gguf model to the HF cache.

    Args:
        repo_id: Hugging Face repository ID.
        filename: Name of the .gguf file.
        cache_dir: Optional custom cache directory.

    Returns:
        Path to the downloaded file as a string.
    """
    downloader = ModelDownloader(repo_id)
    return str(downloader.download_gguf_model(filename, cache_dir))


if __name__ == "__main__":
    # Example usage
    try:
        repo_id = "RichardErkhov/webbigdata_-_ALMA-7B-Ja-V2-gguf"  # Example repo
        filename = "ALMA-7B-Ja-V2.Q4_K_M.gguf"  # Example .gguf file

        file_path = download_model(repo_id, filename)
        print(f"Model downloaded to: {file_path}")
    except (ValueError, HfHubHTTPError) as e:
        print(f"Error: {e}")
