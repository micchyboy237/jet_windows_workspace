import shutil
import os
import glob
import time
import sys
import threading

from typing import Optional
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from datetime import datetime

from huggingface_hub import HfApi, GitCommitInfo
import logging
import subprocess
from pathlib import Path
from typing import List

HUGGINGFACE_BASE_DIR = os.path.expanduser("~/.cache/huggingface")
MODELS_CACHE_DIR = os.path.join(HUGGINGFACE_BASE_DIR, "hub")
XET_CACHE_DIR = os.path.join(HUGGINGFACE_BASE_DIR, "xet")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_shell_command(command: List[str], description: str) -> Optional[str]:
    """Execute a shell command and log the result."""
    try:
        logger.debug(f"Executing {description}: {' '.join(command)}")
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        logger.debug(f"{description} output: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed: {e.stderr}")
        raise


def download_onnx_model(
    repo_id: str,
    cache_dir: str = MODELS_CACHE_DIR,
    model_file: str = "onnx/model_qint8_arm64.onnx",
    target_file: str = "onnx/model.onnx",
) -> None:
    """
    Download model files and execute post-download shell commands.

    Args:
        repo_id: Repository ID for Hugging Face model.
        cache_dir: Directory to cache downloaded files.
        model_file: Source model file path relative to snapshot directory.
        target_file: Target model file path relative to snapshot directory.
    """
    snapshot_dir = Path(cache_dir) / \
        f"models--{repo_id.replace('/', '--')}" / "snapshots"
    # Dynamically retrieve the latest snapshot hash for the repo
    api = HfApi()
    snapshots: List[GitCommitInfo] = api.list_repo_commits(repo_id)
    if not snapshots:
        raise RuntimeError(f"No snapshots found for repo {repo_id}")
    snapshot_hash = snapshots[0].commit_id
    snapshot_path = snapshot_dir / snapshot_hash

    # Check for any other folders under snapshot_dir and remove if not the current snapshot_hash
    if snapshot_dir.exists() and snapshot_dir.is_dir():
        for folder in snapshot_dir.iterdir():
            if folder.is_dir() and folder.name != snapshot_hash:
                logger.info(f"Removing old snapshot directory: {folder}")
                try:
                    import shutil
                    shutil.rmtree(folder)
                except Exception as e:
                    logger.error(f"Failed to remove {folder}: {e}")

    # Verify available files in the repository and select model file by priority
    logger.info(
        f"Checking available files in repo {repo_id} for snapshot {snapshot_hash}")
    try:
        repo_files = api.list_repo_files(
            repo_id=repo_id, revision=snapshot_hash)
        logger.debug(f"Available files: {repo_files}")

        # Define priority order for model files
        model_file_candidates = [
            "onnx/model_qint8_arm64.onnx",
            "onnx/model_quantized.onnx",
            "onnx/model.onnx",
            "model.onnx"
        ]
        selected_model_file = None
        for candidate in model_file_candidates:
            if candidate in repo_files:
                selected_model_file = candidate
                break

        if selected_model_file is None:
            logger.error(
                f"No suitable model file found in repository {repo_id} for snapshot {snapshot_hash}")
            raise FileNotFoundError(
                f"No suitable model file found in repository {repo_id} snapshot {snapshot_hash}.\nExpected one of {model_file_candidates}, but available files: {repo_files}"
            )
        logger.info(f"Selected model file: {selected_model_file}")
    except Exception as e:
        logger.error(f"Failed to list repository files: {str(e)}")
        raise

    source_path = snapshot_path / selected_model_file
    target_path = snapshot_path / target_file

    # Download only the selected model file
    logger.info(
        "Downloading file %s from repo id: %s...",
        source_path,
        selected_model_file
    )

    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            allow_patterns=[selected_model_file],
            ignore_patterns=[
                "onnx/model_O1.onnx",
                "onnx/model_O2.onnx",
                "onnx/model_O3.onnx",
                "onnx/model_O4.onnx",
                "onnx/model_qint8_avx512.onnx",
                "onnx/model_qint8_avx512_vnni.onnx",
                "onnx/model_quint8_avx2.onnx",
            ],
            local_dir_use_symlinks=False,
            force_download=True,
        )
        logger.info("Download completed")
    except Exception as e:
        logger.error(f"Download failed for {selected_model_file}: {str(e)}")
        raise

    # Verify that the source model file exists
    if not source_path.exists():
        logger.error(
            f"Model file {source_path} does not exist after download.")
        raise FileNotFoundError(
            f"Model file {source_path} not found in snapshot {snapshot_hash}")

    # Execute shell commands
    try:
        # ls -l <source_path>
        run_shell_command(
            ["ls", "-l", str(source_path)],
            f"Listing file {source_path}",
        )

        # Check if target_path exists and log
        if target_path.exists():
            logger.warning(
                f"Target file {target_path} already exists. Overwriting with cp -f.")

        # Ensure target directory exists before copying
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        # cp -f -L <source_path> <target_path>
        run_shell_command(
            ["cp", "-f", "-L", str(source_path), str(target_path)],
            f"Copying {source_path} to {target_path}",
        )

        # rm <source_path>
        run_shell_command(
            ["rm", str(source_path)],
            f"Removing {source_path}",
        )

        # Remove all symlinks in onnx folder except model.onnx and their referenced files
        onnx_dir = snapshot_path / "onnx"
        if onnx_dir.exists() and onnx_dir.is_dir():
            for item in onnx_dir.iterdir():
                if item.name != "model.onnx" and item.is_symlink():
                    try:
                        # Get the real path of the symlink
                        real_path = item.resolve()
                        logger.info(
                            f"Removing symlink {item} pointing to {real_path}")
                        item.unlink()  # Remove the symlink
                        if real_path.exists():
                            logger.info(
                                f"Removing referenced file {real_path}")
                            real_path.unlink()  # Remove the referenced file
                    except Exception as e:
                        logger.error(
                            f"Failed to remove symlink {item} or its target: {e}")

        # ls -l <onnx_dir> only if onnx_dir exists
        if onnx_dir.exists() and onnx_dir.is_dir():
            run_shell_command(
                ["ls", "-l", str(onnx_dir)],
                f"Listing directory {onnx_dir}",
            )
        else:
            logger.info(
                "onnx/ directory does not exist, skipping ls -l command")

        # du -sh <target_path>
        run_shell_command(
            ["du", "-sh", str(target_path)],
            f"Checking size of {target_path}",
        )

        logger.info("All shell commands executed successfully")
    except Exception as e:
        logger.error(f"Shell command execution failed: {str(e)}")
        raise



def has_onnx_model_in_repo(repo_id: str, token: Optional[str] = None) -> bool:
    """
    Check if any ONNX model (standard model.onnx, model_*_arm64.onnx, or model*quantized*onnx) exists in a Hugging Face model repository.
    Checks the local cache first, then falls back to the remote repository if no local models are found.

    Args:
        repo_id (str): The ID of the repository (e.g., "sentence-transformers/all-MiniLM-L6-v2").
        token (Optional[str]): Hugging Face API token for private repositories.

    Returns:
        bool: True if a standard, ARM64, or quantized ONNX model is found, False otherwise.
    """

    try:
        logger.info(f"Checking for ONNX models in repository: {repo_id}")

        # Check local cache first
        local_onnx_paths = get_onnx_model_paths(
            repo_id, cache_dir=MODELS_CACHE_DIR, token=token)
        if local_onnx_paths:
            logger.info(
                f"ONNX model(s) found in local cache for {repo_id}: {local_onnx_paths}")
            return True

        # Fall back to remote repository check
        logger.info(
            f"No ONNX models found in local cache, checking remote repository: {repo_id}")
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo_id, token=token)
        logger.debug(f"Files found in {repo_id}: {repo_files}")
        has_onnx = (
            "model.onnx" in repo_files or
            any(file.startswith("model_") and file.endswith("_arm64.onnx") for file in repo_files) or
            any("quantized" in file and file.endswith(".onnx")
                for file in repo_files)
        )
        logger.info(
            f"ONNX model (standard, ARM64, or quantized) found in {repo_id}: {has_onnx}")
        return has_onnx
    except Exception as e:
        logger.error(
            f"Error checking ONNX models in repository {repo_id}: {str(e)}")
        return False


def get_onnx_model_paths(repo_id: str, cache_dir: str = MODELS_CACHE_DIR, token: Optional[str] = None) -> List[str]:
    """
    Retrieve a list of ONNX model file paths (standard, ARM64, or quantized) in the local Hugging Face cache for a repository.

    Args:
        repo_id (str): The ID of the repository (e.g., "sentence-transformers/all-MiniLM-L6-v2").
        token (Optional[str]): Hugging Face API token (unused for local checks but included for consistency).

    Returns:
        List[str]: List of absolute ONNX model file paths found in the local cache.
    """
    try:
        logger.info(
            f"Retrieving local ONNX model paths for repository: {repo_id}")
        # Convert repo_id to cache folder name (e.g., "sentence-transformers/all-MiniLM-L6-v2" to cache folder)
        repo_folder_name = repo_id.replace("/", "--")
        repo_path = os.path.join(cache_dir, f"models--{repo_folder_name}")

        # Check if the repo exists in the local cache
        if not os.path.exists(repo_path):
            logger.warning(
                f"Repository {repo_id} not found in local cache at {repo_path}")
            return []

        # Walk through the repo directory to find ONNX files
        onnx_paths = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if (
                    file == "model.onnx" or
                    (file.startswith("model_") and file.endswith("_arm64.onnx")) or
                    ("quantized" in file and file.endswith(".onnx"))
                ):
                    full_path = os.path.join(root, file)
                    onnx_paths.append(full_path)

        logger.info(
            f"Found {len(onnx_paths)} ONNX model paths for {repo_id}: {onnx_paths}")
        return sorted(onnx_paths)
    except Exception as e:
        logger.error(
            f"Error retrieving local ONNX model paths for repository {repo_id}: {str(e)}")
        return []

class ProgressBar:
    """Custom progress bar implementation mimicking tqdm behavior."""

    _lock = threading.RLock()  # class-level lock

    @classmethod
    def get_lock(cls):
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock

    def __init__(self, iterable=None, total: Optional[int] = None, desc: str = "",
                 unit: str = "it", unit_scale: bool = False, **kwargs):
        self.iterable = iterable
        self.total = total or (len(iterable) if iterable is not None else None)
        self.unit = unit
        self.unit_scale = unit_scale
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
        self._last_update = 0
        self._width = 50  # Width of the progress bar
        self._iterator = iter(iterable) if iterable is not None else None

    def __iter__(self):
        """Make ProgressBar iterable."""
        return self

    def __next__(self):
        """Iterate over the underlying iterable and update progress."""
        if self._iterator is None:
            raise StopIteration
        try:
            item = next(self._iterator)
            self.update(1)
            return item
        except StopIteration:
            self.close()
            raise

    def update(self, n: int) -> None:
        """Update progress bar by n units."""
        self.current += n
        self._refresh()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the progress bar and print newline."""
        print("", flush=True)

    def _refresh(self):
        """Refresh the progress bar display."""
        if not sys.stdout.isatty():
            return

        if self.total is None:
            # Simple counter for unknown total
            print(f"\r{self.desc}: {self._format_size(self.current)}",
                  end="", flush=True)
            return

        # Calculate progress
        progress = min(self.current / self.total, 1.0)
        filled = int(self._width * progress)
        bar = "█" * filled + "-" * (self._width - filled)

        # Format size and speed
        size_str = self._format_size(self.current)
        speed = self.current / \
            max((datetime.now() - self.start_time).total_seconds(), 0.001)
        speed_str = self._format_size(speed) + "/s"

        # Update display
        percent = progress * 100
        print(
            f"\r{self.desc}: |{bar}| {percent:.1f}% {size_str} {speed_str}", end="", flush=True)

    def _format_size(self, size: float) -> str:
        """Format size with appropriate units."""
        if not self.unit_scale:
            return f"{size:.1f}{self.unit}"

        for unit in ["", "K", "M", "G", "T"]:
            if size < 1000:
                return f"{size:.1f}{unit}{self.unit}"
            size /= 1000
        return f"{size:.1f}P{self.unit}"


def remove_cache_locks(cache_dir: str = MODELS_CACHE_DIR, max_attempts: int = 5, wait_interval: float = 0.1) -> None:
    """
    Remove all lock files from the specified cache directory with retries.

    Args:
        cache_dir (str): Path to the cache directory
        max_attempts (int): Maximum number of attempts to remove lock files
        wait_interval (float): Wait time between retry attempts in seconds
    """
    try:
        lock_pattern = os.path.join(cache_dir, "**", "*.lock")
        for attempt in range(1, max_attempts + 1):
            lock_files = glob.glob(lock_pattern, recursive=True)
            if not lock_files:
                logger.debug("No lock files found in cache directory")
                return
            for lock_file in lock_files:
                try:
                    os.remove(lock_file)
                    logger.debug(f"Removed lock file: {lock_file}")
                except OSError as e:
                    logger.warning(
                        f"Attempt {attempt}: Failed to remove lock file {lock_file}: {str(e)}")
            if lock_files and attempt < max_attempts:
                time.sleep(wait_interval)
        if glob.glob(lock_pattern, recursive=True):
            logger.warning(
                "Some lock files could not be removed after maximum attempts")
    except Exception as e:
        logger.error(f"Error while removing lock files: {str(e)}")
        raise


def remove_download_cache() -> None:
    shutil.rmtree(XET_CACHE_DIR, ignore_errors=True)
    remove_cache_locks()


def _has_safetensors_in_repo(repo_id: str) -> bool:
    """
    Quick check if the repository contains any safetensors file.
    Uses the lightweight `list_repo_files` API – no download required.
    """
    from huggingface_hub import list_repo_files

    try:
        files = list_repo_files(repo_id=repo_id, repo_type="model")
        return any(f.endswith(".safetensors") for f in files)
    except Exception as e:
        logger.warning(f"Could not list files for {repo_id}: {e}")
        return False


def download_hf_model(
    repo_id: str,
    cache_dir: str = MODELS_CACHE_DIR,
    timeout: float = 300.0,
    clean_cache: bool = False,
) -> None:
    """
    Download a model from Hugging Face Hub.
    - First tries to download only safetensors + essential files.
    - If no safetensors file exists in the repo → automatically downloads the *.bin files instead.
    """
    model_path = repo_id

    if clean_cache:
        logger.info(f"Removing lock files from cache directory: {cache_dir}")
        remove_download_cache()

    # Resolve repo_id string once
    repo_id_str = str(model_path)

    # Determine which patterns to allow
    has_st = _has_safetensors_in_repo(repo_id_str)

    if has_st:
        # Prefer safetensors – ignore the old .bin weights
        allow_patterns = None
        ignore_patterns = [
            "*.bin",
            "*.h5",
            "*.msgpack",
            "*.onnx",
            "onnx/*.onnx",
            "onnx/*/*.onnx",
            "openvino/*",
        ]
        logger.info(f"Repository {repo_id_str} contains safetensors → ignoring *.bin files")
    else:
        # No safetensors → download the legacy .bin weights (and everything else except ONNX/OpenVINO)
        allow_patterns = None
        ignore_patterns = [
            "*.onnx",
            "onnx/*.onnx",
            "onnx/*/*.onnx",
            "openvino/*",
        ]
        logger.info(f"No safetensors found in {repo_id_str} → downloading *.bin weights")

    try:
        snapshot_download(
            repo_id=repo_id_str,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            force_download=False,
            etag_timeout=20.0,
            local_dir_use_symlinks="auto",
            max_workers=4,
            resume_download=True,
            tqdm_class=ProgressBar,
        )
    except HfHubHTTPError as e:
        logger.error(f"Failed to download model from {repo_id_str}: {str(e)}")
        raise
    except TimeoutError:
        logger.error(f"Download timed out after {timeout} seconds for {repo_id_str}")
        raise


def download_hf_space(
    repo_id: str,
    cache_dir: str = MODELS_CACHE_DIR,
    allow_patterns: List[str] = ["ckpt/*"],
    ignore_patterns: Optional[List[str]] = None,
    clean_cache: bool = False,
    force_download: bool = False,
) -> None:
    """
    Download files from a Hugging Face Space repository.
    Uses snapshot_download with repo_type="space".

    Args:
        repo_id: Space repository ID (e.g., "litagin/Japanese-Ero-Voice-Classifier").
        cache_dir: Directory to cache downloaded files.
        allow_patterns: List of patterns to include (e.g., ["ckpt/*", "app.py"]).
        ignore_patterns: List of patterns to exclude.
        clean_cache: Whether to remove lock files and XET cache before starting.
        force_download: Force re-download even if files exist locally.
    """
    if clean_cache:
        remove_download_cache()

    logger.info(f"Starting download of Space: {repo_id}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="space",  # Required for Spaces
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            force_download=force_download,
            resume_download=True,
            local_dir_use_symlinks=False,
            max_workers=4,
            etag_timeout=20.0,
            tqdm_class=ProgressBar,
        )
        logger.info(f"Space {repo_id} downloaded successfully")
    except HfHubHTTPError as e:
        logger.error(f"HTTP error downloading Space {repo_id}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to download Space {repo_id}: {str(e)}")
        raise


if __name__ == "__main__":
    repo_id = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
    cache_dir = MODELS_CACHE_DIR
    clean_cache = False

    logger.info(f"Downloading files from repo id: {repo_id}...")

    try:
        download_hf_model(repo_id, clean_cache=clean_cache)

        if has_onnx_model_in_repo(repo_id):
            download_onnx_model(repo_id)

        # Do not clean cache after successful download unless explicitly requested
        # remove_download_cache()

        logger.info("Download completed")
    except Exception as e:
        logger.info(f"Downloading files from repo id (space): {repo_id}...")

        try:
            download_hf_space(repo_id, clean_cache=clean_cache)
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise
