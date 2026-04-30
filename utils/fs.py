import os
import shutil
import hashlib
from pathlib import Path

from platformdirs import user_cache_dir, user_log_dir, user_runtime_dir


APP_CACHE_DIR = user_cache_dir(appname="expressive", appauthor="newcomer00")
APP_LOG_DIR = user_log_dir(appname="expressive", appauthor="newcomer00")
APP_RUNTIME_DIR = user_runtime_dir(appname="expressive", appauthor="newcomer00")

APP_CACHE_PATH = Path(APP_CACHE_DIR)
APP_LOG_PATH = Path(APP_LOG_DIR)
APP_RUNTIME_PATH = Path(APP_RUNTIME_DIR)


def calculate_file_hash(file_path):
    """Calculate the SHA-256 hash of a file.

    This is useful for caching, ensuring that identical files are recognized.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: 16-character SHA-256 hash of the file contents.
    """
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()[:16]


def calculate_args_hash(*args, **kwargs):
    """Calculate a short hash of the given arguments for use in cache keys.

    Args:
        *args: Positional values to hash.
        **kwargs: Keyword values to hash.

    Returns:
        str: 16-character hash of the combined arguments.
    """
    import joblib
    return joblib.hash((args, sorted(kwargs.items())))[:16]


def clear_cache():
    """Clear the cache directory.

    Removes all cached pitch extraction data.
    """
    if os.path.exists(APP_CACHE_DIR):
        shutil.rmtree(APP_CACHE_DIR)
