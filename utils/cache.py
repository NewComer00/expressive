import os
import shutil
import hashlib

from platformdirs import user_cache_dir


CACHE_DIR = user_cache_dir(appname="expressive", appauthor="newcomer00")


def calculate_file_hash(file_path):
    """Calculate the SHA-256 hash of a file.

    This is useful for caching, ensuring that identical files are recognized.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: SHA-256 hash of the file contents.
    """
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def clear_cache():
    """Clear the cache directory.

    Removes all cached pitch extraction data.
    """
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
