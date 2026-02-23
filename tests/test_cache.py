"""Tests for cache utilities."""

import pytest

from utils.cache import CACHE_DIR, calculate_file_hash, clear_cache


class TestCalculateFileHash:
    """Test file hash calculation."""

    def test_calculate_file_hash_basic(self, tmp_path):
        """Test calculating hash of a simple file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash_result = calculate_file_hash(str(test_file))

        # SHA-256 of "Hello, World!"
        expected_hash = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        assert hash_result == expected_hash

    def test_calculate_file_hash_empty_file(self, tmp_path):
        """Test calculating hash of an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        hash_result = calculate_file_hash(str(test_file))

        # SHA-256 of empty string
        expected_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert hash_result == expected_hash

    def test_calculate_file_hash_binary_file(self, tmp_path):
        """Test calculating hash of a binary file."""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")

        hash_result = calculate_file_hash(str(test_file))

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA-256 produces 64 hex characters

    def test_calculate_file_hash_large_file(self, tmp_path):
        """Test calculating hash of a large file (tests chunking)."""
        test_file = tmp_path / "large.bin"
        # Create a file larger than the chunk size (8192 bytes)
        large_data = b"A" * 10000
        test_file.write_bytes(large_data)

        hash_result = calculate_file_hash(str(test_file))

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_calculate_file_hash_same_content_same_hash(self, tmp_path):
        """Test that identical files produce the same hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        content = "Same content"
        file1.write_text(content)
        file2.write_text(content)

        hash1 = calculate_file_hash(str(file1))
        hash2 = calculate_file_hash(str(file2))

        assert hash1 == hash2

    def test_calculate_file_hash_different_content(self, tmp_path):
        """Test that different files produce different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("Content A")
        file2.write_text("Content B")

        hash1 = calculate_file_hash(str(file1))
        hash2 = calculate_file_hash(str(file2))

        assert hash1 != hash2

    def test_calculate_file_hash_nonexistent_file(self):
        """Test that calculating hash of nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            calculate_file_hash("/nonexistent/file.txt")


class TestClearCache:
    """Test cache clearing functionality."""

    def test_clear_cache_when_exists(self, tmp_path, monkeypatch):
        """Test clearing cache when directory exists."""
        # Mock CACHE_DIR to use temp directory
        mock_cache_dir = tmp_path / "cache"
        mock_cache_dir.mkdir()

        # Create some files in the cache
        (mock_cache_dir / "file1.txt").write_text("data1")
        (mock_cache_dir / "file2.txt").write_text("data2")

        # Patch CACHE_DIR
        monkeypatch.setattr("utils.cache.CACHE_DIR", str(mock_cache_dir))

        # Clear cache
        clear_cache()

        # Verify directory was removed
        assert not mock_cache_dir.exists()

    def test_clear_cache_when_not_exists(self, tmp_path, monkeypatch):
        """Test clearing cache when directory doesn't exist."""
        mock_cache_dir = tmp_path / "nonexistent_cache"

        # Patch CACHE_DIR
        monkeypatch.setattr("utils.cache.CACHE_DIR", str(mock_cache_dir))

        # Should not raise error
        clear_cache()

        # Directory should still not exist
        assert not mock_cache_dir.exists()

    def test_clear_cache_with_subdirectories(self, tmp_path, monkeypatch):
        """Test clearing cache with nested subdirectories."""
        mock_cache_dir = tmp_path / "cache"
        mock_cache_dir.mkdir()

        # Create nested structure
        subdir = mock_cache_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested_file.txt").write_text("nested data")
        (mock_cache_dir / "root_file.txt").write_text("root data")

        # Patch CACHE_DIR
        monkeypatch.setattr("utils.cache.CACHE_DIR", str(mock_cache_dir))

        # Clear cache
        clear_cache()

        # Verify entire tree was removed
        assert not mock_cache_dir.exists()
        assert not subdir.exists()


class TestCacheDir:
    """Test CACHE_DIR constant."""

    def test_cache_dir_is_string(self):
        """Test that CACHE_DIR is a string."""
        assert isinstance(CACHE_DIR, str)

    def test_cache_dir_contains_appname(self):
        """Test that CACHE_DIR contains the app name."""
        assert "expressive" in CACHE_DIR.lower()
