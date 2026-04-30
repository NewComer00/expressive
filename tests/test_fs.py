"""Tests for fs utilities."""

import pytest

from utils.fs import APP_CACHE_DIR, calculate_args_hash, calculate_file_hash, clear_cache


class TestCalculateFileHash:
    """Test file hash calculation."""

    def test_calculate_file_hash_basic(self, tmp_path):
        """Test calculating hash of a simple file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash_result = calculate_file_hash(str(test_file))

        # calculate_file_hash returns only the first 16 hex characters
        expected_hash = "dffd6021bb2bd5b0"
        assert hash_result == expected_hash

    def test_calculate_file_hash_empty_file(self, tmp_path):
        """Test calculating hash of an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        hash_result = calculate_file_hash(str(test_file))

        # First 16 hex chars of SHA-256 of empty string
        expected_hash = "e3b0c44298fc1c14"
        assert hash_result == expected_hash

    def test_calculate_file_hash_returns_16_chars(self, tmp_path):
        """Hash is always truncated to 16 hex characters."""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")

        hash_result = calculate_file_hash(str(test_file))

        assert isinstance(hash_result, str)
        assert len(hash_result) == 16

    def test_calculate_file_hash_large_file(self, tmp_path):
        """Test calculating hash of a large file (tests chunking)."""
        test_file = tmp_path / "large.bin"
        # Create a file larger than the chunk size (8192 bytes)
        large_data = b"A" * 10000
        test_file.write_bytes(large_data)

        hash_result = calculate_file_hash(str(test_file))

        assert isinstance(hash_result, str)
        assert len(hash_result) == 16

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

    def test_calculate_file_hash_only_hex_chars(self, tmp_path):
        """Hash must contain only lowercase hex characters."""
        test_file = tmp_path / "hex.txt"
        test_file.write_text("hex check")
        hash_result = calculate_file_hash(str(test_file))
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_calculate_file_hash_deterministic(self, tmp_path):
        """Calling twice on the same file yields the same result."""
        test_file = tmp_path / "det.txt"
        test_file.write_text("determinism")
        assert calculate_file_hash(str(test_file)) == calculate_file_hash(str(test_file))


class TestCalculateArgsHash:
    """Test argument hash calculation."""

    def test_returns_string(self):
        result = calculate_args_hash(1, 2, key="value")
        assert isinstance(result, str)

    def test_returns_16_chars(self):
        result = calculate_args_hash("a", "b")
        assert len(result) == 16

    def test_same_args_same_hash(self):
        h1 = calculate_args_hash(1, 2, x=3)
        h2 = calculate_args_hash(1, 2, x=3)
        assert h1 == h2

    def test_different_args_different_hash(self):
        h1 = calculate_args_hash(1, 2)
        h2 = calculate_args_hash(1, 3)
        assert h1 != h2

    def test_kwargs_order_invariant(self):
        """Keyword argument order must not change the hash (sorted internally)."""
        h1 = calculate_args_hash(a=1, b=2)
        h2 = calculate_args_hash(b=2, a=1)
        assert h1 == h2

    def test_no_args_returns_string(self):
        result = calculate_args_hash()
        assert isinstance(result, str)
        assert len(result) == 16

    def test_positional_and_keyword_distinguished(self):
        """Positional and keyword args with the same value should differ."""
        h_pos = calculate_args_hash(1)
        h_kw = calculate_args_hash(x=1)
        assert h_pos != h_kw

    def test_none_args_stable(self):
        h1 = calculate_args_hash(None)
        h2 = calculate_args_hash(None)
        assert h1 == h2


class TestClearCache:
    """Test cache clearing functionality."""

    def test_clear_cache_when_exists(self, tmp_path, monkeypatch):
        """Test clearing cache when directory exists."""
        mock_cache_dir = tmp_path / "cache"
        mock_cache_dir.mkdir()

        (mock_cache_dir / "file1.txt").write_text("data1")
        (mock_cache_dir / "file2.txt").write_text("data2")

        monkeypatch.setattr("utils.fs.APP_CACHE_DIR", str(mock_cache_dir))

        clear_cache()

        assert not mock_cache_dir.exists()

    def test_clear_cache_when_not_exists(self, tmp_path, monkeypatch):
        """Test clearing cache when directory doesn't exist."""
        mock_cache_dir = tmp_path / "nonexistent_cache"

        monkeypatch.setattr("utils.fs.APP_CACHE_DIR", str(mock_cache_dir))

        clear_cache()

        assert not mock_cache_dir.exists()

    def test_clear_cache_with_subdirectories(self, tmp_path, monkeypatch):
        """Test clearing cache with nested subdirectories."""
        mock_cache_dir = tmp_path / "cache"
        mock_cache_dir.mkdir()

        subdir = mock_cache_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested_file.txt").write_text("nested data")
        (mock_cache_dir / "root_file.txt").write_text("root data")

        monkeypatch.setattr("utils.fs.APP_CACHE_DIR", str(mock_cache_dir))

        clear_cache()

        assert not mock_cache_dir.exists()
        assert not subdir.exists()


class TestCacheDir:
    """Test APP_CACHE_DIR constant."""

    def test_cache_dir_is_string(self):
        """Test that APP_CACHE_DIR is a string."""
        assert isinstance(APP_CACHE_DIR, str)

    def test_cache_dir_contains_appname(self):
        """Test that APP_CACHE_DIR contains the app name."""
        assert "expressive" in APP_CACHE_DIR.lower()
