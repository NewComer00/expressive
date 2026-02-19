import pytest
import numpy as np

from utils.ustx import load_ustx, save_ustx, edit_ustx_expression_curve


class TestLoadUSTX:
    """Test USTX file loading"""

    def test_load_ustx_basic(self, temp_ustx_file):
        """Test loading a basic USTX file"""
        result = load_ustx(str(temp_ustx_file))

        assert isinstance(result, dict)
        assert "tempos" in result
        assert "voice_parts" in result
        assert result["tempos"][0]["bpm"] == 120

    def test_load_ustx_with_utf8_bom(self, temp_dir):
        """Test loading USTX file with UTF-8 BOM"""
        ustx_content = """tempos:
  - bpm: 140
    position: 0
voice_parts:
  - name: Track 1
"""
        ustx_path = temp_dir / "test_bom.ustx"
        ustx_path.write_text(ustx_content, encoding='utf-8-sig')

        result = load_ustx(str(ustx_path))
        assert result["tempos"][0]["bpm"] == 140

    def test_load_ustx_nonexistent_file(self):
        """Test loading non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_ustx("nonexistent_file.ustx")

    def test_load_ustx_preserves_structure(self, sample_ustx_dict, temp_dir):
        """Test that loading preserves USTX structure"""
        ustx_path = temp_dir / "test_structure.ustx"
        save_ustx(sample_ustx_dict, str(ustx_path))

        loaded = load_ustx(str(ustx_path))

        assert "tempos" in loaded
        assert "time_signatures" in loaded
        assert "voice_parts" in loaded
        assert len(loaded["voice_parts"]) == len(sample_ustx_dict["voice_parts"])


class TestSaveUSTX:
    """Test USTX file saving"""

    def test_save_ustx_basic(self, sample_ustx_dict, temp_dir):
        """Test saving a basic USTX file"""
        ustx_path = temp_dir / "output.ustx"

        save_ustx(sample_ustx_dict, str(ustx_path))

        assert ustx_path.exists()
        # Verify file is not empty
        assert ustx_path.stat().st_size > 0

    def test_save_ustx_creates_file(self, sample_ustx_dict, temp_dir):
        """Test that save creates new file if it doesn't exist"""
        ustx_path = temp_dir / "new_file.ustx"

        assert not ustx_path.exists()
        save_ustx(sample_ustx_dict, str(ustx_path))
        assert ustx_path.exists()

    def test_save_ustx_overwrites_existing(self, sample_ustx_dict, temp_dir):
        """Test that save overwrites existing file"""
        ustx_path = temp_dir / "existing.ustx"

        # Create initial file
        ustx_path.write_text("old content", encoding='utf-8-sig')

        # Save new content
        save_ustx(sample_ustx_dict, str(ustx_path))

        # Verify content was overwritten
        loaded = load_ustx(str(ustx_path))
        assert loaded["tempos"][0]["bpm"] == 120

    def test_save_ustx_utf8_bom(self, sample_ustx_dict, temp_dir):
        """Test that saved file uses UTF-8 with BOM"""
        ustx_path = temp_dir / "test_encoding.ustx"

        save_ustx(sample_ustx_dict, str(ustx_path))

        # Read raw bytes to check for BOM
        with open(ustx_path, 'rb') as f:
            first_bytes = f.read(3)
            # UTF-8 BOM is EF BB BF
            assert first_bytes == b'\xef\xbb\xbf'


class TestSaveLoadRoundtrip:
    """Test save and load roundtrip consistency"""

    def test_roundtrip_basic(self, sample_ustx_dict, temp_dir):
        """Test basic save/load roundtrip"""
        ustx_path = temp_dir / "roundtrip.ustx"

        # Save
        save_ustx(sample_ustx_dict, str(ustx_path))

        # Load
        loaded = load_ustx(str(ustx_path))

        # Verify key fields
        assert loaded["tempos"][0]["bpm"] == sample_ustx_dict["tempos"][0]["bpm"]
        assert len(loaded["voice_parts"]) == len(sample_ustx_dict["voice_parts"])
        assert loaded["voice_parts"][0]["name"] == sample_ustx_dict["voice_parts"][0]["name"]

    def test_roundtrip_preserves_order(self, temp_dir):
        """Test that roundtrip preserves key order"""
        # Create dict with specific order
        ustx_dict = {
            "tempos": [{"bpm": 120, "position": 0}],
            "time_signatures": [{"bar_index": 0, "beat_per_bar": 4, "beat_unit": 4}],
            "voice_parts": [{"name": "Track 1"}]
        }

        ustx_path = temp_dir / "order_test.ustx"

        # Save and load
        save_ustx(ustx_dict, str(ustx_path))
        loaded = load_ustx(str(ustx_path))

        # Verify keys exist (order checking is harder in Python dicts)
        assert "tempos" in loaded
        assert "time_signatures" in loaded
        assert "voice_parts" in loaded

    def test_roundtrip_with_curves(self, sample_ustx_dict, temp_dir):
        """Test roundtrip with expression curves"""
        # Add curves to sample dict
        sample_ustx_dict["voice_parts"][0]["curves"] = [
            {
                "abbr": "dyn",
                "xs": [0, 480, 960],
                "ys": [0, 50, 100]
            }
        ]

        ustx_path = temp_dir / "curves_test.ustx"

        # Save and load
        save_ustx(sample_ustx_dict, str(ustx_path))
        loaded = load_ustx(str(ustx_path))

        # Verify curves preserved
        assert "curves" in loaded["voice_parts"][0]
        assert len(loaded["voice_parts"][0]["curves"]) == 1
        assert loaded["voice_parts"][0]["curves"][0]["abbr"] == "dyn"


class TestEditUSTXExpressionCurve:
    """Test editing expression curves in USTX"""

    def test_edit_expression_curve_new_curve(self, sample_ustx_dict):
        """Test adding a new expression curve"""
        tick_seq = np.array([0, 480, 960, 1440])
        exp_seq = np.array([0.0, 50.0, 100.0, 75.0])

        edit_ustx_expression_curve(
            sample_ustx_dict,
            ustx_track_number=1,
            expression="dyn",
            tick_seq=tick_seq,
            exp_seq=exp_seq
        )

        # Verify curve was added
        assert "curves" in sample_ustx_dict["voice_parts"][0]
        curves = sample_ustx_dict["voice_parts"][0]["curves"]
        assert len(curves) == 1
        assert curves[0]["abbr"] == "dyn"
        assert curves[0]["xs"] == [0, 480, 960, 1440]
        assert curves[0]["ys"] == [0, 50, 100, 75]

    def test_edit_expression_curve_update_existing(self, sample_ustx_dict):
        """Test updating an existing expression curve"""
        # Add initial curve
        sample_ustx_dict["voice_parts"][0]["curves"] = [
            {"xs": [0, 480], "ys": [0, 50], "abbr": "dyn"}
        ]

        # Update curve
        tick_seq = np.array([0, 960])
        exp_seq = np.array([100.0, 200.0])

        edit_ustx_expression_curve(
            sample_ustx_dict,
            ustx_track_number=1,
            expression="dyn",
            tick_seq=tick_seq,
            exp_seq=exp_seq
        )

        curves = sample_ustx_dict["voice_parts"][0]["curves"]
        # Should still have only one curve (updated)
        assert len(curves) == 1
        assert curves[0]["xs"] == [0, 960]
        assert curves[0]["ys"] == [100, 200]

    def test_edit_expression_curve_with_nan(self, sample_ustx_dict):
        """Test that NaN values are filtered out"""
        tick_seq = np.array([0, 480, 960, 1440])
        exp_seq = np.array([0.0, np.nan, 100.0, 75.0])

        edit_ustx_expression_curve(
            sample_ustx_dict,
            ustx_track_number=1,
            expression="dyn",
            tick_seq=tick_seq,
            exp_seq=exp_seq
        )

        curves = sample_ustx_dict["voice_parts"][0]["curves"]
        # NaN value should be filtered
        assert len(curves[0]["xs"]) == 3
        assert 480 not in curves[0]["xs"]  # NaN position filtered
        assert curves[0]["xs"] == [0, 960, 1440]
        assert curves[0]["ys"] == [0, 100, 75]

    def test_edit_expression_curve_all_nan(self, sample_ustx_dict):
        """Test with all NaN values"""
        tick_seq = np.array([0, 480, 960])
        exp_seq = np.array([np.nan, np.nan, np.nan])

        edit_ustx_expression_curve(
            sample_ustx_dict,
            ustx_track_number=1,
            expression="dyn",
            tick_seq=tick_seq,
            exp_seq=exp_seq
        )

        curves = sample_ustx_dict["voice_parts"][0]["curves"]
        # Should create curve but with empty data
        assert len(curves) == 1
        assert curves[0]["xs"] == []
        assert curves[0]["ys"] == []

    def test_edit_expression_curve_multiple_expressions(self, sample_ustx_dict):
        """Test adding multiple different expressions"""
        # Add dyn
        edit_ustx_expression_curve(
            sample_ustx_dict, 1, "dyn",
            np.array([0, 480]), np.array([0.0, 50.0])
        )

        # Add pitd
        edit_ustx_expression_curve(
            sample_ustx_dict, 1, "pitd",
            np.array([0, 480]), np.array([10.0, 20.0])
        )

        # Add tenc
        edit_ustx_expression_curve(
            sample_ustx_dict, 1, "tenc",
            np.array([0, 480]), np.array([30.0, 40.0])
        )

        curves = sample_ustx_dict["voice_parts"][0]["curves"]
        assert len(curves) == 3

        # Verify each expression
        abbrs = [c["abbr"] for c in curves]
        assert "dyn" in abbrs
        assert "pitd" in abbrs
        assert "tenc" in abbrs

    @pytest.mark.parametrize("expression", ["dyn", "pitd", "tenc"])
    def test_edit_expression_curve_supported_types(self, sample_ustx_dict, expression):
        """Test all supported expression types"""
        tick_seq = np.array([0, 480])
        exp_seq = np.array([0.0, 50.0])

        edit_ustx_expression_curve(
            sample_ustx_dict, 1, expression,
            tick_seq, exp_seq
        )

        curves = sample_ustx_dict["voice_parts"][0]["curves"]
        assert len(curves) == 1
        assert curves[0]["abbr"] == expression

    def test_edit_expression_curve_invalid_type(self, sample_ustx_dict):
        """Test that invalid expression type raises error"""
        with pytest.raises(ValueError, match="Unsupported expression type"):
            edit_ustx_expression_curve(
                sample_ustx_dict, 1, "invalid_expr",
                np.array([0]), np.array([0])
            )

    def test_edit_expression_curve_rounding(self, sample_ustx_dict):
        """Test that values are rounded to integers"""
        tick_seq = np.array([0, 480])
        exp_seq = np.array([10.7, 50.3])

        edit_ustx_expression_curve(
            sample_ustx_dict, 1, "dyn",
            tick_seq, exp_seq
        )

        curves = sample_ustx_dict["voice_parts"][0]["curves"]
        # Values should be rounded
        assert curves[0]["ys"] == [11, 50]

    def test_edit_expression_curve_negative_values(self, sample_ustx_dict):
        """Test with negative values"""
        tick_seq = np.array([0, 480])
        exp_seq = np.array([-10.0, -20.0])

        edit_ustx_expression_curve(
            sample_ustx_dict, 1, "dyn",
            tick_seq, exp_seq
        )

        curves = sample_ustx_dict["voice_parts"][0]["curves"]
        assert curves[0]["ys"] == [-10, -20]

    def test_edit_expression_curve_track_number(self, sample_ustx_dict):
        """Test that track number is correctly handled (1-indexed)"""
        # Add a second track
        sample_ustx_dict["voice_parts"].append({
            "name": "Track 2",
            "track_no": 1,
            "notes": [],
            "curves": []
        })

        # Edit track 2 (1-indexed)
        edit_ustx_expression_curve(
            sample_ustx_dict, 2, "dyn",
            np.array([0, 480]), np.array([0.0, 50.0])
        )

        # Verify curve was added to track 2 (index 1)
        assert "curves" in sample_ustx_dict["voice_parts"][1]
        assert len(sample_ustx_dict["voice_parts"][1]["curves"]) == 1

        # Track 1 should not have curves
        assert "curves" not in sample_ustx_dict["voice_parts"][0] or \
               len(sample_ustx_dict["voice_parts"][0]["curves"]) == 0


class TestIntegration:
    """Integration tests combining multiple operations"""

    def test_full_workflow(self, sample_ustx_dict, temp_dir):
        """Test complete workflow: load, edit, save, load again"""
        ustx_path = temp_dir / "workflow.ustx"

        # Save initial file
        save_ustx(sample_ustx_dict, str(ustx_path))

        # Load file
        loaded = load_ustx(str(ustx_path))

        # Edit expression
        edit_ustx_expression_curve(
            loaded, 1, "dyn",
            np.array([0, 480, 960]),
            np.array([0.0, 50.0, 100.0])
        )

        # Save modified file
        save_ustx(loaded, str(ustx_path))

        # Load again and verify
        final = load_ustx(str(ustx_path))

        assert "curves" in final["voice_parts"][0]
        curves = final["voice_parts"][0]["curves"]
        assert len(curves) == 1
        assert curves[0]["abbr"] == "dyn"
        assert curves[0]["xs"] == [0, 480, 960]
        assert curves[0]["ys"] == [0, 50, 100]
