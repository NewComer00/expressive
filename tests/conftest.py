import pytest
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Session-scoped fixtures (created once per test session)
@pytest.fixture(scope="session")
def project_root_dir():
    """Provide project root directory path"""
    return project_root


@pytest.fixture(scope="session")
def examples_dir(project_root_dir):
    """Provide examples directory path"""
    return project_root_dir / "examples"


@pytest.fixture(scope="session")
def has_example_files(examples_dir):
    """Check if example files are available"""
    example_project = examples_dir / "Прекрасное Далеко"
    return (
        example_project.exists() and
        (example_project / "utau.wav").exists() and
        (example_project / "reference.wav").exists() and
        (example_project / "project.ustx").exists()
    )


# Function-scoped fixtures (created for each test)
@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after test"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_ustx_file(temp_dir):
    """Create a temporary USTX file"""
    ustx_content = """tempos:
  - bpm: 120
    position: 0
time_signatures:
  - bar_index: 0
    beat_per_bar: 4
    beat_unit: 4
voice_parts:
  - name: Track 1
    track_no: 0
    notes: []
    curves: []
"""
    ustx_path = temp_dir / "test.ustx"
    ustx_path.write_text(ustx_content, encoding='utf-8-sig')
    return ustx_path


@pytest.fixture
def sample_ustx_dict():
    """Provide a sample USTX dictionary"""
    return {
        'tempos': [
            {'bpm': 120, 'position': 0}
        ],
        'time_signatures': [
            {'bar_index': 0, 'beat_per_bar': 4, 'beat_unit': 4}
        ],
        'voice_parts': [
            {
                'name': 'Track 1',
                'track_no': 0,
                'notes': [
                    {
                        'pos': 0,
                        'dur': 480,
                        'tone': 60,
                        'lyric': 'la'
                    },
                    {
                        'pos': 480,
                        'dur': 480,
                        'tone': 62,
                        'lyric': 'la'
                    }
                ],
                'curves': []
            }
        ]
    }


@pytest.fixture(autouse=True)
def reset_expression_loader_counter():
    """Reset ExpressionLoader ID counter before each test"""
    from expressions.base import ExpressionLoader
    original_counter = ExpressionLoader._id_counter
    ExpressionLoader._id_counter = 0
    yield
    ExpressionLoader._id_counter = original_counter


@pytest.fixture
def clean_expression_table():
    """Clean expression registration table before and after test"""
    from expressions.base import EXPRESSION_LOADER_TABLE
    original_table = EXPRESSION_LOADER_TABLE.copy()
    EXPRESSION_LOADER_TABLE.clear()

    yield

    # Restore original registration table
    EXPRESSION_LOADER_TABLE.clear()
    EXPRESSION_LOADER_TABLE.update(original_table)


# Hooks for test collection and reporting
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_audio: marks tests that require audio files"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU/CUDA"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection"""
    # Skip tests that require audio files if not available
    examples_dir = Path("examples/Прекрасное Далеко")
    has_audio = (
        examples_dir.exists() and
        (examples_dir / "utau.wav").exists()
    )

    skip_audio = pytest.mark.skip(reason="Audio files not available")

    for item in items:
        if "requires_audio" in item.keywords and not has_audio:
            item.add_marker(skip_audio)
