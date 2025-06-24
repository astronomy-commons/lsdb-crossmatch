from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent


@pytest.fixture
def test_data_dir():
    return Path(TEST_DIR) / "data"


@pytest.fixture
def m67_delve_dir(test_data_dir):
    return test_data_dir / "m67" / "delve_cone"


@pytest.fixture
def m67_ps1_dir(test_data_dir):
    return test_data_dir / "m67" / "ps1_cone"
