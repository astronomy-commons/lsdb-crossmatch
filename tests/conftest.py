from pathlib import Path

import pytest
from dask.distributed import Client, LocalCluster

TEST_DIR = Path(__file__).parent


@pytest.fixture(scope="session", name="dask_client", autouse=True)
def dask_client():
    """Create a single client for use by all unit test cases."""
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, dashboard_address=":0")
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


@pytest.fixture
def test_data_dir():
    return Path(TEST_DIR) / "data"


@pytest.fixture
def m67_delve_dir(test_data_dir):
    return test_data_dir / "m67" / "delve_cone"


@pytest.fixture
def m67_ps1_dir(test_data_dir):
    return test_data_dir / "m67" / "ps1_cone"
