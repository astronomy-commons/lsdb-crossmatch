import lsdb
import nested_pandas as npd
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch


def test_default_crossmatch(m67_delve_dir, m67_ps1_dir):
    # Determine which inputs need to be computed
    left_data = lsdb.open_catalog(m67_ps1_dir)
    right_data = lsdb.open_catalog(m67_delve_dir)

    # Perform the crossmatch
    result = lsdb.crossmatch(
        left_data,
        right_data,
        suffixes=["_left", "_right"],
        algorithm=KdTreeCrossmatch,
        radius_arcsec=0.01 * 3600,
    ).compute()

    # Assertions
    assert isinstance(result, npd.NestedFrame)
    assert 160_000 > len(result) > 150_000
