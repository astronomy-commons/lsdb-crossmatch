import lsdb
import nested_pandas as npd
import pytest

from lsdb_crossmatch.mag_difference_crossmatch import MagnitudeDifferenceCrossmatch


def test_mag_difference_crossmatch(m67_delve_small_dir, m67_ps1_small_dir, xmatch_mags):
    small_ps1 = lsdb.open_catalog(m67_ps1_small_dir)
    small_delve = lsdb.open_catalog(m67_delve_small_dir)

    xmatched = lsdb.crossmatch(
        small_ps1,
        small_delve,
        suffixes=("_ps1", "_delve"),
        algorithm=MagnitudeDifferenceCrossmatch,
        radius_arcsec=3600,
        left_mag_col="rMeanPSFMag",
        right_mag_col="MAG_PSF_R",
    ).compute()

    assert isinstance(xmatched, npd.NestedFrame)

    for _, correct_row in xmatch_mags.iterrows():
        assert correct_row["id_ps1"] in xmatched["objID_ps1"].to_numpy()
        xmatch_row = xmatched[xmatched["objID_ps1"] == correct_row["id_ps1"]]
        assert xmatch_row["QUICK_OBJECT_ID_delve"].to_numpy() == correct_row["id_delve"]
        assert xmatch_row["_dist_arcsec"].to_numpy() == pytest.approx(correct_row["_dist_arcsec"])
        assert xmatch_row["_magnitude_difference"].to_numpy() == pytest.approx(correct_row["_mag_diff"])
