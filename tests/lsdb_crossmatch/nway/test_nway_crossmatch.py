import lsdb
import nested_pandas as npd
import pandas as pd

from lsdb_crossmatch.nway.nway_crossmatch import NWAYCrossmatch

pd.set_option("display.max_rows", None)


def test_mag_difference_crossmatch(m67_delve_small_dir, m67_ps1_small_dir, xmatch_mags):
    small_ps1 = lsdb.open_catalog(m67_ps1_small_dir)
    small_delve = lsdb.open_catalog(m67_delve_small_dir)

    xmatched = lsdb.crossmatch(
        small_ps1,
        small_delve,
        suffixes=("_ps1", "_delve"),
        algorithm=NWAYCrossmatch,
        radius_arcsec=60,
        left_mag_columns=["gMeanPSFMag", "rMeanPSFMag", "iMeanPSFMag", "zMeanPSFMag"],
        right_mag_columns=["MAG_PSF_G", "MAG_PSF_R", "MAG_PSF_I", "MAG_PSF_Z"],
        ra_error_col_left="raMeanErr",
        dec_error_col_left="decMeanErr",
        ra_error_col_right="MAGERR_PSF_G",  ## Well beans
        dec_error_col_right="MAGERR_PSF_I",
        match_flag=2,
    ).compute()

    assert isinstance(xmatched, npd.NestedFrame)

    for _, correct_row in xmatch_mags.iterrows():
        assert correct_row["id_ps1"] in xmatched["objID_ps1"].to_numpy()
        xmatch_row = xmatched[xmatched["objID_ps1"] == correct_row["id_ps1"]]
        assert correct_row["id_delve"] in xmatch_row["QUICK_OBJECT_ID_delve"].to_numpy()
