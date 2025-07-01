import lsdb
import nested_pandas as npd
import numpy as np
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch

from lsdb_crossmatch.mag_difference_crossmatch import MyCrossmatchAlgorithm


def test_mag_difference_crossmatch(m67_delve_dir, m67_ps1_dir):
    left_data = lsdb.open_catalog(m67_ps1_dir)
    right_data = lsdb.open_catalog(m67_delve_dir)

    id_col_left = "objID_left"
    id_col_right = "QUICK_OBJECT_ID_right"

    left_mag_col = "rMeanPSFMag"
    right_mag_col = "MAG_PSF_R"

    kdtree_result = lsdb.crossmatch(
        left_data,
        right_data,
        suffixes=["_left", "_right"],
        algorithm=KdTreeCrossmatch,
        radius_arcsec=0.01 * 3600,
        n_neighbors=5,
    ).compute()

    result = lsdb.crossmatch(
        left_data,
        right_data,
        suffixes=["_left", "_right"],
        algorithm=MyCrossmatchAlgorithm,
        radius_arcsec=0.01 * 3600,
        left_mag_col=left_mag_col,
        right_mag_col=right_mag_col,
        n_neighbors=5,
    ).compute()

    assert isinstance(result, npd.NestedFrame)

    if not result.empty:
        assert len(result) == len(result[id_col_left].unique())
        assert result[id_col_left].value_counts().max() == 1

    if not kdtree_result.empty:
        kdtree_result["magnitude_difference"] = np.abs(
            kdtree_result[right_mag_col + "_right"] - kdtree_result[left_mag_col + "_left"]
        )

    target_left_id = 122650089529714672

    if not kdtree_result.empty and target_left_id in kdtree_result[id_col_left].values:
        potential_matches_for_target = kdtree_result[kdtree_result[id_col_left] == target_left_id]

        if not potential_matches_for_target.empty:
            potential_matches_for_target = potential_matches_for_target.sort_values(
                by="magnitude_difference"
            ).reset_index(drop=True)

            expected_best_right_id = potential_matches_for_target.iloc[0][id_col_right]

            actual_match_from_my_algo = result[result[id_col_left] == target_left_id]

            if not actual_match_from_my_algo.empty:
                actual_matched_right_id = actual_match_from_my_algo.iloc[0][id_col_right]

                assert actual_matched_right_id == expected_best_right_id
        else:
            print(f"No potential matches found in kdtree_result for {target_left_id}.")
    else:
        print(f"Target left ID {target_left_id} not found in kdtree_result.")
