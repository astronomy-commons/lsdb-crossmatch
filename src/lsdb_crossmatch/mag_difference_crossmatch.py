from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
from lsdb.core.crossmatch.kdtree_match import KdTreeCrossmatch
from lsdb.core.crossmatch.kdtree_utils import _find_crossmatch_indices, _get_chord_distance

if TYPE_CHECKING:
    from lsdb.catalog import Catalog


class MagnitudeDifferenceCrossmatch(KdTreeCrossmatch):
    """Cross-matching algorithm that extends KdTreeCrossmatch to include
    magnitude difference calculations and filtering.
    """

    extra_columns = pd.DataFrame(
        {
            "_dist_arcsec": pd.Series(dtype=pd.ArrowDtype(pa.float64())),
            "_magnitude_difference": pd.Series(dtype=pd.ArrowDtype(pa.float64())),
        }
    )

    @classmethod
    def validate(
        cls,
        left: Catalog,
        right: Catalog,
        left_mag_col: str,
        right_mag_col: str,
        radius_arcsec: float = 1,
        n_neighbors: int = 1,
    ):  # pylint: disable=too-many-arguments,arguments-renamed,too-many-positional-arguments
        super().validate(left, right, n_neighbors=n_neighbors, radius_arcsec=radius_arcsec)

        if left_mag_col not in left.columns:
            raise ValueError(f"Left catalog must have column '{left_mag_col}'")
        if right_mag_col not in right.columns:
            raise ValueError(f"Right catalog must have column '{right_mag_col}'")

    def _calculate_magnitude_differences(
        self, all_matches_df: pd.DataFrame, left_mag_col: str, right_mag_col: str
    ) -> pd.DataFrame:
        all_matches_df["left_mag"] = self.left.iloc[all_matches_df["left_idx"]][left_mag_col].to_numpy()
        all_matches_df["right_mag"] = self.right.iloc[all_matches_df["right_idx"]][right_mag_col].to_numpy()
        all_matches_df["_magnitude_difference"] = np.abs(
            all_matches_df["right_mag"] - all_matches_df["left_mag"]
        )
        return all_matches_df

    def _select_best_matches(self, all_matches_df: pd.DataFrame) -> pd.DataFrame:
        best_match_indices_in_all_matches_df = all_matches_df.groupby("left_idx")[
            "_magnitude_difference"
        ].idxmin()
        return all_matches_df.loc[best_match_indices_in_all_matches_df].reset_index(drop=True)

    # pylint: disable=arguments-differ
    def perform_crossmatch(
        self,
        left_mag_col: str,
        right_mag_col: str,
        radius_arcsec: float,
        n_neighbors: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        max_d_chord = _get_chord_distance(radius_arcsec)

        left_xyz, right_xyz = self._get_point_coordinates()

        chord_distances_all, left_idx_all, right_idx_all = _find_crossmatch_indices(
            left_xyz=left_xyz,
            right_xyz=right_xyz,
            n_neighbors=n_neighbors,
            max_distance=max_d_chord,
        )

        arc_distances_all = np.degrees(2.0 * np.arcsin(0.5 * chord_distances_all)) * 3600

        all_matches_df = pd.DataFrame(
            {
                "left_idx": left_idx_all,
                "right_idx": right_idx_all,
                "arc_dist_arcsec": arc_distances_all,
            }
        )

        all_matches_df = self._calculate_magnitude_differences(all_matches_df, left_mag_col, right_mag_col)
        final_matches_df = self._select_best_matches(all_matches_df)

        final_left_indices = final_matches_df["left_idx"].to_numpy()
        final_right_indices = final_matches_df["right_idx"].to_numpy()
        final_distances = final_matches_df["arc_dist_arcsec"].to_numpy()
        final_magnitude_differences = final_matches_df["_magnitude_difference"].to_numpy()

        extra_columns = pd.DataFrame(
            {
                "_dist_arcsec": pd.Series(final_distances, dtype=pd.ArrowDtype(pa.float64())),
                "_magnitude_difference": pd.Series(
                    final_magnitude_differences, dtype=pd.ArrowDtype(pa.float64())
                ),
            }
        )

        return final_left_indices, final_right_indices, extra_columns
