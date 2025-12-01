from __future__ import annotations

from typing import TYPE_CHECKING

import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
from lsdb.core.crossmatch.crossmatch_args import CrossmatchArgs
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

    def __init__(
        self,
        *,
        left_mag_col: str,
        right_mag_col: str,
        n_neighbors: int = 1,
        radius_arcsec: float = 1.0,
        min_radius_arcsec: float = 0.0,
    ):
        super().__init__(n_neighbors, radius_arcsec, min_radius_arcsec)
        self.left_mag_col = left_mag_col
        self.right_mag_col = right_mag_col

    def validate(self, left: Catalog, right: Catalog):
        super().validate(left, right)
        if self.left_mag_col not in left.columns:
            raise ValueError(f"Left catalog must have column '{self.left_mag_col}'")
        if self.right_mag_col not in right.columns:
            raise ValueError(f"Right catalog must have column '{self.right_mag_col}'")

    # pylint: disable=too-many-positional-arguments
    def _calculate_magnitude_differences(
        self,
        all_matches_df: pd.DataFrame,
        left_df: npd.NestedFrame,
        right_df: npd.NestedFrame,
    ) -> pd.DataFrame:
        all_matches_df["left_mag"] = left_df.iloc[all_matches_df["left_idx"]][self.left_mag_col].to_numpy()
        all_matches_df["right_mag"] = right_df.iloc[all_matches_df["right_idx"]][
            self.right_mag_col
        ].to_numpy()
        all_matches_df["_magnitude_difference"] = np.abs(
            all_matches_df["right_mag"] - all_matches_df["left_mag"]
        )
        return all_matches_df

    def _select_best_matches(self, all_matches_df: pd.DataFrame) -> pd.DataFrame:
        best_match_indices_in_all_matches_df = all_matches_df.groupby("left_idx")[
            "_magnitude_difference"
        ].idxmin()
        return all_matches_df.loc[best_match_indices_in_all_matches_df].reset_index(drop=True)

    def perform_crossmatch(
        self, crossmatch_args: CrossmatchArgs
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        max_d_chord = _get_chord_distance(self.radius_arcsec)
        min_d_chord = _get_chord_distance(self.min_radius_arcsec)

        left_xyz, right_xyz = self._get_point_coordinates(
            crossmatch_args.left_df,
            crossmatch_args.left_catalog_info,
            crossmatch_args.right_df,
            crossmatch_args.right_catalog_info,
        )

        chord_distances_all, left_idx_all, right_idx_all = _find_crossmatch_indices(
            left_xyz=left_xyz,
            right_xyz=right_xyz,
            n_neighbors=self.n_neighbors,
            max_distance=max_d_chord,
            min_distance=min_d_chord,
        )

        arc_distances_all = np.degrees(2.0 * np.arcsin(0.5 * chord_distances_all)) * 3600

        all_matches_df = pd.DataFrame(
            {
                "left_idx": left_idx_all,
                "right_idx": right_idx_all,
                "arc_dist_arcsec": arc_distances_all,
            }
        )

        all_matches_df = self._calculate_magnitude_differences(
            all_matches_df, crossmatch_args.left_df, crossmatch_args.right_df
        )
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
