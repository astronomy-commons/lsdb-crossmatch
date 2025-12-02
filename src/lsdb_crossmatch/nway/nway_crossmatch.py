import hats.pixel_math.healpix_shim as hp
import numpy as np
import pandas as pd
import pyarrow as pa
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
from lsdb.core.crossmatch.crossmatch_args import CrossmatchArgs
from nwaylib import nway_match


def _series_of(pa_dtype):
    return pd.Series(dtype=pd.ArrowDtype(pa_dtype))


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-branches,too-many-locals
class NWAYCrossmatch(AbstractCrossmatchAlgorithm):
    """
    Nway crossmatch algorithm.

    match_tables: list of catalogues, each a dict with entries:
    - name (short catalog name, no spaces, used in output columns)
    - ra (RA in degrees)
    - dec (dec in degrees)
    - error (positional error in arcsec)
    - area (sky area covered by the catalogue in square degrees)
    - mags (list of additional columns to consider as priors)
    - magnames (short name for each entry in mags, used in output columns)
    - maghists (list of prior information for each entry in mags)
      use None to automatically build a histogram of target/non-target sources.
      Otherwise, supply the histogram manually: bins_lo, bins_hi, hist_sel, hist_all.

    match_radius: maximum radius in arcsec to consider.
        More distant counterparts are cut off.
        Set to a very large value, e.g. 5 times the largest positional error.
        Setting to larger values does not change the result, but
        setting to smaller values improves performance.

    prior_completeness: expected fraction of sources in the primary catalogue
        that are expected to have a counterpart (e.g., 90%).
        If an array is passed, completeness for each catalog. First
        entry has to be 1.
    """

    extra_columns = pd.DataFrame(
        {
            "Catalog_separation": _series_of(pa.float64()),
            "Separation_max": _series_of(pa.float64()),
            "ncat": _series_of(pa.int64()),
            "dist_bayesfactor_uncorrected": _series_of(pa.float64()),
            "dist_bayesfactor": _series_of(pa.float64()),
            "dist_post": _series_of(pa.float64()),
            "p_single": _series_of(pa.float64()),
            "match_flag": _series_of(pa.int64()),
            "prob_has_match": _series_of(pa.float64()),
            "prob_this_match": _series_of(pa.float64()),
        }
    )

    def __init__(
        self,
        *,
        radius_arcsec: float,
        pos_err_col_left: str = None,
        pos_err_col_right: str = None,
        ra_error_col_left: str = None,
        dec_error_col_left: str = None,
        ra_error_col_right: str = None,
        dec_error_col_right: str = None,
        left_mag_columns: list = None,
        right_mag_columns: list = None,
        prior_completeness: float = 1,
        match_flag: int = 0,
    ):
        self.radius_arcsec = radius_arcsec
        self.pos_err_col_left = pos_err_col_left
        self.pos_err_col_right = pos_err_col_right
        self.ra_error_col_left = ra_error_col_left
        self.dec_error_col_left = dec_error_col_left
        self.ra_error_col_right = ra_error_col_right
        self.dec_error_col_right = dec_error_col_right
        self.left_mag_columns = left_mag_columns
        self.right_mag_columns = right_mag_columns
        self.prior_completeness = prior_completeness
        self.match_flag = match_flag

    def validate(self, left, right):
        super().validate(left, right)
        if self.radius_arcsec < 0:
            raise ValueError("match radius has to be positive, you silly goose.")

        left_has_pos_err = self.pos_err_col_left is not None
        left_has_ra_dec_err = self.ra_error_col_left is not None and self.dec_error_col_left is not None

        if left_has_pos_err:
            if self.pos_err_col_left not in left.columns:
                raise ValueError(f"Column '{self.pos_err_col_left}' not found in left catalog.")
        elif left_has_ra_dec_err:
            if self.ra_error_col_left not in left.columns:
                raise ValueError(f"Column {self.ra_error_col_left} not found in left catalog.")
            if self.dec_error_col_left not in left.columns:
                raise ValueError(f"Column {self.dec_error_col_left} not found in left catalog.")
        else:
            raise ValueError(
                "For left catalog, either positional error column OR "
                "ra error, dec error and dec must be provided."
            )

        right_has_pos_err = self.pos_err_col_right is not None
        right_has_ra_dec_err = self.ra_error_col_right is not None and self.dec_error_col_right is not None

        if right_has_pos_err:
            if self.pos_err_col_right not in right.columns:
                raise ValueError(f"Column '{self.pos_err_col_right}' not found in right catalog.")
        elif right_has_ra_dec_err:
            if self.ra_error_col_right not in right.columns:
                raise ValueError(f"Column {self.ra_error_col_right} not found in right catalog.")
            if self.dec_error_col_right not in right.columns:
                raise ValueError(f"Column {self.dec_error_col_right} not found in right catalog.")
        else:
            raise ValueError(
                "For right catalog, either positional error column OR "
                "ra error, dec error and dec must be provided."
            )

        if self.left_mag_columns is not None:
            for col in self.left_mag_columns:
                if col not in left.columns:
                    raise ValueError(f"Column {col} not found in left catalog")

        if self.right_mag_columns is not None:
            for col in self.right_mag_columns:
                if col not in right.columns:
                    raise ValueError(f"Column {col} not found in right catalog")

        if self.match_flag not in (0, 1, 2):
            raise ValueError("`match_flag` must be an integer with value 0, 1, or 2")

        if not 0 <= self.prior_completeness <= 1:
            raise ValueError("`prior_completeness` must be between 0 and 1")

    def perform_crossmatch(self, crossmatch_args: CrossmatchArgs):
        """Crossmatch two partitions"""
        left = crossmatch_args.left_df
        left_catalog_name = crossmatch_args.left_catalog_info.catalog_name
        left_ra_column = crossmatch_args.left_catalog_info.ra_column
        left_dec_column = crossmatch_args.left_catalog_info.dec_column
        left_order = crossmatch_args.left_order

        right = crossmatch_args.right_df
        right_catalog_name = crossmatch_args.right_catalog_info.catalog_name
        right_ra_column = crossmatch_args.right_catalog_info.ra_column
        right_dec_column = crossmatch_args.right_catalog_info.dec_column
        right_order = crossmatch_args.right_order

        left_mags = []
        if self.left_mag_columns is not None:
            for col_name in self.left_mag_columns:
                left_mags.append(left[col_name].to_numpy())

        right_mags = []
        if self.right_mag_columns is not None:
            for col_name in self.right_mag_columns:
                right_mags.append(right[col_name].to_numpy())

        if self.pos_err_col_left is not None:
            left_errors = left[self.pos_err_col_left].to_numpy()
        else:
            ra_err_left = left[self.ra_error_col_left].to_numpy()
            dec_err_left = left[self.dec_error_col_left].to_numpy()
            dec_left_radians = np.radians(left[left_dec_column].to_numpy())
            left_errors = np.sqrt((ra_err_left * np.cos(dec_left_radians)) ** 2 + dec_err_left**2)

        if self.pos_err_col_right is not None:
            right_errors = right[self.pos_err_col_right].to_numpy()
        else:
            ra_err_right = right[self.ra_error_col_right].to_numpy()
            dec_err_right = right[self.dec_error_col_right].to_numpy()
            dec_right_radians = np.radians(right[right_dec_column].to_numpy())
            right_errors = np.sqrt((ra_err_right * np.cos(dec_right_radians)) ** 2 + dec_err_right**2)

        tables = [
            {
                "name": left_catalog_name,
                "ra": left[left_ra_column].to_numpy(),
                "dec": left[left_dec_column].to_numpy(),
                "error": left_errors,
                "area": hp.order2pixarea(left_order),
                "mags": left_mags,
                "magnames": self.left_mag_columns,
                "maghists": [],
            },
            {
                "name": right_catalog_name,
                "ra": right[right_ra_column].to_numpy(),
                "dec": right[right_dec_column].to_numpy(),
                "error": right_errors,
                "area": hp.order2pixarea(right_order),
                "mags": right_mags,
                "magnames": self.right_mag_columns,
                "maghists": [],
            },
        ]

        results = nway_match(tables, self.radius_arcsec, self.prior_completeness)
        return self._clean_nway_results(results, left_catalog_name, right_catalog_name)

    def _clean_nway_results(self, results, left_catalog_name, right_catalog_name):
        old_sep_col_name = f"Separation_{left_catalog_name}_{right_catalog_name}"

        results = results.rename(columns={old_sep_col_name: "Catalog_separation"})

        for col_name, expected_dtype in self.extra_columns.dtypes.items():
            if col_name in results.columns and results[col_name].dtype != expected_dtype:
                results[col_name] = results[col_name].astype(expected_dtype)

        results = results.reset_index(drop=True)

        results = results.query(f"{right_catalog_name} != -1")

        if self.match_flag == 1:
            results = results.query("match_flag == 1")
        elif self.match_flag == 2:
            results = results.query("match_flag != 0")

        left_idx = results[left_catalog_name].to_numpy()
        right_idx = results[right_catalog_name].to_numpy()

        results = results.drop(columns=[left_catalog_name, right_catalog_name])
        return left_idx, right_idx, results
