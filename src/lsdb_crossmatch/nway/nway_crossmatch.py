import hats.pixel_math.healpix_shim as hp
import numpy as np
import pandas as pd
import pyarrow as pa
from lsdb.core.crossmatch.abstract_crossmatch_algorithm import AbstractCrossmatchAlgorithm
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

    @classmethod
    def validate(
        cls,
        left,
        right,
        radius_arcsec,
        pos_err_col_left=None,
        pos_err_col_right=None,
        ra_error_col_left=None,
        dec_error_col_left=None,
        ra_error_col_right=None,
        dec_error_col_right=None,
        left_mag_columns: list = None,
        right_mag_columns: list = None,
        prior_completeness=1,
        match_flag=0,
    ):  # pylint: disable=arguments-differ
        super().validate(left, right)
        if radius_arcsec < 0:
            raise ValueError("match radius has to be positive, you silly goose.")

        left_has_pos_err = pos_err_col_left is not None
        left_has_ra_dec_err = ra_error_col_left is not None and dec_error_col_left is not None

        if left_has_pos_err:
            if pos_err_col_left not in left.columns:
                raise ValueError(f"Column '{pos_err_col_left}' not found in left catalog.")
        elif left_has_ra_dec_err:
            if ra_error_col_left not in left.columns:
                raise ValueError(f"Column {ra_error_col_left} not found in left catalog.")
            if dec_error_col_left not in left.columns:
                raise ValueError(f"Column {dec_error_col_left} not found in left catalog.")
        else:
            raise ValueError(
                "For left catalog, either positional error column OR "
                "ra error, dec error and dec must be provided."
            )

        right_has_pos_err = pos_err_col_right is not None
        right_has_ra_dec_err = ra_error_col_right is not None and dec_error_col_right is not None

        if right_has_pos_err:
            if pos_err_col_right not in right.columns:
                raise ValueError(f"Column '{pos_err_col_right}' not found in right catalog.")
        elif right_has_ra_dec_err:
            if ra_error_col_right not in right.columns:
                raise ValueError(f"Column {ra_error_col_right} not found in right catalog.")
            if dec_error_col_right not in right.columns:
                raise ValueError(f"Column {dec_error_col_right} not found in right catalog.")
        else:
            raise ValueError(
                "For right catalog, either positional error column OR "
                "ra error, dec error and dec must be provided."
            )

        if left_mag_columns is not None:
            for col in left_mag_columns:
                if col not in left.columns:
                    raise ValueError(f"Column {col} not found in left catalog")

        if right_mag_columns is not None:
            for col in right_mag_columns:
                if col not in right.columns:
                    raise ValueError(f"Column {col} not found in right catalog")

        if match_flag not in (0, 1, 2):
            raise ValueError("`match_flag` must be an integer with value 0, 1, or 2")

        if not 0 <= prior_completeness <= 1:
            raise ValueError("`prior_completeness` must be between 0 and 1")

    def perform_crossmatch(
        self,
        radius_arcsec,
        pos_err_col_left=None,
        pos_err_col_right=None,
        ra_error_col_left=None,
        dec_error_col_left=None,
        ra_error_col_right=None,
        dec_error_col_right=None,
        left_mag_columns: list = None,
        right_mag_columns: list = None,
        prior_completeness=0.9,
        match_flag=0,
    ):  # pylint: disable=arguments-differ
        """Crossmatch two partitions"""
        left_catalog_name = self.left_catalog_info.catalog_name
        right_catalog_name = self.right_catalog_info.catalog_name

        left_mags = []
        if left_mag_columns is not None:
            for col_name in left_mag_columns:
                left_mags.append(self.left[col_name].to_numpy())
        else:
            left_mag_columns = []

        right_mags = []
        if right_mag_columns is not None:
            for col_name in right_mag_columns:
                right_mags.append(self.right[col_name].to_numpy())
        else:
            right_mag_columns = []

        if pos_err_col_left is not None:
            left_errors = self.left[pos_err_col_left].to_numpy()
        else:
            ra_err_left = self.left[ra_error_col_left].to_numpy()
            dec_err_left = self.left[dec_error_col_left].to_numpy()
            dec_left_radians = np.radians(self.left[self.left_catalog_info.dec_column].to_numpy())
            left_errors = np.sqrt((ra_err_left * np.cos(dec_left_radians)) ** 2 + dec_err_left**2)

        if pos_err_col_right is not None:
            right_errors = self.right[pos_err_col_right].to_numpy()
        else:
            ra_err_right = self.right[ra_error_col_right].to_numpy()
            dec_err_right = self.right[dec_error_col_right].to_numpy()
            dec_right_radians = np.radians(self.right[self.right_catalog_info.dec_column].to_numpy())
            right_errors = np.sqrt((ra_err_right * np.cos(dec_right_radians)) ** 2 + dec_err_right**2)

        tables = [
            {
                "name": left_catalog_name,
                "ra": self.left[self.left_catalog_info.ra_column].to_numpy(),
                "dec": self.left[self.left_catalog_info.dec_column].to_numpy(),
                "error": left_errors,
                "area": hp.order2pixarea(self.left_order),
                "mags": left_mags,
                "magnames": left_mag_columns,
                "maghists": [],
            },
            {
                "name": right_catalog_name,
                "ra": self.right[self.right_catalog_info.ra_column].to_numpy(),
                "dec": self.right[self.right_catalog_info.dec_column].to_numpy(),
                "error": right_errors,
                "area": hp.order2pixarea(self.right_order),
                "mags": right_mags,
                "magnames": right_mag_columns,
                "maghists": [],
            },
        ]

        results = nway_match(tables, radius_arcsec, prior_completeness)
        return self._clean_nway_results(results, left_catalog_name, right_catalog_name, match_flag)

    def _clean_nway_results(self, results, left_catalog_name, right_catalog_name, match_flag):
        old_sep_col_name = f"Separation_{left_catalog_name}_{right_catalog_name}"

        results = results.rename(columns={old_sep_col_name: "Catalog_separation"})

        for col_name, expected_dtype in self.extra_columns.dtypes.items():
            if col_name in results.columns and results[col_name].dtype != expected_dtype:
                results[col_name] = results[col_name].astype(expected_dtype)

        results = results.reset_index(drop=True)

        results = results.query(f"{right_catalog_name} != -1")

        if match_flag == 1:
            results = results.query("match_flag == 1")
        elif match_flag == 2:
            results = results.query("match_flag != 0")

        left_idx = results[left_catalog_name].to_numpy()
        right_idx = results[right_catalog_name].to_numpy()

        results = results.drop(columns=[left_catalog_name, right_catalog_name])
        return left_idx, right_idx, results
