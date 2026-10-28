"""Run nway on N catalogs for a single aligned HEALPix pixel.

This is the "calling nway" building block of the N-way crossmatch
(astronomy-commons/lsdb-crossmatch#32). It knows nothing about alignment or
Dask: given the partition data that falls in one aligned pixel, it builds the
nway input tables, calls ``nwaylib.nway_match`` and returns per-catalog row
indices plus nway's probability columns. The plumbing that iterates over
aligned pixels lives in #31.

Current assumptions (as described in #32, to be revisited later):

- Source densities are computed per pixel: every catalog's sky area is the
  area of the aligned pixel.
- Magnitude histograms are built automatically by nway from the rows in the
  pixel, not from the whole catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import hats.pixel_math.healpix_shim as hp
import numpy as np
import pandas as pd
from hats.pixel_math import HealpixPixel
from lsdb.operations.functions.merge_catalog_functions import filter_by_spatial_index_to_pixel
from nwaylib import nway_match

NO_COUNTERPART = -1


@dataclass(frozen=True)
class NwayCatalogConfig:
    """Describes how to read one catalog's partition for nway.

    Parameters
    ----------
    name : str
        Short catalog name with no spaces. nway uses it in output column names,
        so names must be unique across the catalogs in one crossmatch.
    ra_column, dec_column : str
        Position columns, in degrees.
    pos_err_column : str, optional
        Circular positional error, in arcsec.
    ra_err_column, dec_err_column : str, optional
        Per-axis positional errors, in arcsec. Used when ``pos_err_column`` is
        not given; combined as ``sqrt((ra_err * cos(dec))**2 + dec_err**2)``,
        matching the existing two-catalog ``NWAYCrossmatch``.
    mag_columns : list of str, optional
        Columns used as magnitude priors.
    """

    name: str
    ra_column: str
    dec_column: str
    pos_err_column: str | None = None
    ra_err_column: str | None = None
    dec_err_column: str | None = None
    mag_columns: list[str] = field(default_factory=list)

    def validate(self, columns) -> None:
        """Raise ``ValueError`` if a configured column is missing from ``columns``."""
        if " " in self.name:
            raise ValueError(f"Catalog name '{self.name}' must not contain spaces.")
        if self.pos_err_column is None and (self.ra_err_column is None or self.dec_err_column is None):
            raise ValueError(
                f"Catalog '{self.name}': give either pos_err_column or both ra_err_column and dec_err_column."
            )
        wanted = [self.ra_column, self.dec_column, *self.mag_columns]
        if self.pos_err_column is not None:
            wanted.append(self.pos_err_column)
        else:
            wanted += [self.ra_err_column, self.dec_err_column]
        missing = [col for col in wanted if col not in columns]
        if missing:
            raise ValueError(f"Catalog '{self.name}' is missing columns: {missing}")

    def positional_errors(self, df: pd.DataFrame) -> np.ndarray:
        """Positional error per row, in arcsec."""
        if self.pos_err_column is not None:
            return df[self.pos_err_column].to_numpy(dtype=float)
        ra_err = df[self.ra_err_column].to_numpy(dtype=float)
        dec_err = df[self.dec_err_column].to_numpy(dtype=float)
        dec_rad = np.radians(df[self.dec_column].to_numpy(dtype=float))
        return np.sqrt((ra_err * np.cos(dec_rad)) ** 2 + dec_err**2)


def pixel_area_deg2(order: int) -> float:
    """Area of one HEALPix pixel at ``order``, in square degrees (the unit nway expects)."""
    return float(hp.order2pixarea(order, degrees=True))


def prepare_primary(primary_df: pd.DataFrame, primary_pixel: HealpixPixel, aligned_pixel: HealpixPixel):
    """Keep only the primary rows that fall inside the aligned pixel.

    The aligned pixel can be smaller than the primary's partition pixel, in
    which case only part of the partition belongs to this pixel.
    """
    if aligned_pixel.order <= primary_pixel.order:
        return primary_df
    return filter_by_spatial_index_to_pixel(primary_df, aligned_pixel.order, aligned_pixel.pixel)


def prepare_secondary(df: pd.DataFrame | None, margin_df: pd.DataFrame | None) -> pd.DataFrame:
    """Combine a secondary catalog's partition with its margin.

    Counterparts of primary sources near the pixel edge can sit just outside
    the pixel, so secondaries are matched against partition + margin.
    """
    frames = [frame for frame in (df, margin_df) if frame is not None and len(frame) > 0]
    if not frames:
        return df if df is not None else margin_df
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames)


def build_nway_table(df: pd.DataFrame, config: NwayCatalogConfig, area_deg2: float) -> dict:
    """Build one entry of nway's ``match_tables`` list."""
    # nway overwrites -99 with NaN in place, so give it its own float copies.
    mags = [df[col].to_numpy(dtype=float, copy=True) for col in config.mag_columns]
    return {
        "name": config.name,
        "ra": df[config.ra_column].to_numpy(dtype=float),
        "dec": df[config.dec_column].to_numpy(dtype=float),
        "error": config.positional_errors(df),
        "area": area_deg2,
        "mags": mags,
        "magnames": list(config.mag_columns),
        # One entry per magnitude column; None means "let nway build the
        # histogram". An empty list would make nway silently skip the priors.
        "maghists": [None] * len(mags),
    }


# pylint: disable=too-many-arguments
def nway_crossmatch_pixel(
    frames: list[pd.DataFrame],
    configs: list[NwayCatalogConfig],
    aligned_order: int,
    *,
    radius_arcsec: float,
    prior_completeness: float | list[float] = 1.0,
    match_flag: int = 0,
    keep_unmatched: bool = False,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Run nway on N already-prepared frames from one aligned pixel.

    Parameters
    ----------
    frames : list of pd.DataFrame
        One frame per catalog, primary first. The primary should already be
        filtered to the aligned pixel (``prepare_primary``) and secondaries
        combined with their margins (``prepare_secondary``).
    configs : list of NwayCatalogConfig
        One config per frame, in the same order.
    aligned_order : int
        HEALPix order of the aligned pixel; its area is used for every catalog.
    radius_arcsec : float
        Maximum separation considered by nway.
    prior_completeness : float or list of float
        Passed to nway. A list needs one value per catalog, starting with 1.
    match_flag : {0, 1, 2}
        0 keeps every candidate combination, 1 only the best match per primary
        source, 2 the best match and nway's secondary alternatives.
    keep_unmatched : bool
        If False (default), drop rows with no counterpart in any secondary.

    Returns
    -------
    indices : list of np.ndarray
        One array per catalog with positional row indices into the matching
        frame (``frames[k].iloc[indices[k]]``), ``-1`` where that catalog has
        no counterpart in the row.
    extra : pd.DataFrame
        nway's per-row columns (separations, ``ncat``, ``match_flag``,
        ``prob_has_match`` (CLI ``p_any``), ``prob_this_match`` (CLI ``p_i``), ...).
    """
    if len(frames) != len(configs):
        raise ValueError(f"Got {len(frames)} frames but {len(configs)} configs.")
    if len(frames) < 2:
        raise ValueError("nway needs at least two catalogs.")
    names = [config.name for config in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"Catalog names must be unique, got {names}.")
    if radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive.")
    if match_flag not in (0, 1, 2):
        raise ValueError("match_flag must be 0, 1 or 2.")
    for frame, config in zip(frames, configs):
        config.validate(frame.columns)

    if len(frames[0]) == 0 or all(len(frame) == 0 for frame in frames[1:]):
        return _empty_result(names)

    area = pixel_area_deg2(aligned_order)
    tables = [build_nway_table(frame, config, area) for frame, config in zip(frames, configs)]
    # store_mag_hists=False: otherwise nway writes histogram .pdf/.txt files to the working directory.
    results = nway_match(tables, radius_arcsec, prior_completeness, store_mag_hists=False)
    return _clean_results(results, names, match_flag, keep_unmatched)


def _clean_results(results: pd.DataFrame, names: list[str], match_flag: int, keep_unmatched: bool):
    # nway keeps the primary row number in the index (see the pandas 3 note in
    # NWAYCrossmatch._clean_nway_results); bring it back as a column.
    if names[0] in results.index.names:
        results = results.reset_index(level=names[0])
    results = results.reset_index(drop=True)

    if not keep_unmatched:
        has_counterpart = (results[names[1:]] != NO_COUNTERPART).any(axis=1)
        results = results[has_counterpart]
    if match_flag == 1:
        results = results[results["match_flag"] == 1]
    elif match_flag == 2:
        results = results[results["match_flag"] != 0]

    indices = [results[name].to_numpy(dtype=np.int64) for name in names]
    extra = results.drop(columns=names).reset_index(drop=True)
    return indices, extra


def _empty_result(names: list[str]):
    indices = [np.array([], dtype=np.int64) for _ in names]
    return indices, pd.DataFrame()
