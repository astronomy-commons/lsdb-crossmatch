"""Run nway on N catalogs for a single aligned HEALPix pixel.

This is the "calling nway" building block of the N-way crossmatch
(astronomy-commons/lsdb-crossmatch#32). It knows nothing about alignment or
Dask: given the partition data that falls in one aligned pixel, it builds the
nway input tables, calls ``nwaylib.nway_match`` and returns per-catalog row
indices plus nway's probability columns. The plumbing that iterates over
aligned pixels lives in #31.

Source densities: nway turns each table's ``area`` into a source density
(rows / area). The nway CLI sees whole catalogs, so its densities are global.
Here each call only sees one pixel's rows, so callers should pass each
catalog's global density (``source_densities``); the area handed to nway is
then derived from it so that nway uses those densities. Without it, the aligned
pixel's area is used, which is only an approximation (see #33).

Magnitude priors are out of scope for now (positional matching first, see #33);
magnitude columns are optional and default to none.
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


def prepare_primary(
    primary_df: pd.DataFrame,
    primary_pixel: HealpixPixel,
    aligned_pixel: HealpixPixel,
    *,
    spatial_index_order: int | None = None,
):
    """Keep only the primary rows that fall inside the aligned pixel.

    The aligned pixel can be smaller than the primary's partition pixel, in
    which case only part of the partition belongs to this pixel.

    ``spatial_index_order`` is the HEALPix order of the catalog's spatial index
    (from its catalog info); ``None`` means the default of 29.
    """
    if aligned_pixel.order <= primary_pixel.order:
        return primary_df
    return filter_by_spatial_index_to_pixel(
        primary_df, aligned_pixel.order, aligned_pixel.pixel, spatial_index_order=spatial_index_order
    )


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
    source_densities: list[float] | None = None,
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
        HEALPix order of the aligned pixel. Only used for the area when
        ``source_densities`` is not given.
    radius_arcsec : float
        Maximum separation considered by nway.
    prior_completeness : float or list of float
        Passed to nway. A list needs one value per catalog, starting with 1.
    match_flag : {0, 1, 2}
        0 keeps every candidate combination, 1 only the best match per primary
        source, 2 the best match and nway's secondary alternatives.
    keep_unmatched : bool
        If False (default), drop rows with no counterpart in any secondary.
    source_densities : list of float, optional
        Global source density of each catalog (rows per square degree over
        the whole catalog), in the same order as ``frames``. Recommended:
        without it every catalog is assumed to fill the aligned pixel, which
        overestimates the density of secondaries stored at a lower order.

    Returns
    -------
    indices : list of np.ndarray
        One array per catalog with positional row indices into the matching
        frame (``frames[k].iloc[indices[k]]``), ``-1`` where that catalog has
        no counterpart in the row.
    extra : pd.DataFrame
        nway's per-row columns (separations, ``ncat``, ``match_flag``,
        ``prob_has_match`` (CLI ``p_any``), ``prob_this_match`` (CLI ``p_i``), ...).
        The columns are always the same for a given set of catalogs, also when
        the result is empty or some catalogs have no rows in this pixel.
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

    if source_densities is not None:
        if len(source_densities) != len(frames):
            raise ValueError(f"Got {len(source_densities)} source densities for {len(frames)} catalogs.")
        if any(density <= 0 for density in source_densities):
            raise ValueError("source_densities must be positive.")
    completeness = _per_catalog_completeness(prior_completeness, len(frames))

    # nway fails if any catalog is empty, so only match the non-empty ones and
    # put the empty ones back afterwards as "no counterpart".
    present = [0] + [k for k in range(1, len(frames)) if len(frames[k]) > 0]
    if len(frames[0]) == 0 or len(present) == 1:
        return _empty_result(names)

    tables = [
        build_nway_table(
            frames[k], configs[k], _nway_area(len(frames[k]), k == 0, aligned_order, source_densities, k)
        )
        for k in present
    ]
    # store_mag_hists=False: otherwise nway writes histogram .pdf/.txt files to the working directory.
    results = nway_match(tables, radius_arcsec, completeness[present], store_mag_hists=False)
    indices, extra = _clean_results(results, [names[k] for k in present], match_flag, keep_unmatched)
    return _restore_missing_catalogs(indices, extra, names, present)


def _per_catalog_completeness(prior_completeness, n_catalogs: int) -> np.ndarray:
    """Expand nway's prior completeness to one value per catalog.

    nway splits a single value evenly over the secondaries as ``c**(1/(n-1))``.
    Doing that here, for all catalogs, keeps the per-catalog values unchanged
    when empty catalogs are left out of the nway call.
    """
    if np.ndim(prior_completeness) == 0:
        share = float(prior_completeness) ** (1.0 / (n_catalogs - 1))
        return np.array([1.0] + [share] * (n_catalogs - 1))
    completeness = np.asarray(prior_completeness, dtype=float)
    if len(completeness) != n_catalogs:
        raise ValueError(f"prior_completeness needs one value per catalog, got {len(completeness)}.")
    return completeness


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _nway_area(n_rows, is_primary, aligned_order, source_densities, catalog_index) -> float:
    """Area to give nway for one catalog, in square degrees.

    With a known global density, pick the area so that nway's own density
    matches it: nway uses ``n / area`` for the primary and ``(n + 1) / area``
    for secondaries (one extra for "source may be absent").
    """
    if source_densities is None:
        return pixel_area_deg2(aligned_order)
    counted_rows = n_rows if is_primary else n_rows + 1
    return counted_rows / source_densities[catalog_index]


def _restore_missing_catalogs(indices, extra, names, present):
    """Put catalogs that were left out of the nway call back into the result."""
    if len(present) == len(names):
        return indices, extra
    n_rows = len(extra)
    by_name = dict(zip([names[k] for k in present], indices))
    full_indices = [by_name.get(name, np.full(n_rows, NO_COUNTERPART, dtype=np.int64)) for name in names]
    expected = list(_extra_columns(names))
    # Keep any columns nway added beyond the fixed set (e.g. magnitude bias columns).
    others = [column for column in extra.columns if column not in expected]
    full_extra = extra.reindex(columns=expected + others)
    return full_indices, full_extra


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


# Columns nway adds after the per-catalog index columns, apart from separations.
_NWAY_SUMMARY_COLUMNS = {
    "Separation_max": "float64",
    "ncat": "int64",
    "dist_bayesfactor_uncorrected": "float64",
    "dist_bayesfactor": "float64",
    "dist_post": "float64",
    "p_single": "float64",
    "match_flag": "int64",
    "prob_has_match": "float64",
    "prob_this_match": "float64",
}


def _extra_columns(names: list[str]) -> dict[str, str]:
    """nway's per-row columns (name -> dtype) for these catalogs, in nway's order."""
    separations = {
        f"Separation_{names[i]}_{names[j]}": "float64"
        for i in range(len(names))
        for j in range(i + 1, len(names))
    }
    return {**separations, **_NWAY_SUMMARY_COLUMNS}


def _empty_result(names: list[str]):
    indices = [np.array([], dtype=np.int64) for _ in names]
    columns = _extra_columns(names)
    extra = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns.items()})
    return indices, extra
