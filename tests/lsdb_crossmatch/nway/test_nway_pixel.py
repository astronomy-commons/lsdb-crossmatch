import numpy as np
import pandas as pd
import pytest
from hats.pixel_math import HealpixPixel
from hats.pixel_math.spatial_index import compute_spatial_index
from nwaylib import UndersampledException, nway_match

from lsdb_crossmatch.nway.nway_pixel import (
    NwayCatalogConfig,
    build_nway_table,
    nway_crossmatch_pixel,
    pixel_area_deg2,
    prepare_primary,
    prepare_secondary,
)

ARCSEC = 1 / 3600
ALIGNED_ORDER = 7


def make_catalogs(n=30, n_field=60, n_in_c=15, use_mags=False, seed=42):
    """Three small catalogs of the same sources with known true counterparts.

    Every primary source has a counterpart in B; only the first ``n_in_c`` have
    one in C. B also has unrelated field sources, fainter than the true
    counterparts so a magnitude prior has something to learn. Rows of B and C
    are shuffled so a match can't come out right by accident of order.
    """
    rng = np.random.default_rng(seed)
    ra = 150.1 + rng.uniform(0, 0.05, n)
    dec = 2.2 + rng.uniform(0, 0.05, n)
    true_id = np.arange(n)

    primary = pd.DataFrame({"ra": ra, "dec": dec, "err": 1.0, "true_id": true_id})

    b = pd.DataFrame(
        {
            "ra_b": np.concatenate(
                [ra + rng.normal(0, 0.2 * ARCSEC, n), 150.1 + rng.uniform(0, 0.05, n_field)]
            ),
            "dec_b": np.concatenate(
                [dec + rng.normal(0, 0.2 * ARCSEC, n), 2.2 + rng.uniform(0, 0.05, n_field)]
            ),
            "ra_err": 0.2,
            "dec_err": 0.2,
            "mag_b": np.concatenate([rng.uniform(18, 20, n), rng.uniform(20, 24, n_field)]),
            "true_id": np.concatenate([true_id, np.full(n_field, -1)]),
        }
    ).sample(frac=1, random_state=1)

    c = pd.DataFrame(
        {
            "ra_c": ra[:n_in_c] + rng.normal(0, 0.3 * ARCSEC, n_in_c),
            "dec_c": dec[:n_in_c] + rng.normal(0, 0.3 * ARCSEC, n_in_c),
            "err": 0.3,
            "true_id": true_id[:n_in_c],
        }
    ).sample(frac=1, random_state=2)

    configs = [
        NwayCatalogConfig(name="A", ra_column="ra", dec_column="dec", pos_err_column="err"),
        NwayCatalogConfig(
            name="B",
            ra_column="ra_b",
            dec_column="dec_b",
            ra_err_column="ra_err",
            dec_err_column="dec_err",
            mag_columns=["mag_b"] if use_mags else [],
        ),
        NwayCatalogConfig(name="C", ra_column="ra_c", dec_column="dec_c", pos_err_column="err"),
    ]
    return [primary, b, c], configs


@pytest.fixture
def three_catalogs():
    return make_catalogs()


def test_pixel_area_is_in_square_degrees():
    # Whole sky is ~41253 deg^2 over 12 * 4**order pixels.
    full_sky = 4 * np.pi * (180 / np.pi) ** 2
    for order in (0, 5, 10):
        assert pixel_area_deg2(order) == pytest.approx(full_sky / (12 * 4**order))


def test_build_nway_table_passes_area_in_deg2_and_uses_mag_priors():
    frames, configs = make_catalogs(use_mags=True)
    table = build_nway_table(frames[1], configs[1], pixel_area_deg2(ALIGNED_ORDER))

    assert table["area"] == pytest.approx(pixel_area_deg2(ALIGNED_ORDER))
    # One histogram slot per magnitude column, otherwise nway's zip() skips the priors.
    assert len(table["mags"]) == len(table["maghists"]) == len(table["magnames"]) == 1
    # Combined error from per-axis errors at dec ~2.2 deg.
    assert np.allclose(table["error"], np.hypot(0.2 * np.cos(np.radians(frames[1]["dec_b"])), 0.2))


def test_three_catalog_best_matches_are_correct(three_catalogs):
    frames, configs = three_catalogs
    indices, extra = nway_crossmatch_pixel(
        frames, configs, ALIGNED_ORDER, radius_arcsec=5, prior_completeness=0.9, match_flag=1
    )
    primary, b, c = frames
    a_idx, b_idx, c_idx = indices

    # One best match per primary source.
    assert len(a_idx) == len(primary)
    assert sorted(a_idx) == list(range(len(primary)))

    true_a = primary["true_id"].to_numpy()[a_idx]
    assert (b["true_id"].to_numpy()[b_idx] == true_a).all()

    has_c = c_idx != -1
    assert has_c.sum() == len(c)
    assert (c["true_id"].to_numpy()[c_idx[has_c]] == true_a[has_c]).all()

    assert {"prob_has_match", "prob_this_match", "match_flag", "ncat"} <= set(extra.columns)
    assert len(extra) == len(a_idx)
    assert (extra["ncat"].to_numpy() == np.where(has_c, 3, 2)).all()


def test_matches_direct_nway_call(three_catalogs):
    """The wrapper must not change nway's numbers, only reshape them."""
    frames, configs = three_catalogs
    indices, extra = nway_crossmatch_pixel(
        frames, configs, ALIGNED_ORDER, radius_arcsec=5, prior_completeness=0.9, keep_unmatched=True
    )

    area = pixel_area_deg2(ALIGNED_ORDER)
    tables = [build_nway_table(f, cfg, area) for f, cfg in zip(frames, configs)]
    direct = nway_match(tables, 5, 0.9, store_mag_hists=False).reset_index(level="A").reset_index(drop=True)

    for name, idx in zip(["A", "B", "C"], indices):
        assert np.array_equal(idx, direct[name].to_numpy())
    for col in ("prob_has_match", "prob_this_match", "match_flag"):
        assert np.allclose(extra[col].to_numpy(), direct[col].to_numpy())


def test_match_flag_filters(three_catalogs):
    frames, configs = three_catalogs
    kwargs = {"radius_arcsec": 5, "prior_completeness": 0.9}
    _, all_rows = nway_crossmatch_pixel(frames, configs, ALIGNED_ORDER, match_flag=0, **kwargs)
    _, best = nway_crossmatch_pixel(frames, configs, ALIGNED_ORDER, match_flag=1, **kwargs)
    _, best_and_alt = nway_crossmatch_pixel(frames, configs, ALIGNED_ORDER, match_flag=2, **kwargs)

    assert (best["match_flag"] == 1).all()
    assert (best_and_alt["match_flag"] != 0).all()
    assert len(best) <= len(best_and_alt) <= len(all_rows)


def test_drops_rows_without_any_counterpart(three_catalogs):
    frames, configs = three_catalogs
    indices, _ = nway_crossmatch_pixel(frames, configs, ALIGNED_ORDER, radius_arcsec=5)
    secondaries = np.vstack(indices[1:])
    assert (secondaries != -1).any(axis=0).all()


def test_empty_secondaries_return_empty(three_catalogs):
    frames, configs = three_catalogs
    empty = [frames[0], frames[1].iloc[0:0], frames[2].iloc[0:0]]
    indices, extra = nway_crossmatch_pixel(empty, configs, ALIGNED_ORDER, radius_arcsec=5)
    assert all(len(idx) == 0 for idx in indices)
    assert len(extra) == 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"radius_arcsec": -1}, "positive"),
        ({"radius_arcsec": 5, "match_flag": 3}, "match_flag"),
    ],
)
def test_invalid_arguments(three_catalogs, kwargs, message):
    frames, configs = three_catalogs
    with pytest.raises(ValueError, match=message):
        nway_crossmatch_pixel(frames, configs, ALIGNED_ORDER, **kwargs)


def test_invalid_configs(three_catalogs):
    frames, configs = three_catalogs
    with pytest.raises(ValueError, match="unique"):
        nway_crossmatch_pixel(frames, [configs[0], configs[1], configs[1]], ALIGNED_ORDER, radius_arcsec=5)
    with pytest.raises(ValueError, match="frames"):
        nway_crossmatch_pixel(frames[:2], configs, ALIGNED_ORDER, radius_arcsec=5)
    missing = NwayCatalogConfig(name="C", ra_column="nope", dec_column="dec_c", pos_err_column="err")
    with pytest.raises(ValueError, match="missing columns"):
        nway_crossmatch_pixel(frames, [configs[0], configs[1], missing], ALIGNED_ORDER, radius_arcsec=5)
    no_err = NwayCatalogConfig(name="C", ra_column="ra_c", dec_column="dec_c")
    with pytest.raises(ValueError, match="pos_err_column"):
        nway_crossmatch_pixel(frames, [configs[0], configs[1], no_err], ALIGNED_ORDER, radius_arcsec=5)


def test_prepare_primary_filters_to_smaller_aligned_pixel():
    rng = np.random.default_rng(0)
    ra = rng.uniform(150.0, 150.5, 500)
    dec = rng.uniform(2.0, 2.5, 500)
    df = pd.DataFrame({"ra": ra, "dec": dec}, index=compute_spatial_index(ra, dec))

    def pixel_of(index, order):
        return np.asarray(index) >> (2 * (29 - order))

    order = 6
    partition_pixel = HealpixPixel(order, int(np.bincount(pixel_of(df.index, order)).argmax()))
    partition = df[pixel_of(df.index, order) == partition_pixel.pixel]

    # Same-order alignment keeps everything.
    assert len(prepare_primary(partition, partition_pixel, partition_pixel)) == len(partition)

    # A child pixel keeps only its own rows; the four children together keep them all.
    children = [HealpixPixel(order + 1, partition_pixel.pixel * 4 + k) for k in range(4)]
    kept = [prepare_primary(partition, partition_pixel, child) for child in children]
    assert sum(len(k) for k in kept) == len(partition)
    for child, rows in zip(children, kept):
        assert (pixel_of(rows.index, child.order) == child.pixel).all()


def test_prepare_secondary_concatenates_margin():
    main = pd.DataFrame({"x": [1, 2]})
    margin = pd.DataFrame({"x": [3]})
    assert prepare_secondary(main, margin)["x"].tolist() == [1, 2, 3]
    assert prepare_secondary(main, None)["x"].tolist() == [1, 2]
    assert prepare_secondary(main, margin.iloc[0:0])["x"].tolist() == [1, 2]


def test_magnitude_priors_are_applied():
    """With enough matches in the pixel, nway builds the magnitude histogram and uses it."""
    frames, configs = make_catalogs(n=150, n_field=300, n_in_c=80, use_mags=True)
    kwargs = {"radius_arcsec": 5, "prior_completeness": 0.9, "keep_unmatched": True}
    _, with_mags = nway_crossmatch_pixel(frames, configs, ALIGNED_ORDER, **kwargs)

    plain_configs = [configs[0], NwayCatalogConfig(**{**configs[1].__dict__, "mag_columns": []}), configs[2]]
    _, without_mags = nway_crossmatch_pixel(frames, plain_configs, ALIGNED_ORDER, **kwargs)

    assert "bias_B_mag_b" in with_mags.columns
    assert "bias_B_mag_b" not in without_mags.columns
    assert not np.allclose(with_mags["prob_this_match"], without_mags["prob_this_match"])


def test_magnitude_priors_need_enough_matches_in_the_pixel():
    """nway refuses to build an automatic histogram from fewer than 100 secure matches.

    Sparse catalogs (e.g. Chandra) will hit this in many pixels, which is why the
    histograms will likely need to be built from the whole catalog (see #32).
    """
    frames, configs = make_catalogs(use_mags=True)
    with pytest.raises(UndersampledException):
        nway_crossmatch_pixel(frames, configs, ALIGNED_ORDER, radius_arcsec=5)
