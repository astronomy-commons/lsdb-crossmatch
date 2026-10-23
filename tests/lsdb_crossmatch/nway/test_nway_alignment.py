import numpy as np
import pytest
from hats.pixel_math import HealpixPixel
from hats.pixel_tree import PixelAlignmentType, align_trees
from hats.pixel_tree.pixel_tree import PixelTree

from lsdb_crossmatch.nway.nway_alignment import NWayAlignment, align_n_trees

INNER = PixelAlignmentType.INNER
LEFT = PixelAlignmentType.LEFT
OUTER = PixelAlignmentType.OUTER


@pytest.fixture
def small_trees():
    """Three hand-made trees inside / around order 0 pixel 4.

    a: the whole of (0, 4)
    b: three of the four order-1 children of (0, 4): 16, 17, 18
    c: two order-2 grandchildren in 16 (64, 65), all of 17, and (0, 5) next door
    """
    a = PixelTree.from_healpix([(0, 4)])
    b = PixelTree.from_healpix([(1, 16), (1, 17), (1, 18)])
    c = PixelTree.from_healpix([(2, 64), (2, 65), (1, 17), (0, 5)])
    return [a, b, c]


def rows(alignment):
    """Mapping as a list of per-aligned-pixel tuples: (cat0, cat1, ..., aligned)."""
    return [tuple(tuple(p) for p in alignment.pixel_mapping[:, i]) for i in range(len(alignment))]


NONE = (-1, -1)


def test_inner_keeps_sky_covered_by_all(small_trees):
    alignment = align_n_trees(small_trees, INNER)
    assert rows(alignment) == [
        ((0, 4), (1, 16), (2, 64), (2, 64)),
        ((0, 4), (1, 16), (2, 65), (2, 65)),
        ((0, 4), (1, 17), (1, 17), (1, 17)),
    ]


def test_left_keeps_all_of_the_first_catalog(small_trees):
    alignment = align_n_trees(small_trees, LEFT)
    assert rows(alignment) == [
        ((0, 4), (1, 16), (2, 64), (2, 64)),
        ((0, 4), (1, 16), (2, 65), (2, 65)),
        ((0, 4), (1, 16), NONE, (2, 66)),
        ((0, 4), (1, 16), NONE, (2, 67)),
        ((0, 4), (1, 17), (1, 17), (1, 17)),
        ((0, 4), (1, 18), NONE, (1, 18)),
        ((0, 4), NONE, NONE, (1, 19)),
    ]


def test_outer_keeps_sky_covered_by_any(small_trees):
    alignment = align_n_trees(small_trees, OUTER)
    assert rows(alignment)[:-1] == rows(align_n_trees(small_trees, LEFT))
    assert rows(alignment)[-1] == (NONE, NONE, (0, 5), (0, 5))


@pytest.mark.parametrize("alignment_type", [INNER, LEFT, OUTER])
def test_two_trees_match_hats(small_trees, alignment_type):
    a, _, c = small_trees
    ours = align_n_trees([a, c], alignment_type)
    theirs = align_trees(a, c, alignment_type)

    assert ours.n_catalogs == 2
    assert [p.pixel for p in ours.aligned_pixels()] == list(theirs.pixel_mapping["aligned_Npix"])
    assert np.array_equal(ours.pixel_tree.tree, theirs.pixel_tree.tree)
    for k, (order_col, pixel_col) in enumerate(
        [("primary_Norder", "primary_Npix"), ("join_Norder", "join_Npix")]
    ):
        expected = [
            None if o is None else HealpixPixel(int(o), int(p))
            for o, p in zip(theirs.pixel_mapping[order_col], theirs.pixel_mapping[pixel_col])
        ]
        assert ours.catalog_pixels(k) == expected


def test_accessors(small_trees):
    alignment = align_n_trees(small_trees, LEFT)
    assert alignment.n_catalogs == 3
    assert alignment.pixel_mapping.shape == (4, 7, 2)
    assert alignment.pixel_mapping.dtype == np.int64
    assert alignment.alignment_type == LEFT

    assert alignment.aligned_pixels()[-1] == HealpixPixel(1, 19)
    assert alignment.catalog_pixels(1)[-1] is None
    assert alignment.catalog_pixels(-1) == alignment.catalog_pixels(2)
    assert np.array_equal(alignment.catalog_pixels_array(0), alignment.pixel_mapping[0])
    assert np.array_equal(alignment.aligned_pixels_array(), alignment.pixel_mapping[-1])
    assert alignment.pixel_tree.get_healpix_pixels() == alignment.aligned_pixels()

    with pytest.raises(IndexError):
        alignment.catalog_pixels(3)


def test_order_of_input_trees_only_changes_columns(small_trees):
    """Aligning in a different order gives the same rows with the catalog columns permuted."""
    a, b, c = small_trees
    abc = align_n_trees([a, b, c], INNER)
    acb = align_n_trees([a, c, b], INNER)
    reordered = {(r[0], r[2], r[1], r[3]) for r in rows(acb)}
    assert set(rows(abc)) == reordered


def test_no_overlap_gives_empty_alignment():
    a = PixelTree.from_healpix([(0, 1)])
    b = PixelTree.from_healpix([(0, 2)])
    c = PixelTree.from_healpix([(0, 3)])
    alignment = align_n_trees([a, b, c], INNER)
    assert len(alignment) == 0
    assert alignment.pixel_mapping.shape == (4, 0, 2)
    assert alignment.aligned_pixels() == []


def test_invalid_inputs(small_trees):
    with pytest.raises(ValueError, match="at least two"):
        align_n_trees(small_trees[:1])
    with pytest.raises(ValueError, match="RIGHT"):
        align_n_trees(small_trees, PixelAlignmentType.RIGHT)
    with pytest.raises(ValueError, match="shape"):
        NWayAlignment(small_trees[0], np.zeros((3, 2)), INNER)


def random_tree(rng, max_order=4, n_splits=12):
    """A random valid tree: start from some order 0 pixels and repeatedly split / drop leaves."""
    # Draw from only a few order 0 pixels so the trees overlap often enough to be interesting.
    leaves = {(0, int(p)) for p in rng.choice(3, size=rng.integers(2, 4), replace=False)}
    for _ in range(n_splits):
        order, pixel = sorted(leaves)[rng.integers(len(leaves))]
        if order < max_order:
            leaves.remove((order, pixel))
            children = [(order + 1, 4 * pixel + k) for k in range(4)]
            keep = [child for child in children if rng.random() < 0.8] or children[:1]
            leaves.update(keep)
    return PixelTree.from_healpix(sorted(leaves))


def covers(outer, inner):
    """True if HEALPix pixel ``outer`` contains (or equals) pixel ``inner``."""
    return outer[0] <= inner[0] and inner[1] >> (2 * (inner[0] - outer[0])) == outer[1]


def sky_at(tree, order):
    """Set of order-``order`` pixels covered by ``tree``."""
    cells = set()
    for o, p in tree.pixels:
        shift = 2 * (order - o)
        cells.update(range(p << shift, (p + 1) << shift))
    return cells


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("alignment_type", [INNER, LEFT, OUTER])
def test_random_trees_invariants(seed, alignment_type):
    rng = np.random.default_rng(seed)
    trees = [random_tree(rng) for _ in range(rng.integers(2, 5))]
    alignment = align_n_trees(trees, alignment_type)

    # Every catalog pixel listed for an aligned pixel is really in that catalog and contains it.
    for k, tree in enumerate(trees):
        catalog_pixels = {tuple(p) for p in tree.pixels}
        for catalog_pixel, aligned_pixel in zip(
            alignment.catalog_pixels_array(k), alignment.aligned_pixels_array()
        ):
            if tuple(catalog_pixel) != (-1, -1):
                assert tuple(catalog_pixel) in catalog_pixels
                assert covers(catalog_pixel, aligned_pixel)

    # The aligned pixels cover exactly the sky the alignment type promises.
    order = 4
    covered = [sky_at(tree, order) for tree in trees]
    if alignment_type == INNER:
        expected = set.intersection(*covered)
    elif alignment_type == LEFT:
        expected = covered[0]
    else:
        expected = set.union(*covered)
    assert sky_at(alignment.pixel_tree, order) == expected

    # A catalog is marked missing exactly where it doesn't cover the aligned pixel.
    for k, tree in enumerate(trees):
        for catalog_pixel, aligned_pixel in zip(
            alignment.catalog_pixels_array(k), alignment.aligned_pixels_array()
        ):
            aligned_sky = sky_at(PixelTree.from_healpix([tuple(aligned_pixel)]), order)
            assert (tuple(catalog_pixel) != (-1, -1)) == aligned_sky.issubset(covered[k])
