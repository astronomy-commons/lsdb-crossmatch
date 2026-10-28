"""Align the pixel trees of N catalogs (astronomy-commons/lsdb-crossmatch#30).

hats' ``PixelAlignment`` aligns exactly two trees. ``NWayAlignment`` is its
N-catalog equivalent: for every pixel of the aligned tree it records which
pixel of each input catalog overlaps it.

The alignment is built by folding ``hats.pixel_tree.align_trees`` over the
input trees: align tree 0 with tree 1, align the result with tree 2, and so on,
carrying the earlier catalogs' pixels along at every step.
"""

from __future__ import annotations

import numpy as np
from hats.pixel_math import HealpixPixel
from hats.pixel_tree import PixelAlignment, PixelAlignmentType, align_trees
from hats.pixel_tree.pixel_tree import PixelTree

MISSING_PIXEL = (-1, -1)


class NWayAlignment:
    """How the pixel trees of N catalogs align with each other.

    Attributes
    ----------
    pixel_tree : PixelTree
        The aligned tree, made of the smallest overlapping pixels of all inputs.
    pixel_mapping : np.ndarray
        ``int64`` array of shape ``(n_catalogs + 1, n_aligned_pixels, 2)``. Index
        ``[k, i]`` holds the ``(order, pixel)`` of catalog ``k`` that overlaps
        aligned pixel ``i``, or ``(-1, -1)`` if catalog ``k`` has no pixel there.
        The last entry along the first axis (``[-1, i]``) is the aligned pixel itself.
    alignment_type : PixelAlignmentType
        The alignment type used at every pairwise step.
    moc : MOC or None
        Not used yet; kept for parity with ``PixelAlignment``.
    """

    def __init__(
        self,
        aligned_tree: PixelTree,
        pixel_mapping: np.ndarray,
        alignment_type: PixelAlignmentType,
        moc=None,
    ) -> None:
        pixel_mapping = np.asarray(pixel_mapping, dtype=np.int64)
        if pixel_mapping.ndim != 3 or pixel_mapping.shape[2] != 2:
            raise ValueError(
                "pixel_mapping must have shape (n_catalogs + 1, n_aligned_pixels, 2), "
                f"got {pixel_mapping.shape}"
            )
        self.pixel_tree = aligned_tree
        self.pixel_mapping = pixel_mapping
        self.alignment_type = alignment_type
        self.moc = moc

    @property
    def n_catalogs(self) -> int:
        """Number of input catalogs."""
        return self.pixel_mapping.shape[0] - 1

    def __len__(self) -> int:
        """Number of aligned pixels."""
        return self.pixel_mapping.shape[1]

    def catalog_pixels_array(self, catalog_index: int) -> np.ndarray:
        """``(order, pixel)`` of catalog ``catalog_index`` for every aligned pixel, ``(-1, -1)`` if none."""
        if not -self.n_catalogs <= catalog_index < self.n_catalogs:
            raise IndexError(f"catalog_index {catalog_index} out of range for {self.n_catalogs} catalogs")
        return self.pixel_mapping[catalog_index % self.n_catalogs]

    def aligned_pixels_array(self) -> np.ndarray:
        """``(order, pixel)`` of every aligned pixel."""
        return self.pixel_mapping[-1]

    def catalog_pixels(self, catalog_index: int) -> list[HealpixPixel | None]:
        """Catalog ``catalog_index``'s pixel for every aligned pixel, ``None`` if it has none."""
        return _to_healpix_list(self.catalog_pixels_array(catalog_index))

    def aligned_pixels(self) -> list[HealpixPixel]:
        """Every aligned pixel as a ``HealpixPixel``."""
        return _to_healpix_list(self.aligned_pixels_array())


def align_n_trees(
    trees: list[PixelTree], alignment_type: PixelAlignmentType = PixelAlignmentType.INNER
) -> NWayAlignment:
    """Align the pixel trees of N catalogs.

    Parameters
    ----------
    trees : list of PixelTree
        At least two trees. The first is treated as the primary (left) catalog.
    alignment_type : PixelAlignmentType
        Applied at every pairwise step:

        - INNER keeps sky covered by every catalog,
        - LEFT keeps sky covered by the first catalog,
        - OUTER keeps sky covered by any catalog.

        RIGHT is rejected because chaining it is ambiguous with more than two trees.

    Returns
    -------
    NWayAlignment
    """
    alignment_type = PixelAlignmentType(alignment_type)
    if alignment_type == PixelAlignmentType.RIGHT:
        raise ValueError("RIGHT alignment is not supported for N-way alignment; reorder the trees instead.")
    if len(trees) < 2:
        raise ValueError("Need at least two trees to align.")

    first = align_trees(trees[0], trees[1], alignment_type)
    mapping = np.stack(
        [
            _column_pairs(
                first, PixelAlignment.PRIMARY_ORDER_COLUMN_NAME, PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME
            ),
            _column_pairs(
                first, PixelAlignment.JOIN_ORDER_COLUMN_NAME, PixelAlignment.JOIN_PIXEL_COLUMN_NAME
            ),
            _column_pairs(
                first, PixelAlignment.ALIGNED_ORDER_COLUMN_NAME, PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME
            ),
        ]
    )
    aligned_tree = first.pixel_tree

    for tree in trees[2:]:
        step = align_trees(aligned_tree, tree, alignment_type)
        mapping = _extend_mapping(mapping, step)
        aligned_tree = step.pixel_tree

    return NWayAlignment(aligned_tree, mapping, alignment_type)


def _extend_mapping(mapping: np.ndarray, step: PixelAlignment) -> np.ndarray:
    """Add one catalog to ``mapping`` using a pairwise alignment of the previous aligned tree and it.

    In ``step`` the "primary" pixel is a pixel of the previous aligned tree. We
    look that pixel up in ``mapping`` to find which pixel of every earlier
    catalog it came from, then append the new catalog's pixel and the new
    aligned pixel.
    """
    previous_aligned = mapping[-1]
    previous_rows = {tuple(pixel): row for row, pixel in enumerate(previous_aligned)}

    step_primary = _column_pairs(
        step, PixelAlignment.PRIMARY_ORDER_COLUMN_NAME, PixelAlignment.PRIMARY_PIXEL_COLUMN_NAME
    )
    step_join = _column_pairs(
        step, PixelAlignment.JOIN_ORDER_COLUMN_NAME, PixelAlignment.JOIN_PIXEL_COLUMN_NAME
    )
    step_aligned = _column_pairs(
        step, PixelAlignment.ALIGNED_ORDER_COLUMN_NAME, PixelAlignment.ALIGNED_PIXEL_COLUMN_NAME
    )

    n_earlier = mapping.shape[0] - 1
    earlier = np.full((n_earlier, len(step_primary), 2), MISSING_PIXEL, dtype=np.int64)
    for i, primary in enumerate(step_primary):
        if tuple(primary) == MISSING_PIXEL:
            # Only possible for OUTER: sky covered by the new catalog alone.
            continue
        earlier[:, i] = mapping[:n_earlier, previous_rows[tuple(primary)]]

    return np.concatenate([earlier, step_join[np.newaxis], step_aligned[np.newaxis]])


def _column_pairs(alignment: PixelAlignment, order_column: str, pixel_column: str) -> np.ndarray:
    """``(n, 2)`` int64 array of (order, pixel) from a ``PixelAlignment`` mapping, ``-1`` for missing."""
    frame = alignment.pixel_mapping
    if len(frame) == 0:
        return np.empty((0, 2), dtype=np.int64)
    # hats stores missing pixels as None in object columns.
    pairs = frame[[order_column, pixel_column]].astype(object).fillna(-1)
    return pairs.to_numpy(dtype=np.int64)


def _to_healpix_list(pairs: np.ndarray) -> list[HealpixPixel | None]:
    return [
        None if tuple(pair) == MISSING_PIXEL else HealpixPixel(int(pair[0]), int(pair[1])) for pair in pairs
    ]
