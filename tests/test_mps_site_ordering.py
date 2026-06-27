"""Unit tests for the row / snake / hilbert MPS site orderings."""
from __future__ import annotations

import numpy as np
import pytest

from tnwf.data.mps_ordering import (
    get_ordering,
    hilbert_ordering,
    row_ordering,
    snake_ordering,
)


@pytest.mark.parametrize("ordering", ["row", "snake", "hilbert"])
@pytest.mark.parametrize("side", [2, 4, 8])
def test_ordering_is_bijection(ordering, side):
    """Every pixel index appears exactly once."""
    perm = get_ordering(ordering, side, side)
    assert perm.shape == (side * side,)
    assert sorted(perm.tolist()) == list(range(side * side))


@pytest.mark.parametrize("ordering", ["row", "snake", "hilbert"])
@pytest.mark.parametrize("side", [2, 4, 8])
def test_roundtrip_preserves_data(ordering, side):
    """Applying perm then inv_perm restores the original vector."""
    rng = np.random.default_rng(0)
    flat = rng.standard_normal((10, side * side)).astype(np.float32)
    perm = get_ordering(ordering, side, side)
    inv_perm = np.argsort(perm)
    permuted = flat[:, perm]
    restored = permuted[:, inv_perm]
    np.testing.assert_array_equal(flat, restored)


def test_row_is_identity():
    assert np.array_equal(row_ordering(4, 4), np.arange(16))


def test_snake_4x4_first_two_rows():
    """Row 0: 0,1,2,3   Row 1: 7,6,5,4  (reversed)."""
    perm = snake_ordering(4, 4)
    assert np.array_equal(perm[:4], np.array([0, 1, 2, 3]))
    assert np.array_equal(perm[4:8], np.array([7, 6, 5, 4]))


def test_hilbert_2x2_is_z_curve():
    """For n=2 the Hilbert curve is a U-shape: (0,0) → (0,1) → (1,1) → (1,0)
    which in row-major flat = 0, 2, 3, 1."""
    perm = hilbert_ordering(2, 2)
    assert np.array_equal(perm, np.array([0, 2, 3, 1]))


def test_hilbert_locality_8x8():
    """For a Hilbert curve, consecutive sites should be 4-connected on the
    grid (Chebyshev distance ≤ 1). This isn't true for row-major at the
    row boundary, so it's a real check of Hilbert behaviour."""
    perm = hilbert_ordering(8, 8)
    coords = np.array([(p // 8, p % 8) for p in perm])
    for i in range(1, len(coords)):
        d = np.max(np.abs(coords[i] - coords[i - 1]))
        assert d == 1, f"Hilbert sites {i-1}, {i} are not adjacent: {coords[i-1]} → {coords[i]}"


def test_hilbert_requires_power_of_two():
    with pytest.raises(ValueError, match="power-of-two"):
        hilbert_ordering(3, 3)


def test_hilbert_requires_square():
    with pytest.raises(ValueError, match="h == w"):
        hilbert_ordering(4, 8)


def test_unknown_ordering_raises():
    with pytest.raises(ValueError, match="Unknown ordering"):
        get_ordering("rainbow", 4, 4)
