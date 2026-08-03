"""Distance metrics for distribution comparison.

Only the estimators the pipeline reports are exported here. ``wasserstein_2``
and ``grid_pmf_indices`` stay in their modules — they are called internally by
``wasserstein_2_subsampled`` and ``nll_from_density_grid`` respectively — but
they are not part of this package's surface.
"""
from tnwf.metrics.mmd import mmd_rbf
from tnwf.metrics.nll import density_from_psi, nll_from_density_grid
from tnwf.metrics.sw import sliced_wasserstein
from tnwf.metrics.wasserstein import wasserstein_2_subsampled

__all__ = [
    "sliced_wasserstein",
    "wasserstein_2_subsampled",
    "mmd_rbf",
    "density_from_psi",
    "nll_from_density_grid",
]
