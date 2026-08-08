"""Distance metrics for distribution comparison.

Only the estimators the pipeline reports are exported here. ``grid_pmf_indices``
stays in its module — it is called internally by ``nll_from_density_grid`` — but
it is not part of this package's surface.
"""
from tnwf.metrics.mmd import mmd_rbf
from tnwf.metrics.nll import density_from_psi, nll_from_density_grid
from tnwf.metrics.sw import sliced_wasserstein

__all__ = [
    "sliced_wasserstein",
    "mmd_rbf",
    "density_from_psi",
    "nll_from_density_grid",
]
