"""Shared helpers for running the resourced-cross recipe through the Modal HP
drivers (modal/gmm_{d}d_hp*.py).

Lives in the tnwf package so it ships to every Modal image via
``add_local_python_source("tnwf")`` and is importable from inside the remote
``run_one`` functions.

The recipe adds three knobs on top of the published (method, N, K, D_max, seed)
sweep, all defaulting to the original coupled/off behaviour so legacy CSVs and
grids reproduce Table 1 exactly:

  D_V          operator bond for exp(iβV) (decoupled from D_out; default = D_max)
  D_out        output/product bond (default = D_max)
  n_v_substeps split exp(iβV) → exp(i(β/M)V) applied M times (default 1)
  n_global     global-pivot re-seeding rounds of the cross (default 0)
"""
from __future__ import annotations


def resourced_kwargs(method: str, D_max: int, D_V: int = -1, D_out: int = -1,
                     n_v_substeps: int = 1, n_global: int = 0):
    """Return (method_kwargs, out_suffix) for a resourced cell.

    method_kwargs is ready to pass to run(..., method_kwargs=...). out_suffix is
    "" when all knobs are at their defaults (so the on-disk path matches the
    original D_max-keyed layout) and otherwise encodes the knobs to avoid
    colliding with the baseline cell.
    """
    if method == "jam":
        return {}, ""
    D_V = D_max if D_V < 0 else D_V
    D_out = D_max if D_out < 0 else D_out
    resourced = (D_V != D_max) or (D_out != D_max) or (n_v_substeps != 1) or (n_global != 0)
    suffix = f"_DV{D_V}_DO{D_out}_M{n_v_substeps}_g{n_global}" if resourced else ""
    kwargs = {
        "D_max": D_max, "D_V": D_V, "D_out": D_out, "D_init": D_out,
        "n_v_substeps": n_v_substeps, "n_global": n_global,
    }
    return kwargs, suffix


def grid_row(r: dict) -> tuple:
    """Build a run_one starmap tuple from a CSV row, reading optional resourced
    columns (D_V, D_out, n_v_substeps, n_global) with coupled/off defaults.

    Returns (method, N, K, D_max, seed, D_V, D_out, n_v_substeps, n_global).
    """
    Dm = int(r["D_max"])

    def g(key, default):
        v = r.get(key)
        return int(v) if v not in (None, "") else default

    return (r["method"], int(r["N"]), int(r["K"]), Dm, int(r["seed"]),
            g("D_V", Dm), g("D_out", Dm), g("n_v_substeps", 1), g("n_global", 0))
