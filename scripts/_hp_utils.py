"""Shared HP-sweep helpers used by the make_fig_* scripts."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tnwf.coords import CoordConventionError  # noqa: E402

METHODS = ["jam", "dense", "tci_tdvp1", "tci_tdvp2"]
WAVE_METHODS = ["dense", "tci_tdvp1", "tci_tdvp2"]
TDVP_METHODS = ["tci_tdvp1", "tci_tdvp2"]

METHOD_LABEL = {
    "jam":        "JAM",
    "dense":      "Dense",
    "tci_tdvp1":  "TCI+1TDVP",
    "tci_tdvp2":  "TCI+2TDVP",
}
METHOD_COLORS = {
    "jam":        "tab:gray",
    "dense":      "black",
    "tci_tdvp1":  "tab:green",
    "tci_tdvp2":  "tab:red",
}
METHOD_MARKERS = {
    "jam":        "v",
    "dense":      "P",
    "tci_tdvp1":  "^",
    "tci_tdvp2":  "o",
}

NK_D_PATTERN = re.compile(r"^N(\d+)_K(\d+)_D(\d+)$")
NK_PATTERN   = re.compile(r"^N(\d+)_K(\d+)$")


def _safe_load(path: Path) -> dict | None:
    """Open a seed*.npz; return None if it's mid-write or corrupted.

    Reads `sw_endpoint`, the endpoint sliced-Wasserstein recomputed from
    node-centred samples with a recorded projection seed (see
    `scripts/migrate_archive.py`). A pre-migration archive raises rather than
    silently supplying the half-cell-biased `sw`, which is the failure mode
    that put two conventions into one paper.

    Two timers are exposed. `time` is `total_time`, which brackets the whole
    Trotter loop and so includes the per-step metric callback; `evolve_time`
    sums `step_times` and covers the evolution alone. At d=2 the metric
    callback is over 90% of `total_time`, which is enough to break monotonicity
    in a cost-vs-dimension fit, so anything comparing methods on runtime wants
    `evolve_time`.
    """
    import zipfile
    try:
        z = np.load(path, allow_pickle=True)
        if "sw_endpoint" not in z.files:
            raise CoordConventionError(
                f"{path} predates the node-centred coordinate convention. "
                f"Its `sw` carries a half-cell sampling bias. Run "
                f"`python scripts/migrate_archive.py --src <v1> --dst <v2>`."
            )
        step_times = (np.asarray(z["step_times"], dtype=float)
                      if "step_times" in z.files else None)
        out = {
            "sw":   float(np.asarray(z["sw_endpoint"])),
            "mmd":  float(np.asarray(z["mmd_endpoint"])),
            "sw_mc_std": (float(np.asarray(z["sw_endpoint_mc_std"]))
                          if "sw_endpoint_mc_std" in z.files else float("nan")),
            "chi":  int(np.asarray(z["chi_max"]).max()) if "chi_max" in z.files else 0,
            "time": float(np.asarray(z["total_time"])) if "total_time" in z.files else float("nan"),
            "evolve_time": (float(step_times.sum()) if step_times is not None
                            and step_times.size else float("nan")),
            "d":    int(np.asarray(z["d"])) if "d" in z.files else 0,
        }
        return out
    except (zipfile.BadZipFile, OSError, EOFError, KeyError) as e:
        print(f"[skip {path}] {type(e).__name__}: {e}")
        return None


def collect(results_dir: Path, method: str) -> list[dict]:
    """Load all (N, K, D) cells for an MPS method, mean over seeds.

    Each cell dict has: N, K, D, sw, mmd, chi, time, n_seeds.
    Returns [] if the method directory is missing or has no parseable cells.
    """
    base = Path(results_dir) / method
    if not base.exists():
        return []
    rows = []
    for sub in sorted(base.iterdir()):
        m = NK_D_PATTERN.match(sub.name)
        if not m:
            continue
        N, K, D = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sw, mmd, chi, time_seeds, evolve_seeds, mc_std = [], [], [], [], [], []
        for f in sorted(sub.glob("seed*.npz")):
            rec = _safe_load(f)
            if rec is None:
                continue
            sw.append(rec["sw"])
            mmd.append(rec["mmd"])
            chi.append(rec["chi"])
            mc_std.append(rec["sw_mc_std"])
            if not np.isnan(rec["time"]):
                time_seeds.append(rec["time"])
            if not np.isnan(rec["evolve_time"]):
                evolve_seeds.append(rec["evolve_time"])
        if sw:
            rows.append({
                "N": N, "K": K, "D": D,
                "sw": float(np.mean(sw)),
                # Spread across seeds within the cell. Reported alongside the
                # best cell because cells cluster tightly near the minimum, so
                # a bare argmin over ~35 of them is selection on noise.
                "sw_std": float(np.std(sw)) if len(sw) > 1 else float("nan"),
                "sw_mc_std": float(np.nanmean(mc_std)) if mc_std else float("nan"),
                "mmd": float(np.mean(mmd)),
                "chi": float(np.mean(chi)),
                "time": float(np.mean(time_seeds)) if time_seeds else float("nan"),
                "evolve_time": (float(np.mean(evolve_seeds)) if evolve_seeds
                                else float("nan")),
                "n_seeds": len(sw),
            })
    return rows


def collect_reference(results_dir: Path, method: str) -> list[dict]:
    """Load a D-invariant reference method (JAM or Dense) keyed by (N, K).

    JAM lives at `N{N}_K{K}/`; Dense lives at `N{N}_K{K}_D{D}/` but its
    result is D-independent, so we average across D per (N, K).
    """
    base = Path(results_dir) / method
    if not base.exists():
        return []
    bins: dict[tuple[int, int], dict] = {}
    for sub in sorted(base.iterdir()):
        m_nkd = NK_D_PATTERN.match(sub.name)
        m_nk  = NK_PATTERN.match(sub.name)
        if m_nkd:
            N, K = int(m_nkd.group(1)), int(m_nkd.group(2))
        elif m_nk:
            N, K = int(m_nk.group(1)), int(m_nk.group(2))
        else:
            continue
        for f in sorted(sub.glob("seed*.npz")):
            rec = _safe_load(f)
            if rec is None:
                continue
            d = bins.setdefault((N, K), {"sw": [], "mmd": [], "time": [],
                                         "evolve_time": [], "sw_mc_std": []})
            d["sw"].append(rec["sw"])
            d["sw_mc_std"].append(rec["sw_mc_std"])
            d["mmd"].append(rec["mmd"])
            if not np.isnan(rec["time"]):
                d["time"].append(rec["time"])
            if not np.isnan(rec["evolve_time"]):
                d["evolve_time"].append(rec["evolve_time"])
    return [
        {"N": N, "K": K,
         "sw":   float(np.mean(d["sw"])),
         "sw_std": float(np.std(d["sw"])) if len(d["sw"]) > 1 else float("nan"),
         "sw_mc_std": float(np.nanmean(d["sw_mc_std"])) if d["sw_mc_std"] else float("nan"),
         "mmd":  float(np.mean(d["mmd"])),
         "time": float(np.mean(d["time"])) if d["time"] else float("nan"),
         "evolve_time": (float(np.mean(d["evolve_time"])) if d["evolve_time"]
                         else float("nan")),
         "n_seeds": len(d["sw"])}
        for (N, K), d in bins.items()
    ]


def best_classical_sw(results_dir: Path) -> dict:
    """Return {'jam': sw_jam_best, 'dense': sw_dense_best, 'best': min}.

    Used as the threshold benchmark in cost-to-target plots and as the
    reference horizontal lines on sensitivity plots.
    """
    out = {"jam": float("nan"), "dense": float("nan"), "best": float("nan")}
    jam = collect_reference(Path(results_dir), "jam")
    if jam:
        out["jam"] = min(r["sw"] for r in jam)
    dense = collect_reference(Path(results_dir), "dense")
    if dense:
        out["dense"] = min(r["sw"] for r in dense)
    finite = [v for v in (out["jam"], out["dense"]) if not np.isnan(v)]
    if finite:
        out["best"] = min(finite)
    return out


# ── Memory model ──────────────────────────────────────────────────────────

def mps_param_count(N: float, chi: float, d: int) -> float:
    """Approx. # complex doubles in a length-d MPS with physical dim N and
    uniform bond χ. Edges are (1, N, χ) and (χ, N, 1); interior cores are
    (χ, N, χ). Total = 2·N·χ + (d−2)·N·χ² for d ≥ 2.
    """
    return 2.0 * N * chi + max(d - 2, 0) * N * chi * chi


def memory_fraction(N: float, chi: float, d: int) -> float:
    """MPS memory / Dense memory. <1 means MPS is more compact."""
    return mps_param_count(N, chi, d) / max(N ** d, 1.0)


def compression_factor(N: float, chi: float, d: int) -> float:
    """Dense memory / MPS memory (the inverse of memory_fraction)."""
    return (N ** d) / max(mps_param_count(N, chi, d), 1.0)
