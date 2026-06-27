"""Shared HP-sweep helpers used by the make_fig_* scripts."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

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
    """Open a seed*.npz; return None if it's mid-write or corrupted."""
    import zipfile
    try:
        z = np.load(path, allow_pickle=True)
        out = {
            "sw":   float(z["sw"][-1]),
            "mmd":  float(z["mmd"][-1]),
            "chi":  int(np.asarray(z["chi_max"]).max()) if "chi_max" in z.files else 0,
            "time": float(np.asarray(z["total_time"])) if "total_time" in z.files else float("nan"),
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
        sw, mmd, chi, time_seeds = [], [], [], []
        for f in sorted(sub.glob("seed*.npz")):
            rec = _safe_load(f)
            if rec is None:
                continue
            sw.append(rec["sw"])
            mmd.append(rec["mmd"])
            chi.append(rec["chi"])
            if not np.isnan(rec["time"]):
                time_seeds.append(rec["time"])
        if sw:
            rows.append({
                "N": N, "K": K, "D": D,
                "sw": float(np.mean(sw)),
                "mmd": float(np.mean(mmd)),
                "chi": float(np.mean(chi)),
                "time": float(np.mean(time_seeds)) if time_seeds else float("nan"),
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
            d = bins.setdefault((N, K), {"sw": [], "mmd": [], "time": []})
            d["sw"].append(rec["sw"])
            d["mmd"].append(rec["mmd"])
            if not np.isnan(rec["time"]):
                d["time"].append(rec["time"])
    return [
        {"N": N, "K": K,
         "sw":   float(np.mean(d["sw"])),
         "mmd":  float(np.mean(d["mmd"])),
         "time": float(np.mean(d["time"])) if d["time"] else float("nan"),
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
