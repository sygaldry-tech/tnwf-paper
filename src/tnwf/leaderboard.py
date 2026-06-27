"""Aggregate per-run results into a sortable leaderboard.

Scans `results/**/seed*.npz` (and `results/**/N*_K*/seed*.npz` for sweep dirs),
groups by config = (dataset, method, N, d, K, sub_dir), aggregates SW/MMD/NLL/χ
mean ± std across seeds, and emits CSV + Markdown sorted by SW (ascending).

Usage:
    python -m tnwf.leaderboard                    # rebuild from results/
    python -m tnwf.leaderboard --metric nll       # sort by NLL instead
    python -m tnwf.leaderboard --update-after PATH/seed*.npz   # incremental update
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

LEADERBOARD_CSV = "results/leaderboard.csv"
LEADERBOARD_MD = "results/leaderboard.md"
DEFAULT_RESULTS_DIR = "data"      # npz pipeline outputs live in data/, not results/

NK_PATTERN = re.compile(r"^N(\d+)_K(\d+)$")
D_PATTERN = re.compile(r"^d(\d+)$")
N_PATTERN = re.compile(r"^N(\d+)$")


@dataclass
class Row:
    dataset: str
    method: str
    N: int
    d: int
    K: int
    L: float
    sub: str                    # sweep-cell tag (e.g. "N16_K8" or "" for top-level)
    n_seeds: int
    sw_mean: float
    sw_std: float
    mmd_mean: float
    mmd_std: float
    nll_mean: float
    nll_std: float
    chi_mean: float
    chi_max: int
    time_mean: float = float("nan")    # total wall-clock seconds, mean over seeds
    time_std: float = float("nan")
    # Per-seed arrays — keyed by seed index (so paired tests can align across methods)
    sw_by_seed: dict[int, float] = field(default_factory=dict)
    mmd_by_seed: dict[int, float] = field(default_factory=dict)

    def csv_header(self) -> list[str]:
        keys = list(asdict(self).keys())
        return [k for k in keys if k not in ("sw_by_seed", "mmd_by_seed")]

    def csv_row(self) -> list[str]:
        d = asdict(self)
        return [str(d[k]) for k in self.csv_header()]


def _scan_results(root: Path) -> dict[tuple, list[Path]]:
    """Group all seed*.npz under root by (dataset, method, sub_dir)."""
    groups: dict[tuple, list[Path]] = defaultdict(list)
    for npz in root.rglob("seed*.npz"):
        try:
            rel = npz.relative_to(root)                       # dataset/method[/sub]/seedX.npz
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 3:
            continue
        dataset, method, *rest = parts[:-1]                   # strip seedX.npz
        sub = "/".join(rest) if rest else ""
        groups[(dataset, method, sub)].append(npz)
    return groups


def _aggregate(npz_paths: list[Path], dataset: str | None = None) -> Row | None:
    """Aggregate seed runs in npz_paths. If `dataset` is given, it overrides
    the value stored inside the npz files — this is what the directory-based
    scan uses, so e.g. results/swiss_roll_NK/... gets dataset='swiss_roll_NK'
    even though the npz files were written with dataset='swiss_roll_2d'.
    """
    if not npz_paths:
        return None
    sorted_paths = sorted(npz_paths)
    runs = [np.load(p, allow_pickle=True) for p in sorted_paths]
    sample = runs[0]

    # Map each run to its seed (parsed from "seedN.npz")
    import re as _re
    seed_re = _re.compile(r"seed(\d+)\.npz$")
    seeds = []
    for p in sorted_paths:
        m = seed_re.search(str(p.name))
        seeds.append(int(m.group(1)) if m else len(seeds))

    sw = np.array([float(r["sw"][-1]) for r in runs])
    mmd = np.array([float(r["mmd"][-1]) for r in runs])
    # NLL may be missing on older npzs that pre-date target-data NLL.
    if "nll" in sample.files:
        nll = np.array([float(r["nll"][-1]) for r in runs])
    else:
        nll = np.array([float("nan")])
    chi = np.array([int(r["chi_max"][-1]) for r in runs])
    if "total_time" in sample.files:
        times = np.array([float(np.asarray(r["total_time"])) for r in runs])
    else:
        times = np.array([float("nan")])

    sw_by_seed = {s: float(v) for s, v in zip(seeds, sw)}
    mmd_by_seed = {s: float(v) for s, v in zip(seeds, mmd)}

    return Row(
        dataset=dataset if dataset is not None else str(sample["dataset"]),
        method=str(sample["method"]),
        N=int(sample["N"]),
        d=int(sample["d"]),
        K=int(sample["K"]),
        L=float(sample["L"]),
        sub="",                                                # filled by caller
        n_seeds=len(runs),
        sw_mean=float(sw.mean()),
        sw_std=float(sw.std()),
        mmd_mean=float(mmd.mean()),
        mmd_std=float(mmd.std()),
        nll_mean=float(np.nanmean(nll)),
        nll_std=float(np.nanstd(nll)),
        chi_mean=float(chi.mean()),
        chi_max=int(chi.max()),
        time_mean=float(np.nanmean(times)),
        time_std=float(np.nanstd(times)),
        sw_by_seed=sw_by_seed,
        mmd_by_seed=mmd_by_seed,
    )


def rebuild(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> list[Row]:
    """Scan results_dir and return aggregated leaderboard rows."""
    root = Path(results_dir)
    if not root.exists():
        return []
    rows: list[Row] = []
    for (dataset, method, sub), paths in _scan_results(root).items():
        row = _aggregate(paths, dataset=dataset)
        if row is None:
            continue
        row.sub = sub
        rows.append(row)
    return rows


def write_csv(rows: list[Row], path: str | Path = LEADERBOARD_CSV) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("(empty)\n", encoding="utf-8")
        return
    header = rows[0].csv_header()
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(r.csv_row()))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


_MEDAL = {1: "🥇", 2: "🥈", 3: "🥉"}


def _paired_pvalue(metric_by_seed_a: dict, metric_by_seed_b: dict) -> tuple[float, int]:
    """Paired one-sided t-test on common seeds. Returns (p, n_pairs).
    H1: metric_a < metric_b (i.e. method a is better, lower-is-better metric).
    """
    common = sorted(set(metric_by_seed_a) & set(metric_by_seed_b))
    if len(common) < 3:
        return float("nan"), len(common)
    a = np.array([metric_by_seed_a[s] for s in common])
    b = np.array([metric_by_seed_b[s] for s in common])
    diff = a - b
    if diff.std(ddof=1) < 1e-12:
        return (0.5, len(common)) if diff.mean() == 0 else (0.0 if diff.mean() < 0 else 1.0, len(common))
    from scipy import stats as _stats
    t, p_two = _stats.ttest_rel(a, b)
    p_one = p_two / 2 if t < 0 else 1 - p_two / 2     # one-sided: a < b
    return float(p_one), len(common)


def _sig_marker(p: float) -> str:
    if np.isnan(p):
        return "—"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "·"
    return "ns"

# Pretty method labels for the table cells.
_METHOD_LABELS = {
    "jam":        "JAM",
    "dense":      "Dense",
    "tci_tdvp1":  "TCI+TDVP1",
    "tci_tdvp2":  "TCI+TDVP2",
}

# Datasets shown in this order; anything else falls through alphabetically.
_DATASET_ORDER = ["swiss_roll_2d", "swiss_roll_NK",
                  "gmm_2d", "gmm_3d",
                  "gmm_d_scaling", "gmm_N_scaling"]

# Pretty display names for dataset section titles.
_DATASET_TITLES = {
    "swiss_roll_2d":  "Swiss Roll (2D)",
    "swiss_roll_NK":  "Swiss Roll (2D) — N × K phase diagram",
    "gmm_2d":         "Gaussian Mixture (2D)",
    "gmm_3d":         "Gaussian Mixture (3D)",
    "gmm_d_scaling":  "GMM — Scaling with d",
    "gmm_N_scaling":  "GMM — Scaling with N",
}

_SORT_ARROW = " ↓"
_SORT_HEADER_BY_KEY = {
    "sw_mean":  "SW",
    "mmd_mean": "MMD",
    "chi_mean": "χ̄ (max)",
}


def _ordered_datasets(by_dataset: dict[str, list]) -> list[str]:
    known = [ds for ds in _DATASET_ORDER if ds in by_dataset]
    rest = sorted(ds for ds in by_dataset if ds not in _DATASET_ORDER)
    return known + rest


def _pretty_sub_title(sub: str) -> str:
    """`N16_K8` → 'N = 16, K = 8'; '' → 'Default config'."""
    if not sub:
        return "Default config"
    parts = []
    for tok in sub.split("_"):
        if tok.startswith("N") and tok[1:].isdigit():
            parts.append(f"N = {tok[1:]}")
        elif tok.startswith("K") and tok[1:].isdigit():
            parts.append(f"K = {tok[1:]}")
        elif tok.startswith("d") and tok[1:].isdigit():
            parts.append(f"d = {tok[1:]}")
        elif tok.startswith("D") and tok[1:].isdigit():
            parts.append(f"D = {tok[1:]}")
        else:
            parts.append(tok)
    return ", ".join(parts)


def _sub_sort_key(sub: str) -> tuple:
    """Numeric sort key: '' first, then by (N, K, d, D) parsed from the name."""
    if not sub:
        return (0,)                                    # default config first
    nums = {}
    for tok in sub.split("_"):
        if len(tok) >= 2 and tok[0] in "NKdD" and tok[1:].isdigit():
            nums[tok[0]] = int(tok[1:])
    # Order: N, K, d, D — yields ascending sweep traversal
    return (1, nums.get("N", 0), nums.get("K", 0),
            nums.get("d", 0), nums.get("D", 0), sub)


def _render_scan_matrix(by_sub: dict[str, list], sort_by: str) -> str:
    """Compact matrix: rows = method, columns = sub-cell, cell = sort metric mean.

    Lets you scan across N (or N×K) at a glance. Shows just the metric mean
    (one number per cell), with the per-method best bolded.
    """
    metric_key = sort_by.replace("_mean", "_mean")     # already _mean
    # Column order = sorted sub-cells
    sub_order = sorted(by_sub, key=_sub_sort_key)
    # Row order = methods seen, in METHOD_LABELS order then any extras
    seen_methods = set()
    for rows in by_sub.values():
        seen_methods.update(r.method for r in rows)
    method_order = [m for m in _METHOD_LABELS if m in seen_methods]
    method_order += sorted(seen_methods - set(method_order))

    metric_label = _SORT_HEADER_BY_KEY.get(sort_by, sort_by)
    header_cells = ["method"] + [_pretty_sub_title(s) for s in sub_order]
    out_lines = [
        f"#### Quick scan — {metric_label} mean across configs",
        "",
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(["---"] * len(header_cells)) + "|",
    ]
    # For column-wise best (within each sub-cell, lowest metric across methods)
    col_best = {}
    for sub in sub_order:
        sub_rows = by_sub[sub]
        col_best[sub] = min(getattr(r, metric_key) for r in sub_rows)

    for method in method_order:
        cells = [_METHOD_LABELS.get(method, method)]
        for sub in sub_order:
            row = next((r for r in by_sub[sub] if r.method == method), None)
            if row is None:
                cells.append("—")
                continue
            v = getattr(row, metric_key)
            txt = f"{v:.4f}"
            if v == col_best[sub]:
                txt = f"**{txt}**"
            cells.append(txt)
        out_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(out_lines) + "\n"


def _render_dataset_block(
    name: str, rows: list[Row], sort_by: str, level: int = 2
) -> str:
    """Render one dataset's rows as one or more grouped tables."""
    if not rows:
        return ""

    # Header
    title = _DATASET_TITLES.get(name, name)
    head = "#" * level + f" {title}\n\n"

    # Group by sub (so gmm_3d's N×K cells become separate sub-sections)
    by_sub: dict[str, list[Row]] = {}
    for r in rows:
        by_sub.setdefault(r.sub, []).append(r)

    sort_header = _SORT_HEADER_BY_KEY.get(sort_by, sort_by)

    out = [head]
    # ── Quick-scan matrix at the top: rows = method, cols = sub-cell ───────
    if len(by_sub) > 1:
        out.append(_render_scan_matrix(by_sub, sort_by) + "\n")

    for sub_name in sorted(by_sub, key=_sub_sort_key):
        sub_rows = sorted(by_sub[sub_name], key=lambda r: getattr(r, sort_by))
        any_row = sub_rows[0]
        max_seeds = max(r.n_seeds for r in sub_rows)
        config = f"N={any_row.N}, d={any_row.d}, K={any_row.K}, L={any_row.L:g}, n_seeds≤{max_seeds}"
        sub_label = _pretty_sub_title(sub_name)
        out.append(f"### {sub_label}\n_{config}_\n")

        # Best (min) per metric — used for **bold** highlight
        best_sw = min(r.sw_mean for r in sub_rows)
        best_mmd = min(r.mmd_mean for r in sub_rows)

        # Identify the leader (lowest sort_by) for paired comparisons
        leader_row = sub_rows[0]
        leader_seedmap = (
            leader_row.sw_by_seed if sort_by == "sw_mean"
            else leader_row.mmd_by_seed
        )

        cols_raw = ["rank", "method", "SW", "MMD",
                    "p_paired vs #1", "χ̄ (max)", "time (s)", "seeds"]
        # Append arrow to whichever column is the sort key
        cols = [
            (c + _SORT_ARROW) if c == sort_header else c for c in cols_raw
        ]
        widths = [4, 12, 22, 22, 18, 11, 12, 5]
        out.append(
            "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
        )
        out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
        for i, r in enumerate(sub_rows, 1):
            medal = _MEDAL.get(i, "")
            rank = f"{medal}{i:>2}".strip() or str(i)
            sw_s = f"{r.sw_mean:.4f} ± {r.sw_std:.4f}"
            mmd_s = f"{r.mmd_mean:.4f} ± {r.mmd_std:.4f}"
            if r.sw_mean == best_sw:
                sw_s = f"**{sw_s}**"
            if r.mmd_mean == best_mmd:
                mmd_s = f"**{mmd_s}**"
            method_label = _METHOD_LABELS.get(r.method, r.method)

            # Paired p-value vs the leader on the sort metric
            if i == 1:
                p_cell = "(leader)"
            else:
                seedmap = (
                    r.sw_by_seed if sort_by == "sw_mean"
                    else r.mmd_by_seed
                )
                p_one, n_pairs = _paired_pvalue(leader_seedmap, seedmap)
                marker = _sig_marker(p_one)
                p_cell = f"p={p_one:.3f} {marker}" if not np.isnan(p_one) else f"n={n_pairs}"

            time_s = ("—" if np.isnan(r.time_mean)
                      else f"{r.time_mean:.2f} ± {r.time_std:.2f}")
            cells = [
                rank,
                method_label,
                sw_s,
                mmd_s,
                p_cell,
                f"{r.chi_mean:>4.1f} ({r.chi_max})",
                time_s,
                str(r.n_seeds),
            ]
            out.append(
                "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
            )
        out.append("")
    return "\n".join(out) + "\n"


def to_markdown(rows: list[Row], sort_by: str = "sw_mean") -> str:
    if not rows:
        return "_(no results yet)_\n"

    by_dataset: dict[str, list[Row]] = {}
    for r in rows:
        by_dataset.setdefault(r.dataset, []).append(r)

    out = []
    for ds in _ordered_datasets(by_dataset):
        out.append(_render_dataset_block(ds, by_dataset[ds], sort_by))
    return "\n".join(out)


def write_markdown(rows: list[Row], path: str | Path = LEADERBOARD_MD,
                   sort_by: str = "sw_mean") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(rows)
    n_datasets = len({r.dataset for r in rows})
    n_seeds_total = sum(r.n_seeds for r in rows)
    header = (
        f"# tnWF leaderboard\n\n"
        f"**{n_total} configs** across **{n_datasets} datasets**, "
        f"**{n_seeds_total} total runs**. "
        f"Sort: `{sort_by}` (ascending = better; lower SW / MMD is better). "
        f"🥇🥈🥉 mark the top three within each (dataset, sub-cell). "
        f"`p_paired vs #1`: one-sided paired t-test against the leader on common seeds; "
        f"markers: *** p<0.001, ** p<0.01, * p<0.05, · p<0.10, ns otherwise.\n\n"
    )
    p.write_text(header + to_markdown(rows, sort_by=sort_by), encoding="utf-8")


def update_after_batch(results_dir: str | Path = DEFAULT_RESULTS_DIR,
                       sort_by: str = "sw_mean") -> str:
    """Convenience: scan, write csv + md, return the markdown string.

    Note: long-running runner processes import this module once at startup, so
    leaderboard *layout* edits won't take effect until the runner restarts.
    To always get the latest layout, runners can shell out to
    `python -m tnwf.leaderboard` instead of calling this in-process — see
    `update_after_batch_subprocess`.
    """
    rows = rebuild(results_dir)
    write_csv(rows)
    write_markdown(rows, sort_by=sort_by)
    return to_markdown(rows, sort_by=sort_by)


def update_after_batch_subprocess(sort_by: str = "sw_mean") -> None:
    """Like update_after_batch but spawns a fresh `python -m tnwf.leaderboard`.

    The fresh process always loads the current module source, so a long-running
    runner picks up leaderboard.py edits without restart. Slower (~100 ms per
    call) but immune to the stale-import issue.
    """
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "-m", "tnwf.leaderboard", "--metric", sort_by],
        check=False,
        capture_output=True,
    )


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--metric", default="sw_mean",
                   choices=["sw_mean", "mmd_mean", "chi_mean"])
    p.add_argument("--print", action="store_true",
                   help="print markdown to stdout in addition to writing files")
    args = p.parse_args()

    rows = rebuild(args.results_dir)
    write_csv(rows)
    write_markdown(rows, sort_by=args.metric)
    print(f"wrote {LEADERBOARD_CSV} ({len(rows)} rows)")
    print(f"wrote {LEADERBOARD_MD} (sort: {args.metric})")
    if args.print:
        print()
        print(to_markdown(rows, sort_by=args.metric))


if __name__ == "__main__":
    _cli()
