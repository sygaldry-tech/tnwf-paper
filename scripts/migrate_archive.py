"""Migrate the results archive from the v1 to the v2 coordinate convention.

Why this exists
---------------
Releases up to v1 wrote Born samples dithered to the *right* of the grid node
(``x = (i + U(0,1))·dx``) while ψ is stored as its values *at* the nodes, so
every stored sample carries a rigid ``+dx/2`` bias on every axis. See
``src/tnwf/coords.py``. The shift is exactly invertible, so no run has to be
repeated: the archive can be rescored in place of re-simulated.

What it does, per results file
------------------------------
* shifts ``samples_T``, and ``samples_per_step`` where the record still carries
  it, by ``-dx/2`` (grid methods only);
* recomputes the endpoint metrics from the corrected samples with a **fixed,
  recorded** projection seed, as ``sw_endpoint`` / ``mmd_endpoint``;
* keeps the original arrays verbatim as ``sw_legacy_v1`` / ``mmd_legacy_v1``;
* stamps ``coord_convention`` so the result is self-describing.

Two deliberate choices worth knowing
------------------------------------
**The endpoint is recomputed from ``samples_T``, not from the trajectory.**
``_record`` draws a fresh ``n_samples`` cloud per step and scores that, while
``samples_per_step`` stores only a 200-sample subsample. Recomputing the
trajectory from those snapshots would silently change estimator — the n=200
finite-sample floor is 2–4x the n=500 one — and would read as a spurious
collapse at the final step. So the per-step array is carried over untouched as
``sw_legacy_v1`` and only the endpoint, which is the only thing any paper float
reads, is recomputed at the matching sample size.

**``sw`` and ``mmd`` are not written.** A consumer that has not been updated
should fail loudly rather than silently read a stale, biased number under a
familiar key. ``_hp_utils`` reads ``sw_endpoint``.

Usage
-----
    python scripts/migrate_archive.py --src /path/to/v1/data --dst /path/to/v2/data
    python scripts/migrate_archive.py --src ... --dst ... --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tnwf import coords  # noqa: E402
from tnwf.metrics.mmd import mmd_rbf  # noqa: E402
from tnwf.metrics.sw import sliced_wasserstein  # noqa: E402

#: Projection seed for every recomputed endpoint. One global constant, not a
#: per-file seed: the consumer is an argmin over cells, and common random
#: numbers make cell-to-cell comparisons far more stable than independent
#: draws would.
SW_SEED = 20240917
SW_PROJECTIONS = 128

#: Extra projection seeds used to report the estimator's own Monte-Carlo
#: spread, so a reader can tell a real argmin move from selection noise.
SW_NOISE_REPLICATES = 32

#: Keys a results file must have before we will touch it. Anything else
#: (traj.npz, checkpoints, CSVs) is copied through untouched.
#:
#: ``samples_per_step`` is deliberately absent: the published archive strips it
#: (see scripts/package_data_archive.py), so requiring it would make this script
#: refuse the very records it documents. It is still shifted when present.
REQUIRED = {"samples_T", "target", "sw", "mmd",
            "N", "d", "K", "L", "method"}

#: Sample arrays to shift, if the record carries them.
SHIFT_KEYS = ("samples_T", "samples_per_step")

# Gate tolerance on the offset, in units of dx.
#
# The offset is measured as (mean(samples) - mean(target)) / dx. Its standard
# error scales as 1/dx, so a file at N=256 (dx=0.031) carries ~1500x less
# information than one at N=16 (dx=0.5) — an unweighted group mean is dominated
# by exactly the files that know least. Hence inverse-variance weighting, and a
# tolerance that widens when a group genuinely has no power (e.g. the
# single-file d=7 and d=8 groups).
OFFSET_TOL = 0.25
OFFSET_TOL_SIGMA = 3.0

# Below this combined standard error a group is considered to have enough
# power for its verdict to mean anything; above it we say so out loud rather
# than reporting a pass that is really an absence of evidence.
LOW_POWER_SE = 0.20


def _sw(a: np.ndarray, b: np.ndarray, seed: int = SW_SEED) -> float:
    return float(sliced_wasserstein(a, b, n_projections=SW_PROJECTIONS,
                                    rng=np.random.default_rng(seed)))


def is_results_file(path: Path) -> tuple[bool, object]:
    """True if this .npz is a per-cell results record we should migrate."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception:
        return False, None
    return REQUIRED.issubset(set(z.files)), z


def group_of(path: Path, src: Path) -> str:
    """`<dataset>/<method>` — the granularity the offset gates operate at."""
    rel = path.relative_to(src).parts
    return "/".join(rel[:2]) if len(rel) >= 2 else rel[0]


def measure(src: Path) -> tuple[dict, list]:
    """Pass 1: read every file, measure per-group offsets and boundary atoms.

    Runs to completion before a single byte is written, so a bad allowlist
    entry is caught by the data rather than by trust.
    """
    stats: dict[str, dict] = defaultdict(
        lambda: {"ratio": [], "se": [], "dx": [], "abs": [],
                 "boundary": 0, "n": 0, "methods": set()})
    todo = []
    for path in sorted(src.rglob("*.npz")):
        ok, z = is_results_file(path)
        if not ok:
            continue
        method = str(np.asarray(z["method"]).item())
        L, N = float(z["L"]), int(z["N"])
        dx = L / N
        s = np.asarray(z["samples_T"], np.float64)
        t = np.asarray(z["target"], np.float64)
        n, d = s.shape
        # Mean displacement averaged over axes, and its standard error.
        offset = float((s.mean(0) - t.mean(0)).mean())
        se = float(np.sqrt((s.var(0) / n + t.var(0) / n).sum()) / d)
        g = group_of(path, src)
        st = stats[g]
        st["ratio"].append(offset / dx)
        st["se"].append(se / dx)
        st["dx"].append(dx)
        st["abs"].append(offset)
        # JAM clamps to exactly [0, L]; grid Born samples cannot produce
        # either endpoint exactly. A group-level discriminator, not a
        # per-file one — it fires on only ~18% of individual jam files.
        st["boundary"] += int(np.count_nonzero((s <= 0.0) | (s >= L)))
        st["n"] += 1
        st["methods"].add(method)
        todo.append((path, method, dx))
    for st in stats.values():
        w = 1.0 / np.maximum(np.asarray(st["se"]), 1e-12) ** 2
        st["offset_dx"] = float(np.sum(w * np.asarray(st["ratio"])) / np.sum(w))
        st["offset_se"] = float(1.0 / np.sqrt(np.sum(w)))
    return stats, todo


def check_gates(stats: dict) -> tuple[list[str], list[str]]:
    """Return (failures, warnings). A non-empty failure list blocks the write."""
    problems, warnings = [], []
    for g, st in sorted(stats.items()):
        if len(st["methods"]) != 1:
            problems.append(f"{g}: mixed methods {sorted(st['methods'])}")
            continue
        method = next(iter(st["methods"]))
        try:
            expect = coords.shift_for_method(method)
        except coords.CoordConventionError as e:
            problems.append(f"{g}: {e}")
            continue

        off, se = st["offset_dx"], st["offset_se"]
        tol = max(OFFSET_TOL, OFFSET_TOL_SIGMA * se)
        if abs(off - expect) > tol:
            problems.append(
                f"{g}: offset {off:+.3f} ± {se:.3f} dx is {abs(off - expect):.3f} "
                f"from the {expect:+.1f} expected for method {method!r} "
                f"(tolerance {tol:.3f}, n={st['n']} files)")
        if se > LOW_POWER_SE:
            warnings.append(
                f"{g}: offset {off:+.3f} ± {se:.3f} dx — too little power to "
                f"confirm the convention (n={st['n']} files); relying on the "
                f"method allowlist alone")

        # Grid Born samples cannot land exactly on the domain boundary; the
        # JAM integrator clamps there. A single boundary atom in a group we
        # are about to shift means the allowlist is wrong.
        if expect and st["boundary"]:
            problems.append(
                f"{g}: {st['boundary']} samples sit exactly on the domain "
                f"boundary, which grid Born sampling cannot produce — is "
                f"{method!r} really grid-sampled?")

        # Independent check, and the sharpest one where several grid sizes are
        # present: a convention bias is a fixed multiple of dx, whereas a
        # genuine flow displacement is a fixed distance in world units. So the
        # ratio should be flat in dx for grid methods and grow like 1/dx for
        # continuous ones.
        #
        # Restricted to files whose own standard error is small enough to say
        # anything. Without that restriction the test reads |ratio| off
        # large-N files that are pure noise (se up to 1.5 dx at N=256) and
        # reports a trend that is an artifact of taking |.| of noise.
        dxs = np.asarray(st["dx"])
        ses = np.asarray(st["se"])
        ratios = np.asarray(st["ratio"])
        powered = ses < OFFSET_TOL
        if powered.sum() >= 8 and len(np.unique(dxs[powered])) >= 3:
            r = float(np.corrcoef(np.log(dxs[powered]),
                                  np.log(np.abs(ratios[powered]) + 1e-9))[0, 1])
            if expect and r < -0.5:
                problems.append(
                    f"{g}: offset/dx grows as dx shrinks (log-log r={r:+.2f} "
                    f"over {int(powered.sum())} files with power), which is the "
                    f"signature of an absolute displacement, not a half-cell "
                    f"convention — refusing to shift {method!r}")
    return problems, warnings


def migrate_file(path: Path, out: Path, method: str, dx: float) -> dict:
    z = np.load(path, allow_pickle=True)
    payload = {k: z[k] for k in z.files}

    if coords.CONV_KEY in z.files:
        conv = str(np.asarray(z[coords.CONV_KEY]).item())
        if conv == coords.CONV_CELL:
            return {"status": "already-v2"}

    shift = -coords.shift_for_method(method) * dx
    for key in SHIFT_KEYS:
        if key not in payload:
            continue
        arr = np.asarray(payload[key], np.float64) + shift
        payload[key] = arr.astype(np.float32)

    s = np.asarray(payload["samples_T"], np.float64)
    t = np.asarray(payload["target"], np.float64)

    payload["sw_legacy_v1"] = payload.pop("sw")
    payload["mmd_legacy_v1"] = payload.pop("mmd")
    payload["sw_endpoint"] = np.asarray(_sw(s, t))
    payload["mmd_endpoint"] = np.asarray(float(mmd_rbf(s, t)))
    payload["sw_endpoint_mc_std"] = np.asarray(float(np.std(
        [_sw(s, t, seed=SW_SEED + 1 + i) for i in range(SW_NOISE_REPLICATES)])))
    payload["sw_proj_seed"] = np.asarray(SW_SEED)
    payload["sw_n_projections"] = np.asarray(SW_PROJECTIONS)
    payload["sw_endpoint_n"] = np.asarray(s.shape[0])
    payload[coords.CONV_KEY] = np.asarray(coords.CONV_CELL)
    payload[coords.SCHEMA_KEY] = np.asarray(coords.SCHEMA)

    out.parent.mkdir(parents=True, exist_ok=True)
    # The suffix must stay ".npz": savez_compressed appends one otherwise, and
    # the rename would then chase a file that was never written.
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, out)

    # np.savez_compressed silently drops anything it cannot store, so verify
    # the round-trip rather than assume it.
    back = np.load(out, allow_pickle=True)
    missing = set(payload) - set(back.files)
    if missing:
        raise RuntimeError(f"{out}: keys lost on write: {sorted(missing)}")
    return {"status": "migrated",
            "sw_v1": float(np.asarray(z["sw"])[-1]),
            "sw_v2": float(payload["sw_endpoint"]),
            "mc_std": float(payload["sw_endpoint_mc_std"])}


def write_manifest(dst: Path) -> int:
    rows = []
    for p in sorted(dst.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.tsv":
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append(f"{p.relative_to(dst).as_posix()}\t{p.stat().st_size}\t{h}")
    (dst / "MANIFEST.tsv").write_text(
        "# path\tsize_bytes\tsha256\n" + "\n".join(rows) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = a.src.resolve()
    print(f"[1/4] measuring {src} ...", flush=True)
    stats, todo = measure(src)
    for g, st in sorted(stats.items()):
        print(f"      {g:34s} n={st['n']:4d}  "
              f"offset={st['offset_dx']:+.3f} ± {st['offset_se']:.3f} dx"
              f"  boundary_atoms={st['boundary']}")

    print("[2/4] gates ...", flush=True)
    problems, warnings = check_gates(stats)
    for w in warnings:
        print(f"      WARN {w}")
    if problems:
        print("\nGATE FAILURES — nothing written:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print(f"      all {len(stats)} groups pass; {len(todo)} files to migrate")

    if a.dry_run:
        print("[dry-run] stopping before any write.")
        return

    dst = a.dst.resolve()
    print(f"[3/4] copying pass-through files to {dst} ...", flush=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    print("[4/4] migrating results files ...", flush=True)
    todo_set = {p for p, _, _ in todo}
    deltas, done = [], 0
    for path, method, dx in todo:
        r = migrate_file(path, dst / path.relative_to(src), method, dx)
        if r["status"] == "migrated":
            deltas.append(r["sw_v2"] - r["sw_v1"])
        done += 1
        if done % 250 == 0:
            print(f"      {done}/{len(todo)}", flush=True)

    # Every results file must now resolve, and nothing else should have moved.
    for path in sorted(dst.rglob("*.npz")):
        z = np.load(path, allow_pickle=True)
        if REQUIRED.issubset(set(z.files)) or "sw_endpoint" in z.files:
            assert coords.convention_of(z, path) == coords.CONV_CELL, path

    n = write_manifest(dst)
    d = np.asarray(deltas)
    print(f"\nmigrated {len(deltas)} results files ({len(todo_set)} candidates)")
    print(f"  SW change: median {np.median(d):+.4f}  "
          f"p10 {np.percentile(d, 10):+.4f}  p90 {np.percentile(d, 90):+.4f}")
    print(f"  MANIFEST.tsv: {n} files")


if __name__ == "__main__":
    main()
