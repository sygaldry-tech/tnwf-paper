"""Robust resourced-cross gate runner (unbuffered, per-cell flush, OOM-safe).

Each cell is wrapped in try/except so an OOM/SIGKILL on one heavy cell doesn't
lose the others' results. Writes every cell line to stdout (flushed) AND appends
to an results file. Run with `python -u`.

Usage:
  python -u scripts/resourced_gate.py --d 4 --dataset gmm_4d --method tci_tdvp1 \
      --out results/resourced_gate_d4.txt --seeds 0 \
      --cells "N32,K8,DV16,DO16,M1,g0;N32,K8,DV64,DO64,M1,g0;..."
"""
from __future__ import annotations
import argparse, sys, time, traceback
import numpy as np
from tnwf.pipelines.run_evolution import run


def parse_cell(s):
    p = {kv[:2]: kv[2:] for kv in s.split(",")}  # crude: keys N,K,DV->'DV', etc.
    # explicit parse
    d = {}
    for tok in s.split(","):
        tok = tok.strip()
        if tok.startswith("DV"): d["DV"] = int(tok[2:])
        elif tok.startswith("DO"): d["DO"] = int(tok[2:])
        elif tok.startswith("N"): d["N"] = int(tok[1:])
        elif tok.startswith("K"): d["K"] = int(tok[1:])
        elif tok.startswith("M"): d["M"] = int(tok[1:])
        elif tok.startswith("g"): d["g"] = int(tok[1:])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", default="tci_tdvp1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--nsc", type=int, default=2, help="n_sweeps_cross")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", required=True, help="semicolon-separated cells")
    a = ap.parse_args()
    cells = [parse_cell(c) for c in a.cells.split(";") if c.strip()]
    fh = open(a.out, "a")
    def emit(line):
        print(line); sys.stdout.flush()
        fh.write(line + "\n"); fh.flush()
    emit(f"# d={a.d} {a.dataset} {a.method} nsc={a.nsc} seeds={a.seeds}")
    for c in cells:
        N, K = c["N"], c["K"]
        DV, DO = c.get("DV", 16), c.get("DO", c.get("DV", 16))
        M, g = c.get("M", 1), c.get("g", 0)
        tag = f"N{N}_K{K}_DV{DV}_DO{DO}_M{M}_g{g}"
        sws = []
        try:
            t0 = time.time()
            for s in a.seeds:
                out = run(method=a.method, dataset=a.dataset, V_source="analytic",
                          seed=s, N=N, d=a.d, K=K, n_samples=500, save=False,
                          method_kwargs=dict(D_max=DO, D_V=DV, D_out=DO, D_init=DO,
                                             n_sweeps=2, n_sweeps_cross=a.nsc,
                                             n_v_substeps=M, n_global=g))
                sws.append(float(out["sw"][-1]))
            chi = int(out["chi_max"][-1])
            dt = time.time() - t0
            emit(f"{tag}: SW={np.mean(sws):.3f} per_seed={np.round(sws,3).tolist()} "
                 f"chi={chi} t={dt:.0f}s")
        except Exception as e:
            emit(f"{tag}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc()
    fh.close()


if __name__ == "__main__":
    main()
