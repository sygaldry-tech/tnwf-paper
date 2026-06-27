"""Fetch + preprocess the Moon et al. 2019 embryoid body scRNA-seq trajectory.

Downloads the Mendeley dataset v6n743h5ng/1 (~hundreds of MB) from its S3
mirror, library-size-normalises and log1p-transforms each cell, fits PCA-5
on the pooled matrix, bins cells into the five day-windows
(Day 00-03 / 06-09 / 12-15 / 18-21 / 24-27 ↔ samples
T0_1A / T2_3B / T4_5C / T6_7D / T8_9E), subsamples ``--n_per_step`` cells
per window, and writes::

    data/eb_5d/raw/eb_5d_pca5.npz       (5, n_per_step, 5) snapshots
    data/eb_5d/raw/eb_5d_pca_basis.npz  PCA components, mean, per-dim std
    data/eb_5d/raw/eb_5d_log_norm.npz   post-norm, pre-PCA cache (large)

Run this once before invoking the petals-style training / sweep scripts:

    uv run python scripts/eb_5d/fetch_raw.py
    uv run python scripts/eb_5d/train_jam.py
    uv run python scripts/eb_5d/run_all_methods.py
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse
from sklearn.decomposition import PCA

SAMPLE_DIRS = ["T0_1A", "T2_3B", "T4_5C", "T6_7D", "T8_9E"]
LABELS = ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"]
# Mendeley dataset DOI: 10.17632/v6n743h5ng.1 ("Embryoid body data for PHATE")
# Direct S3 mirror returns 403; use the public-api file-listing endpoint and
# pull the signed `download_url` for the scRNAseq.zip entry (~162 MB).
MENDELEY_FILES_API = (
    "https://data.mendeley.com/public-api/datasets/"
    "v6n743h5ng/files?folder_id=root&version=1"
)
TARGET_FILENAME = "scRNAseq.zip"


# Mendeley's public-api rejects the default `Python-urllib/x.y` User-Agent
# with 403; a browser-style UA succeeds.
_UA_HEADER = {"User-Agent": "Mozilla/5.0 (compatible; tnwf-eb-fetch/1.0)"}


def _urlopen(url: str):
    req = urllib.request.Request(url, headers=_UA_HEADER)
    return urllib.request.urlopen(req)


def _resolve_download_url() -> tuple[str, int]:
    """Query Mendeley public-api for the scRNAseq.zip signed download URL."""
    print(f"[fetch] querying Mendeley public-api → {MENDELEY_FILES_API}")
    with _urlopen(MENDELEY_FILES_API) as resp:
        files = json.loads(resp.read().decode("utf-8"))
    for entry in files:
        if entry["filename"] == TARGET_FILENAME:
            return entry["content_details"]["download_url"], entry["size"]
    raise RuntimeError(
        f"{TARGET_FILENAME!r} not found in Mendeley dataset listing."
    )


def _download(url: str, out_path: Path, expected_size: int = 0) -> None:
    print(f"[fetch] downloading {url}\n        → {out_path}")
    with _urlopen(url) as resp, open(out_path, "wb") as f:
        total = expected_size or int(resp.headers.get("Content-Length", 0))
        chunk_size = 1 << 20
        read = 0
        while True:
            buf = resp.read(chunk_size)
            if not buf:
                break
            f.write(buf)
            read += len(buf)
            if total:
                pct = 100.0 * read / total
                print(f"        {read / 1e6:.1f} / {total / 1e6:.1f} MB "
                      f"({pct:.1f}%)", end="\r", flush=True)
        print()


def _load_10x_sample(sample_dir: Path) -> "scipy.sparse.csr_matrix":
    """Load matrix.mtx as cells × genes sparse float32."""
    mtx_path = sample_dir / "matrix.mtx"
    M = scipy.io.mmread(str(mtx_path))                  # genes × cells
    return scipy.sparse.csr_matrix(M).T.astype(np.float32)


def _normalize_and_log(X_sparse) -> np.ndarray:
    """Library-size normalise to 10k counts per cell, then log1p."""
    counts = np.asarray(X_sparse.sum(axis=1)).ravel()
    counts[counts == 0] = 1.0
    scaling = 1e4 / counts
    X_dense = X_sparse.multiply(scaling[:, None]).toarray().astype(np.float32)
    return np.log1p(X_dense)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/eb_5d/raw")
    p.add_argument("--n_per_step", type=int, default=2000,
                   help="Cells per timepoint snapshot (default 2000).")
    p.add_argument("--n_components", type=int, default=5,
                   help="PCA components (default 5; matches d=5 in DATASET_DEFAULTS).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_download", action="store_true",
                   help="Assume the zip is already at out_dir/v6n743h5ng-1.zip.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / TARGET_FILENAME
    extract_dir = out_dir / "extracted"

    if not args.skip_download and not zip_path.exists():
        url, expected_size = _resolve_download_url()
        _download(url, zip_path, expected_size=expected_size)
    elif not zip_path.exists():
        raise FileNotFoundError(
            f"Expected {zip_path}; remove --skip_download or place the file there."
        )

    if not extract_dir.exists():
        print(f"[fetch] extracting → {extract_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    # Resolve each sample directory under the extracted tree (nesting varies).
    sample_paths = []
    for s in SAMPLE_DIRS:
        candidates = list(extract_dir.rglob(s))
        if not candidates:
            raise FileNotFoundError(f"sample dir {s!r} not found under {extract_dir}")
        sample_paths.append(candidates[0])

    print("[fetch] loading + normalising 10x matrices...")
    log_norm_per_sample: list[np.ndarray] = []
    timepoint_ids: list[np.ndarray] = []
    for k, (label, path) in enumerate(zip(LABELS, sample_paths)):
        X_sparse = _load_10x_sample(path)
        Xn = _normalize_and_log(X_sparse)
        print(f"  {label}: n_cells={Xn.shape[0]}, n_genes={Xn.shape[1]}")
        log_norm_per_sample.append(Xn)
        timepoint_ids.append(np.full(Xn.shape[0], k, dtype=np.int64))

    X_all = np.concatenate(log_norm_per_sample, axis=0)
    t_all = np.concatenate(timepoint_ids, axis=0)

    cache_path = out_dir / "eb_5d_log_norm.npz"
    print(f"[fetch] caching post-norm matrix → {cache_path}")
    np.savez_compressed(cache_path, X=X_all, t=t_all, labels=np.array(LABELS))

    print(f"[fetch] fitting PCA(n_components={args.n_components})...")
    pca = PCA(n_components=args.n_components, random_state=args.seed)
    Z = pca.fit_transform(X_all)
    Z_std_vec = Z.std(axis=0)
    Z_norm = Z / Z_std_vec[None, :]                     # per-dim unit stddev

    rng = np.random.default_rng(args.seed)
    snapshots = np.empty((5, args.n_per_step, args.n_components), dtype=np.float32)
    for k in range(5):
        mask = t_all == k
        cells = Z_norm[mask]
        if cells.shape[0] < args.n_per_step:
            raise RuntimeError(
                f"timepoint {k} ({LABELS[k]}): only {cells.shape[0]} cells "
                f"(< n_per_step={args.n_per_step})."
            )
        idx = rng.choice(cells.shape[0], size=args.n_per_step, replace=False)
        snapshots[k] = cells[idx].astype(np.float32)

    out_path = out_dir / "eb_5d_pca5.npz"
    print(f"[fetch] saving snapshots → {out_path} ({snapshots.shape})")
    np.savez_compressed(
        out_path,
        snapshots=snapshots,
        labels=np.array(LABELS),
        n_per_step=args.n_per_step,
    )

    basis_path = out_dir / "eb_5d_pca_basis.npz"
    print(f"[fetch] saving PCA basis → {basis_path}")
    np.savez_compressed(
        basis_path,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        std=Z_std_vec.astype(np.float32),
    )
    print("[fetch] done.")


if __name__ == "__main__":
    main()
