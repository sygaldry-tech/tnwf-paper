"""Package the results archive for deposit, and emit its checksum + metadata.

Produces `tnwf-paper-data.tar.gz`, whose members are all rooted at `data/` so
that unpacking at the repository root puts the sweeps exactly where the figure
and table generators look for them.

This script is the single definition of what ships. The staging tree stays a
complete record of the sweeps; the reductions below are applied here, on the way
into the tarball, so they are reviewable in one place and nothing is destroyed.

Four deliberate choices:

* Files that the repository already tracks under `data/` are **excluded**.
  `data/exact_sw_floor.json` and `data/COORD_CONVENTIONS.tsv` are source, not
  bulk data; shipping them in the archive too would mean unpacking silently
  overwrites tracked files, which is fine while the copies agree and a
  confusing mess the moment they don't.
* Runs of methods the paper does not report (`aci`, `tci_als`) are **excluded**.
  No figure or table generator reads them -- `METHODS` in `scripts/_hp_utils.py`
  omits both -- and they were a third of the payload.
* `samples_per_step` is **stripped** from each sweep record. It is a
  `(K+1, 200, d)` trajectory subsample and the largest member by far, but no
  figure or table opens it; the endpoint cloud the generators actually plot is
  `samples_T`, which is kept. Every metric, the bond-dimension trajectory, the
  timings and the pre-migration `*_legacy_v1` values are kept as well.
* `MANIFEST.tsv` is regenerated to describe exactly what the tarball contains,
  rather than copied from the source tree, so `make verify-data` on an unpacked
  copy checks the archive rather than the tree it was built from.

Reproducibility: members are added in sorted order with normalized metadata
(fixed mtime, uid/gid 0, mode 0644/0755), and gzip is invoked with mtime=0, so
byte-identical inputs give a byte-identical tarball and therefore a stable
sha256.

Usage:
    uv run python scripts/package_data_archive.py --src /path/to/data-v2
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat

ROOT = Path(__file__).resolve().parents[1]

#: Rooted at data/ so the archive unpacks at the repository root.
PREFIX = "data"

#: Fixed timestamp for every member (2026-01-01T00:00:00Z), for reproducibility.
FIXED_MTIME = 1767225600

#: The same instant as a zip DOS timestamp, for rewritten npz members.
FIXED_ZIP_DATE = (2026, 1, 1, 0, 0, 0)

#: Sweep sub-directories whose method the paper does not report.
EXCLUDE_METHODS = {"aci", "tci_als"}

#: npz members dropped on the way into the archive. See the module docstring.
DROP_MEMBERS = {"samples_per_step"}


def repo_tracked_under_data() -> set[str]:
    """Paths the repository already tracks under data/, relative to data/."""
    import subprocess
    out = subprocess.run(["git", "ls-files", "data/"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return {p[len("data/"):] for p in out if p.startswith("data/")}


#: Filesystem and editor debris that must never reach a public archive. The
#: staging tree lives on a Mac, so simply browsing it in Finder -- or running
#: `ls` in some shells -- creates .DS_Store files inside it. Nothing filtered
#: them before, so a rebuild would have shipped them to Zenodo; the only reason
#: the published archive is clean is that it predates the ones now on disk.
CRUFT = {".DS_Store", "Thumbs.db", "desktop.ini", ".directory"}
CRUFT_PREFIXES = ("._",)          # macOS AppleDouble resource forks
CRUFT_SUFFIXES = (".swp", ".swo", "~", ".orig", ".rej")


def is_cruft(rel: str) -> bool:
    name = Path(rel).name
    return (name in CRUFT
            or name.startswith(CRUFT_PREFIXES)
            or name.endswith(CRUFT_SUFFIXES)
            or "__MACOSX" in Path(rel).parts)


def collect(src: Path, exclude: set[str]) -> list[tuple[Path, str]]:
    files = []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        if rel == "MANIFEST.tsv" or rel in exclude:
            continue
        if EXCLUDE_METHODS.intersection(Path(rel).parts):
            continue
        if is_cruft(rel):
            continue
        files.append((p, rel))
    return files


def write_npz_fixed_time(dst: Path, arrays: dict) -> None:
    """`np.savez_compressed` with a fixed zip timestamp.

    numpy stamps every zip entry with the local time at write, so a plain
    `savez_compressed` would make each rebuild produce different bytes and thus a
    different archive sha256 -- defeating the reproducibility this script
    otherwise guarantees.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            npformat.write_array(buf, np.asanyarray(arr), allow_pickle=False)
            zi = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_ZIP_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue())


def materialize(files: list[tuple[Path, str]], workdir: Path) -> tuple[list[tuple[Path, str]], int]:
    """Rewrite npz members that carry a dropped array; pass everything else through.

    Returns the effective (path, rel) list and the number of records rewritten.
    Member order is preserved so a stripped record still reads in its original
    key order.
    """
    out, n = [], 0
    for p, rel in files:
        if p.suffix != ".npz":
            out.append((p, rel))
            continue
        with np.load(p, allow_pickle=False) as z:
            keep = [k for k in z.files if k not in DROP_MEMBERS]
            if len(keep) == len(z.files):
                out.append((p, rel))
                continue
            arrays = {k: z[k] for k in keep}
        dst = workdir / rel
        write_npz_fixed_time(dst, arrays)
        out.append((dst, rel))
        n += 1
    return out, n


def build_manifest(files: list[tuple[Path, str]]) -> bytes:
    rows = ["# path\tsize_bytes\tsha256"]
    for p, rel in files:
        rows.append(f"{rel}\t{p.stat().st_size}\t"
                    f"{hashlib.sha256(p.read_bytes()).hexdigest()}")
    return ("\n".join(rows) + "\n").encode()


def norm(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    ti.mtime = FIXED_MTIME
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mode = 0o755 if ti.isdir() else 0o644
    return ti


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="migrated (v2) archive tree, i.e. the contents of data/")
    ap.add_argument("--out", default="tnwf-paper-data.tar.gz", type=Path)
    a = ap.parse_args()

    src = a.src.resolve()
    exclude = repo_tracked_under_data()
    files = collect(src, exclude)
    raw_bytes = sum(p.stat().st_size for p, _ in files)

    # Strip into a scratch tree rather than in place: the staging directory is
    # the complete record and must not be mutated by packaging.
    with tempfile.TemporaryDirectory(prefix="tnwf-pack-") as tmp:
        files, n_stripped = materialize(files, Path(tmp))
        kept_bytes = sum(p.stat().st_size for p, _ in files)
        manifest = build_manifest(files)

        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tf:
            mi = tarfile.TarInfo(f"{PREFIX}/MANIFEST.tsv")
            mi.size = len(manifest)
            tf.addfile(norm(mi), io.BytesIO(manifest))
            for p, rel in files:
                ti = tf.gettarinfo(str(p), arcname=f"{PREFIX}/{rel}")
                with p.open("rb") as fh:
                    tf.addfile(norm(ti), fh)

    out = a.out if a.out.is_absolute() else ROOT / a.out
    with open(out, "wb") as fh:
        # filename="" matters: GzipFile otherwise derives the gzip header's
        # FNAME field from fileobj.name, so building to a different output path
        # changes the compressed bytes and therefore the published sha256, even
        # though the tar stream underneath is identical.
        with gzip.GzipFile(filename="", fileobj=fh, mode="wb",
                           compresslevel=9, mtime=0) as gz:
            gz.write(raw.getvalue())

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    size = out.stat().st_size
    print(f"wrote {out}")
    print(f"  members : {len(files) + 1} ({len(files)} data files + MANIFEST.tsv)")
    if exclude:
        print(f"  excluded (tracked in the repository): {sorted(exclude)}")
    print(f"  excluded (methods not reported): {sorted(EXCLUDE_METHODS)}")
    print(f"  stripped {sorted(DROP_MEMBERS)} from {n_stripped} records: "
          f"{raw_bytes / 1e6:.1f} MB -> {kept_bytes / 1e6:.1f} MB uncompressed")
    print(f"  size    : {size / 1e6:.1f} MB")
    print(f"  sha256  : {digest}")

    meta = {"archive": out.name, "size_bytes": size, "sha256": digest,
            "n_files": len(files) + 1, "prefix": PREFIX + "/"}
    (out.parent / "tnwf-paper-data.sha256").write_text(f"{digest}  {out.name}\n")
    (out.parent / "tnwf-paper-data.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  checksum: {out.parent / 'tnwf-paper-data.sha256'}")


if __name__ == "__main__":
    main()
