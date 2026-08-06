"""Package the results archive for deposit, and emit its checksum + metadata.

Produces `tnwf-paper-data.tar.gz`, whose members are all rooted at `data/` so
that unpacking at the repository root puts the sweeps exactly where the figure
and table generators look for them.

Two deliberate choices:

* Files that the repository already tracks under `data/` are **excluded**.
  `data/exact_sw_floor.json` and `data/COORD_CONVENTIONS.tsv` are source, not
  bulk data; shipping them in the archive too would mean unpacking silently
  overwrites tracked files, which is fine while the copies agree and a
  confusing mess the moment they don't.
* `MANIFEST.tsv` is regenerated to describe exactly what the tarball contains,
  rather than copied from the source tree, so `make verify-data` on an unpacked
  copy checks the archive rather than the tree it was built from.

Reproducibility: members are added in sorted order with normalised metadata
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Rooted at data/ so the archive unpacks at the repository root.
PREFIX = "data"

#: Fixed timestamp for every member (2026-01-01T00:00:00Z), for reproducibility.
FIXED_MTIME = 1767225600


def repo_tracked_under_data() -> set[str]:
    """Paths the repository already tracks under data/, relative to data/."""
    import subprocess
    out = subprocess.run(["git", "ls-files", "data/"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return {p[len("data/"):] for p in out if p.startswith("data/")}


def collect(src: Path, exclude: set[str]) -> list[tuple[Path, str]]:
    files = []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        if rel == "MANIFEST.tsv" or rel in exclude:
            continue
        files.append((p, rel))
    return files


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
    print(f"  size    : {size / 1e6:.1f} MB")
    print(f"  sha256  : {digest}")

    meta = {"archive": out.name, "size_bytes": size, "sha256": digest,
            "n_files": len(files) + 1, "prefix": PREFIX + "/"}
    (out.parent / "tnwf-paper-data.sha256").write_text(f"{digest}  {out.name}\n")
    (out.parent / "tnwf-paper-data.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  checksum: {out.parent / 'tnwf-paper-data.sha256'}")


if __name__ == "__main__":
    main()
