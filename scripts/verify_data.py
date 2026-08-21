"""Verify an unpacked data archive against its MANIFEST.tsv.

The README used to tell you to run `shasum -a 256` on the tarball and compare
against "the expected digest published with the archive" -- but no digest ships
in this repository, so the step printed a hash and compared it to nothing.

This checks what can actually be checked locally: every file listed in
`data/MANIFEST.tsv` is present, the right size, and hashes to the recorded
sha256. It also reports files present on disk but absent from the manifest.

(The manifest travels inside the archive, so this establishes internal
consistency, not provenance. For provenance, compare the tarball checksum
against the one Zenodo displays on the record page.)

Usage:
    make verify-data
    uv run python scripts/verify_data.py [--data data]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", type=Path)
    a = ap.parse_args()

    manifest = a.data / "MANIFEST.tsv"
    if not manifest.exists():
        raise SystemExit(
            f"{manifest} not found. Unpack the data archive at the repository "
            f"root first (see the Data section of README.md).")

    listed, missing, wrong_size, wrong_hash = set(), [], [], []
    for line in manifest.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rel, size, digest = line.split("\t")
        listed.add(rel)
        p = a.data / rel
        if not p.exists():
            missing.append(rel)
            continue
        if p.stat().st_size != int(size):
            wrong_size.append(rel)
            continue
        if sha256(p) != digest:
            wrong_hash.append(rel)

    on_disk = {p.relative_to(a.data).as_posix()
               for p in a.data.rglob("*")
               if p.is_file() and p.name != "MANIFEST.tsv"}
    extra = sorted(on_disk - listed)

    print(f"manifest lists {len(listed)} files under {a.data}/")
    for label, items in (("missing", missing), ("wrong size", wrong_size),
                         ("hash mismatch", wrong_hash)):
        if items:
            print(f"  {label}: {len(items)}")
            for r in items[:10]:
                print(f"    {r}")
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")
    if extra:
        print(f"  not in manifest: {len(extra)} (harmless if you generated them)")
        for r in extra[:5]:
            print(f"    {r}")

    bad = len(missing) + len(wrong_size) + len(wrong_hash)
    print("OK — every listed file matches." if bad == 0
          else f"FAILED — {bad} file(s) do not match the manifest.")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
