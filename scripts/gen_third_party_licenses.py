"""Regenerate `licenses/THIRD_PARTY_LICENSES.md` from `uv.lock`.

Resolution order per package: installed distribution metadata (offline and
authoritative for what is actually in the environment), then the PyPI JSON API
for packages not installed on this platform -- chiefly the Linux-only CUDA
stack that `torch` pulls in.

This is a starting point for review, not a legal determination. What it is good
for is answering two questions quickly: has anything copyleft or proprietary
entered the dependency set, and has that changed since the last release.

Usage:
    uv run python scripts/gen_third_party_licenses.py
    uv run python scripts/gen_third_party_licenses.py --check   # non-zero if anything needs review
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Outbound license of this repository. Compatibility is judged against it.
OUTBOUND = "MIT"

# Classification of license identifiers. Deliberately coarse: the job is to
# separate "no action" from "a human should look at this".
PERMISSIVE = {
    "mit", "mit license", "mit-cmu", "bsd", "bsd license", "bsd-2-clause",
    "bsd-3-clause", "3-clause bsd license", "apache-2.0", "apache 2.0",
    "apache license 2.0", "apache software license", "isc",
    "isc license (iscl)", "psf-2.0", "python software foundation license",
    "zlib", "0bsd", "cc0-1.0", "unlicense", "the unlicense (unlicense)",
    "hpnd", "historical permission notice and disclaimer (hpnd)",
}
WEAK_COPYLEFT = {
    "mpl-2.0", "mozilla public license 2.0 (mpl 2.0)", "lgpl", "lgpl-2.1",
    "lgpl-3.0", "gnu lesser general public license v2 (lgplv2)",
    "gnu lesser general public license v3 (lgplv3)",
}
STRONG_COPYLEFT = {
    "gpl", "gpl-2.0", "gpl-3.0", "agpl-3.0",
    "gnu general public license v2 (gplv2)",
    "gnu general public license v3 (gplv3)",
    "gnu affero general public license v3 (agplv3)",
}


#: NVIDIA ships the CUDA runtime wheels with *empty* license metadata on PyPI:
#: no `license_expression`, no `License ::` classifier, no free-text `license`.
#: The authoritative text is the NVIDIA Software License Agreement bundled in
#: the wheel itself. Reporting these as UNKNOWN would be accurate but useless,
#: so they are classified by package family, and the inference is stated in the
#: report rather than hidden here. It is corroborated by three siblings in the
#: same family that *do* declare a license: cuda-bindings
#: (LicenseRef-NVIDIA-SOFTWARE-LICENSE), nvidia-cublas
#: (LicenseRef-NVIDIA-Proprietary) and nvidia-cusparselt-cu13 (NVIDIA
#: Proprietary Software).
NVIDIA_FAMILY = re.compile(r"^(nvidia-|cuda-)")
NVIDIA_INFERRED = "NVIDIA CUDA EULA (inferred; PyPI metadata empty)"


def classify(lic: str) -> str:
    if not lic or lic.lower().strip() in ("unknown", "none", ""):
        return "UNKNOWN"
    low = lic.lower().strip()
    if low in PERMISSIVE:
        return "permissive"
    if low in STRONG_COPYLEFT or re.search(r"\bagpl|\bgpl-?[23]|gplv[23]", low):
        return "strong-copyleft"
    if low in WEAK_COPYLEFT or "mozilla" in low or "lgpl" in low:
        return "weak-copyleft"
    if "proprietary" in low or "nvidia" in low:
        return "proprietary"
    # Compound expressions ("Apache-2.0 AND MIT") are permissive iff every
    # clause is. An OR of a permissive option is also fine, but we keep the
    # stricter AND-style reading and let anything else fall through to review.
    parts = [p.strip() for p in re.split(r"\s+(?:and|or)\s+|;", low) if p.strip()]
    if len(parts) > 1 and all(p in PERMISSIVE for p in parts):
        return "permissive"
    return "review"


def from_installed(name: str) -> str | None:
    try:
        m = metadata.distribution(name).metadata
    except Exception:
        return None
    if m.get("License-Expression"):
        return m["License-Expression"]
    cls = [c.split("::")[-1].strip() for c in (m.get_all("Classifier") or [])
           if c.startswith("License ::")]
    if cls:
        return "; ".join(sorted(set(cls)))
    lic = (m.get("License") or "").strip()
    if not lic:
        return None
    # Some projects paste their entire license text into this field.
    return lic.splitlines()[0][:60]


def from_pypi(name: str) -> str | None:
    try:
        with urllib.request.urlopen(
                f"https://pypi.org/pypi/{name}/json", timeout=15) as r:
            info = json.load(r)["info"]
    except Exception:
        return None
    if info.get("license_expression"):
        return info["license_expression"]
    cls = [c.split("::")[-1].strip() for c in info.get("classifiers", [])
           if c.startswith("License ::")]
    if cls:
        return "; ".join(sorted(set(cls)))
    lic = (info.get("license") or "").strip()
    return lic.splitlines()[0][:60] if lic else None


def locked_packages() -> list[tuple[str, str]]:
    lock = (ROOT / "uv.lock").read_text()
    pkgs = re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', lock)
    return [(n, v) for n, v in pkgs if n != "tnwf"]


def platform_markers() -> dict[str, str]:
    """Map package -> the environment marker any dependent gates it behind.

    A dependency that only installs on one platform has a different practical
    footprint from one that always installs, so the report states it.
    """
    lock = (ROOT / "uv.lock").read_text()
    out: dict[str, str] = {}
    for name, marker in re.findall(
            r'\{ name = "([^"]+)", marker = "([^"]*)" \}', lock):
        out.setdefault(name, marker)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any package needs review")
    ap.add_argument("--out", default="licenses/THIRD_PARTY_LICENSES.md")
    a = ap.parse_args()

    markers = platform_markers()
    rows, unresolved = [], 0
    for name, ver in locked_packages():
        lic, src = from_installed(name), "installed"
        if not lic:
            lic, src = from_pypi(name), "pypi"
        if not lic and NVIDIA_FAMILY.match(name):
            lic, src = NVIDIA_INFERRED, "family-rule"
        if not lic:
            unresolved += 1
            src = "unresolved"
        rows.append({"name": name, "version": ver, "license": lic or "UNKNOWN",
                     "class": classify(lic or ""), "source": src,
                     "marker": markers.get(name, "")})

    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(r["class"], []).append(r)
    flagged = [r for r in rows if r["class"] != "permissive"]
    ORDER = ("permissive", "weak-copyleft", "strong-copyleft", "proprietary",
             "review", "UNKNOWN")

    out = [
        "# Third-Party Dependency Licenses",
        "",
        "> Generated by `scripts/gen_third_party_licenses.py` from `uv.lock`.",
        f"> Outbound license of this repository: **{OUTBOUND}** (see `LICENSE`).",
        "> License strings come from installed distribution metadata where",
        "> available, otherwise the PyPI JSON API. A starting point for review,",
        "> **not a legal determination**.",
        "",
        "**This repository vendors no third-party code.** Every package below is a",
        "declared dependency resolved from PyPI at install time, not redistributed",
        "here, so obligations that attach on redistributing a combined work do not",
        "arise from this repository on its own.",
        "",
        "The NVIDIA CUDA wheels carry *empty* license metadata on PyPI. They are",
        "classified here by package family, not by a declared identifier -- an",
        "inference, corroborated by three siblings that do declare one",
        "(`cuda-bindings`, `nvidia-cublas`, `nvidia-cusparselt-cu13`). All of them",
        "are gated behind `sys_platform == 'linux'` as transitive dependencies of",
        "`torch`, so they are not installed on macOS or Windows and are never",
        "fetched by the CPU-only reproduction path this repository documents.",
        "",
        f"Total packages: **{len(rows)}**",
        "",
        "## Summary",
        "",
        "| Category | Count |",
        "|---|---|",
    ]
    out += [f"| {k} | {len(buckets[k])} |" for k in ORDER if k in buckets]
    out += ["", "## Needs review", ""]
    if flagged:
        out += ["| Package | Version | License | Category | Installed when |",
                "|---|---|---|---|---|"]
        out += [f"| `{r['name']}` | {r['version']} | {r['license']} | {r['class']} "
                f"| {r['marker'] or 'always'} |"
                for r in sorted(flagged, key=lambda r: (r["class"], r["name"]))]
    else:
        out.append("None -- every dependency resolves to a permissive license.")
    out += ["", "## All packages", "",
            "| Package | Version | License | Category | Source | Installed when |",
            "|---|---|---|---|---|---|"]
    out += [f"| `{r['name']}` | {r['version']} | {r['license']} | {r['class']} "
            f"| {r['source']} | {r['marker'] or 'always'} |"
            for r in sorted(rows, key=lambda r: r["name"])]

    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out) + "\n")

    for k in ORDER:
        if k in buckets:
            print(f"  {k:16s} {len(buckets[k]):3d}")
    print(f"\nwrote {p}")
    if unresolved:
        print(f"  note: {unresolved} package(s) could not be resolved")
    if a.check and flagged:
        print(f"\n{len(flagged)} package(s) need review:")
        for r in sorted(flagged, key=lambda r: (r["class"], r["name"])):
            print(f"  {r['name']:24s} {r['license']:44s} [{r['class']}]")
        sys.exit(1)


if __name__ == "__main__":
    main()
