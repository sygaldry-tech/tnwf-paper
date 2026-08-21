# Third-Party Notices

This software depends on the packages listed below. Each is reproduced
in full in this directory, verbatim and unmodified, as their licenses
require. Nothing here is a link standing in for a license text.

tnwf itself is MIT — see [`../LICENSE`](../LICENSE). This repository
contains no third-party source code; these are the terms of the
dependencies it installs.

## Run on CPU

**This release is intended to run on CPU, and does not need a GPU.**
`torch` is pinned to the PyTorch CPU index (see `[tool.uv.sources]` in
`pyproject.toml`), so a default install pulls no NVIDIA package and the
figures and tables all regenerate on CPU.

Installing `torch` from the default index instead — which
`pyproject.toml` describes for anyone wanting GPU training — resolves
roughly 19 additional packages (`nvidia-*`, `cuda-*`, `triton`) under the
NVIDIA CUDA Toolkit EULA and NVIDIA Software License Agreement. Those are
proprietary terms between you and NVIDIA. This release neither
redistributes those packages nor grants any license to them, and no
notice here covers them.

## Dynamically linked components carrying copyleft

`scipy` ships three shared libraries inside its wheel, used by dynamic
linking and unmodified:

| Library | License |
|---|---|
| `libgfortran.5.dylib` | GPL-3.0-or-later WITH GCC-exception-3.1 |
| `libgcc_s.1.1.dylib` | GPL-3.0-or-later WITH GCC-exception-3.1 |
| `libquadmath.0.dylib` | LGPL-2.1-or-later |

The GCC Runtime Library Exception exists to permit linking into non-GPL
software, so the GPL components impose no copyleft obligation on this
code. libquadmath is LGPL-2.1-or-later, satisfied by dynamic linking
against an unmodified library plus this notice.

Full texts: the GPLv3 and the GCC Runtime Library Exception are inside
[`scipy-LICENSE.txt`](scipy-LICENSE.txt); the LGPL-2.1 body is in
[`LGPL-2.1.txt`](LGPL-2.1.txt).

**numpy is deliberately not listed here.** Its license file declares the
same components, but the installed wheel bundles no shared libraries at
all — it links Apple Accelerate. Repeating the declaration would claim we
redistribute binaries we do not. numpy's license is still reproduced
verbatim, including those stale declarations, because editing a copyright
holder's notice is worse than reproducing it as written.

## Scope and platform

License text can only be read from installed wheels, so these notices
describe the platform they were generated on (macOS, arm64). A Linux
install resolves different wheels and may bundle different components —
numpy there typically bundles OpenBLAS and libgfortran, which it does not
here. The dependency *audit* accompanying this release is lock-driven
precisely to avoid that blind spot, but no lock file records license
text. Regenerate on a platform if you need its exact set.

Each package's own primary license is reproduced. Licenses for components
bundled *inside* a dependency are reproduced only where that dependency
actually ships the component here — in practice scipy.

## Packages

| Package | Version | License | Copyright | Text |
|---|---|---|---|---|
| colorama | 0.4.6 | BSD-3-Clause | Copyright (c) 2010 Jonathan Hartley | [`colorama-LICENSE.txt`](colorama-LICENSE.txt) |
| contourpy | 1.3.3 | BSD-3-Clause | Copyright (c) 2021-2025, ContourPy Developers. | [`contourpy-LICENSE`](contourpy-LICENSE) |
| cycler | 0.12.1 | BSD-3-Clause | Copyright (c) 2015, matplotlib project | [`cycler-LICENSE`](cycler-LICENSE) |
| filelock | 3.29.0 | MIT | Copyright (c) 2025 Bernát Gábor and contributors | [`filelock-LICENSE`](filelock-LICENSE) |
| fonttools | 4.62.1 | MIT | Copyright (c) 2017 Just van Rossum | [`fonttools-LICENSE`](fonttools-LICENSE) |
| fsspec | 2026.4.0 | BSD-3-Clause | Copyright (c) 2018, Martin Durant | [`fsspec-LICENSE`](fsspec-LICENSE) |
| iniconfig | 2.3.0 | MIT | Copyright (c) 2010 - 2023 Holger Krekel and others | [`iniconfig-LICENSE`](iniconfig-LICENSE) |
| jinja2 | 3.1.6 | — | Copyright 2007 Pallets | [`jinja2-LICENSE.txt`](jinja2-LICENSE.txt) |
| joblib | 1.5.3 | BSD-3-Clause | Copyright (c) 2008-2021, The joblib developers. | [`joblib-LICENSE.txt`](joblib-LICENSE.txt) |
| kiwisolver | 1.5.0 | BSD-3-Clause | Copyright (c) 2013-2026, Nucleic Development Team | [`kiwisolver-LICENSE`](kiwisolver-LICENSE) |
| markupsafe | 3.0.3 | BSD-3-Clause | Copyright 2010 Pallets | [`markupsafe-LICENSE.txt`](markupsafe-LICENSE.txt) |
| matplotlib | 3.10.9 | PSF-2.0 (matplotlib license, PSF-derived) | Copyright (c) 1997, 2009, American Mathematical Society (http://www.ams.org).; Copyright (C) 1994, 1995, Basil K. Malyshev. All Rights Reserved.; Copyright (c) 2002 Cynthia Brewer, Mark Harrower, and The Pennsylvania State University.; Copyright (c) 2009 Pierre Raybaut | [`matplotlib-LICENSE`](matplotlib-LICENSE) |
| mpmath | 1.3.0 | BSD-3-Clause | Copyright (c) 2005-2021 Fredrik Johansson and mpmath contributors | [`mpmath-LICENSE`](mpmath-LICENSE) |
| networkx | 3.6.1 | BSD-3-Clause | Copyright (c) 2004-2025, NetworkX Developers | [`networkx-LICENSE.txt`](networkx-LICENSE.txt) |
| numpy | 2.4.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Copyright (c) 2005-2025, NumPy Developers.; Copyright (c) 2011-2014, The OpenBLAS Project; Copyright (c) 1992-2013 The University of Tennessee and The University; Copyright (c) 2000-2013 The University of California Berkeley. All | [`numpy-LICENSE.txt`](numpy-LICENSE.txt) |
| packaging | 26.2 | Apache-2.0 OR BSD-2-Clause | *(no notice in file; author: Donald Stufft <donald@stufft.io>)* | [`packaging-LICENSE`](packaging-LICENSE) |
| pillow | 12.3.0 | MIT-CMU | Copyright © 1997-2011 by Secret Labs AB; Copyright © 1995-2011 by Fredrik Lundh and contributors; Copyright © 2010 by Jeffrey 'Alex' Clark and contributors; Copyright (c) 2016, Alliance for Open Media. All rights reserved. | [`pillow-LICENSE`](pillow-LICENSE) |
| pluggy | 1.6.0 | MIT | Copyright (c) 2015 holger krekel (rather uses bitbucket/hpk42) | [`pluggy-LICENSE`](pluggy-LICENSE) |
| pygments | 2.20.0 | BSD-2-Clause | Copyright (c) 2006-2022 by the respective authors (see AUTHORS file). | [`pygments-LICENSE`](pygments-LICENSE) |
| pyparsing | 3.3.2 | MIT | Copyright (c) 2003-2025  Paul McGuire | [`pyparsing-LICENSE`](pyparsing-LICENSE) |
| pytest | 9.0.3 | MIT | Copyright (c) 2004 Holger Krekel and others | [`pytest-LICENSE`](pytest-LICENSE) |
| python-dateutil | 2.9.0.post0 | Apache-2.0 OR BSD-3-Clause | Copyright 2017- Paul Ganssle <paul@ganssle.io>; Copyright 2017- dateutil contributors (see AUTHORS file); Copyright (c) 2003-2011 - Gustavo Niemeyer <gustavo@niemeyer.net>; Copyright (c) 2012-2014 - Tomi Pieviläinen <tomi.pievilainen@iki.fi> | [`python-dateutil-LICENSE`](python-dateutil-LICENSE) |
| scikit-learn | 1.8.0 | BSD-3-Clause | Copyright (c) 2007-2024 The scikit-learn developers.; Copyright (c) 2003-2019 University of Illinois at Urbana-Champaign. | [`scikit-learn-COPYING`](scikit-learn-COPYING) |
| scipy | 1.17.1 | BSD-3-Clause (bundles copyleft components — see note) | Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.; Copyright (c) 2011-2014, The OpenBLAS Project; Copyright (c) 1992-2013 The University of Tennessee and The University; Copyright (c) 2000-2013 The University of California Berkeley. All | [`scipy-LICENSE.txt`](scipy-LICENSE.txt) |
| setuptools | 84.0.0 | MIT | *(no notice in file; author: Python Packaging Authority <distutils-sig@python.org>)* | [`setuptools-LICENSE`](setuptools-LICENSE) |
| six | 1.17.0 | MIT | Copyright (c) 2010-2024 Benjamin Peterson | [`six-LICENSE`](six-LICENSE) |
| sympy | 1.14.0 | BSD-3-Clause | Copyright (c) 2006-2023 SymPy Development Team; Copyright (c) 2006-2018 SymPy Development Team,; Copyright (c) 2014 Matthew Rocklin; Copyright (c) 2009-2023, PyDy Authors | [`sympy-LICENSE`](sympy-LICENSE) |
| threadpoolctl | 3.6.0 | BSD-3-Clause | Copyright (c) 2019, threadpoolctl contributors | [`threadpoolctl-LICENSE`](threadpoolctl-LICENSE) |
| torch | 2.13.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | Copyright (c) 2016-     Facebook, Inc            (Adam Paszke); Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala); Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert); Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu) | [`torch-LICENSE`](torch-LICENSE) |
| typing-extensions | 4.15.0 | PSF-2.0 | Copyright (c) 1991 - 1995, Stichting Mathematisch Centrum Amsterdam, | [`typing-extensions-LICENSE`](typing-extensions-LICENSE) |

30 packages — every third-party entry in `uv.lock`.

## Supporting license texts

Terms of components bundled inside a dependency rather than
dependencies themselves, so they are not counted above.

- [`LGPL-2.1.txt`](LGPL-2.1.txt) — LGPL-2.1, for scipy's libquadmath

Generated by `gen_notices.py` in the project's license-audit
directory. Regenerate with `make notices` there rather than editing
this file by hand.
