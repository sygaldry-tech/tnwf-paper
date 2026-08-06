"""Grid-coordinate conventions for Born-sample readout.

The wavefunction is represented by its values *at* grid nodes ``x_i = i·dx``:
``make_grid`` uses ``linspace(0, L, N, endpoint=False)``, ``build_V_mpo``
queries V at those same nodes, the pseudospectral K-step treats the array as
function samples, and the trained MPS-V indexes ``(x/dx) % N``. So ``|ψ_i|²``
is the mass of the cell *centred* on ``x_i``, and a Born sample drawn in cell
``i`` must be dithered symmetrically about the node:

    x = (i + U(-1/2, +1/2))·dx        # cell_centre_v2   (unbiased)

Releases up to and including v1 of the data archive dithered to the right,

    x = (i + U(0, 1))·dx              # node_lower_left_v1

which puts every sample a rigid ``+dx/2`` above where it belongs on every
axis. Some downstream scripts compensated with an explicit ``- dx/2``; others
did not, so two conventions were live at once under the same symbol.

This module is the single place that shift exists. Everything that reads
sample coordinates asks :func:`resolve_shift` what to add, and nothing
subtracts ``dx/2`` by hand.

The key to why this cannot double-correct is the *semantics* of the marker.
``CONV_CELL`` means "these coordinates need no further shift" — **not** "this
file was shifted". That reading is true of three different kinds of file for
three different reasons, with no special-casing anywhere:

* freshly generated grid samples, which are now dithered correctly at source;
* JAM samples, which are continuous ODE particles that were never binned and
  so were never biased;
* derived quantities such as ``traj.npz``'s ``sw_unbiased``, which had the
  correction folded in when they were written.

Applying the shift and stamping ``CONV_CELL`` is one atomic operation, so a
second pass over an already-migrated file reads ``CONV_CELL`` and gets 0.0.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np

# Key names as they appear inside .npz files.
CONV_KEY = "coord_convention"
SCHEMA_KEY = "tnwf_schema"
SCHEMA = 2

#: Samples dithered to the right of the node: ``x = (i + U(0,1))·dx``. Carries
#: a rigid ``+dx/2`` bias per axis and needs ``-dx/2`` to be read correctly.
CONV_NODE = "node_lower_left_v1"

#: Samples already on the node-centred convention: ``x = (i + U(-½,+½))·dx``,
#: or coordinates that were never grid-binned at all. Needs no further shift.
CONV_CELL = "cell_centre_v2"

_VALID = (CONV_NODE, CONV_CELL)

#: Methods whose ``samples_T`` come from binned Born readout and therefore
#: carry the bias under :data:`CONV_NODE`.
#:
#: This is an allowlist on purpose. The tempting form is ``method != "jam"``,
#: but a denylist fails *open*: a method added later would be silently shifted
#: whether or not it grid-samples. An allowlist fails closed — an unknown
#: method raises rather than being quietly mangled.
GRID_METHODS = frozenset({"dense", "tci_tdvp1", "tci_tdvp2", "tci_als", "aci"})

#: Methods that integrate continuous particles and are never grid-binned.
CONTINUOUS_METHODS = frozenset({"jam"})


class CoordConventionError(RuntimeError):
    """Raised when a file's coordinate convention cannot be established.

    Deliberately fatal. The dangerous failure mode is an un-migrated file
    reaching a consumer that no longer compensates, which would silently
    reintroduce the half-cell bias; a hard error makes that loud instead.
    """


def _load_sidecar(table_path: Path) -> list[tuple[str, str]]:
    """Parse a ``COORD_CONVENTIONS.tsv`` glob table (``pattern<TAB>convention``)."""
    rows: list[tuple[str, str]] = []
    for raw in table_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, _, conv = line.partition("\t")
        conv = conv.strip()
        if conv not in _VALID:
            raise CoordConventionError(
                f"{table_path}: unknown convention {conv!r} for pattern {pattern!r}"
            )
        rows.append((pattern.strip(), conv))
    return rows


def convention_for_path(path: Path | str, table_path: Path | str | None = None) -> str | None:
    """Look a path up in the checked-in sidecar table, or None if unlisted.

    The table covers artifacts we deliberately do not rewrite — shipped demo
    npz files and ``traj.npz`` records whose metrics were already corrected —
    so their convention is recorded rather than inferred.
    """
    path = Path(path)
    if table_path is None:
        table_path = Path(__file__).resolve().parents[2] / "data" / "COORD_CONVENTIONS.tsv"
    table_path = Path(table_path)
    if not table_path.exists():
        return None
    posix = path.as_posix()
    for pattern, conv in _load_sidecar(table_path):
        if fnmatch.fnmatch(posix, pattern) or fnmatch.fnmatch(posix, f"*/{pattern}"):
            return conv
    return None


def convention_of(z, path: Path | str | None = None) -> str:
    """Return the coordinate convention of an opened ``.npz`` (or its path).

    Resolution order: the in-file stamp, then the sidecar glob table. If
    neither answers, raise — never guess.
    """
    if z is not None and CONV_KEY in getattr(z, "files", ()):
        conv = str(np.asarray(z[CONV_KEY]).item())
        if conv not in _VALID:
            raise CoordConventionError(f"{path}: unknown {CONV_KEY} {conv!r}")
        return conv
    if path is not None:
        conv = convention_for_path(path)
        if conv is not None:
            return conv
    raise CoordConventionError(
        f"{path}: no {CONV_KEY!r} stamp and no entry in data/COORD_CONVENTIONS.tsv. "
        "Refusing to assume a convention — run scripts/migrate_archive.py, or add "
        "the file to the sidecar table if it is intentionally left unmigrated."
    )


def resolve_shift(z, dx: float, path: Path | str | None = None) -> float:
    """Return the additive correction to apply to this file's coordinates.

    ``0.0`` when the coordinates are already node-centred, ``-dx/2`` when they
    still carry the v1 right-dither. Never returns a silent default.
    """
    conv = convention_of(z, path)
    return 0.0 if conv == CONV_CELL else -0.5 * float(dx)


def shift_for_method(method: str) -> float:
    """Return the *multiple of dx* by which a v1 file of this method is biased.

    ``0.5`` for grid-sampled methods, ``0.0`` for continuous ones. Raises on an
    unrecognised method rather than assuming either.
    """
    if method in GRID_METHODS:
        return 0.5
    if method in CONTINUOUS_METHODS:
        return 0.0
    raise CoordConventionError(
        f"unknown method {method!r}: add it to coords.GRID_METHODS or "
        "coords.CONTINUOUS_METHODS before migrating archives that contain it."
    )
