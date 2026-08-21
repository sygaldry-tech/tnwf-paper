"""Train an MPS-V velocity potential (thin CLI wrapper over ``tnwf.mps_v.train``).

Fits V_t as a real tensor train by JAM / action matching (self-contained from
data — no teacher). The resulting checkpoint is consumed by the 2-site TDVP
V-step bypass; see ``scripts/mps_v/generate.py``.

Usage:
    python scripts/mps_v/train.py --dataset gmm_8d --N 32 --D 8 --seed 0 \
        --out examples/checkpoints/mps_v_gmm_d8.pt
"""
from tnwf.mps_v.train import _main

if __name__ == "__main__":
    _main()
