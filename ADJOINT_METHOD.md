# Adjoint Method for the Wavefunction-Flow Trotter Chain

We have a wavefunction-flow pipeline that takes a position-diagonal
potential $V_t(x)$ and runs it through a Trotter chain on an MPS-encoded
$\psi$ to produce samples at the snapshot times $t_k = k/K$. The standard
way to obtain $V_t$ is JAM training: fit an MLP $V_t(x;\theta)$ by
regressing on action-matching targets over the dense grid. JAM is
"pipeline-agnostic" — it doesn't know that the downstream Trotter chain
truncates $\psi$ to bond cap $D_{\max}$, so at high $N$ or low $D_{\max}$
the $V_t$ JAM picks is mismatched to the pipeline actually used at
inference.

**Pipeline-native training** fixes this by minimising sample-quality loss
*through the deployed pipeline*. The loss is the Hellinger amplitude
discrepancy

$$
\mathcal{L}(V_t) = \sum_k \big\||\psi_{t_k}(V_t)| - \sqrt{\hat q_{t_k}}\big\|^2,
$$

where $\psi_{t_k}(V_t)$ is the MPS Trotter output. Training requires
$\partial \mathcal{L}/\partial V_t$ through the truncated MPS chain. The
adjoint method described below computes this gradient at $\mathcal{O}(K)$
memory and without SVD-gradient instability.

The companion note [`RIEMANNIAN_ADAM.md`](RIEMANNIAN_ADAM.md) covers the
*optimizer update* that consumes this gradient when $V_t$ is itself
parameterised as an MPS. The two pieces are independent and complementary:
**same adjoint backward, different optimizer step** is the apples-to-apples
comparison in the figures and tables.

## Pipeline overview

Forward:

```
ψ_0 = √q̂_0  →  [V-substep, K-substep × 8 letters] × K segments
            →  {ψ_{t_1}, …, ψ_{t_K}}
```

$\psi$ is kept as an MPS throughout (truncated to bond cap $D_{\max}$
after each V-substep). The V-substep applies $e^{i\beta V_t}$ to $\psi$
via TT-cross compression and Hadamard product (bond cap $D_V$). The
K-substep applies $e^{i\alpha K}$ as a site-local FFT (no bond growth).

## The adjoint method

We need $\partial \mathcal{L}/\partial V_t$ through a long Trotter chain
on truncated MPS. The naive route — reverse-mode autograd through the
whole chain — has two problems: it stores an $\mathcal{O}(K)$-deep tape
of MPS-shaped intermediates, and it runs the gradient of the SVD inside
each truncation step, which is ill-conditioned when singular values are
nearly degenerate. The adjoint method (Pontryagin 1962;
Chen et al. 2018 in the Neural-ODE form) sidesteps both.

### Discrete-time formulation

Write the forward chain as a sequence of state transitions

$$
\psi_k = F_k(\psi_{k-1}, V_t), \qquad k = 1, \dots, K,
$$

and a loss that decomposes as a sum of per-snapshot terms,

$$
\mathcal{L} = \sum_{k=1}^{K} \phi_k(\psi_k),
\qquad
\phi_k(\psi) = \big\| |\psi| - \sqrt{\hat q_{t_k}} \big\|^2.
$$

The discrete-time adjoint introduces a costate (Lagrange multiplier)
$\lambda_k$ — same shape as $\psi_k$ — that propagates *backward*:

$$
\lambda_K \;=\; \frac{\partial \phi_K}{\partial \psi_K^*}, \qquad
\lambda_{k-1} \;=\; \left(\frac{\partial F_k}{\partial \psi_{k-1}}\right)^{\!\!\dagger} \lambda_k
                  \;+\; \frac{\partial \phi_{k-1}}{\partial \psi_{k-1}^*}.
$$

Reading right to left: the costate at time $k{-}1$ is the costate at
time $k$ pulled back by the linearised dynamics, plus the direct
sensitivity of the local loss term. The total gradient with respect to
$V_t$ then accumulates as

$$
\frac{\partial \mathcal{L}}{\partial V_t} \;=\;
\sum_{k=1}^{K} \left(\frac{\partial F_k}{\partial V_t}\right)^{\!\!\dagger} \lambda_k.
$$

This is just the discrete-time Pontryagin maximum principle with $V_t$
playing the role of the control and $\psi$ the state.

### What the substep adjoints look like

Each Trotter segment is itself an eight-letter product of V-substeps and
K-substeps; the adjoint chain runs through all eight letters per
segment.

* **V-substep**: $\psi \to e^{i\beta V_t} \odot \psi$ is elementwise on
  the grid, so its linearisation is diagonal:
  $\partial F / \partial \psi = \mathrm{diag}(e^{i\beta V_t})$, and
  $\lambda \to e^{-i\beta V_t} \odot \lambda$. The local contribution
  to the gradient at this substep is, with real $V_t$,
  $$
  \frac{\partial \mathcal{L}}{\partial V_t(x)} \;\mathrel{+}=\;
  2 \beta \cdot \mathrm{Im}\bigl(\lambda(x) \cdot \psi_{\text{after}}(x)^*\bigr).
  $$
  This is the Wirtinger gradient at a single grid cell; the constant
  $2\beta$ comes from the chain rule through $e^{i\beta V_t}$.

* **K-substep**: $\psi \to e^{i\alpha K} \psi$ is unitary and the
  pseudo-spectral $K$ is realised by FFT. Its adjoint is the same
  operation with $\alpha \to -\alpha$, so $\lambda \to e^{-i\alpha K}
  \lambda$. No contribution to $\partial \mathcal{L}/\partial V_t$.

* **MPS truncation**: the V-substep grows the bond by $D_V$ and is then
  truncated back to $D_{\max}$. At the level of the linearised
  dynamics, *at fixed truncation pattern*, this is just an orthogonal
  projection onto a fixed subspace, whose adjoint is the same
  projection. We never differentiate the truncation itself; the
  singular-value gradient never appears. This is the crucial reason
  the adjoint method is well-conditioned where reverse-mode autograd
  through TDVP-style code is not (Liao et al. 2019).

### What the adjoint actually buys

1. **Memory**: $\mathcal{O}(K)$ cached MPS snapshots (each
   $\mathcal{O}(d \cdot D_{\max}^2 \cdot N)$) versus
   $\mathcal{O}(K)$ autograd tapes (each containing every intermediate
   tensor in the substeps, the truncation SVD, and the cross
   interpolant). Asymptotically the cache wins by orders of magnitude.
2. **Numerical stability**: no SVD gradient, no near-degenerate
   spectrum to regularise around.
3. **Validation**: we verified bit-exact equivalence to
   `torch.autograd` on dense $\psi$ (~$10^{-15}$ relative) and TT-cross-tolerance
   equivalence on MPS $\psi$ (~$10^{-8}$, set by the cross
   approximation error rather than the adjoint formulation itself).

The continuous-time limit of all of this is the Neural-ODE adjoint
(Chen et al. 2018); ours is the discrete-time version applied to a
quantum Trotter chain.

## Composition with the optimizer step

The adjoint method (gradient computation) and the parameter update step
are independent and complementary. The adjoint produces a Euclidean
gradient with respect to the dense $V_{\text{grid}}$ entries; if $V_t$
is dense, that gradient feeds straight into a plain Euclidean optimiser.
If $V_t$ is itself parameterised as an MPS, the dense-to-core projection
turns the dense gradient into an MPS-shaped Euclidean gradient — and the
Riemannian Adam update of [`RIEMANNIAN_ADAM.md`](RIEMANNIAN_ADAM.md)
treats *that* as its input, projects it to the Stiefel tangent space,
and retracts.

## Files

| Path                                                       | Role                                         |
| ---------------------------------------------------------- | -------------------------------------------- |
| `scripts/exploration/train_v_mps_end_to_end.py`            | Adjoint forward+backward through the chain   |
| `scripts/exploration/make_table_eb_w2.py`                  | EB W₂ table columns (1-step prediction)      |

## References

* L. S. Pontryagin et al., *The Mathematical Theory of Optimal
  Processes* (1962) — discrete-time adjoint / Pontryagin maximum
  principle.
* R. T. Q. Chen et al., "Neural Ordinary Differential Equations"
  (2018) — continuous-time adjoint, modern ML framing.
* H.-J. Liao et al., "Differentiable Programming Tensor Networks"
  (2019) — SVD gradient stability in TN.
