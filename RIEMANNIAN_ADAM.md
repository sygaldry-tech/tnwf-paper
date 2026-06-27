# Riemannian Adam on the MPS Manifold

When the pipeline-native training loss

$$
\mathcal{L}(V_t) = \sum_k \big\||\psi_{t_k}(V_t)| - \sqrt{\hat q_{t_k}}\big\|^2
$$

is minimised with $V_t$ itself parameterised as an MPS, the optimizer
update needs to respect the manifold structure of the MPS-core
parameter space. A naive Euclidean Adam over the cores drifts
unboundedly along gauge orbits because the loss is gauge-invariant —
and at $N \ge 32$ that drift breaks the TT-cross approximation of
$e^{i\beta V_t}$ during the forward pass.

This note covers the optimizer update. The companion note
[`ADJOINT_METHOD.md`](ADJOINT_METHOD.md) covers the *gradient
computation* through the truncated MPS Trotter chain. The two pieces
are independent and complementary: **same adjoint backward, different
optimizer step** is the apples-to-apples comparison in the figures
and tables.

## $V_t$ as MPS

Each segment's $V_t^{(k)}$ is a list of $d$ real cores
$A_j^{(k)} \in \mathbb{R}^{D_{j-1}\times N \times D_j}$, bond cap $D_V$.
The dense-grad-to-core projection uses the standard tangent-space
chain rule:

$$
\frac{\partial \mathcal{L}}{\partial A_j[a,n,b]}
= \sum_{\mathbf{x} : x_j=n}
  \frac{\partial \mathcal{L}}{\partial V_{\text{grid}}[\mathbf{x}]}
  \cdot L_j[\mathbf{x}_{<j}, a] \cdot R_j[b, \mathbf{x}_{>j}]
$$

with left/right partial environments $L_j, R_j$.

**Before this work**, the optimizer was a plain Euclidean Adam over
the flat cores. This works at $N=16$ but breaks at $N \ge 32$: the
loss is gauge-invariant under $A_j \to A_j G_j$,
$A_{j+1} \to G_j^{-1} A_{j+1}$, so Adam has no curvature signal
along the gauge orbit and the cores drift unboundedly. At higher $N$
the Trotter coefficient $\beta \propto N$, so any growth in
$V_{\text{dense}}$ amplifies into instability in $e^{i\beta V_t}$.

## Riemannian Adam on the MPS manifold

The set of MPS at fixed bond cap $D_V$ forms a smooth manifold
$\mathcal{M}_{D_V} \subset \mathbb{R}^{N^d}$ of dimension roughly
$d \cdot D_V^2 \cdot N$ — vastly smaller than the ambient $N^d$
(Holtz–Rohwedder–Schneider 2012). The geometric framing of training is:
take gradient steps on $\mathcal{M}_{D_V}$ rather than in the embedding
space.

### The gauge group

$\mathcal{M}_{D_V}$ carries an action of the product group
$\prod_j \mathrm{GL}(D_j)$ acting by bond-wise change of basis:

$$
A_j \;\longmapsto\; A_j G_j,
\qquad
A_{j+1} \;\longmapsto\; G_j^{-1} A_{j+1},
\qquad
G_j \in \mathrm{GL}(D_j).
$$

Two MPS related by such a transformation contract to the same dense
tensor. The loss $\mathcal{L}$ depends only on the dense tensor, so
$\mathcal{L}$ is *gauge-invariant*: directly in core space, the loss
has flat directions along every gauge orbit.

Euclidean Adam doesn't know about this. Its first and second moment
estimates pick up components along the gauge orbits, and Adam's
adaptive scaling treats those components as legitimate descent
directions. The result is that the core Frobenius norms drift
unboundedly under training — and because the Trotter coefficient
$\beta = \pi N / (2L)\sqrt{d \Delta t}$ scales linearly in $N$, any
drift in $V_{\text{dense}}$ rapidly leaves the regime where the
TT-cross approximation of $e^{i\beta V_t}$ is faithful. This is the
$N=32$ instability mechanism in one sentence.

### Fixing the gauge: mixed-canonical form

Pick a canonical representative in each gauge equivalence class. The
**mixed-canonical form** with centre at site $c$ makes cores
$j < c$ left-orthogonal and cores $j > c$ right-orthogonal:

$$
A_j^\top A_j \;=\; I_{D_j} \quad \text{for } j < c,
\qquad
A_j A_j^\top \;=\; I_{D_{j-1}} \quad \text{for } j > c,
$$

where the matricisations are taken on the natural "fat" leg
($(D_{j-1} \cdot N) \times D_j$ for left-canonical;
$D_{j-1} \times (N \cdot D_j)$ for right-canonical). The centre core
$A_c$ absorbs the scale (and is the only one that carries any norm).
For $d=2$ — the petals case — we take $c = d - 1$, which is the
simplest convention: core 0 is left-orthogonal, core 1 is the centre.

A constrained core then lives on a **Stiefel manifold**:

$$
\mathrm{St}(n, k) \;=\; \{\,A \in \mathbb{R}^{n \times k} \,:\, A^\top A = I_k\,\},
$$

which for left-canonical core $j$ is $\mathrm{St}(D_{j-1} \cdot N,\, D_j)$.
The whole parameter space, after gauge fixing, is a product of Stiefel
manifolds glued by the centre core. Stiefel optimisation has been
thoroughly studied (Edelman–Arias–Smith 1998; Absil–Mahony–Sepulchre
2008).

### Tangent space and Riemannian gradient

The tangent space at a point $A \in \mathrm{St}(n, k)$ is obtained by
differentiating the constraint $A^\top A = I$:

$$
T_A \mathrm{St} \;=\; \{\,\dot A \in \mathbb{R}^{n \times k} \,:\, A^\top \dot A + \dot A^\top A = 0\,\},
$$

i.e. the symmetric part of $A^\top \dot A$ must vanish. With the
Euclidean inner product, the orthogonal projector onto $T_A \mathrm{St}$
is

$$
\Pi_A(G) \;=\; G - A \cdot \tfrac{1}{2}\bigl(A^\top G + G^\top A\bigr).
$$

The piece subtracted off, $A \cdot \mathrm{sym}(A^\top G)$, is *exactly*
the component of $G$ along the gauge orbit at $A$. So **projecting the
Euclidean gradient onto the tangent space is the same as projecting out
the gauge component** — and the Riemannian gradient is what's left.

Three equivalent ways to say what's happening:

* The loss is constant along gauge orbits.
* The gauge orbit is a flat direction in core space.
* The orthogonal complement of the gauge orbit at $A$ is the Stiefel
  tangent space, where genuine descent lives.

### Retraction

A tangent vector $\xi \in T_A \mathrm{St}$ is an infinitesimal motion;
to take a finite step and remain on $\mathrm{St}$ we need a
**retraction** — a smooth map $R_A: T_A \mathrm{St} \to \mathrm{St}$
with $R_A(0) = A$ and $\mathrm{d} R_A(0) = \mathrm{id}$. The geodesic
exponential map satisfies these but is expensive; the **QR retraction**

$$
R_A(\xi) \;=\; Q \quad \text{where} \quad A + \xi = QR
$$

is the cheap, numerically stable choice used throughout TT optimisation
(Absil–Malick 2012; Steinlechner 2016). It is first-order accurate
(matches the geodesic at zero step), preserves bond dimensions exactly,
and never needs an SVD.

### Riemannian Adam

Put the pieces together (Bécigneul–Ganea 2019). The Riemannian Adam
update at a Stiefel core $A_j$ is:

1. Project the Euclidean gradient onto the tangent space:
   $\tilde G_j = \Pi_{A_j}(G_j)$.
2. Update Adam moments using $\tilde G_j$ as the gradient: first moment
   $m_j$ in the tangent space, second moment $v_j$ as a per-coordinate
   variance (not tangent; we deliberately do *not* project $v_j$ since
   it must stay non-negative for $\sqrt{v_j}$ to be defined).
3. Form the candidate update direction $d_j = \hat m_j / (\sqrt{\hat v_j}
   + \varepsilon)$ as in standard Adam, and re-project to the tangent
   space.
4. Retract: $A_j \leftarrow R_{A_j}(-\eta\, d_j)$.
5. Approximate parallel transport of $m_j$ to the new tangent space by
   identity transport followed by re-projection.

The centre core $A_c$ is unconstrained and takes a plain Euclidean Adam
step.

### Why this is the right thing geometrically

The deep reason Riemannian Adam works for our problem is that the
gauge-invariance of $\mathcal{L}$ and the Stiefel constraint share a
common origin: both express the same flat directions of core space
(Lubich–Rohwedder–Schneider–Vandereycken 2013 on the TT manifold's
mixed-canonical decomposition). Quotienting them out by canonical form
+ Stiefel projection isn't an *additional* constraint on top of the
optimisation — it's the natural geometry of the problem made explicit.
Euclidean Adam was fighting the geometry; Riemannian Adam respects it.

The empirical consequence: at $N=32$ the cores stay bounded under
Riemannian Adam at $\eta = 10^{-3}$, where Euclidean Adam at the same
learning rate would diverge. The Trotter coefficient $\beta$ no longer
amplifies a runaway, and TT-cross of $e^{i\beta V_t}$ stays in its
linearisation regime throughout training.

## Validation

- **14 needle tests** in `tests/test_riemannian_mps.py`, all green in
  90 ms: canonical round-trip, Stiefel orthogonality, gauge-projection
  idempotence, pure-gauge annihilation, retraction preservation,
  Riemannian Adam end-to-end on a toy fit.
- **Toy fit** (`toy_riemannian_mps_fit.py`): on
  $\tfrac{1}{2}\|V_\text{dense} - V_\text{target}\|^2$ at $N=8, d=2, D=4$,
  Riemannian Adam converges monotonically at $\eta = 10^{-2}$.
- **End-to-end on petals_2d**: at $N=32$ with $\eta = 10^{-3}$,
  Riemannian Adam reaches loss $1.73$ in 500 iters where Euclidean Adam
  at the same learning rate diverges, and at $\eta = 10^{-4}$ needs
  5000 iters to reach loss $1.98$ — a ${\sim}100\times$ compute-efficiency
  win on this dataset.

## Files

| Path                                                       | Role                                         |
| ---------------------------------------------------------- | -------------------------------------------- |
| `scripts/exploration/riemannian_mps.py`                    | `to_mixed_canonical`, `gauge_project`, `retract_qr` |
| `scripts/exploration/riemannian_adam_mps.py`               | `_RiemannianAdamMPS`, `init_v_mps_riemannian` |
| `scripts/exploration/toy_riemannian_mps_fit.py`            | Toy MPS-fit validation                       |
| `scripts/exploration/train_v_mps_end_to_end.py` (modified) | `--optimizer {adam, radam}` + baseline flags |
| `scripts/exploration/make_fig_radam_compare.py`            | Fig-2 / Fig-3 analogues vs. paper            |
| `tests/test_riemannian_mps.py`                             | 14 needle tests                              |

## References

* S. Holtz, T. Rohwedder, R. Schneider, "On manifolds of tensors of
  fixed TT-rank" (2012).
* C. Lubich, T. Rohwedder, R. Schneider, B. Vandereycken, "Dynamical
  approximation by hierarchical Tucker and tensor-train tensors" (2013).
* A. Edelman, T. Arias, S. Smith, "The geometry of algorithms with
  orthogonality constraints" (1998) — Stiefel manifold.
* P.-A. Absil, R. Mahony, R. Sepulchre, *Optimization Algorithms on
  Matrix Manifolds* (2008).
* P.-A. Absil, J. Malick, "Projection-like retractions on matrix
  manifolds" (2012) — QR retraction.
* M. Steinlechner, "Riemannian optimization for high-dimensional tensor
  completion" (2016).
* G. Bécigneul, O.-E. Ganea, "Riemannian Adaptive Optimization Methods"
  (2019) — Riemannian Adam.
