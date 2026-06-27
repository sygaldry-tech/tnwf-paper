# Theory: closed forms for the Gaussian → Gaussian-mixture flow

For the Gaussian-source → Gaussian-mixture target case used throughout this
repo, the action-matching potential $V_t(x)$, the interpolant density
$p_t(x)$, and (the magnitude of) the wavefunction $\psi_t$ all have closed
forms. This means we don't need to train a neural net to evaluate $V_t$ on
the grid — we can use the analytic expression directly. This isolates Trotter
and MPS-compression errors from JAM training noise.

## Setup

The conservative wavefunction-flow pipeline (Layden et al. 2025) trains a
scalar potential $V_t(x)$ such that

$$
\nabla V_t(x) \;=\; \mathbb{E}[\,x_1 - x_0 \mid x_t = x\,]
$$

where $x_t = (1-t)\,x_0 + t\,x_1$ is the linear interpolant between

- source $x_0 \sim N(0, \sigma_0^2 I)$
- target $x_1 \sim \sum_k w_k\, N(c_k,\, \sigma_k^2 I)$

The wavefunction $\psi_t(x)$ evolves under $\partial_t \psi = -i H_t^c \psi$
with $H_t^c = i[K, V_t]$; the construction guarantees $|\psi_t(x)|^2 = p_t(x)$.

## 1. The interpolant density $p_t(x)$ is a Gaussian mixture

Each component of the source–target pair $(x_0, x_1)$ is independent Gaussian,
so the marginal of $x_t$ along each mixture branch $k$ is

$$
x_t \mid k \;\sim\; N\!\big(t\,c_k,\;\;[(1-t)^2 \sigma_0^2 + t^2 \sigma_k^2]\,I\big).
$$

Mixing over $k$:

$$
\boxed{\;p_t(x) \;=\; \sum_k w_k \, N\!\big(x;\; \mu_k(t),\; \Sigma_k(t)\big)\;}
$$

with

$$
\mu_k(t) = t\,c_k,
\qquad
\Sigma_k(t) = \big[(1-t)^2 \sigma_0^2 + t^2 \sigma_k^2\big] \, I.
$$

This is closed-form, smooth, evaluable analytically on any grid.

## 2. The action-matching potential $V_t(x)$

Using $x_t = (1-t)x_0 + t\,x_1$ and rearranging gives
$x_1 - x_0 = (x_1 - x_t)/(1-t)$, so the AM velocity reduces to a posterior
expectation:

$$
\nabla V_t(x) \;=\; \mathbb{E}[\,x_1 - x_0 \mid x_t = x\,]
\;=\; \frac{\mathbb{E}[\,x_1 \mid x_t = x\,] - x}{1 - t}.
$$

Using Tweedie's identity on $u = t x_1$ with noise
$\varepsilon \sim N(0, (1-t)^2 \sigma_0^2 I)$ so $x_t = u + \varepsilon$:

$$
\mathbb{E}[\,t x_1 \mid x_t = x\,] \;=\; x + (1-t)^2 \sigma_0^2 \,\nabla \log p_t(x).
$$

Substituting and simplifying:

$$
\boxed{\;\nabla V_t(x) \;=\; \frac{x}{t} \;+\; \frac{(1-t)\,\sigma_0^2}{t}\,\nabla \log p_t(x)\;}
$$

Integrating once (the field is conservative by construction):

$$
\boxed{\;V_t(x) \;=\; \frac{\|x\|^2}{2t} \;+\; \frac{(1-t)\,\sigma_0^2}{t}\,\log p_t(x) \;+\; C(t)\;}
$$

So the AM potential has **two pieces**: a quadratic confinement term that
depends only on $t$, and a log-density term. The log-density term is what
produces the multi-modal target structure; the quadratic term anchors the
flow at the source. Both are closed-form for any GMM target.

> Numerically: $V_t \to \infty$ as $t \to 0^+$ (training avoids $t=0$ via
> $t \in [t_{\rm eps}, 1)$). At $t=1$ the second term vanishes and
> $V_t(x) = \|x\|^2/2 + C(1)$ — pure confinement at the target time.

> The additive $C(t)$ is irrelevant: in the V-step $e^{i\beta V_t(x)}$ it
> contributes a global phase that drops out of $|\psi|^2$.

Concretely, with isotropic component variances
$\Sigma_k(t) = s_k(t)^2 I$ where $s_k(t)^2 = (1-t)^2 \sigma_0^2 + t^2 \sigma_k^2$,

$$
\log p_t(x) \;=\; \log \sum_k w_k\, (2\pi s_k^2)^{-d/2}
                       \exp\!\big(-\|x - t c_k\|^2 / (2 s_k^2)\big).
$$

## 3. The V-step operator $e^{i\beta V_t}$

Since $V_t$ is closed-form pointwise, the V-step operator
$\mathrm{diag}\big(e^{i\beta V_t(x)}\big)$ is closed-form on the grid. Whether
this operator is **low rank as an MPS** is a separate question — and the
answer is *generally no for arbitrary mixtures*. For orthogonal-mode
mixtures (target modes at $\pm c\,e_j$, $j = 1, \ldots, d$, used in this
repo), there is partial separability that makes the bond dimension empirically
manageable, but no exact low-rank structure.

## 4. The wavefunction $\psi_t$

The Layden construction guarantees $|\psi_t|^2 = p_t$. The magnitude is
therefore closed-form,

$$
|\psi_t(x)| \;=\; \sqrt{p_t(x)}.
$$

The **phase** $\arg \psi_t$ does *not* have a clean closed form — it depends
on the specific Trotter splitting and is what the [K, V] flow has to compute.
The phase carries the dynamical information; the magnitude is fixed by
construction.

## Implications for the paper

Three concrete uses of these closed forms:

### (a) Eliminate JAM as a confounder

Replace the trained MLP $V_t$ with the analytic
$V_t \propto \log p_t$ in `tnwf.jam.train.make_V_fn` (or via a new
`analytic_V_fn`). Methods get tested against the *exact* potential, not an
MLP approximation. The remaining error is Trotter + MPS-compression only —
JAM training noise, capacity, and seed-to-seed variance all vanish.

### (b) Ground-truth validation

At any $t$, the true $p_t(x)$ is computable in closed form. We can compare
each method's $|\psi_t|^2$ on the grid to $p_t$ directly via
KL or total-variation distance — a *dimensionless per-cell error* that's
sharper than sample-based SW or MMD (no sampling noise from drawing
$n$ samples).

### (c) Decompose the error budget

The current numerical error is a mixture of:
- JAM training error (depends on iterations, hidden width)
- JAM seed variance (different runs of training)
- Trotter splitting error $O(\Delta t^2)$ at order 2, $O(\Delta t^4)$ at
  order 4
- Grid quantisation $O(L/N)$
- MPS truncation $O(\sigma_{D+1})$ where $\sigma$ are Schmidt singular values

Using analytic $V_t$ removes the first two terms entirely, isolating
the **method-intrinsic error** of TDVP1 / TDVP2.

## Suggested API change

A new helper `tnwf.theory.analytic_V_fn(dataset_config)` that returns a
`V_fn(x_grid, t)` callable matching the signature of `make_V_fn(model)`. The
runner then accepts a `--V_source {jam, analytic}` flag. The pipelines remain
identical otherwise; only the $V_t$ source changes.

## Related: when the source is *not* Gaussian

For non-Gaussian sources (e.g. swiss roll), $p_0$ has no closed form, so the
posterior $x_1 \mid x_t$ does not either, and we lose the analytic $V_t$.
JAM remains the right tool for those datasets.

For the swiss roll dataset specifically, however, JAM's trained $V_t$ can
still be compared *qualitatively* against the gradient field induced by the
data (e.g. by checking whether $\nabla V_t$ points toward the nearest data
manifold), but no analytic ground truth exists.
