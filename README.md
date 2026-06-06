# SAN and Furious

A small [HCIPy](https://hcipy.org)-based testbed for **focal-plane speckle nulling** behind a
coronagraph. It implements and compares four estimators that drive a deformable mirror (DM) to
dig a dark hole, all built on the same per-pixel phase-shifting interferometry idea:

1. **SAN** — Speckle Area Nulling (symmetric 5-frame phase shifting).
2. **SAF** (`furious_step`) — *SAN and Furious*: per-pixel least squares that reuses the whole
   command **history**.
3. **`min_step`** — a non-SAN 3-frame iteration that reuses the corrected image as a reference.
4. **`ff_step`** — a *Fast & Furious*–style 1-probe iteration that borrows the missing quadrature
   from the command history.

All equations below use a single, shared notation.

---

## 1. Optical model and notation

The DM probe basis is a set of Fourier modes, one per dark-zone pixel, so **mode index and
dark-zone pixel index coincide** (call it $j$). A dark-zone point at focal position $\mathbf{x}$
maps to the pupil *angular* spatial frequency

$$
\mathbf{p} = \frac{2\pi\,\mathbf{x}}{\lambda F},
$$

where $\lambda$ is the wavelength and $F$ the focal length (the $2\pi$ is required because HCIPy
builds modes as $\cos(\mathbf{p}\cdot\mathbf{u})$, with no $2\pi$ inside).

For small commands the focal field at pixel $j$ is **linear in the DM command**:

$$
E_j(w) = E_{0,j} + k'_j\, w_j ,
$$

| symbol | meaning |
|---|---|
| $w_j \in \mathbb{C}$ | modal command for mode $j$, in units of the probe amplitude $a_p$. Cosine coefficient $=\operatorname{Re} w_j$, sine coefficient $=\operatorname{Im} w_j$ |
| $E_{0,j}$ | static (aberration) speckle field at pixel $j$ with **zero** command |
| $k'_j \in \mathbb{C}$ | field-per-unit-command gain at pixel $j$ |
| $R_j \equiv E_{0,j} + k'_j w_j$ | **residual field** at the current operating point |
| $I_j \equiv \lvert R_j\rvert^2$ | measured normalized intensity at pixel $j$ |
| $\kappa_j^2 \equiv \lvert k'_j\rvert^2$ | per-pixel gain magnitude (the probe self-intensity) |
| $a_p$ | probe amplitude (m of DM surface); the unit of $w$ |
| $g$ | loop gain |
| $\mu \in (0,1]$ | exponential forgetting factor (`forget`) |

The goal at every pixel is to command $w_j$ so that $E_j(w)=0$, i.e. the **ideal correction**

$$
\boxed{\;\Delta w_j = -\,\frac{R_j}{k'_j} = -\,\frac{R_j\,k_j'^*}{\kappa_j^2}\;}
$$

Every algorithm below is just a different way of measuring $R_j k_j'^*$ (and, where needed,
$\kappa_j^2$) from intensity images. Because a single image gives only $\lvert R_j\rvert^2$, phase
information must be created by **probing** with the DM — the coronagraph nulls the on-axis
reference that ordinary phase retrieval would otherwise use.

---

## 2. SAN — Speckle Area Nulling

*(`san_step`, `san_correction_step`; Nishikawa 2022, Oya 2017)*

Around the current command $w$, take a reference image and four **symmetric** probe images,
using a cosine probe ($\pm 1$ in $a_p$ units) and a sine/quadrature probe ($\pm i$):

$$
I_0 = \lvert R_j\rvert^2,\qquad
I^{c}_{\pm} = \lvert R_j \pm k'_j\rvert^2,\qquad
I^{s}_{\pm} = \lvert R_j \pm i\,k'_j\rvert^2 .
$$

Symmetric sums and differences isolate the two quadratures **and** the gain:

$$
I^{c}_{+}-I^{c}_{-} = 4\,\operatorname{Re}(R_j k_j'^*),\qquad
I^{c}_{+}+I^{c}_{-}-2I_0 = 2\kappa_j^2 ,
$$

so the (dimensionless) SAN ratios are

$$
\mathrm{san}^c_j = \frac{I^{c}_{+}-I^{c}_{-}}{\,I^{c}_{+}+I^{c}_{-}-2I_0\,}
= \frac{2\operatorname{Re}(R_j k_j'^*)}{\kappa_j^2},
\qquad
\mathrm{san}^s_j = \frac{2\operatorname{Im}(R_j k_j'^*)}{\kappa_j^2}.
$$

The applied correction is exactly the ideal step:

$$
\Delta w_j = -\tfrac{1}{2}\,g\,\bigl(\mathrm{san}^c_j + i\,\mathrm{san}^s_j\bigr)
= -\,g\,\frac{R_j k_j'^*}{\kappa_j^2}.
$$

The symmetry is what lets SAN cancel the probe self-term $\kappa_j^2$ **without knowing it**; the
denominator also yields $\kappa_j^2$ for free. **Cost: 4 fresh exposures/iteration** (the reference
is reused), no history, no prior calibration.

---

## 3. SAF — *SAN and Furious* (least-squares over history)

*(`furious_step`, `step`)*

Instead of discarding probe data, record every applied command $w^{(n)}$ with its image
$I_j^{(n)}$ and solve each pixel by **least-squares phase shifting**. Expanding the linear model,

$$
I_j^{(n)} = \lvert E_{0,j}\rvert^2 + \kappa_j^2\,\lvert w_j^{(n)}\rvert^2
+ 2\operatorname{Re}(E_{0,j}k_j'^*)\,\underbrace{\operatorname{Re}(w_j^{(n)})}_{c_n}
+ 2\operatorname{Im}(E_{0,j}k_j'^*)\,\underbrace{\operatorname{Im}(w_j^{(n)})}_{s_n}.
$$

Move the known $\kappa_j^2\lvert w_j^{(n)}\rvert^2$ to the left, $y_j^{(n)} = I_j^{(n)} -
\kappa_j^2\lvert w_j^{(n)}\rvert^2$, leaving a 3-parameter linear model in $a=(a_0,a_1,a_2)$.
With exponential weights $\mu_n = \mu^{\,N-n}$ (newest frame weight 1), the per-pixel weighted
normal equations are

$$
\begin{bmatrix} a_0\\ a_1\\ a_2 \end{bmatrix}
=
\left(\sum_n \mu_n
\begin{bmatrix} 1\\ c_n\\ s_n\end{bmatrix}
\begin{bmatrix} 1 & c_n & s_n\end{bmatrix}\right)^{-1}
\sum_n \mu_n
\begin{bmatrix} 1\\ c_n\\ s_n\end{bmatrix} y_j^{(n)} .
$$

The residual estimate and the **absolute** target command that nulls the static field are

$$
E_{0,j}k_j'^* = \tfrac12\,(a_1 + i\,a_2),
\qquad
w_j^{\mathrm{target}} = -\,\frac{a_1 + i\,a_2}{2\kappa_j^2},
\qquad
\Delta w_j = g\,\bigl(w_j^{\mathrm{target}} - w_j\bigr).
$$

$\kappa_j^2$ is calibrated by a seed SAN step. The forgetting factor $\mu$ is essential: with
$\mu=1$ the fit pools stale frames and is biased toward the wrong null by inter-mode cross-talk;
$\mu<1$ down-weights old frames so the fit re-linearizes around the current state. `step()`
re-probes every `probe_every_n` iterations and runs the furious solve in between.

---

## 4. `min_step` — non-SAN 3-frame iteration

Solving for $R_j$ needs only **3 frames**, not SAN's 5. Reuse the current corrected image as the
reference ($I_0 = \lvert R_j\rvert^2$, free) and apply two **one-sided** probes of size $\epsilon$:

$$
I^{\mathrm{re}} = \lvert R_j + \epsilon\,k'_j\rvert^2 = I_0 + \kappa_j^2\epsilon^2 + 2\epsilon\operatorname{Re}(R_j k_j'^*),
$$

$$
I^{\mathrm{im}} = \lvert R_j + i\epsilon\,k'_j\rvert^2 = I_0 + \kappa_j^2\epsilon^2 + 2\epsilon\operatorname{Im}(R_j k_j'^*).
$$

Hence

$$
\operatorname{Re}(R_j k_j'^*) = \frac{I^{\mathrm{re}} - I_0 - \kappa_j^2\epsilon^2}{2\epsilon},
\qquad
\operatorname{Im}(R_j k_j'^*) = \frac{I^{\mathrm{im}} - I_0 - \kappa_j^2\epsilon^2}{2\epsilon},
$$

$$
\Delta w_j = -\,g\,\frac{\operatorname{Re}(R_j k_j'^*) + i\operatorname{Im}(R_j k_j'^*)}{\kappa_j^2}.
$$

Because the probes are one-sided, the $\kappa_j^2\epsilon^2$ self-term must be **subtracted
explicitly**; 3 one-sided frames cannot also solve for $\kappa_j^2$ (that would be 4 unknowns), so
$\kappa_j^2$ is taken from a single seed SAN step. **Cost: 2 fresh exposures/iteration.** This is
the most exposure-efficient method in the noiseless limit, precisely because it recycles the
already-taken corrected image as the phase-shifting reference.

---

## 5. `ff_step` — *Fast & Furious*–style 1-probe iteration

This ports the temporal-diversity idea of Fast & Furious (Korkiakoski et al.): supply the
**missing quadrature from the command history** so only one fresh probe is needed.

Let $Z_j \equiv R_j k_j'^*$ be the unknown residual at the current point. For any recorded frame
$k$ with command $w^{(k)}$, define the command change to *now*, $\delta_j^{(k)} = w_j - w_j^{(k)}$,
so that $R_j^{(k)} = R_j - k'_j\delta_j^{(k)}$ and

$$
I_j^{(k)} = I_{0,j} + \kappa_j^2\,\lvert\delta_j^{(k)}\rvert^2 - 2\operatorname{Re}\!\bigl(Z_j\,\delta_j^{(k)*}\bigr).
$$

Each frame therefore gives **one real, linear constraint** on $(\operatorname{Re}Z_j,
\operatorname{Im}Z_j)$:

$$
\underbrace{\operatorname{Re}(\delta_j^{(k)})}_{p_k}\operatorname{Re}Z_j
+ \underbrace{\operatorname{Im}(\delta_j^{(k)})}_{q_k}\operatorname{Im}Z_j
= \underbrace{\tfrac12\bigl(\kappa_j^2\lvert\delta_j^{(k)}\rvert^2 - (I_j^{(k)} - I_{0,j})\bigr)}_{g_j^{(k)}} .
$$

Stacking the fresh probe and the recent history gives a per-pixel **weighted $2\times2$** solve:

$$
\begin{bmatrix} \operatorname{Re}Z_j\\ \operatorname{Im}Z_j \end{bmatrix}
=
\left(\sum_k \mu_k
\begin{bmatrix} p_k\\ q_k\end{bmatrix}
\begin{bmatrix} p_k & q_k\end{bmatrix}\right)^{-1}
\sum_k \mu_k
\begin{bmatrix} p_k\\ q_k\end{bmatrix} g_j^{(k)},
\qquad
\Delta w_j = -\,g\,\frac{Z_j}{\kappa_j^2}.
$$

The single fresh probe **alternates** between the real ($+\epsilon$) and imaginary ($+i\epsilon$)
quadrature each iteration, so both directions stay sampled even as the corrections (and their
diversity) shrink. Clustered frames ($\delta_j^{(k)}\to 0$ near convergence) contribute
$p_k,q_k,g_j^{(k)}\to 0$ and **self-cancel**, so they neither help nor bias; a tiny Tikhonov ridge
stabilizes pixels that have only seen one quadrature. **Cost: 1 fresh exposure/iteration**;
$\kappa_j^2$ from a seed SAN step.

---

## 6. Summary and findings

| Method | fresh exposures / iter | frames used | history reuse | $\kappa^2_j$ calibration | per-pixel solver |
|---|:--:|---|:--:|---|---|
| SAN (`san_step`) | 4 | 5 symmetric | no | self (free) | closed form |
| SAF (`furious_step`) | 4 on probe iters, ~0 between | all history | yes | seed SAN | $3\times3$ LSQ (absolute) |
| `min_step` | 2 | 1 reference + 2 one-sided | no | seed SAN | closed form |
| `ff_step` | 1 | history + 1 fresh | yes | seed SAN | $2\times2$ LSQ (differential) |

The `COMPARE_EFFICIENCY` panel plots dark-zone mean contrast vs **cumulative exposures** on the
same static aberration. In the **noiseless** simulation the ordering by exposure efficiency is

$$
\texttt{min\_step} \;\lesssim\; \texttt{ff\_step} \approx \texttt{SAN} \;\lesssim\; \texttt{SAF},
$$

i.e. the 3-frame `min_step` is most efficient, while history reuse (SAF, FF) does **not** beat
fresh full-information probing. This is expected: least-squares pooling of past frames pays off
mainly under **measurement noise** (it averages frames down), which a noiseless forward model does
not exercise. All methods reach a few $\times 10^{-9}$ mean contrast.

---

## 7. Running

```bash
conda activate san
python run.py
```

Key flags at the top of `run.py`:

- `USE_SAF` — main loop uses the SAF estimator (`True`) or the original SAN step (`False`).
- `SAVE_MOVIE` / `MOVIE_FILENAME` / `MOVIE_FPS` — save the convergence as an animated GIF.
- `COMPARE_EFFICIENCY` — run the contrast-vs-exposures comparison of all four estimators.
- `num_iterations`, `gain`, and the `SANAndFurious(forget=…, probe_every_n=…)` constructor
  arguments tune the loop.
