"""Demo driver for the `san` package.

Builds a single :class:`san.CoronagraphModel` forward model and exercises the
speckle-nulling algorithms in :mod:`san.algorithms`:

  1. CONVERGENCE  -- run one algorithm to a deep null, optionally save a movie.
  2. EFFICIENCY   -- contrast vs *cumulative exposures* for every algorithm.
  3. MAINTENANCE  -- hold a dark hole against an injected drift with few probes.

The optics, deformable mirror and Fourier-mode machinery all live in the package
now; this file is just experiment glue.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from san import (
    CoronagraphModel,
    SpeckleAreaNulling,
    SANAndFurious,
    MinStepNulling,
    FastAndFurious,
    FastAndFuriousNoProbe,
)

# ---------------------------------------------------------------------------
# Forward model (defaults reproduce the original hcipy EFC-demo configuration).
# ---------------------------------------------------------------------------
model = CoronagraphModel(seed=1)
dz_mask = model.dz_mask

# Toggles for the three demos below.
SAVE_MOVIE = True
MOVIE_FILENAME = 'san_convergence.gif'
MOVIE_FPS = 2
COMPARE_EFFICIENCY = True
MAINTENANCE_DEMO = True


# ===========================================================================
# 1. CONVERGENCE
# ===========================================================================
num_iterations = 10
gain = 1.0

nuller = SANAndFurious(model, gain=gain)   # swap in any algorithm class here
img_initial = model.image()


def render_frame(iteration, img_now, history):
    """Render one movie frame (initial PSF | current PSF | convergence curve).

    Fixed figsize, color scale and axis limits keep every frame the same pixel
    size and layout, which the .gif encoder requires.  Returns an RGB array.
    """
    fig = plt.figure(figsize=[15, 5], dpi=90)

    ax1 = fig.add_subplot(131)
    ax1.set_title('Initial')
    ax1.imshow(np.log10(img_initial), cmap="inferno", vmax=-2, vmin=-10)

    ax2 = fig.add_subplot(132)
    ax2.set_title(f'SAN iteration {iteration}')
    ax2.imshow(np.log10(img_now), cmap="inferno", vmax=-2, vmin=-10)

    ax3 = fig.add_subplot(133)
    ax3.set_title('Dark-zone mean contrast')
    ax3.semilogy(range(len(history)), history, 'o-')
    ax3.set_xlim(-0.5, num_iterations + 0.5)
    ax3.set_ylim(10**np.floor(np.log10(min(history)) - 0.5),
                 10**np.ceil(np.log10(max(history)) + 0.5))
    ax3.set_xlabel('iteration')
    ax3.set_ylabel('mean contrast')
    ax3.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


contrast_history = [nuller.contrast]
print(f"iter  0: dark-zone mean contrast = {contrast_history[0]:.3e}")

frames = []
if SAVE_MOVIE:
    frames.append(render_frame(0, img_initial, contrast_history))

for k in range(num_iterations):
    nuller.step()
    contrast_history.append(nuller.contrast)
    print(f"iter {k + 1:2d}: dark-zone mean contrast = {contrast_history[-1]:.3e}")
    if SAVE_MOVIE:
        frames.append(render_frame(k + 1, nuller.last_image, contrast_history))

img_corrected = nuller.last_image

if SAVE_MOVIE:
    import imageio.v2 as imageio
    # hold the final frame a little longer so the converged state is visible.
    durations = [1.0 / MOVIE_FPS] * len(frames)
    durations[-1] = 2.0
    imageio.mimsave(MOVIE_FILENAME, frames, duration=durations, loop=0)
    print(f"saved movie: {os.path.abspath(MOVIE_FILENAME)} ({len(frames)} frames)")

plt.figure(figsize=[15, 5])
plt.subplot(131)
plt.title('Initial')
plt.imshow(np.log10(img_initial), cmap="inferno", vmax=-2, vmin=-10)
plt.colorbar()
plt.subplot(132)
plt.title(f'After {num_iterations} SAN iterations')
plt.imshow(np.log10(img_corrected), cmap="inferno", vmax=-2, vmin=-10)
plt.colorbar()
plt.subplot(133)
plt.title('Dark-zone mean contrast')
plt.semilogy(range(len(contrast_history)), contrast_history, 'o-')
plt.xlabel('iteration')
plt.ylabel('mean contrast')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()


# ===========================================================================
# 2. EFFICIENCY: contrast vs *cumulative exposures*
# ===========================================================================
# The fair currency for a real instrument is exposures (probe images), not
# iterations.  Each estimator is rerun from scratch on the SAME static aberration;
# we plot how deep the dark zone gets per exposure spent.  In a noiseless sim plain
# SAN is already very exposure-efficient; the least-squares history reuse is
# expected to pull ahead mainly once measurement noise is present.
if COMPARE_EFFICIENCY:
    def efficiency_curve(make_nuller, n_iter):
        est = make_nuller()
        exposures = [est.n_exposures]
        contrast = [est.contrast]
        for _ in range(n_iter):
            est.step()
            exposures.append(est.n_exposures)
            contrast.append(est.contrast)
        return np.array(exposures), np.array(contrast)

    curves = [
        ('SAN (5-frame, probe every iter)', efficiency_curve(lambda: SpeckleAreaNulling(model, gain=gain), 14)),
        ('SAF LSQ (probe_every_n=1)',       efficiency_curve(lambda: SANAndFurious(model, gain=gain, probe_every_n=1), 14)),
        ('non-SAN 3-frame (MinStep)',       efficiency_curve(lambda: MinStepNulling(model, gain=gain), 22)),
        ('Fast & Furious 1-probe',          efficiency_curve(lambda: FastAndFurious(model, gain=gain), 34)),
        ('FnF correction-only',             efficiency_curve(lambda: FastAndFuriousNoProbe(model, gain=gain), 20)),
    ]

    plt.figure(figsize=[8, 6])
    for label, (exposures, contrast) in curves:
        plt.semilogy(exposures, contrast, 'o-', markersize=4, label=label)
    plt.title('Speckle nulling efficiency: contrast vs exposures')
    plt.xlabel('cumulative exposures (forward-model evaluations)')
    plt.ylabel('dark-zone mean contrast')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 3. MAINTENANCE: hold a dark hole against drift with few probe frames
# ===========================================================================
# The science goal is not the deepest null but holding moderate contrast while
# spending as few *dedicated probe* frames as possible (every probe frame is lost
# science time).  A low-order quasi-static drift is injected with the model's
# disturbance DM, and a single drift realization is replayed for every strategy:
#   - open loop          : freeze the seed command, take only science frames (0 probes).
#   - SAN every N steps   : re-probe every N steps, hold in between (4 probes/event).
#   - science-frame+dither: never take a dedicated probe; put a tiny known dither on
#                           each science frame and run differential least squares.
if MAINTENANCE_DEMO:
    rng = np.random.default_rng(3)
    T_maint = 36
    K_drift = 6            # quasi-static drift lives in a low-dim mode subspace
    drift_sigma = 0.015    # per-step random-walk amplitude (a_probe units)
    dither_amp = 0.05      # micro-dither for the science-frame estimator (a_probe units)

    a_probe = model.a_probe
    nfreq = model.num_frequencies
    cos_probe, sin_probe = model.cos_probe, model.sin_probe
    cos_phase, sin_phase = model.cos_phase, model.sin_phase

    def fwd_drift(actuators, drift_cmd):
        """Forward model with the disturbance DM set to a given drift command."""
        model.set_disturbance(drift_cmd)
        return model.image(actuators)

    # one quasi-static drift realization (low-dim random walk), replayed per strategy
    drift_dirs = (rng.standard_normal((K_drift, nfreq))
                  + 1j * rng.standard_normal((K_drift, nfreq)))
    _amp = np.zeros(K_drift)
    drift_seq = []
    for _ in range(T_maint):
        _amp = _amp + drift_sigma * rng.standard_normal(K_drift)
        dwm = (_amp[:, None] * drift_dirs).sum(0)
        drift_seq.append(model.command(dwm).copy())

    # seed a dark hole at zero drift (shared starting point for every strategy)
    model.set_disturbance(np.zeros(model.num_actuators))
    seed = SpeckleAreaNulling(model, gain=1.0)
    for _ in range(6):
        seed.step()
    act_seed = seed.actuators.copy()
    w_seed = seed.w.copy()
    kap = seed.kappa2.copy()
    histW0 = [w.copy() for w in seed.prior_corrections]   # seed frames bootstrap diversity
    histI0 = [im.copy() for im in seed.prior_images]

    def maint_open():
        c = []
        for t in range(T_maint):
            c.append(fwd_drift(act_seed, drift_seq[t])[dz_mask].mean())
        return np.array(c), 0

    def maint_sparse(N):
        act = act_seed.copy(); c = []; probes = 0
        for t in range(T_maint):
            d = drift_seq[t]
            if t % N == 0:
                I0 = fwd_drift(act, d)[dz_mask]
                Icp = fwd_drift(act + cos_probe, d)[dz_mask]; Icm = fwd_drift(act - cos_probe, d)[dz_mask]
                Isp = fwd_drift(act + sin_probe, d)[dz_mask]; Ism = fwd_drift(act - sin_probe, d)[dz_mask]
                probes += 4
                mc = (Icp + Icm - 2 * I0) / 2; ms = (Isp + Ism - 2 * I0) / 2
                dw = -0.5 * ((Icp - Icm) / (2 * mc) + 1j * (Isp - Ism) / (2 * ms))
                act = act + model.command(dw)
            c.append(fwd_drift(act, d)[dz_mask].mean())
        return np.array(c), probes

    def maint_science(forget=0.7, gain=0.7, dither=dither_amp):
        act = act_seed.copy(); w = w_seed.copy()
        Wh = [x.copy() for x in histW0]; Ih = [x.copy() for x in histI0]; c = []
        for t in range(T_maint):
            d = drift_seq[t]
            # tiny known dither on the science frame, alternating quadrature for diversity
            dd = dither if t % 2 == 0 else 1j * dither
            actd = act + (dither * cos_probe if t % 2 == 0 else dither * sin_probe)
            I0 = fwd_drift(actd, d)[dz_mask]
            Wh.append((w + dd).copy()); Ih.append(I0.copy())

            W = np.asarray(Wh); I = np.asarray(Ih); nfr = W.shape[0]
            wt = (forget ** np.arange(nfr)[::-1])[:, None]
            delta = (w + dd)[None, :] - W; p = delta.real; q = delta.imag
            gg = 0.5 * (kap[None, :] * np.abs(delta) ** 2 - (I - I0[None, :]))
            M = np.empty((nfreq, 2, 2))
            M[:, 0, 0] = (wt * p * p).sum(0)
            M[:, 0, 1] = M[:, 1, 0] = (wt * p * q).sum(0)
            M[:, 1, 1] = (wt * q * q).sum(0)
            b = np.stack([(wt * p * gg).sum(0), (wt * q * gg).sum(0)], axis=-1)
            r = 1e-6 * (M[:, 0, 0] + M[:, 1, 1]) + 1e-30
            M[:, 0, 0] += r; M[:, 1, 1] += r
            Z = np.linalg.solve(M, b)
            # null the UNDITHERED command: R(w) k'* = Z - kap*dd  (subtract known dither)
            dw = -gain * ((Z[:, 0] + 1j * Z[:, 1]) - kap * dd) / kap
            w = w + dw
            act = act + model.command(dw)
            c.append(fwd_drift(act, d)[dz_mask].mean())   # contrast at the (undithered) command
        return np.array(c), 0

    maint = [
        ('open loop (freeze)',     *maint_open()),
        ('SAN every step',         *maint_sparse(1)),
        ('SAN every 4 steps',      *maint_sparse(4)),
        ('SAN every 8 steps',      *maint_sparse(8)),
        ('science-frame + dither', *maint_science()),
    ]

    plt.figure(figsize=[13, 5])
    ax1 = plt.subplot(121)
    for label, c, probes in maint:
        ax1.semilogy(range(T_maint), c, 'o-', markersize=3, label=f'{label} [{probes} probes]')
    ax1.set_title('Maintaining contrast against drift')
    ax1.set_xlabel('maintenance step (science frame)')
    ax1.set_ylabel('dark-zone mean contrast')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = plt.subplot(122)
    for label, c, probes in maint:
        ax2.loglog(max(probes, 0.5), c.mean(), 'o', markersize=9)
        ax2.annotate(label, (max(probes, 0.5), c.mean()),
                     textcoords='offset points', xytext=(6, 4), fontsize=8)
    ax2.set_title('Cost of maintenance: mean contrast vs probe frames')
    ax2.set_xlabel('total dedicated probe frames (less = more science time)')
    ax2.set_ylabel('mean maintained contrast')
    ax2.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()
