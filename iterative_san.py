import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def dot(a, b):
    """Complex-plane vector dot product Re(conj(a) b)."""
    return a.real * b.real + a.imag * b.imag

def measure(E, read_sigma=0.0):
    """Return measured intensity I = |E|^2 (counts), with optional additive
    Gaussian read noise"""
    I = abs(E)**2
    if read_sigma > 0.0:
        I = I + np.random.normal(0.0, read_sigma)
    return float(I)

def projection_sigma(read_sigma, g):
    """Std of one projection scalar P (counts), read noise only.
    P = (I_a - I_b + g^2|Etilde|^2)/(2g); var(P) = 2 read_sigma^2 / (2g)^2.
    This needs to be changed when dealing with multiple pixels to take an empirical std."""
    return read_sigma / (g * np.sqrt(2.0)) if read_sigma > 0 else 0.0

# ----------------------------------------------------------------------
# Step 0 -- SAN iteration (cost: 5 frames) -> seed estimate Etilde_0
# ----------------------------------------------------------------------
def san_seed(E0, deltaE1, deltaE2, read_sigma=0.0):
    I0  = measure(E0, read_sigma)
    I1p = measure(E0 + deltaE1, read_sigma)
    I1m = measure(E0 - deltaE1, read_sigma)
    I2p = measure(E0 + deltaE2, read_sigma)
    I2m = measure(E0 - deltaE2, read_sigma)

    p = (I1p - I1m) / (2.0 * (I1p + I1m - 2.0 * I0))
    q = (I2p - I2m) / (2.0 * (I2p + I2m - 2.0 * I0))

    Etilde_0 = p * deltaE1 + q * deltaE2     # in sqrt(counts), since deltaE in sqrt(counts)
    return Etilde_0, I0

# ----------------------------------------------------------------------
# Step 1 -- bootstrap: single correction -> two-circle reconstruction + sign GUESS
# ----------------------------------------------------------------------
def bootstrap(Etilde_0, I0, I1, g, s0=+1):
    """Reconstruct E_0 from one applied correction (-g Etilde_0); guess perp sign s0.
    The guess is overwritten by the unambiguous 2x2 solve at the next step."""
    proj0 = (I0 - I1 + g**2 * abs(Etilde_0)**2) / (2.0 * g)   # = Etilde_0 . E_0
    unit  = Etilde_0 / abs(Etilde_0)
    perp  = 1j * unit                                         # 90-deg rotate
    E0_par = proj0 / abs(Etilde_0)                            # scalar component along unit
    E0_perp_mag = np.sqrt(max(0.0, I0 - E0_par**2))
    E0_rec = E0_par * unit + s0 * E0_perp_mag * perp
    Etilde_1 = E0_rec - g * Etilde_0                          # propagate to current frame
    return Etilde_1

# ----------------------------------------------------------------------
# General iteration -- two-projection solve with a noise-tied Tikhonov ridge
# ----------------------------------------------------------------------
def solve_prev_field(Etilde_im2, Etilde_im1, I_im2, I_im1, I_i, g, lam):
    """Solve the true field E_{i-1} from the two most recent corrections.

    lam is the absolute Tikhonov ridge (counts), set by the caller as k*sigma_P.
    When the two probe directions collinearize (off-axis residual in the noise),
    M^T M has a near-zero eigenvalue; the ridge floors it, so the well-determined
    (parallel) component is solved cleanly while the ill-determined (perpendicular)
    component is pulled toward zero -- the correct answer in that limit.
    """
    proj_newest = (I_im1 - I_i  + g**2 * abs(Etilde_im1)**2) / (2.0 * g)  # Etilde_{i-1}.E_{i-1}
    proj_slid   = (I_im2 - I_im1 - g**2 * abs(Etilde_im2)**2) / (2.0 * g)  # Etilde_{i-2}.E_{i-1}

    M = np.array([[Etilde_im2.real, Etilde_im2.imag],
                  [Etilde_im1.real, Etilde_im1.imag]])
    b = np.array([proj_slid, proj_newest])

    A = M.T @ M + lam * np.eye(2)
    x = np.linalg.solve(A, M.T @ b)
    return x[0] + 1j * x[1]

# ----------------------------------------------------------------------
# Driver: run the closed loop
# ----------------------------------------------------------------------
def run(E0_true, deltaE1, deltaE2, g, n_iter, read_sigma=0.0, s0=+1,
        ridge_k=2.0, lam_floor=1e-9, clip_cap=3.0, seed=0, quiet=False):
    """Closed-loop iterative SAN with a noise-tied Tikhonov ridge.

      ridge_k   : ridge factor; lam = ridge_k * sigma_P (sigma_P from image noise).
      lam_floor : tiny absolute ridge used when read_sigma == 0, so the collinear
                  (noiseless) inversion is still well-posed.
      clip_cap  : stability guard rail; caps a stroke at clip_cap x recent mean stroke.
    """

    sigma_P = projection_sigma(read_sigma, g)
    lam = ridge_k * sigma_P if sigma_P > 0 else lam_floor

    E_true = E0_true                         # the optical system's hidden state
    Etilde_hist = []                         # commanded estimates (the "register")
    I_hist = []                              # measured intensities

    def say(*a):
        if not quiet:
            print(*a)

    say(f"  [ridge] sigma_P = {sigma_P:.4g} counts,  lam = {lam:.4g} counts  (k={ridge_k})")

    # --- Step 0: SAN seed ---
    Etilde_0, I0 = san_seed(E_true, deltaE1, deltaE2, read_sigma)
    I_hist.append(I0)
    Etilde_hist.append(Etilde_0)
    say(f"  SAN seed:  Etilde_0 = {Etilde_0:+.3f},  true E_0 = {E_true:+.3f},  |E_0|={abs(E_true):.3f}")

    # apply -g Etilde_0, measure I_1
    E_true = E_true - g * Etilde_0
    I1 = measure(E_true, read_sigma)
    I_hist.append(I1)

    # --- Step 1: bootstrap with sign guess ---
    Etilde_1 = bootstrap(Etilde_0, I0, I1, g, s0=s0)
    Etilde_hist.append(Etilde_1)
    say(f"  bootstrap: Etilde_1 = {Etilde_1:+.3f}  (sign guess s0={s0:+d}),  true E_1 = {E_true:+.3f},  |E_1|={abs(E_true):.3f}")

    # --- Steps 2..n: two-projection solve, no guess ---
    clip_hist = [abs(g * Etilde_0), abs(g * Etilde_1)]   # running record of stroke sizes
    for i in range(2, n_iter + 1):
        Etilde_im1 = Etilde_hist[-1]          # Etilde_{i-1}
        # apply -g Etilde_{i-1}, measure I_i
        E_true = E_true - g * Etilde_im1
        Ii = measure(E_true, read_sigma)
        I_hist.append(Ii)

        # solve previous true field E_{i-1} from last two corrections
        Etilde_im2 = Etilde_hist[-2]
        I_im2, I_im1, I_i = I_hist[-3], I_hist[-2], I_hist[-1]
        E_prev_solved = solve_prev_field(Etilde_im2, Etilde_im1, I_im2, I_im1, I_i, g, lam)

        # propagate to current frame -> new estimate
        Etilde_i = E_prev_solved - g * Etilde_im1

        # fnf-style guard: clip a runaway estimate to the recent stroke scale,
        # so one ill-conditioned solve cannot poison the register it feeds.
        stroke = abs(g * Etilde_i)
        cap = clip_cap * np.mean(clip_hist[-3:])
        if stroke > cap and stroke > 0:
            Etilde_i *= cap / stroke
        clip_hist.append(abs(g * Etilde_i))

        Etilde_hist.append(Etilde_i)

        say(f"  iter {i:2d}:   |E|={abs(E_true):.4e}   I={Ii:.4e}   "
            f"Etilde_i={Etilde_i:+.4f}   E_prev_solved={E_prev_solved:+.4f}")

    return I_hist, Etilde_hist


def _floor_note():
    print("    (note: once |E|^2 falls to the read-noise level, the intensity")
    print("     differences feeding the projections are noise-dominated and the")
    print("     solve cannot dig deeper without more flux, frame averaging, or a")
    print("     deliberate cross-axis probe -- the noise floor.)")


if __name__ == "__main__":

    # one pixel; true field has both real and imag parts (a genuine 2D phasor)
    E0_true = 8.0 + 6.0j          # |E0|^2 = 100 counts
    deltaE1 = 10.0 + 0.0j         # sine probe  -> real axis,  |deltaE1|^2 = 100
    deltaE2 = 0.0 + 10.0j         # cosine probe-> imag axis,  |deltaE2|^2 = 100
    g = 0.5

    print("\n=== NOISELESS (clean geometric convergence down the field's own line) ===")
    I_n, _ = run(E0_true, deltaE1, deltaE2, g, n_iter=10, read_sigma=0.0, s0=+1)

    print("\n=== WRONG bootstrap guess s0=-1 (noiseless): step-2 solve overwrites it ===")
    run(E0_true, deltaE1, deltaE2, g, n_iter=6, read_sigma=0.0, s0=-1)

    # Realistic flux: scale the field up so there is SNR headroom for many decades.
    print("\n=== BRIGHT SPECKLE, read noise, NOISE-TIED ridge (k=2) ===")
    scale = 1.0e3                 # |E0|^2 = 1e8 counts (HDR-like)
    I_b, _ = run(scale*E0_true, scale*deltaE1, scale*deltaE2, g,
                 n_iter=30, read_sigma=5.0, s0=+1, ridge_k=2.0)
    _floor_note()

    # --- convergence plot (contrast = I / I_0) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cn = np.array(I_n) / I_n[0]
    cb = np.array(I_b) / I_b[0]
    ax.semilogy(range(len(cn)), np.clip(cn, 1e-12, None), "o-", label="noiseless", color="#1b6")
    ax.semilogy(range(len(cb)), np.clip(cb, 1e-12, None), "s-",
                label="bright speckle + read noise (noise-tied ridge)", color="#36c")
    ax.set_xlabel("frame index (1 science frame per correction)")
    ax.set_ylabel(r"contrast  $I_k / I_0$")
    ax.set_title("Iterative SAN-seeded control, single pixel")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("iterative_san_convergence.png", dpi=130)
    print("\nsaved plot -> iterative_san_convergence.png")
