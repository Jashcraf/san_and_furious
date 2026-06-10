
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
model = CoronagraphModel(iwa_ld=3, owa_ld=12, seed=1)
dz_mask = model.dz_mask

# Toggles for the three demos below.
SAVE_MOVIE = True
MOVIE_FILENAME = 'min_convergence.gif'
MOVIE_FPS = 2
COMPARE_EFFICIENCY = True
MAINTENANCE_DEMO = True

num_iterations = 10
num_sansteps = 2
gain = 1.0

# Test original 5-step algorithm from Oya+ 2017
# nuller = SpeckleAreaNulling(model, gain=gain) 
nuller = MinStepNulling(model, gain=gain) 
# nuller = SANAndFurious(model, gain=gain) 
# nuller = FastAndFuriousNoProbe(model, gain=gain)
img_initial = model.image()


def render_frame(iteration, img_now, history, algo="MinStep"):
    """Render one movie frame (initial PSF | current PSF | convergence curve).

    Fixed figsize, color scale and axis limits keep every frame the same pixel
    size and layout, which the .gif encoder requires.  Returns an RGB array.
    """
    fig = plt.figure(figsize=[15, 5], dpi=90)

    ax1 = fig.add_subplot(131)
    ax1.set_title('Initial')
    ax1.imshow(np.log10(img_initial), cmap="inferno", vmax=-2, vmin=-10)

    ax2 = fig.add_subplot(132)
    ax2.set_title(f'{algo} iteration {iteration}')
    ax2.imshow(np.log10(img_now), cmap="inferno", vmax=-2, vmin=-10)

    ax3 = fig.add_subplot(133)
    ax3.set_title('Dark-zone mean contrast')
    ax3.semilogy(range(len(history)), history, 'o-')
    ax3.set_xlim(-0.5, num_sansteps * num_iterations + 0.5)
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

for i in range(num_sansteps):
    nuller.san_step()

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