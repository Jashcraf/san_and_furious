import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from san import (
    CoronagraphModel,
    SpeckleAreaNulling,
    MinStepNulling,
    FastAndFuriousNoProbe,
)

# --- inputs ---
SAN_ITERS = 3
SAF_ITERS = 10
# --------------

# init model 
model = CoronagraphModel(iwa_ld=3, owa_ld=12, seed=1)
dz_mask = model.dz_mask
img_initial = model.image()

# Init furious nuller
nuller = FastAndFuriousNoProbe(model, forget=1)
contrast_history = [nuller.contrast]

# warm up with SAN steps
for i in range(SAN_ITERS):
    nuller.san_step()
    contrast_history.append(nuller.contrast)
    print(np.array(nuller.prior_corrections).shape)

post_san = nuller.last_image

# Run psi
for i in range(SAF_ITERS):
    nuller.psi_step()
    contrast_history.append(nuller.contrast)

post_psi = nuller.last_image

norm = LogNorm(vmin=1e-8, vmax=1e-4)
plt.figure(figsize=[15, 5])
plt.subplot(131)
plt.title("3 SAN iters")
plt.imshow(post_san, norm=norm, cmap="inferno")
plt.colorbar()
plt.subplot(132)
plt.title("3 PSI iter")
plt.imshow(post_psi, norm=norm, cmap="inferno")
plt.colorbar()
plt.subplot(133)
plt.plot(contrast_history, marker="o")
plt.xlabel("Algorithm Iteration")
plt.ylabel("Contrast")
plt.yscale("log")
plt.axvline(x=SAN_ITERS, linestyle="dashed", color="k")
plt.show()