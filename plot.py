import re
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from matplotlib.patches import Circle
from matplotlib.colors import FuncNorm

PATH = "./to_plot" # Path to the folder containing fits to plot

fig, axs = plt.subplots(3, 3, figsize=(15, 15))

row = 0
for file in os.listdir(PATH):

    m = re.search('l_Pup_spec1_output(Dirac|Model)_.*_fov([0-9]+)mas_gamma([0-9]+)mas_mu1e([0-9]+).fits', file)

    source = m.group(1)
    fov = int(m.group(2))
    gamma = int(m.group(3))
    mu = int(m.group(4))

    if source != "Dirac":
        continue

    with fits.open(os.path.join(PATH,file)) as hdul:
        dirac = np.flip(hdul[0].data, axis=1)
        n = dirac.shape[0]

    with fits.open(os.path.join(PATH,file.replace("Dirac","Model"))) as hdul:
        reconstruction = hdul[0].data
        prior = hdul[3].data

    vmax = max(np.max(reconstruction), np.max(dirac))

    im = axs[row,0].imshow(prior, cmap='hot', norm=FuncNorm((np.sqrt,lambda x: x**2)))
    im.set_clim(0,1)
    cbar = plt.colorbar(im, ax=axs[row,0])
    cbar.set_label('Normalized intensity')
    im = axs[row,1].imshow(reconstruction, cmap='hot', norm=FuncNorm((np.sqrt,lambda x: x**2), vmax=vmax))
    cbar = plt.colorbar(im, ax=axs[row,1])
    cbar.set_label('Normalized intensity')
    im = axs[row,2].imshow(dirac, cmap='hot', norm=FuncNorm((np.sqrt,lambda x: x**2), vmax=vmax))
    cbar = plt.colorbar(im, ax=axs[row,2])
    cbar.set_label('Normalized intensity')


    for ax in axs[row]:
        ax.text(1, n-2, f"$\mu = 1e{mu}$", color='white', fontsize=12, ha='left', va='top')
        gamma_px = n*gamma/fov
        ax.add_patch(Circle((n-2-gamma_px, 1+gamma_px), gamma_px, color='white', fill=False, linewidth=1, linestyle='dashed'))
        
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels = [str(int((int(label.replace("−","-"))-n//2)/n*fov)) for label in labels]
        ax.set_xticklabels(labels)
        labels = [item.get_text() for item in ax.get_yticklabels()]
        labels = [str(int((int(label.replace("−","-"))-n//2)/n*fov)) for label in labels]
        ax.set_yticklabels(labels)

    axs[row,0].set_ylabel(f"$\gamma = {gamma} mas$")

    row += 1

axs[0,0].set_title("Model fit prior")
axs[0,1].set_title("Reconstruction using model fit prior")
axs[0,2].set_title("Reconstruction using dirac prior")

fig.suptitle("MiRA image reconstruction of l Puppis with FoV=30mas", fontsize=16)

fig.savefig("comparison.png", dpi=300)
# plt.show()
