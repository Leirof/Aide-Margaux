import re
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['image.origin'] = 'lower'
from matplotlib.patches import Circle
from matplotlib.colors import FuncNorm

PATH = "./to_plot" # Path to the folder containing fits to plot

fig, axs = plt.subplots(3, 3, figsize=(19, 15))

row = 0 # Will be used to iterate over the rows of the plot
for file in os.listdir(PATH):

    # Extracting parameters from the file name
    m = re.search('l_Pup_spec1_output(Dirac|Model)_.*_fov([0-9]+)mas_gamma([0-9]+)mas_mu1e([0-9]+).fits', file)

    # Parse (=treat) the parameters and save them in variables
    source = m.group(1)
    fov = int(m.group(2))
    gamma = int(m.group(3))
    mu = int(m.group(4))

    # Considering only the Dirac files (and then looking at the corresponding model files)
    if source != "Dirac":
        continue

    # Loading the current file (dirac)
    with fits.open(os.path.join(PATH,file)) as hdul:
        dirac = np.flip(hdul[0].data, axis=1) # Only interested in the reconstruction
        dirac /= np.max(dirac) # Normalizing the intensity
        n = dirac.shape[0] # Size of the image

    # The corresponding model file
    with fits.open(os.path.join(PATH,file.replace("Dirac","Model"))) as hdul:
        reconstruction = hdul[0].data # Reconstruction
        reconstruction /= np.max(reconstruction) # Normalizing the intensity
        prior = hdul[3].data # Prior
        prior /= np.max(prior) # Normalizing the intensity

    vmax = max(np.max(reconstruction), np.max(dirac)) # Putting reconstructions images on the same scale

    # Plot of the prior
    im = axs[row,0].imshow(prior, cmap='hot', norm=FuncNorm((np.sqrt,lambda x: x**2)))
    divider = make_axes_locatable(axs[row,0]) # Some black magic to have a nice colorbar
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, ax=axs[row,0], cax=cax)
    cbar.set_label('Normalized intensity')

    # Plot of the reconstruction using the model
    im = axs[row,1].imshow(reconstruction, cmap='hot', norm=FuncNorm((np.sqrt,lambda x: x**2), vmax=vmax))
    divider = make_axes_locatable(axs[row,1]) # Some black magic to have a nice colorbar
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, ax=axs[row,1], cax=cax)
    cbar.set_label('Normalized intensity')

    # Plot of the reconstruction using the dirac
    im = axs[row,2].imshow(dirac, cmap='hot', norm=FuncNorm((np.sqrt,lambda x: x**2), vmax=vmax))
    divider = make_axes_locatable(axs[row,2]) # Some black magic to have a nice colorbar
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, ax=axs[row,2], cax=cax)
    cbar.set_label('Normalized intensity')

    # Storing latex symbols inside variables
    alpha = r'$\alpha$'
    delta = r'$\delta$'

    for ax in axs[row]:

        # Mu text
        ax.text(1, n-2, f"$\mu = 1e{mu}$", color='white', fontsize=12, ha='left', va='top')

        # Cross rule to have gamma in pixels
        gamma_px = n*gamma/fov
        # Plotting the circle
        ax.add_patch(Circle((n-2-gamma_px, 1+gamma_px), gamma_px, color='white', fill=False, linewidth=1, linestyle='dashed'))
        
        # Changing ticks from pixel to mas
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels = [str(int((int(label.replace("−","-"))-n//2)/n*fov)) for label in labels]
        ax.set_xticklabels(labels)
        labels = [item.get_text() for item in ax.get_yticklabels()]
        labels = [str(int((int(label.replace("−","-"))-n//2)/n*fov)) for label in labels]
        ax.set_yticklabels(labels)

        # Labels
        ax.set_xlabel(f"{alpha} (mas)")
        ax.set_ylabel(f"{delta} (mas)")

    # Title on the left of the first column
    axs[row,0].set_ylabel(f"$\gamma = {gamma} mas$\n\n{delta} (mas)")

    # Incrementing the row (to go to the next one in the next iteration)
    row += 1

# Titles for the columns
axs[0,0].set_title("Model fit prior")
axs[0,1].set_title("Reconstruction using model fit prior")
axs[0,2].set_title("Reconstruction using dirac prior")

# Title for the whole plot
fig.suptitle("MiRA image reconstruction of l Puppis with FoV=30mas", fontsize=16, va='top')

# Saving the plot
fig.savefig("comparison.png", dpi=300)
# plt.show()
