import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.fft import rfft, rfftfreq
from utils.dataset import data_path, fig_path, get_df
from utils.masks import alternate_mask

# Reading dataset

df = get_df()

# In order to compute likelihood function, we need Power Spectral Density, 
# which is the Fourier transform of ACF (Wiener-Kinchin theorem). 
# To perform the calculations we use RFFT, which discards negative frequences
# for real input

acf = df["acf"].values
psd = rfft(acf)

# Computing sample spacing and the corresponding sample frequences

t = df["t"].values
dt = t[1]-t[0]
n = acf.size
f = rfftfreq(n, d=dt)

# Plot

fig, ax = plt.subplots()
ax.set_ylabel("PSD")
ax.set_xlabel("Hz")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(f, np.absolute(psd), color="black", linewidth=0.5)
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig(os.path.join(fig_path, "psd.pdf"), format="pdf")