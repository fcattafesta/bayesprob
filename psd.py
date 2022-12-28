import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.fft import rfft, fftfreq

from utils.dataset import data_path, fig_path, get_df

# Reading dataset

df = get_df()

# In order to compute likelihood function, we need Power Spectral Density, 
# which is the Fourier transform of ACF (Wiener-Kinchin theorem)

acf = df["acf"].values
psd = rfft(acf)

t = df["t"].values
dt = t[1]-t[0]
n = len(psd)
f = fftfreq(n, d=dt)

# Plot 

fig, ax = plt.subplots()
ax.set_ylabel("PSD")
ax.set_xlabel("")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(f, np.absolute(psd), color="black", linewidth=0.5)
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig(os.path.join(fig_path, "psd.pdf"), format="pdf")