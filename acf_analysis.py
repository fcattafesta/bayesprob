import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import UnivariateSpline

from path import data_path, fig_path
from mean import columns

# Reading dataset

f = h5py.File(os.path.join(data_path, "post_data.hdf5"), "r")
df = pd.DataFrame(f.get("H-H1_GWOSC_4KHZ_R1-1126257415-4096"), columns=columns)
f.close()

# Normalzing ACF

t = df["t"].values

y_var = np.var(df["y"])
acf_norm = df["acf"] / (y_var * len(t))

# Plot of ACF_norm

fig, ax = plt.subplots()
ax.set_ylabel("ACF")
ax.set_xlabel("t [s]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(t, acf_norm, color="black", linewidth=0.5)
fig.savefig(os.path.join(fig_path, "acf.pdf"), format="pdf")

# Zooming in 

fig, ax = plt.subplots()
ax.set_ylabel("ACF")
ax.set_xlabel("t [s]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(t[:10000], acf_norm[:10000], color="black", linewidth=0.5)
fig.savefig(os.path.join(fig_path, "acf_zoom.pdf"), format="pdf")