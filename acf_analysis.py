import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from utils.dataset import data_path, fig_path, get_df

# Reading dataset

df = get_df()

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