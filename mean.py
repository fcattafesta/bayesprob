import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import UnivariateSpline
from utils.dataset import data_path, fig_path, get_df, write_data

# Reading dataset

df = get_df()

# First of all, we want to compute the mean of the process as a function of time.

y = df["y"].values
y_sum = 0
y_mean = np.zeros_like(y)
for i, val in enumerate(y):
    y_sum += val
    y_mean[i] = y_sum / (i+1)

# Plot of y_mean

fig, ax = plt.subplots()
ax.set_ylabel("$\overline{y}(t)$")
ax.set_xlabel("t [s]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(df["t"], y_mean, color="black", linewidth=0.5)
fig.savefig(os.path.join(fig_path, "signal_mean.pdf"), format="pdf")

# Normalizing data to get zero-mean process

df["y_norm"] = y - np.mean(y)

# Writing on file

write_data(df)
