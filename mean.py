import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.dataset import data_path, fig_path, get_df, write_data

# Reading dataset

columns = ["y", "t"]

df = get_df(columns=columns)
y = df["y"].values
t = df["t"].values

print(t)

# In order to compute mean function, we slice the sample dataset in n bits 
# and for each of them we compute the mean.

dt = 5 # s
n = int(t[-1] / dt)
n_idx = int(y.size / n)
d = np.empty((n, n_idx))

for i in range(n):
    start = i * n_idx
    stop = (i + 1) * n_idx
    d[i, :] = y[start:stop]

# Computing mean for each bit

y_mean = np.empty(n)

for i in range(n):
    y_mean[i] = np.mean(d[i, :])

# Getting the time of each part

tt_n = t[np.arange(0, n) * n_idx]

# Final value of the mean

mean = np.mean(y_mean)
mean_dev = np.std(y_mean)
s = rf"$\overline{{y}}={mean:.2e} \pm {mean_dev:.2e}$"
print(s)
print(tt_n)
# Plot of y_mean

fig, ax = plt.subplots()
ax.set_ylabel("$\overline{y}(t)$")
ax.set_xlabel("t [s]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(tt_n, y_mean, color="black", linewidth=0.5)
ax.plot(tt_n, (mean + mean_dev)*np.ones_like(tt_n), color="red", ls="--")
ax.plot(tt_n, (mean - mean_dev)*np.ones_like(tt_n), color="red", ls="--")
ax.plot(tt_n, (mean + 2*mean_dev)*np.ones_like(tt_n), color="blue", ls="--")
ax.plot(tt_n, (mean - 2*mean_dev)*np.ones_like(tt_n), color="blue", ls="--")
#ax.text(0.5, 0.9, s, transform=ax.transAxes, horizontalalignment='center',verticalalignment='center')
fig.savefig(os.path.join(fig_path, "mean.pdf"), format="pdf")


# Normalizing data to get zero-mean process if mean is not zero

# df["y_norm"] = y - mean

# Writing on file 
# print(df)
# write_data(df)
