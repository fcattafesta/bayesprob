import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from utils.dataset import get_df, fig_path

# Reading dataset

columns = ["y", "t", "acf"]

df = get_df(columns=columns)
y = df["y"].values
t = df["t"].values

dt = 1 # s
n = int(t[-1] / dt)
n_idx = int(y.size / n)

acf = np.empty((n, n_idx))
act = np.empty(n)
tt = t[:n_idx]

for i in range(n):
    start = i * n_idx
    stop = (i + 1) * n_idx
    x = y[start:stop]
    acf[i, :] = correlate(x, x, mode="full", method="auto")[x.size - 1 :] / (np.var(x) * x.size)
    act[i] = t[np.where(np.abs(acf[i, :]) < 0.01)[0][0]]



fig, ax = plt.subplots()
ax.set_ylabel("ACF")
ax.set_xlabel("t [s]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(tt, acf[0, :], color="black", linewidth=0.5)
ax.plot(tt, acf[1, :], color="red", linewidth=0.5)
ax.plot(tt, acf[2, :], color="blue", linewidth=0.5)
fig.savefig(os.path.join(fig_path, "wss_acf.pdf"), format="pdf")

fig, ax = plt.subplots()
ax.set_ylabel("ACT [s]")
ax.set_xlabel("n")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.scatter(np.arange(0, n), act, color="black", linewidth=0.5, marker=".")
fig.savefig(os.path.join(fig_path, "act.pdf"), format="pdf")