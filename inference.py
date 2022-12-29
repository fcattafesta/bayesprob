import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.dataset import data_path, fig_path, get_df
from utils.model import TaylorF2, matched_filter
from utils.prior import prior

df = get_df(columns=["f", "psd", "fft_y"], filename="fft_data.hdf5")

f = np.real(df["f"].values)
psd = df["psd"].values

fig, ax = plt.subplots()
ax.set_ylabel("$\widetilde{y}$")
ax.set_xlabel("")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(f, np.real(df["fft_y"]), color="black", linewidth=0.5)
ax.set_xscale("log")
fig.savefig(os.path.join(fig_path, "fft_y.pdf"), format="pdf")

M = np.arange(25, 35, step=0.1)
q = np.arange(0.5, 1., step=0.05)
x, y = np.meshgrid(M, q)
log_pr = np.log(prior(x, y))

filtered = matched_filter(psd, f, 25, 0.7) * df["fft_y"]


fig, ax = plt.subplots()
ax.set_ylabel("")
ax.set_xlabel("")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(f, np.real(df["fft_y"]), color="black", linewidth=0.5)
ax.plot(f[1:], np.real(filtered[1:]), color="red", linewidth=0.5)
ax.set_xscale("log")
ax.set_xlim(1, 1000)
fig.savefig(os.path.join(fig_path, "filtered_y.pdf"), format="pdf")