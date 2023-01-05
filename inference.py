import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.dataset import data_path, fig_path, get_df
from utils.model import TaylorF2, matched_filter, log_likelihood
from utils.prior import prior

df = get_df(columns=["f", "y_fft", "psd"], filename="fft_data.hdf5")
y_fft = df["y_fft"].values
f = df["f"].values
psd = df["psd"].values

M = np.arange(25, 35, step=0.1)
q = np.arange(0.5, 1., step=0.05)

x, y = np.array(np.meshgrid(M, q))

logL = np.zeros(x.shape)

idx_nan = np.ones(y_fft.size)

for i in range(y_fft.size):
    l = log_likelihood(y_fft[i], f[i], x, y, psd[i])
    if np.prod(np.isnan(l)):
        idx_nan[i] = i
    else:
        logL += l

fig, ax = plt.subplots()
c = ax.contourf(x, y, logL)
ax.set_ylabel("q")
ax.set_xlabel("$\mathcal{M}$ [$M_\odot$]")
b = fig.colorbar(c)
fig.savefig(os.path.join(fig_path, "loglikelihood.pdf"), format="pdf")