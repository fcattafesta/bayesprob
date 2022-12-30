import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from utils.dataset import get_df, fig_path

df = get_df()

x = df["y_norm"].values

x_var = np.var(x)

n = 10000
n_idx = int(x.size / n)

d = np.empty((n, n_idx))

for i in range(n):
    start = i * n_idx
    stop = (i + 1) * n_idx
    d[i, :] = x[start:stop]

r = np.empty((n_idx, n_idx), dtype=np.float16)

for i in range(n_idx):
    for j in range(n_idx):
        r[i, j] = np.mean(d[:, i] * d[:, j]) / x_var


fig, ax = plt.subplots()
c = ax.imshow(r)
fig.colorbar(c)
fig.savefig(os.path.join(fig_path, "correlation_matrix.pdf"), format="pdf")



