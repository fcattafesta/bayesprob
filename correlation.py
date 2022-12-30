import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.dataset import get_df, fig_path

# Reading dataset

columns = ["y", "t"]

df = get_df(columns=columns)
y = df["y"].values

# In order to see if the process is stationary we must build
# autocorrelation matrix. Under the assumption that points are
# independent from each other, we can split the sample into n
# parts. Then we use definition
#
#                   r_ij = mean(y_i * y_j)
#
# where we take the mean over the splits. 

# Computing variance for normalization

y_var = np.var(y)

# Slicing the sample

n = 10000
n_idx = int(y.size / n)

d = np.empty((n, n_idx))

for i in range(n):
    start = i * n_idx
    stop = (i + 1) * n_idx
    d[i, :] = y[start:stop]

# Computing autocorrelation matrix

r = np.empty((n_idx, n_idx), dtype=np.float16)

for i in range(n_idx):
    for j in range(n_idx):
        r[i, j] = np.mean(d[:, i] * d[:, j]) / y_var

# Visualizing the matrix to see possible patterns

fig, ax = plt.subplots()
c = ax.imshow(r)
fig.colorbar(c)
fig.savefig(os.path.join(fig_path, "correlation_matrix.pdf"), format="pdf")



