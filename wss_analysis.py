import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from path import data_path, fig_path
from mean import columns

f = h5py.File(os.path.join(data_path, "post_data.hdf5"), "r")
df = pd.DataFrame(f.get("H-H1_GWOSC_4KHZ_R1-1126257415-4096"), columns=columns)
f.close()

def mask(arr):
    m = np.zeros_like(arr, dtype=bool)
    for i, elm in enumerate(arr):
        if i % 10000 == 0:
            m[i] = 1
    return m

y_norm = df["y_norm"].values
y_var = np.var(y_norm)
m = mask(y_norm)
y_norm = y_norm[:1000]
n = len(y_norm)
r = np.ones((n, n))

for i in range(n):
    for j in range(n):
        r[i, j] = np.mean(y_norm[i] * y_norm[j]) / y_var

fig, ax = plt.subplots()
c = ax.imshow(r)
fig.colorbar(c)
fig.savefig(os.path.join(fig_path, "correlation_matrix.pdf"), format="pdf")



