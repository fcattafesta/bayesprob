import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

data_path = os.path.join(os.path.dirname(__file__), "data")
fig_path = os.path.join(os.path.dirname(__file__), "figures")

# Reading dataset

f = h5py.File(os.path.join(data_path, "post_data.hdf5"), "r")
X = np.array(f.get("dataset_1"))

t = X[:, 0]
y = X[:, 1]
acf = X[:, 2]

# Normalizing ACF

y_var = np.var(y)

acf = acf[:10000] / (y_var * len(y))

fig, ax = plt.subplots()
ax.scatter(t[:10000], acf, color="black", marker=".")
ax.set_ylabel("ACF")
ax.set_xlabel("t [s]")
ax.grid(True)
fig.savefig(os.path.join(fig_path, "normalized_ACF.pdf"), format="pdf")

plt.show()
