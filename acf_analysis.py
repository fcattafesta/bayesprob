import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import UnivariateSpline

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
acf = acf / (y_var * len(y))

# ACF interpolation for later analysis

acf_f = UnivariateSpline(t, acf, s=0)

# Useful slicing for better visualization

acf = acf[:100000]
t = t[:100000]

fig, ax = plt.subplots()
ax.scatter(t, acf, color="black", marker=".")
ax.set_ylabel("ACF")
ax.set_xlabel("t [s]")
ax.grid(True)
fig.savefig(os.path.join(fig_path, "normalized_ACF.pdf"), format="pdf")

# Since ACF is oscillating, we clearly see that this stochastic process is not stationary.
# Although, we can measure (by eye!) the period of the oscillation to test the WSS
# hypothesis

T = 0.15 # s

# We must evaluate ACF(i * T) to verify if it is constant

max_i = int(t[-1] / T)
i = np.arange(max_i)
tt_i = i * T

fig, ax = plt.subplots()
ax.scatter(tt_i, acf_f(tt_i), color="black", marker=".")
ax.set_ylabel("ACF")
ax.set_xlabel("t [s]")
ax.grid(True)
fig.savefig(os.path.join(fig_path, "WSS_ACF.pdf"), format="pdf")


plt.show()
