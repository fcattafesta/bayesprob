import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import h5py

# Defining data and figures paths

data_path = os.path.join(os.path.dirname(__file__), "data")
fig_path = os.path.join(os.path.dirname(__file__), "figures")

# Data loading using numpy arrays

y = np.loadtxt(os.path.join(data_path, "H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt"))

# Generating x-axis array and normalizing to acquisition TIME

t = np.arange(len(y))
TIME = 4096  # seconds
t = t / TIME

# Plotting signal as a function of time

fig, ax = plt.subplots()
ax.plot(t, y, color="black", linewidth=0.08)
ax.set_ylabel("y(t)")
ax.set_xlabel("t [s]")
ax.grid(True)
fig.savefig(os.path.join(fig_path, "signal.pdf"), format="pdf")

# Computing sample mean as a function of t

y_mean = np.zeros_like(t)
y_sum = 0

for i, dt in enumerate(t):
    y_sum += y[i]
    if dt == 0:
        y_mean[i] = y_sum
    else:
        y_mean[i] = y_sum / dt

fig, ax = plt.subplots()
ax.plot(t, y_mean, color="black")
ax.set_ylabel(r"$\overline{y}(t)$")
ax.set_xlabel("t [s]")
ax.grid(True)
fig.savefig(os.path.join(fig_path, "mean_vs_time.pdf"), format="pdf")

plt.show()

y_mean = np.mean(y)
y_var = np.var(y)
y_std = np.sqrt(y_var)

# print(f"Mean: {y_mean} \nVariance: {y_var}")

# Normalizing sample to have 0 mean

y = y - y_mean

y_mean = np.mean(y)
y_var = np.var(y)
y_std = np.sqrt(y_var)

# print(f"Post-normalization: \nMean: {y_mean} \nVariance: {y_var}")

# Signal plot after normalization

fig, ax = plt.subplots()
ax.plot(t, y, color="black", linewidth=0.08)
ax.set_ylabel("y(t)")
ax.set_xlabel("t [s]")
ax.grid(True)
fig.savefig(os.path.join(fig_path, "normalized_signal.pdf"), format="pdf")

# Computing 2-points autocorrelation with scipy.signal.correlate, as suggested in
# https://numpy.org/doc/stable/reference/generated/numpy.correlate.html. This
# function uses FFT to perform fast computation, as direct calculation is not
# time-sustainable. Why we select only that part of the output?

acf = correlate(y, y, mode="full", method="auto")[(len(y) - 1) :]

# Plotting ACF

fig, ax = plt.subplots()
ax.plot(t, acf, color="black", linewidth=0.08)
ax.set_ylabel("ACF")
ax.set_xlabel("t [s]")
ax.grid(True)
fig.savefig(os.path.join(fig_path, "ACF_full.pdf"), format="pdf")

# Saving normalized sample and ACF for further analysis

X = np.stack([t, y, acf], axis=1)

f = h5py.File(os.path.join(data_path, "post_data.hdf5"), "w")
f.create_dataset("dataset_1", data=X)
f.close()
