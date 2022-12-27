import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
import h5py
from path import data_path, fig_path


y = np.loadtxt(os.path.join(data_path, "H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt"))

# Using pandas DataFrame to create dataset

df = pd.DataFrame(y, columns=["y"])

# Generating x-axis array and normalizing to acquisition TIME

t = np.arange(len(y))
TIME = 4096  # seconds
t = t / TIME
df["t"] = t

# Plotting sample for visualization

fig, ax = plt.subplots()
ax.set_ylabel("y(t)")
ax.set_xlabel("t [s]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(t, y, color="black", linewidth=0.08)
fig.savefig(os.path.join(fig_path, "signal.pdf"), format="pdf")

# Computing 2-points autocorrelation with scipy.signal.correlate, as suggested in
# https://numpy.org/doc/stable/reference/generated/numpy.correlate.html. This
# function uses FFT to perform fast computation, as direct calculation is not
# time-sustainable. We select the second half of the array to have the t > 0
# autocorrelation (see
# https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation)

acf = correlate(y, y, mode="full", method="auto")[(len(y) - 1) :]
df["acf"] = acf

# Saving samples and ACF for further analysis

f = h5py.File(os.path.join(data_path, "post_data.hdf5"), "w")
f.create_dataset("H-H1_GWOSC_4KHZ_R1-1126257415-4096", data=df)
f.close()
