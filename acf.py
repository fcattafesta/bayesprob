import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from utils.dataset import data_path, fig_path, get_df, write_data

# Reading dataset

columns = ["y", "t"]

df = get_df(columns=columns)
t = df["t"].values
y = df["y"].values

# If the process is WSS, we know that autocorrelation matrix is function only of ACF.
# Computing 2-points autocorrelation with scipy.signal.correlate, as suggested in
# https://numpy.org/doc/stable/reference/generated/numpy.correlate.html. This
# function uses FFT to perform fast computation, as direct calculation is not
# time-sustainable. We select the second half of the array to have the t > 0
# autocorrelation (see
# https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation)

acf = correlate(y, y, mode="full", method="auto")[(len(y) - 1) :]

# Normalzing ACF

y_var = np.var(y)
acf = acf / (y * len(y))
df["acf"] = acf

# Plot of ACF_norm

fig, ax = plt.subplots()
ax.set_ylabel("ACF")
ax.set_xlabel("t [s]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(t, acf, color="black", linewidth=0.5)
fig.savefig(os.path.join(fig_path, "acf.pdf"), format="pdf")

# Zooming in 

fig, ax = plt.subplots()
ax.set_ylabel("ACF")
ax.set_xlabel("t [s]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(t[:10000], acf[:10000], color="black", linewidth=0.5)
fig.savefig(os.path.join(fig_path, "acf_zoom.pdf"), format="pdf")

write_data(df)