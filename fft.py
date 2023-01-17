import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from utils.dataset import data_path, fig_path, get_df, write_data

# Reading dataset

columns = ["y", "t", "acf"]

df = get_df(columns=columns)
y = df["y"].values
acf = df["acf"].values * (np.var(y) * y.size)
t = df["t"].values

# In order to compute likelihood function, we need Power Spectral Density, 
# which is the Fourier transform of ACF (Wiener-Kinchin theorem). 
# To perform the calculations we use RFFT, which discards negative frequences
# for real input

psd = rfft(acf)

# Same for sample data

y_fft = rfft(y)

# Computing sample spacing and the corresponding sample frequences

dt = t[1]-t[0]
n = acf.size
f = rfftfreq(n, d=dt)

# PSD plot

fig, ax = plt.subplots()
ax.set_ylabel("PSD")
ax.set_xlabel("Hz")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(f, np.absolute(psd), color="black", linewidth=0.5)
#ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(10, 2000)
fig.savefig(os.path.join(fig_path, "psd.pdf"), format="pdf")

# Data FFT

fig, ax = plt.subplots()
ax.set_ylabel("$\widetilde{y}$")
ax.set_xlabel("Hz")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(f, np.absolute(y_fft), color="black", linewidth=0.5)
#ax.set_xscale("log")
ax.set_xlim(10, 2000)
ax.set_yscale("log")
fig.savefig(os.path.join(fig_path, "y_fft.pdf"), format="pdf")

# Write data

df = pd.DataFrame()
df["f"] = f
df["y_fft"] = y_fft
df["psd"] = psd
print(psd)
write_data(df, filename="fft_data.hdf5")