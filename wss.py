import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, welch
from scipy.fft import rfft, rfftfreq
from utils.dataset import get_df, fig_path

# Reading dataset

columns = ["y", "t", "acf"]

df = get_df(columns=columns)
y = df["y"].values
t = df["t"].values

dt = 8 # s
n = int(t[-1] / dt)
n_idx = int(y.size / n)

acf = np.empty((n, n_idx))
psd = np.empty((n, int(n_idx/2)+1))
act = np.empty(n)
tt = t[:n_idx]

for i in range(n):
    start = i * n_idx
    stop = (i + 1) * n_idx
    x = y[start:stop]
    acf[i, :] = correlate(x, x, mode="full", method="auto")[x.size - 1 :] / (np.var(x) * x.size)
    act[i] = t[np.where(np.abs(acf[i, :]) < 0.01)[0][0]]
    psd[i, :] = np.absolute(rfft(acf[i, :]))

f = rfftfreq(acf.shape[1], d = t[1]-t[0])

f_welch, psd_welch = welch(y, 1/(t[1]-t[0]), window="tukey", nperseg=int(4096/8))

fig, ax = plt.subplots()
ax.set_ylabel("ACF")
ax.set_xlabel("$\Delta t$ [s]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(tt, acf[0, :], color="black", linewidth=0.5)
ax.plot(tt, acf[1, :], color="red", linewidth=0.5)
ax.plot(tt, acf[2, :], color="blue", linewidth=0.5)
fig.savefig(os.path.join(fig_path, "wss_acf.pdf"), format="pdf")

fig, ax = plt.subplots()
ax.set_ylabel("PSD")
ax.set_xlabel("f [Hz]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(f, psd[0, :], color="black", linewidth=0.5)
ax.plot(f, psd[4, :], color="red", linewidth=0.5)
ax.plot(f, psd[8, :], color="blue", linewidth=0.5)
ax.set_xlim(10, 2000)
# ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig(os.path.join(fig_path, "wss_psd.pdf"), format="pdf")

fig, ax = plt.subplots()
ax.set_ylabel("PSD")
ax.set_xlabel("f [Hz]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(f_welch, psd_welch, color="black", linewidth=0.5)
#ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig(os.path.join(fig_path, "psd_welch.pdf"), format="pdf")

fig, ax = plt.subplots()
ax.set_ylabel("ASD")
ax.set_xlabel("f [Hz]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(f_welch, np.sqrt(psd_welch), color="black", linewidth=0.5)
ax.set_xlim(10, 2000)
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig(os.path.join(fig_path, "asd_welch.pdf"), format="pdf")

fig, ax = plt.subplots()
ax.set_ylabel("ACT [s]")
ax.set_xlabel("n")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.scatter(np.arange(0, n), act, color="black", linewidth=0.5, marker=".")
fig.savefig(os.path.join(fig_path, "act.pdf"), format="pdf")

