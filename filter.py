import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.dataset import data_path, fig_path, get_df
from utils.model import TaylorF2

def snr(y_fft, f, x, y, S):
    #num = np.zeros(x.shape)
    #den = np.zeros(x.shape)
    w = np.zeros(x.shape)
    n = np.zeros(x.shape)
    for i in range(1, y_fft.size):
        if i == 0:
            pass
        h = TaylorF2(f[i], x, y)
        w += np.absolute(y_fft[i] * np.conjugate(h) / S[i])
        n += np.absolute(np.absolute(h)**2 / S[i])
        #num += w
        #den += n
    return w/n


df = get_df(columns=["f", "y_fft", "psd"], filename="fft_data.hdf5")
y_fft = df["y_fft"].values
f = df["f"].values
S = df["psd"].values

M = np.arange(25, 35, step=0.1)
q = np.arange(0.5, 1., step=0.05)

x, y = np.meshgrid(M, q)

a = snr(y_fft, f, x, y, S)

#print(a)

fig, ax = plt.subplots()
c = ax.contourf(x, y, a, levels=100)
ax.set_ylabel("q")
ax.set_xlabel("$\mathcal{M}$ [$M_\odot$]")
b = fig.colorbar(c)
b.set_label("SNR")
# fig.savefig(os.path.join(fig_path, "prior.pdf"), format="pdf")
plt.show()







