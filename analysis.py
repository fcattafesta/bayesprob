import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.fft import rfft, rfftfreq
from utils.dataset import data_path, fig_path, get_df, write_data
from utils.model import TaylorF2, matched_filter, log_likelihood
from utils.prior import prior

y = np.loadtxt(os.path.join(data_path, "H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt"))

cut = int(8e6)
y = y[cut:(len(y) - cut)]

t = np.arange(len(y))
TIME = 4096  # seconds
t = t / TIME
dt = t[1]-t[0]

acf = correlate(y, y)[(y.size-1):] / (np.var(y) * y.size)

y_fft = rfft(y)
psd = rfft(acf)
f = rfftfreq(y.size, d=dt)

M = np.arange(25, 35, step=0.1)
q = np.arange(0.5, 1., step=0.05)

x, y = np.array(np.meshgrid(M, q))

logL = np.zeros(x.shape)

idx_nan = np.ones(y_fft.size)

for i in range(y_fft.size):
    l = log_likelihood(y_fft[i], f[i], x, y, psd[i])
    if np.prod(np.isnan(l)):
        idx_nan[i] = i
    else:
        logL += l

print(logL.shape)

fig, ax = plt.subplots()
c = ax.contourf(x, y, logL)
ax.set_ylabel("q")
ax.set_xlabel("$\mathcal{M}$ [$M_\odot$]")
b = fig.colorbar(c)
fig.savefig(os.path.join(fig_path, "loglikelihood.pdf"), format="pdf")