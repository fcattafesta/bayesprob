import os
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

data_path = os.path.join(os.path.dirname(__file__), "data", "data.txt")
figure_path = os.path.join(os.path.dirname(__file__), "figures")

names, m, m_errup, m_errdown, sigma, sigma_errup, sigma_errdown = np.genfromtxt(
    data_path, dtype=None, unpack=True
)

m_err = np.array(list(zip(m_errdown, m_errup))).T
sigma_err = np.array(list(zip(sigma_errdown, sigma_errup))).T

fig, ax = plt.subplots()
ax.set_xlabel("$\sigma_{GC}$ [km $s^{-1}$]")
ax.set_ylabel("$M_{BH}/M_\odot$")
ax.set_yscale("log")
ax.set_ylim(1e6, 1e10)
ax.set_xlim(50, 350)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.grid(True, which="major", linestyle="--", alpha=0.5)
ax.errorbar(
    sigma, m, yerr=m_err, xerr=sigma_err, ls="", color="black", marker=".", capsize=2
)
# for i, name in enumerate(names):
#    ax.annotate(text=name.decode(), xy=(sigma[i], m[i]))
fig.savefig(os.path.join(figure_path, "data_plot.pdf"), format="pdf")
