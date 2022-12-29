import os
import numpy as np
import matplotlib.pyplot as plt 
from utils.dataset import fig_path

def prior(M, q):
    return M * np.power((1 + q) / (q**3), (2./5))

M = np.arange(25, 35, step=0.1)
q = np.arange(0.5, 1., step=0.05)

x, y = np.meshgrid(M, q)

log_pr = np.log(prior(x, y))

fig, ax = plt.subplots()
c = ax.contourf(x, y, log_pr)
ax.set_ylabel("q")
ax.set_xlabel("$\mathcal{M}$ [$M_\odot$]")
b = fig.colorbar(c)
b.set_label("$\propto$logP")
fig.savefig(os.path.join(fig_path, "log_prior.pdf"), format="pdf")