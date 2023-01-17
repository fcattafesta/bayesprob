import os
import numpy as np
import matplotlib.pyplot as plt 
from dataset import fig_path
from scipy.stats.contingency import margins

def prior(M, q):
    return M * np.power((1 + q) / (q**3), (2./5))

if __name__ == "__main__":

    M = np.arange(25, 35, step=0.1)
    q = np.arange(0.5, 1, step=0.05)

    print(M)

    x, y = np.meshgrid(M, q)
    pr = prior(x, y)
    log_pr = np.log(pr)

    fig, ax = plt.subplots()
    c = ax.contourf(x, y, pr, levels=100)
    ax.set_ylabel("q")
    ax.set_xlabel("$\mathcal{M}$ [$M_\odot$]")
    b = fig.colorbar(c)
    b.set_label("$\propto$ P")
    fig.savefig(os.path.join(fig_path, "prior.pdf"), format="pdf")

    p1, p2 = margins(pr)

    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()        


    fig, ax = plt.subplots()
    ax.grid(True, ls="--", alpha=0.5)
    ax.minorticks_on()
    ax.tick_params(direction="in", which="both")
    plt.plot(q, p1, color='blue')
    plt.fill_between(q, p1.flatten(), alpha=0.5, color="blue")
    ax.set_xlabel("q")
    ax.set_ylim(0)
    ax.set_xlim(q[0], q[-1])
    fig.savefig(os.path.join(fig_path, "q_prior.pdf"), format="pdf")

    fig, ax = plt.subplots()
    ax.grid(True, ls="--", alpha=0.5)
    ax.minorticks_on()
    ax.tick_params(direction="in", which="both")
    plt.plot(M, p2.T, color='red')
    plt.fill_between(M, p2.T.flatten(), alpha=0.5, color="red")
    ax.set_ylim(0)
    ax.set_xlabel("$\mathcal{M}$ [$M_\odot$]")
    ax.set_xlim(M[0], M[-1])
    fig.savefig(os.path.join(fig_path, "M_prior.pdf"), format="pdf")
