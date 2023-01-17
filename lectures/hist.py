import os
import numpy as np
from matplotlib import pyplot as plt

N = 10000

# Sampling N standard gaussian numbers

x = np.random.normal(size=N)

# Creating different partitions

bins = [5, 25, 50, 75, 100]

for i, bins in enumerate(bins):

    # Making figures and histogram for each different partition

    plt.figure(i)
    n, edges, _ = plt.hist(
        x, bins=bins, histtype="step", label=f"Hist ({bins} bins)", color="red"
    )

    bincenters = np.array(
        [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]
    )

    # We know that posterior distribution is proportional to a dirichlet
    # distribution. We evaluate expected values for bin centers to see the
    # difference

    alpha = np.ones(bins)

    exp_value = (n + alpha) / (N + alpha.sum())

    plt.plot(
        bincenters,
        exp_value * N,
        linestyle="",
        marker=".",
        color="black",
        label="Dirichlet Exp. Val.",
    )

    plt.text(np.min(bincenters), np.max(n), f"N = {N}")
    plt.legend()
    plt.savefig(
        os.path.join(
            os.path.dirname(__file__), "histograms_fig", f"{N}", f"hist_{bins}_bins.pdf"
        ),
        format="pdf",
    )
