import numpy as np
import sys


def log_posterior(x, mu=0.0, sigma=1.0, min=-10, max=10):
    return log_prior(x, min=min, max=max) + log_likelihood(x, mu=mu, sigma=sigma)


def log_prior(x, min=-10, max=10):
    if x < min or x > max:
        return -np.inf
    return 0.0


def log_likelihood(x, mu=0.0, sigma=1.0):
    r = (x - mu) / sigma
    return -0.5 * r**2


def uniform_proposal(x0, rng):
    return x0 + rng.uniform(-1, 1)


def gaussian_proposal(x0, rng):
    return x0 + rng.normal(0, 0.3)


def metropolis_hastings(
    target, proposal, rng, n=1000, min=-10, max=10, mu=0.0, sigma=1.0
):

    samples = []
    x0 = rng.uniform(min, max)
    logP0 = target(x0, mu=mu, sigma=sigma, min=min, max=max)

    accepted = 0
    rejected = 1

    i = 0

    while i < n:

        xt = proposal(x0, rng)
        logP = target(xt, mu=mu, sigma=sigma, min=min, max=max)
        logr = logP - logP0

        if np.log(rng.uniform(0, 1)) < logr:
            x0 = xt
            logP0 = logP
            samples.append(xt)
            accepted += 1
        else:
            samples.append(x0)
            rejected += 1

        sys.stderr.write(
            "i:{0} acc = {1}\r".format(i, np.float64(accepted / (accepted + rejected)))
        )
        i += 1
    sys.stderr.write("\n")
    return np.array(samples)


def autocorrelation(chain):
    m = np.mean(chain)
    s = np.var(chain)
    xhat = chain - m
    acorr = np.correlate(xhat, xhat, "full")[len(xhat) - 1 :]
    return acorr / s / len(xhat)


def ACT(acf, tolerance=0.01, n=3):
    (t,) = np.where(np.abs(acf) < tolerance)
    return n * t[0]


if __name__ == "__main__":

    rng = np.random.default_rng(4444)
    samples = metropolis_hastings(
        log_posterior, gaussian_proposal, rng, n=10000, mu=3.0, sigma=0.2
    )

    burnin = 1000

    samples = samples[burnin:]

    # save the output and do the post-processing afterwards
    acf = autocorrelation(samples)
    thinning = ACT(acf, tolerance=0.01, n=1)
    print("measured ACT = {0}".format(thinning))

    import matplotlib.pyplot as plt

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(samples, ".b")
    ax.set_xlabel("iteration")

    from scipy.stats import norm

    pdf = norm(3.0, 0.2)
    x = np.linspace(-10, 10, 1000)

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.hist(samples[::thinning], density=True, bins=64)
    ax.plot(x, pdf.pdf(x), "-k")
    ax.set_xlabel("x")
    ax.set_ylabel("pdf(x)")

    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    ax.plot(autocorrelation(samples[::thinning]))
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF(x)")
    plt.show()
