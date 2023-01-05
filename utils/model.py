import numpy as np
import matplotlib.pyplot as plt

def TaylorF2(f, M, q):
    A = np.power(M, (5./6))
    nu = q / (1+q)**2
    m_tot = M * np.power(nu, (-3./5))
    v = np.power(np.pi * m_tot * f, (1./3))
    o_2 = 1 + (20./9) * ((743./336) + (11./4) * nu) * np.power(v, 2)
    o_3 = 16 * np.pi * np.power(v, 3)
    o_4 = 10 * ((3058673./1016064) + (5429./1008) * nu + (617./144) * np.power(nu, 2)) * np.power(v, 4)
    phi = (2 * np.pi * f) - 1 - (np.pi/4) + (3./(128 * nu * np.power(v, 5))) * (o_2 - o_3 + o_4)

    return A * np.power(f, (-7./6)) * np.exp(phi * 1j)

def matched_filter(S, f, M, q):
    return TaylorF2(f, M, q) / np.sqrt(S)

def log_likelihood(y_fft, f, M, q, S):
    return - 0.5 * (y_fft - np.real(TaylorF2(f, M, q)))**2 / S

if __name__ == "__main__":

    f = np.logspace(0, 3, 1000)
    args = [30, 0.7]
    model = TaylorF2(f, *args)

    fig, ax = plt.subplots()
    ax.plot(f, np.real(model))
    ax.set_xscale("log")
    plt.show()