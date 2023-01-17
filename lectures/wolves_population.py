import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

b = 500

M = 20
n = 30
r = 7

N = np.arange(30, b+1)

P_prior = 1./(b - n)

hg = hypergeom(N, M, n)
P = hg.pmf(r) * P_prior

norm = P.sum()
P = P / norm

N_max = N[np.argmax(P)]


plt.plot(N, P, color='black', marker='.',
         ls='', label='Posterior')
plt.vlines(N_max, 0., np.max(P), color='red',
           ls='--', label=''.join([r'$N_{max}$=', f'{N_max}']))
plt.hlines(P_prior, np.min(N), np.max(N), color='blue', ls='--', label='Prior')
plt.ylabel('P(N|Mnr)')
plt.xlabel('N')
plt.grid()
plt.legend()
plt.show()
