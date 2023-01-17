import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

M = 3000
N = 10000
n = 138
P = hypergeom(N, M, n)

r = np.arange(0, n+1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(r, P.pmf(r), 'bo')
plt.show()
