import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate
from scipy.fft import rfft, rfftfreq


np.random.seed(1)

t = np.arange(0, 10000)
n = np.random.normal(size=t.size) + 0.1 * np.sin(2 * np.pi * 0.3 * t)

plt.figure(1)
plt.plot(t, n)

acf = correlate(n, n, mode="full", method="auto")[(n.size-1):] / (np.var(n) * n.size)

plt.figure(2)
plt.plot(acf[:500])

psd = np.absolute(rfft(acf))
f = rfftfreq(acf.size, d=t[1]- t[0])

plt.figure(3)
plt.plot(f, psd)

n_fft = rfft(n)

h_fft = rfft(np.sin(2 * np.pi * 0.3 * t))


w = np.absolute((h_fft / np.sqrt(psd)) * n_fft)

plt.figure(4)
plt.plot(f, w)

plt.show()