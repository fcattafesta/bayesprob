import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate
from scipy.fft import fft, rfft, rfftfreq

np.random.seed(1)

def h(x, a):
    return 0.5*np.sin(2 * np.pi * a * x)

def h_fft(x, a):
    return fft(h(x, a))

x = np.linspace(0, np.pi, 1000)
dx = x[1] - x[0]
n = 100 * np.random.normal(size=x.size) 
d = h(x, 100) + n

plt.figure(1)
plt.plot(x, d)

acf = correlate(d, d, mode="full", method="auto")[(x.size-1):] / (np.var(d) * d.size)

lag = np.arange(0, d.size)

plt.figure(2)
plt.plot(lag, acf)

psd = rfft(acf)
f = rfftfreq(acf.size, d=dx)

d_fft = rfft(d)

plt.figure(3)
plt.plot(f, np.absolute(psd))

def log_likelihood(d, m, a, s):
    return - 0.5 * np.absolute(d - h_fft(m, a))**2 / s

a = np.linspace(0, 1000, 100)

print(psd.size, d_fft.size)

logL = np.zeros(a.size)

for i in range(d_fft.size):
    logL += log_likelihood(d_fft[i], f[i], a, np.absolute(psd[i])) 

plt.figure(4)
plt.plot(a, logL)

a_mle = a[np.argmin(logL)]
print(a_mle)

filter = h(f, 100) / np.sqrt(np.absolute(psd))



signal = filter * d_fft

plt.figure(5)
plt.plot(f, np.absolute(signal/d_fft))

plt.show()