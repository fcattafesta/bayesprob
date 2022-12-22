import os
import numpy as np
import matplotlib.pyplot as plt

data_path = os.path.join(os.path.dirname(__file__), "data", "H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt")

y = np.loadtxt(data_path)

t = np.arange(len(y))

TIME = 4096 # seconds

t = t / TIME

fig, ax = plt.subplots()
ax.plot(t, y)
ax.set_ylabel("y")
ax.set_xlabel("t [s]")
ax.grid(True)
plt.show()
