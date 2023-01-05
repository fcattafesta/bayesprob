import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.dataset import data_path, fig_path, write_data


y = np.loadtxt(os.path.join(data_path, "H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt"))
cut = int(8e6)
y = y[cut:(len(y) - cut)]

# Using pandas DataFrame to create dataset

df = pd.DataFrame(y, columns=["y"])

# Generating x-axis array and normalizing to acquisition TIME

t = np.arange(len(y))
TIME = 4096  # seconds
t = t / TIME
df["t"] = t

# Plotting sample for visualization

fig, ax = plt.subplots()
ax.set_ylabel("y(t)")
ax.set_xlabel("t [s]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(t, y, color="black", linewidth=0.08)
fig.savefig(os.path.join(fig_path, "signal.pdf"), format="pdf")

# Saving samples and ACF for further analysis

write_data(df)
