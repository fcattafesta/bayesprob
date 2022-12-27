import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import UnivariateSpline

from path import data_path, fig_path
from acf_analysis import t, acf_norm

# Since ACF is oscillating, we clearly see that this stochastic process is not stationary.
# Although, we can measure (by eye!) the period of the oscillation to test the WSS
# hypothesis

dt = 0.15 # seconds

max_i = int(t[-1] / dt)
i = np.arange(max_i)
tt_i = i * dt

del i

# In order to evaluate ACF in arbitrary points, we build the ACF spline and evaluate it 
# in the dt multiple points

acf_spline = UnivariateSpline(t, acf_norm, s=0)
del t
acf_wss = acf_spline(tt_i)
del acf_spline, acf_norm

# In fact, if the process is WSS, we should see that ACF values do not change with
# time distance

fig, ax = plt.subplots()
ax.set_ylabel("$ACF(i \cdot \Delta t)$")
ax.set_xlabel("t [s]")
ax.grid(True, ls="--", alpha=0.5)
ax.minorticks_on()
ax.tick_params(direction="in", which="both")
ax.plot(tt_i[:1000], acf_wss[:1000], color="black", linewidth=0.5)
fig.savefig(os.path.join(fig_path, "acf_wss_zoom.pdf"), format="pdf")
