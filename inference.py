import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.dataset import data_path, fig_path, get_df
from utils.model import TaylorF2, matched_filter
from utils.prior import prior

df = get_df(columns=["f", "psd", "y_ftt"], filename="fft_data.hdf5")

M = np.arange(25, 35, step=0.1)
q = np.arange(0.5, 1., step=0.05)