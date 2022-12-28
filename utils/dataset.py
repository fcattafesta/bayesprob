import os
import pandas as pd
import h5py

data_path = os.path.join(os.path.dirname(__file__), "..", "data")
fig_path = os.path.join(os.path.dirname(__file__), "..", "figures")

def get_df(columns=["y", "t", "acf", "y_norm"]):
    f = h5py.File(os.path.join(data_path, "post_data.hdf5"), "r")
    df = pd.DataFrame(f.get("H-H1_GWOSC_4KHZ_R1-1126257415-4096"), columns=columns)
    f.close()
    return df

def write_data(df):
    f = h5py.File(os.path.join(data_path, "post_data.hdf5"), "w")
    f.create_dataset("H-H1_GWOSC_4KHZ_R1-1126257415-4096", data=df)
    f.close()

