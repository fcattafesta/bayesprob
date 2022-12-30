import os
import pandas as pd
import h5py

data_path = os.path.join(os.path.dirname(__file__), "..", "data")
fig_path = os.path.join(os.path.dirname(__file__), "..", "figures")

def get_df(columns=["y", "t", "acf"], filename="post_data.hdf5"):
    f = h5py.File(os.path.join(data_path, filename), "r")
    df = pd.DataFrame(f.get("H-H1_GWOSC_4KHZ_R1-1126257415-4096"), columns=columns)
    f.close()
    return df

def write_data(df, filename="post_data.hdf5"):
    f = h5py.File(os.path.join(data_path, filename), "w")
    f.create_dataset("H-H1_GWOSC_4KHZ_R1-1126257415-4096", data=df)
    f.close()

