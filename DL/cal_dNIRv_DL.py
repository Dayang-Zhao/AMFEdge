import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 
import xarray as xr

# dst_cols = ['Id', 'nirv_scale', 'nirv_magnitude', 'ndwi_scale', 'ndwi_magnitude', 'evi_scale', 'evi_magnitude']

# onset_path = r"F:\Research\AMFEdge\EdgeOnset\Amazon_UndistEdge_Effect_2023.csv"
# onset_df = pd.read_csv(onset_path)[dst_cols]
# end_path = r"F:\Research\AMFEdge\EdgeEnd\Amazon_UndistEdge_Effect_2023.csv"
# end_df = pd.read_csv(end_path)[dst_cols]

# # Calculate the difference in NIRv scale and magnitude between onset and end.
# diff_df = onset_df.merge(end_df, on="Id", suffixes=('_onset', '_end'))
# for col in dst_cols[1:]:
#     diff_df[col + '_diff'] = diff_df[col + '_end'] - diff_df[col + '_onset']

# # Save the result to a new CSV file.
# diff_path = r"F:\Research\AMFEdge\EdgeEnd\Amazon_UndistEdge_Effect_2023_dEndOnset.csv"
# diff_df.to_csv(diff_path, index=False)

path = r"F:\Research\AMFEdge\Meteo\Processed\Amazon_2023_droughtPeriod.nc"
ds = xr.open_dataset(path)
print(ds)