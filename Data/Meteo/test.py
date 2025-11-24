import osgeo
import xarray as xr

import rioxarray as rxr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# path = r"F:\Research\AMFEdge\Meteo\Meta\spei01.nc"
# ds = xr.open_dataset(path)

# path = r"F:\Research\AMFEdge\Meteo\Processed\Amazon_ERA5_1985_2024_monthlyTP.nc"
# ds = xr.open_dataset(path)

# print(ds)

import pandas as pd
from scipy.stats import kendalltau

df = pd.DataFrame({
    'temperature': [15, 16, 20, 25, 30],
    'evaporation': [100, 110, 130, 170, 200]
})

tau, p = kendalltau(df['temperature'], df['evaporation'])
print(f"Kendall tau = {tau:.2f}, p = {p:.3e}")