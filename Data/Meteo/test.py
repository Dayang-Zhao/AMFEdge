import osgeo
import xarray as xr

import rioxarray as rxr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# path = r"F:\Research\AMFEdge\Meteo\Meta\spei01.nc"
# ds = xr.open_dataset(path)

path = r"F:\Research\AMFEdge\Edge\Main\anoVI_Amazon_GLEAM_Edge_Effect_2023.csv"
df1 = pd.read_csv(path)
path = r"F:\Research\AMFEdge\Edge\Main\anoVI_Amazon_Edge_Effect_2023.csv"
df2 = pd.read_csv(path)

df = df1.merge(df2, on='Id', suffixes=('_GLEAM', '_ERA5'))

print(df)