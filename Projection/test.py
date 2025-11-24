import os
import glob

import osgeo
import xarray as xr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r"F:\Research\AMFEdge\Meteo\Processed\Amazon_ERA5_2023_drought_spei_gamma_03.nc"
ds = xr.open_dataset(path)
print(ds)