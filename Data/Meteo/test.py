import osgeo
import xarray as xr

import rioxarray as rxr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r"F:\Research\AMFEdge\CMIP6\metaData\SSP1_26\bcc_csm2_mr_ssp1_2_6_evaporation_including_sublimation_and_transpiration_2015_2100.nc"
ds = xr.open_dataset(path)

# path = r"F:\Research\AMFEdge\AGB\GlobBiomass\Amazon\N00W052_agb.tif"
# ds = rxr.open_rasterio(path)
# print(ds)

# path = r"F:\Research\AMFEdge\CMIP6\Predict\Mnirv_pred_SSP5_85.csv"
# df = pd.read_csv(path)

# MCWD_threshold, HAND_threshold = -300, 15
# dst_df = df[(df['MCWD_mean']<MCWD_threshold) & (df['HAND_mean']>HAND_threshold)]

print(ds)