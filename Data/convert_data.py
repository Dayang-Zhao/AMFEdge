# import warnings
# warnings.filterwarnings("ignore")
# import numpy as np
# import pandas as pd

# import xarray as xr

# # import osgeo
# import rioxarray as rioxr

import os
from typing import Literal

import numpy as np
import pandas as pd
import osgeo #必须加否则会报错
import netCDF4
import xarray as xr
import rioxarray as rioxr
from rasterio.enums import Resampling

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

def datetTif2ds(path)->xr.Dataset:
    """Convert multiple tifs with date 
    (e.g., t2m_20220801) in band names into xr.dataset.
    """
    da = rioxr.open_rasterio(path, masked=True)
    da = da.rename({'x':'lon', 'y': 'lat'})
    band_names = da.attrs['long_name']

    # Split band names.
    band_names = np.char.array(band_names)
    band_names_split = np.char.split(band_names, '_')
    nvars = np.array(["_".join(m[0:-1]) for m in band_names_split])
    time = pd.to_datetime(np.array([m[-1] for m in band_names_split]))
    multiindex = pd.MultiIndex.from_arrays([nvars, time], names=('nvars','time'))
    da['band'] = multiindex
    da = da.unstack('band')
    ds = da.to_dataset('nvars')

    return ds
    

# ------------------------------------------------
def tif2ds(fpath):
    """Convert multiple tifs without date (e.g., t2m) 
    in band names into xr.dataset."""
    da = rioxr.open_rasterio(fpath, masked=True)
    da = da.rename({'x':'lon', 'y': 'lat'})

    if 'long_name' in da.attrs:
        band_names = da.attrs['long_name']

        da['band'] = np.char.array(band_names)
        
    ds = da.to_dataset('band')

    return ds
    

# ------------------------------------------------
def ds2df(ds:xr.Dataset, reset_index=True)->pd.DataFrame:
    """Convert dataset into dataframe.

    Args:
        ds (xr.Dataset): Input dataset.
        reset_index (bool, optional): Whether to convert index 
        into columns. Defaults to True.

    Returns:
        pd.DataFrame: Output dataframe.
    """
    df = ds.to_dataframe()
    if 'spatial_ref' in df.columns:
        df = df.drop('spatial_ref', axis=1)
    df = df.dropna(how='all', axis=0, subset=list(ds.data_vars))
    # Remove the rows with all target columns 0.
    df = df[df[list(ds.data_vars)].all(axis=1)!=0]
    if reset_index:
        df = df.reset_index()

    return df