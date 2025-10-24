import os
import sys
sys.path.append(r"D:\ProgramData\VistualStudioCode\AMFEdge")
import glob 
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import r2_score

import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
from xclim.sdba.adjustment import QuantileDeltaMapping

import matplotlib.pyplot as plt
plt.ion()

import Data.convert_data as cd
import preprocess as pp

ERA5_DS = cd.datetTif2ds(r"F:\Research\AMFEdge\Meteo\Meta\Amazon_1985_2015_monthlyMeteo.tif")
ROOT_DIR = r"F:\Research\AMFEdge\CMIP6\metaData\Hist"
EXP_STR = 'historical'
BANDS_LUT = {'evspsbl':'total_evaporation_sum', 'rsds':'surface_solar_radiation_downwards_sum',
            'pr':'c','ps':'surface_pressure',
            'tas':'c', 'huss':'specific_humidity_2m',
            }
STD_TIME = pd.date_range(start='1985-01-01', end='2014-12-31', freq='MS')

def fetch_fpaths4model(model, rootdir):
    fnames = glob.glob(f"{model}_{EXP_STR}*.nc", root_dir=rootdir)
    fpaths = [os.path.join(rootdir, fname) for fname in fnames]

    return fpaths

def get_qdm_model(model):
    era5_ds = pp.preprocess_era5(ERA5_DS.copy(), std_time=STD_TIME)

    cmip6_fpaths = fetch_fpaths4model(model, ROOT_DIR)
    cmip6_ds = pp.preprocess_model(cmip6_fpaths, std_time=STD_TIME)

    # Dry-season months.
    dst_months = range(1,13)

    # Preprocess CMIP6.
    cmip6_hist_ds = cmip6_ds.sel(
        time=(cmip6_ds['time'].dt.year.isin(range(1985, 2015))&(cmip6_ds['time'].dt.month.isin(dst_months)))
    ).transpose("time", "lat", "lon")

    # Preprocess ERA5.
    era5_hist_ds = era5_ds.sel(
        time=(era5_ds['time'].dt.year.isin(range(1985, 2015))&(era5_ds['time'].dt.month.isin(dst_months)))
    ).transpose("time", "lat", "lon")

    # QDM for each variable.
    add_vars = ['temperature_2m', 'vpd', 'WD']
    mul_vars = ['surface_solar_radiation_downwards_sum', 'total_precipitation_sum', 
                'total_evaporation_sum', ]
    Adjs = []
    for var in add_vars:
        Adj = QuantileDeltaMapping.train(era5_hist_ds[var], cmip6_hist_ds[var], group='time.month', 
                                    nquantiles=50, kind='+')
        Adjs.append(Adj)

    for var in mul_vars:
        Adj = QuantileDeltaMapping.train(era5_hist_ds[var], cmip6_hist_ds[var], group='time.month', 
                                    nquantiles=50, kind='*')
        Adjs.append(Adj)

    Adjs_lut = dict(zip(add_vars+mul_vars, Adjs))

    return Adjs_lut


def cab_qdm_cmip6_by_era5(dqm:dict, cmip6_fut_ds:xr.Dataset):

    # Calibrate.
    outds = []
    for var, Adj in dqm.items():
        ds = Adj.adjust(cmip6_fut_ds[var], interp="linear")
        outds.append(ds)

    outds = xr.Dataset(dict(zip(dqm.keys(), outds)))

    return outds

if __name__ == "__main__":
    models = ['mri_esm2_0', 'cnrm_cm6_1_hr', 'cesm2', 'ukesm1_0_ll',
            'noresm2_mm', 'miroc6', 'taiesm1',
            'kace_1_0_g', 'access_cm2', 'cmcc_cm2_sr5']
    
    # era5_path = r"F:\Research\AMFEdge\Meteo\Meta\Amazon_2004_2024_monthlyMeteo.tif"
    # era5_ds = cd.datetTif2ds(era5_path)
    # results_df = get_qdm_cmip6_by_era5(models, era5_ds) 