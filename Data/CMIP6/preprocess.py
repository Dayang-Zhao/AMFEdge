"""Preprocess ECOSTRESS Tiled data:
1. Process files according to their tiles.
2. Concatenate data of the same tile along the time dimension.
3. Calculate the monthly mean of the data.
4. Save the processed data as GeoTIFF files.
"""
import sys
sys.path.append(r"D:\ProgramData\VistualStudioCode\AMFEdge")
import os

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import glob 
from tqdm import tqdm

import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
import osgeo
import rioxarray as rxr
from xclim.sdba import QuantileDeltaMapping

import matplotlib.pyplot as plt

import GlobVars as gv

BANDS_LUT = {'evspsbl':'total_evaporation_sum', 'rsds':'surface_solar_radiation_downwards_sum',
            'pr':'total_precipitation_sum','ps':'surface_pressure',
            'tas':'temperature_2m', 'huss':'specific_humidity_2m',
            }

MIN_LON, MAX_LON, MIN_LAT, MAX_LAT = -80, -44, -21, 9
RES, HALF_RES = 1, 0.5
NUM_LON, NUM_LAT = int((MAX_LON - MIN_LON) / RES), int((MAX_LAT - MIN_LAT) / RES)
STD_LON = np.linspace(MIN_LON+HALF_RES, MAX_LON-HALF_RES, num=NUM_LON)        
STD_LAT = np.linspace(MAX_LAT-HALF_RES, MIN_LAT+HALF_RES, num=NUM_LAT)
# STD_TIME = pd.date_range(start='2004-01-01', end='2014-12-31', freq='MS')

ROOT_DIR = r"F:\Research\AMFEdge\CMIP6\metaData\SSP5_85"
EXP_STR = 'ssp'
BANDS_LUT = {'evspsbl':'total_evaporation_sum', 'rsds':'surface_solar_radiation_downwards_sum',
            'pr':'c','ps':'surface_pressure',
            'tas':'c', 'huss':'specific_humidity_2m',
            }
STD_TIME = pd.date_range(start='1985-01-01', end='2014-12-31', freq='MS')

def fetch_fpaths4model(model, rootdir):
    fnames = glob.glob(f"{model}_{EXP_STR}*.nc", root_dir=rootdir)
    fpaths = [os.path.join(rootdir, fname) for fname in fnames]

    return fpaths

def cal_vpd(t_da, q_da, p_da):
    """Calculate VPD from temperature, specific humidity, and pressure."""
    # Convert temperature from Kelvin to Celsius.
    t_da = t_da - 273.15

    # Calculate saturation vapor pressure (es) in kPa.
    es = 0.61094 * np.exp((17.625 * t_da) / (t_da + 243.04))

    # Calculate actual vapor pressure (ea) in kPa.
    ea = q_da * p_da * 0.001 / (0.622 + 0.378 * q_da)

    # Calculate VPD in kPa.
    vpd = es - ea

    return vpd

def sum_to_mean(ds, var):
    """
    Convert the flux to the sum data.
    """
    # time_diff = (ds['time'].diff(dim="time") / np.timedelta64(1, "s")).values
    # time_diff = xr.DataArray(time_diff, dims=["time"], coords={"time": ds['time'][1:]})

    # sum_da = ds[var].isel(time=slice(1, None)) * time_diff
    # sum_da = sum_da.reindex(time=ds['time'])
    # ds[var] = sum_da

    # kg/m2/s = mm/s -> mm
    ds[var] = ds[var]*30*24*3600

    return ds

def convert_unit(ds):

    # Convert PAR data from W/m² to J/m².
    dst_var = 'surface_solar_radiation_downwards_sum'
    ds = sum_to_mean(ds, dst_var)

    # Convert precipitation data from kg/m²/s to mm.
    dst_var = 'total_precipitation_sum'
    ds = sum_to_mean(ds, dst_var)

    # Convert ET data from kg/m²/s to mm.
    dst_var = 'total_evaporation_sum'
    ds = sum_to_mean(ds, dst_var)

    return ds

def preprocess_era5(ds:xr.Dataset, std_time=STD_TIME):

    ds['total_evaporation_sum'] = ds['total_evaporation_sum'] * -1000
    ds['total_precipitation_sum'] = ds['total_precipitation_sum'] * 1000

    ds['WD'] = (ds['total_precipitation_sum'] - ds['total_evaporation_sum'])

    # Interpolate to standard grid if needed.
    outds = ds.coarsen(lon=10, lat=10, boundary='pad').mean()
    outds = outds.interp(lon=gv.STD_LON, lat=gv.STD_LAT, method='slinear')
    outds = outds.interp(time=std_time, method='nearest')

    # Add unit attributes.
    unit_dict = {
        'total_evaporation_sum': 'mm',
        'surface_solar_radiation_downwards_sum': 'J/m²',
        'total_precipitation_sum': 'mm',
        'temperature_2m': 'K',
        'specific_humidity_2m': 'kg/kg',
        'surface_pressure': 'Pa',
        'vpd': 'kPa',
        'WD': 'mm'
    }
    for var in outds.data_vars:
        if var in unit_dict:
            outds[var].attrs['units'] = unit_dict[var]

    # Mask value=0 data.
    outds = outds.where(outds != 0)

    return outds

def preprocess_model(paths, std_time=STD_TIME):
    """Preprocess each model: merge and calculate VPD."""

    # Open all datasets and merge them.
    da_list = []
    for path in paths:
        ds = xr.open_dataset(path)
        
        dst_var = list(set(ds.data_vars) & set(BANDS_LUT.keys()))
        da_list.append(ds[dst_var])

    ds = xr.merge(da_list)

    # Rename dimensions.
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})

    # Convert longitude from [0, 360] to [-180, 180] if needed.
    ds['lon'] = ((ds['lon'] + 180) % 360) - 180

    # Adjust the time dim as datetime64.
    if isinstance(ds.indexes['time'], CFTimeIndex):
        ds['time'] = pd.to_datetime(ds.indexes['time'].to_datetimeindex())

    # Linearly regression.
    ds['WD'] = (ds['pr'] - ds['evspsbl'])*30*24*3600
    dst_ds = ds.isel(time=ds['time.month'].isin([6, 7, 8, 9, 10]))
    fit = dst_ds.polyfit(dim='time', deg=1)


    # Rename variables according to ERA5.
    ds = ds.rename({v: BANDS_LUT[v] for v in BANDS_LUT.keys() if v in ds.data_vars})

    # Calculate VPD.
    vpd = cal_vpd(ds['temperature_2m'], ds['specific_humidity_2m'], ds['surface_pressure'])
    ds['vpd'] = vpd

    # Convert units.
    ds = convert_unit(ds)

    # Calculate water deficit (WD).
    ds['WD'] = (ds['total_precipitation_sum'] - ds['total_evaporation_sum'])

    # Interpolate to standard grid if needed.
    outds = ds.interp(lon=STD_LON, lat=STD_LAT, method='slinear')
    outds = outds.interp(time=std_time, method='nearest')

    # Drop unnecessary dimensions.
    if 'height' in outds.coords:
        outds = outds.drop_vars('height')

    # Add unit attributes.
    unit_dict = {
        'total_evaporation_sum': 'mm',
        'surface_solar_radiation_downwards_sum': 'J/m²',
        'total_precipitation_sum': 'mm',
        'temperature_2m': 'K',
        'specific_humidity_2m': 'kg/kg',
        'surface_pressure': 'Pa',
        'vpd': 'kPa',
        'WD': 'mm'
    }
    for var in outds.data_vars:
        if var in unit_dict:
            outds[var].attrs['units'] = unit_dict[var]

    # Sort variables.
    band_names = list(BANDS_LUT.values()) + ['vpd', 'WD']
    outds = outds[band_names]
    outds.attrs['band_names'] = tuple(band_names)

    return outds

if __name__ == "__main__":
    """Main function to preprocess all models."""
    # List of CMIP6 models to process.
    models = ['mri_esm2_0', 'cnrm_cm6_1_hr', 'cesm2', 'ukesm1_0_ll',
             'noresm2_mm', 'miroc6', 'taiesm1',
             'kace_1_0_g', 'access_cm2', 'cmcc_cm2_sr5']

    # Preprocess each model.
    models = ['noresm2_mm']
    for model in models:
        paths = fetch_fpaths4model(model, ROOT_DIR)
        ds = preprocess_model(paths, std_time=STD_TIME)

