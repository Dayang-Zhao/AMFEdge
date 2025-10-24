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

import matplotlib.pyplot as plt
plt.ion()

import Data.convert_data as cd
import GlobVars as gv

ROOT_DIR = r"F:\Research\AMFEdge\CMIP6\metaData\Hist"
EXP_STR = 'historical'
BANDS_LUT = {'evspsbl':'total_evaporation_sum', 'rsds':'surface_solar_radiation_downwards_sum',
            'pr':'total_precipitation_sum','ps':'surface_pressure',
            'tas':'temperature_2m', 'huss':'specific_humidity_2m',
            }
STD_TIME = pd.date_range(start='2004-01-01', end='2014-12-31', freq='MS')

def mean_to_sum(ds, var):
    """
    Convert the flux to the sum data.
    """
    ds[var] = ds[var]*30*24*3600

    return ds

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

def preprocess_era5(ds:xr.Dataset):

    ds['total_evaporation_sum'] = ds['total_evaporation_sum'] * -1

    ds['WD'] = (ds['total_precipitation_sum'] - ds['total_evaporation_sum'])

    # Interpolate to standard grid if needed.
    outds = ds.coarsen(lon=10, lat=10, boundary='pad').mean()
    outds = outds.interp(lon=gv.STD_LON, lat=gv.STD_LAT, method='slinear')
    outds = outds.interp(time=STD_TIME, method='nearest')

    # Mask value=0 data.
    outds = outds.where(outds != 0)

    return outds

def convert_unit(ds):

    # Convert PAR data from W/m² to J/m².
    dst_var = 'surface_solar_radiation_downwards_sum'
    ds = mean_to_sum(ds, dst_var)

    # Convert precipitation data from kg/m²/s to m/m².
    dst_var = 'total_precipitation_sum'
    ds = mean_to_sum(ds, dst_var)
    ds[dst_var] = ds[dst_var] / 1000

    # Convert ET data from kg/m²/s to m/m².
    dst_var = 'total_evaporation_sum'
    ds = mean_to_sum(ds, dst_var)
    ds[dst_var] = ds[dst_var] / 1000

    return ds

def preprocess_model(paths):
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
    outds = ds.interp(lon=gv.STD_LON, lat=gv.STD_LAT, method='slinear')
    outds = outds.interp(time=STD_TIME, method='nearest')

    # Drop unnecessary dimensions.
    if 'height' in outds.coords:
        outds = outds.drop_vars('height')

    # Sort variables.
    band_names = list(BANDS_LUT.values()) + ['vpd', 'WD']
    outds = outds[band_names]
    outds.attrs['band_names'] = tuple(band_names)

    return outds

def linear_reg(xds, yds, var):
    xda = xds[var]
    yda = yds[var]

    # Commom mask.
    common_mask = xr.where(np.isfinite(xda) & np.isfinite(yda), 1, np.nan)
    xda = xda * common_mask
    yda = yda * common_mask

    x = xda.values.ravel()
    y = yda.values.ravel()

    # 去除 NaN
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Linear regression.
    slope, intercept, r, p, stderr = linregress(x, y)

    # Calculate rmse.
    rmse = np.sqrt(np.mean((y - x) ** 2))

    # Calculate centered rmse.
    x2 = x-x.mean()
    y2 = y-y.mean()
    crmse = np.sqrt(np.mean((y2 - x2) ** 2))

    results = {"slope": slope, "intercept": intercept,
               "r_value": r, "p_value": p, "stderr": stderr, 
               'r2':r**2, 'rmse': rmse, 'crmse': crmse}

    return results

def get_ds_res(ds):
    """Get the resolution of the dataset."""
    lon_res = ds['lon'].diff(dim='lon').mean().item()
    lat_res = ds['lat'].diff(dim='lat').mean().item()
    return lon_res, lat_res

def cal_rmse_cmip6_by_era5(cmip6_ds, era5_ds):

    # Dry-season months.
    dst_months = [6, 7, 8, 9, 10]

    # Preprocess CMIP6.
    cmip6_ds = cmip6_ds.sel(
        time=(cmip6_ds['time'].dt.year.isin(range(2004, 2015))&(cmip6_ds['time'].dt.month.isin(dst_months)))
    ).transpose("time", "lat", "lon")

    # Preprocess ERA5.
    era5_ds = era5_ds.sel(
        time=(era5_ds['time'].dt.year.isin(range(2004, 2015))&(era5_ds['time'].dt.month.isin(dst_months)))
    ).transpose("time", "lat", "lon")

    # Linear regression for each variable.
    results = {}
    dst_vars = list(era5_ds.data_vars)
    for var in dst_vars:
        results[var] = linear_reg(cmip6_ds, era5_ds, var)

    results_df = pd.DataFrame(results).T
    results_df['var'] = results_df.index

    return results_df

def fetch_fpaths4model(model, rootdir):
    fnames = glob.glob(f"{model}_{EXP_STR}*.nc", root_dir=rootdir)
    fpaths = [os.path.join(rootdir, fname) for fname in fnames]

    return fpaths

def main(models, era5_ds):
    era5_ds = preprocess_era5(era5_ds)

    results = []
    good_models = []
    for model in models:
        try:
            cmip6_fpaths = fetch_fpaths4model(model, ROOT_DIR)
            cmip6_ds = preprocess_model(cmip6_fpaths)

            # Remove models with spatial resolution > 2 degrees.
            lon_res, lat_res = get_ds_res(cmip6_ds)
            if lon_res > 2 or lat_res > 2:
                print(f"Model {model} has resolution > 2 degrees, skipping.")
                continue
            else:
                result_df = cal_rmse_cmip6_by_era5(cmip6_ds, era5_ds)
                result_df['model'] = model
                results.append(result_df)
                good_models.append(model)
        except Exception as e:
            print(f"Error processing model {model}: {e}")
            continue
    results_df = pd.concat(results, axis=0, ignore_index=True)

    return results_df

if __name__ == "__main__":
    models = ['access_cm2', 'awi_cm_1_1_mr', 'bcc_csm2_mr', 'cesm2', 
              'cmcc_cm2_sr5', 'cmcc_esm2', 'cnrm_cm6_1_hr', 'ec_earth3_veg_lr', 
              'fgoals_f3_l', 'fio_esm_2_0', 'gfdl_esm4', 'iitm_esm', 'inm_cm5_0', 
              'ipsl_cm6a_lr', 'kace_1_0_g', 'miroc6', 'mpi_esm1_2_lr', 'mri_esm2_0', 
              'noresm2_mm', 'taiesm1', 'ukesm1_0_ll']
    
    era5_path = r"F:\Research\AMFEdge\Meteo\Meta\Amazon_2004_2024_monthlyMeteo.tif"
    era5_ds = cd.datetTif2ds(era5_path)
    results_df = main(models, era5_ds) 

    outpath = r"F:\Research\AMFEdge\CMIP6\rmse_cmip6_era5_2004_2014.csv"
    results_df.to_csv(outpath, index=False)