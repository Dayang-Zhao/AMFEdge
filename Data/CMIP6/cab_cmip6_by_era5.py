import sys
sys.path.append(r"D:\ProgramData\VistualStudioCode\AMFEdge")

import numpy as np
import pandas as pd
from scipy.stats import linregress

import xarray as xr

import matplotlib.pyplot as plt
plt.ion()

import Data.convert_data as cd

def mean_to_sum(ds, var):
    """
    Convert the flux to the sum data.
    """
    # time_diff = (ds['time'].diff(dim="time") / np.timedelta64(1, "s")).values
    # time_diff = xr.DataArray(time_diff, dims=["time"], coords={"time": ds['time'][1:]})

    # sum_da = ds[var].isel(time=slice(1, None)) * time_diff
    # sum_da = sum_da.reindex(time=ds['time'])
    ds[var] = ds[var]*30*24*3600

    return ds

def preprocess_era5(ds):

    # Mask value=0 data.
    ds = ds.where(ds != 0)

    return ds

def preprocess_cmip6(ds):

    # Convert PAR data from W/m² to J/m².
    dst_var = 'surface_solar_radiation_downwards_sum'
    ds = mean_to_sum(ds, dst_var)
    ds[dst_var] = ds[dst_var]

    # Convert precipitation data from kg/m²/s to m/m².
    dst_var = 'total_precipitation_sum'
    ds = mean_to_sum(ds, dst_var)
    ds[dst_var] = ds[dst_var] / 1000

    # Convert ET data from kg/m²/s to m/m².
    dst_var = 'total_evaporation_sum'
    ds = mean_to_sum(ds, dst_var)
    ds[dst_var] = -1 * ds[dst_var] / 1000

    return ds

def linear_reg(xds, yds, var):
    xda = xds[var]
    yda = yds[var]
    x = xda.values.ravel()
    y = yda.values.ravel()

    # 去除 NaN
    mask = np.isfinite(x) & np.isfinite(y)
    slope, intercept, r, p, stderr = linregress(x[mask], y[mask])

    results = {
        "slope": slope,
        "intercept": intercept,
        "r_value": r,
        "p_value": p,
        "stderr": stderr
    }
    # Calculate the fitted values.
    fitted_values = slope * x[mask] + intercept
    slope, intercept, r, p, stderr = linregress(fitted_values, y[mask])

    return results

def linear_cab(ds, cab_df, vars):
    for var in vars:
        slope = cab_df.loc[cab_df['var'] == var, 'slope'].values[0]
        intercept = cab_df.loc[cab_df['var'] == var, 'intercept'].values[0]

        cab_da = ds[var] * slope + intercept
        ds[var] = cab_da

    return ds

def cab_cmip6_by_era5(cmip6_path, era5_path, dst_vars):

    # Convert tiff data with dates into xarray dataset.
    dst_months = [6, 7, 8, 9, 10]
    cmip6_ds = cd.datetTif2ds(cmip6_path)
    cmip6_ds = cmip6_ds.sel(
        time=(cmip6_ds['time'].dt.year.isin(range(2004, 2015))&(cmip6_ds['time'].dt.month.isin(dst_months)))
    )
    cmip6_ds['WD'] = (cmip6_ds['total_precipitation_sum'] - cmip6_ds['total_evaporation_sum']) * 30 * 24 * 3600
    # cmip6_ds = preprocess_cmip6(cmip6_ds)
    era5_ds = cd.datetTif2ds(era5_path)
    era5_ds = era5_ds.sel(
        time=(era5_ds['time'].dt.year.isin(range(2004, 2015))&(era5_ds['time'].dt.month.isin(dst_months)))
    )
    era5_ds = preprocess_era5(era5_ds)
    era5_ds['WD'] = (era5_ds['total_precipitation_sum'] + era5_ds['total_evaporation_sum']) * 1000

    # Linear regression for each variable.
    results = {}
    for var in dst_vars:
        results[var] = linear_reg(cmip6_ds, era5_ds, var)

    results_df = pd.DataFrame(results).T
    results_df['var'] = results_df.index

    # # Calibrate WD using linearly corrected values.
    # cab_cmip6_ds = linear_cab(cmip6_ds, results_df, ['total_precipitation_sum', 'total_evaporation_sum'])
    # cab_cmip6_ds['WD'] = (cab_cmip6_ds['total_precipitation_sum']+ cab_cmip6_ds['total_evaporation_sum'])*1000
    # era5_ds['WD'] = (era5_ds['total_precipitation_sum']+ era5_ds['total_evaporation_sum'])*1000
    # wd_results = linear_reg(cab_cmip6_ds, era5_ds, 'WD')

    return results_df

if __name__ == "__main__":
    cmip6_path = r"F:\Research\AMFEdge\Meteo\Meta\Amazon_2004_2014_monthlyCmip6_v2.tif"
    era5_path = r"F:\Research\AMFEdge\Meteo\Meta\Amazon_2004_2024_monthlyMeteo.tif"
    dst_vars = ['total_evaporation_sum', 'total_precipitation_sum', 
                'surface_solar_radiation_downwards_sum', 'temperature_2m', 
                'vpd', 'WD']

    results_df = cab_cmip6_by_era5(cmip6_path, era5_path, dst_vars)
    print(results_df)

    outpath = r"F:\Research\AMFEdge\CMIP6\xCmip6_yEra5_cab_2004_2014.csv"
    results_df.to_csv(outpath, index=False)