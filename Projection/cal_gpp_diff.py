import os
import glob

import numpy as np
import pandas as pd
import scipy.stats as stats

import osgeo
import xarray as xr

import matplotlib.pyplot as plt

MIN_LON, MAX_LON, MIN_LAT, MAX_LAT = -80, -44, -21, 9
RES, HALF_RES = 1, 0.5
NUM_LON, NUM_LAT = int((MAX_LON - MIN_LON) / RES), int((MAX_LAT - MIN_LAT) / RES)
STD_LON = np.linspace(MIN_LON+HALF_RES, MAX_LON-HALF_RES, num=NUM_LON)        
STD_LAT = np.linspace(MAX_LAT-HALF_RES, MIN_LAT+HALF_RES, num=NUM_LAT)

ROOT_DIR = r"F:\Research\AMFEdge\CMIP6\metaData"
VAR = 'gpp'

models = ['mri_esm2_0', 'cnrm_cm6_1_hr', 'cesm2', 'ukesm1_0_ll',
            'noresm2_mm', 'miroc6', 'taiesm1',
            'kace_1_0_g', 'access_cm2', 'cmcc_cm2_sr5']
start_period = range(2015, 2025)
end_period = range(2090, 2100)

def get_fpaths(scenario):
    dst_dir = f"{ROOT_DIR}/{scenario}/"
    # fnames = glob.glob(f"*_{scenario}_{VAR}_2015_2100.nc", root_dir=dst_dir)
    # paths = [os.path.join(dst_dir, fname) for fname in fnames]

    paths = []
    for model in models:
        fname = f"{model}_{scenario}_{VAR}_2015_2100.nc"
        path = os.path.join(dst_dir, fname)
        if os.path.exists(path):
            paths.append((model, path))

    return paths

def preprocess(paths):

    das = []
    for model, path in paths:
        da = xr.open_dataset(path)[VAR]

        # Remove singleton coordinates
        da = da.squeeze().reset_coords(drop=True)

        # Select dry season months
        da = da.sel(time=da['time.month'].isin([6,7,8,9]))
        yda = da.groupby('time.year').mean(dim='time', skipna=True)

        # Regrid to standard lat/lon
        da2 = yda.interp(lat=STD_LAT, lon=STD_LON, method='linear')
        da2 = da2.assign_coords({'member': model})
        das.append(da2)

    ave_da = xr.concat(das, dim='member')

    return ave_da

# ---------------- Linear regression to calculate trend and p-value --------------
# def new_linregress(x:np.ndarray, y:np.ndarray):
    
#     flatten_x = x.flatten()
#     flatten_y = y.flatten()

#     # Remove nan.
#     flatten_x2 = flatten_x[~np.isnan(flatten_y)]
#     flatten_y2 = flatten_y[~np.isnan(flatten_y)]

#     if len(np.unique(flatten_x2)) < 4:
#         slope, intercept, r_value, p_value, std_err = (np.nan, np.nan, np.nan, np.nan, np.nan)
#     else:
#         # Wrapper around scipy linregress to use in apply_ufunc
#         slope, intercept, r_value, p_value, std_err = stats.linregress(flatten_x2, flatten_y2)

#     return np.array([slope, intercept, r_value, p_value, std_err])

# def linear_reg_da(da:xr.DataArray, x:str):
#     stat = xr.apply_ufunc(new_linregress, da[x], da, 
#                         input_core_dims=[['year'], ['year']],
#                         output_core_dims=[["parameter"]],
#                         vectorize=True,
#                         dask="parallelized",
#                         output_dtypes=['float64'],
#                         output_sizes={"parameter": 5},
#                         )
    
#     # Label outputs.
#     stat_ds = stat.to_dataset(name='params')
#     stat_ds['slope'] = stat_ds['params'].isel(parameter=0)
#     stat_ds['intercept'] = stat_ds['params'].isel(parameter=1)
#     stat_ds['r_value'] = stat_ds['params'].isel(parameter=2)
#     stat_ds['p_value'] = stat_ds['params'].isel(parameter=3)
#     stat_ds['std_err'] = stat_ds['params'].isel(parameter=4)
#     stat_ds = stat_ds.drop_vars(['params'])

#     return stat_ds

# def main(scenario):
#     # Get historical data.
#     hist_paths = get_fpaths('Hist')
#     hist_da = preprocess(hist_paths)
#     hist_ave_da = hist_da.mean(dim='year', skipna=True)

#     # Get scenario data.
#     paths = get_fpaths(scenario)
#     da = preprocess(paths)
#     prec_da = da*100/hist_ave_da
#     reg_ds = linear_reg_da(prec_da, 'year')

#     return reg_ds

# ---------------- Calculate difference between end and start period --------------
def cal_diff(da:xr.DataArray):
    diff = (da.sel(year=da['year'].isin(end_period)).mean(dim='year', skipna=True) 
            - da.sel(year=da['year'].isin(start_period)).mean(dim='year', skipna=True))

    return diff

def cal_sign_prec(da:xr.DataArray):
    # Caculate the percentage of consistent sign trends among models.
    pos_sign = ((da > 0)).sum(dim='member')
    neg_sign = ((da < 0)).sum(dim='member')
    total = len(da['member'])
    pos_prec = pos_sign / total
    neg_prec = neg_sign / total
    prec = xr.where(pos_prec > neg_prec, pos_prec, neg_prec)

    return prec

def main(scenario):
    # Get historical data.
    hist_paths = get_fpaths('Hist')
    hist_da = preprocess(hist_paths)
    hist_ave_da = hist_da.mean(dim='year', skipna=True)

    # Get scenario data.
    paths = get_fpaths(scenario)
    da = preprocess(paths)
    prec_da = da*100/hist_ave_da

    # Calculate difference between end and start period
    diff_da = cal_diff(prec_da)
    mdiff_da = diff_da.mean(dim='member', skipna=True)

    # Calculate sign consistency
    sign_prec_da = cal_sign_prec(diff_da)

    outds = xr.Dataset({
        'diff': mdiff_da,
        'sign_prec': sign_prec_da,
    })

    return outds

if __name__ == "__main__":
    scenarios = ['SSP1_26', 'SSP2_45', 'SSP5_85']

    outdss = []
    for scenario in scenarios:
        reg_ds = main(scenario)
        outdss.append(reg_ds)

    outds = xr.concat(outdss, dim='scenario')
    outds = outds.assign_coords({'scenario': scenarios})

    outpath = r"F:\Research\AMFEdge\CMIP6\gpp_diff_dryseason_2015@2090.nc"
    outds.to_netcdf(outpath)