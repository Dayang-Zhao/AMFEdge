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
import concurrent.futures

import matplotlib.pyplot as plt

import GlobVars as gv

ROOT_DIR = r"F:\Research\AMFEdge\CMIP6\metaData"
OUT_ROOTDIR = r"F:\Research\AMFEdge\CMIP6\Processed"
BANDS_LUT = {'evspsbl':'total_evaporation_sum', 'rsds':'surface_solar_radiation_downwards_sum',
            'pr':'total_precipitation_sum','ps':'surface_pressure',
            'tas':'temperature_2m', 'huss':'specific_humidity_2m',
            }
CAB_PATH  = r"F:\Research\AMFEdge\CMIP6\xCmip6_yEra5_cab_2004_2014.csv"
CAB_DF = pd.read_csv(CAB_PATH)

# STD_TIME = pd.date_range(start='2015-01-01', end='2100-12-31', freq='MS')
STD_TIME = pd.date_range(start='2004-01-01', end='2014-12-31', freq='MS')
EXP_STR = 'historical'

def save_as_tiff(ds, out_prefix):
    """Save the dataset as a GeoTIFF file."""
    times = ds['time'].values
    ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat').rio.write_crs("EPSG:4326")

    for time in times:
        time_str = pd.to_datetime(time).strftime('%Y%m%d')
        outpath = f"{out_prefix}_{time_str}.tif"

        # Save the dataset as a GeoTIFF file.
        dst_ds = ds.sel(time=time, drop=True).to_array(dim='band')

        outda = dst_ds.rio.write_nodata(-9999)\
            .rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
            .rio.write_crs("EPSG:4326")
        outda.rio.to_raster(outpath, compress="ZSTD", driver='GTiff')
        print(f"Saved {outpath}")

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

def linear_cab(ds, cab_df, vars):
    for var in vars:
        slope = cab_df.loc[cab_df['var'] == var, 'slope'].values[0]
        intercept = cab_df.loc[cab_df['var'] == var, 'intercept'].values[0]

        cab_da = ds[var] * slope + intercept
        ds[var] = cab_da

    return ds

def preprocess_model(paths):
    """Preprocess each model: merge and calculate VPD."""

    # Open all datasets and merge them.
    da_list = []
    for path in paths:
        ds = xr.open_dataset(path)
        
        dst_var = list(set(ds.data_vars) & set(BANDS_LUT.keys()))
        da_list.append(ds[dst_var])

        # lon_res, lat_res = get_ds_res(ds)
        # print(f"Variable {dst_var} resolution: lon_res={lon_res}, lat_res={lat_res}")

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

    # Linear calibration with ERA5.
    # ds = linear_cab(ds, CAB_DF, CAB_DF['var'].tolist())

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

def get_ds_res(ds):
    """Get the resolution of the dataset."""
    lon_res = ds['lon'].diff(dim='lon').mean().item()
    lat_res = ds['lat'].diff(dim='lat').mean().item()
    return lon_res, lat_res

def main(models, rootdir=ROOT_DIR, outdir=OUT_ROOTDIR):
    """Main function to preprocess all models."""
    # Preprocess each model.
    ds_list = []
    good_models = []
    for model in models:
        try:
            print(f"Processing model: {model}")
            paths = fetch_fpaths4model(model, rootdir)

            # Preprocess the model.
            ds = preprocess_model(paths)

            # Remove models with spatial resolution > 2 degrees.
            lon_res, lat_res = get_ds_res(ds)
            if lon_res > 2 or lat_res > 2:
                print(f"Model {model} has resolution > 2 degrees, skipping.")
                continue
            else:
                ds_list.append(ds)
                
                good_models.append(model)
        except Exception as e:
            print(f"Error processing model {model}: {e}")
            continue
    print(good_models)

    # Calculate the average and std across all models.
    if ds_list:
        combined_ds = xr.concat(ds_list, dim='model')
        avg_ds = combined_ds.mean(dim='model', keep_attrs=True, skipna=True)
        std_ds = combined_ds.std(dim='model', keep_attrs=True, skipna=True)
        avg_ds = avg_ds.rename(dict(zip(avg_ds.data_vars, [f"{var}_avg" for var in avg_ds.data_vars])))
        std_ds = std_ds.rename(dict(zip(std_ds.data_vars, [f"{var}_std" for var in std_ds.data_vars])))
        outds = xr.merge([avg_ds, std_ds])

        # Save output datasets as tiff.
        out_prefname = f"{outdir}/cmip6"
        # save_as_tiff(outds, out_prefname)
    else:
        print("No datasets to process.")

if __name__ == '__main__':
    models = ['access_cm2', 'awi_cm_1_1_mr', 'bcc_csm2_mr', 'canesm5',
              'cesm2', 'cmcc_cm2_sr5', 'cmcc_esm2',
              'cnrm_cm6_1_hr', 'ec_earth3_veg_lr',
              'fgoals_f3_l', 'fgoals_g3', 'fio_esm_2_0', 'gfdl_esm4',
              'hadgem3_gc31_mm', 'iitm_esm',
              'inm_cm5_0', 'ipsl_cm6a_lr', 'kace_1_0_g',
              'mcm_ua_1_0', 'miroc6', 'mpi_esm1_2_lr', 'mri_esm2_0',
              'nesm3', 'noresm2_mm', 'taiesm1', 'ukesm1_0_ll']
    experiments = ['Hist'] # 'Hist', 'SSP1_26', 'SSP2_45', 'SSP5_85'
    
    for experiment in experiments:
        root_dir = os.path.join(ROOT_DIR, experiment)
        out_dir = os.path.join(OUT_ROOTDIR, experiment)
        main(models, rootdir=root_dir, outdir=out_dir)