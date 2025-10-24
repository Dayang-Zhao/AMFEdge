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
import preprocess as pp
import Data.CMIP6.cab_cmip6_xclim as cab

ROOT_DIR = r"F:\Research\AMFEdge\CMIP6\metaData"
OUT_ROOTDIR = r"F:\Research\AMFEdge\CMIP6\Processed\QDM"
BANDS_LUT = {'evspsbl':'total_evaporation_sum', 'rsds':'surface_solar_radiation_downwards_sum',
            'pr':'total_precipitation_sum','ps':'surface_pressure',
            'tas':'temperature_2m', 'huss':'specific_humidity_2m',
            }

MIN_LON, MAX_LON, MIN_LAT, MAX_LAT = -80, -44, -21, 9
RES, HALF_RES = 1, 0.5
NUM_LON, NUM_LAT = int((MAX_LON - MIN_LON) / RES), int((MAX_LAT - MIN_LAT) / RES)
STD_LON = np.linspace(MIN_LON+HALF_RES, MAX_LON-HALF_RES, num=NUM_LON)        
STD_LAT = np.linspace(MAX_LAT-HALF_RES, MIN_LAT+HALF_RES, num=NUM_LAT)
STD_TIME = pd.date_range(start='2015-01-01', end='2100-12-31', freq='MS')
# STD_TIME = pd.date_range(start='1985-01-01', end='2014-12-31', freq='MS')

EXP_STR = 'SSP'

def fetch_fpaths4model(model, rootdir):
    fnames = glob.glob(f"{model}_{EXP_STR}*.nc", root_dir=rootdir)
    fpaths = [os.path.join(rootdir, fname) for fname in fnames]

    return fpaths

def get_ds_res(ds):
    """Get the resolution of the dataset."""
    lon_res = ds['lon'].diff(dim='lon').mean().item()
    lat_res = ds['lat'].diff(dim='lat').mean().item()
    return lon_res, lat_res

def save_as_tiff(ds, out_prefix):
    """Save the dataset as a GeoTIFF file."""
    vars = list(ds.data_vars)
    ds = ds.transpose("time", "lat", "lon")

    for var in vars:
        outpath = f"{out_prefix}_{var}.tif"

        # Save the dataset as a GeoTIFF file.
        dst_da = ds[var]

        outda = dst_da.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
            .rio.write_crs("EPSG:4326")
        outda.rio.to_raster(outpath, compress="ZSTD", driver='GTiff')
        print(f"Saved {outpath}")

def main(models, rootdir=ROOT_DIR, outdir=OUT_ROOTDIR):
    """Main function to preprocess all models."""
    # Preprocess each model.
    good_models = []
    for model in models:
        try:
            print(f"Processing model: {model}")
            paths = fetch_fpaths4model(model, rootdir)

            # Preprocess the model.
            ds = pp.preprocess_model(paths, std_time=STD_TIME)

            # Bias correction using ERA5.
            qdm = cab.get_qdm_model(model)
            cds = cab.cab_qdm_cmip6_by_era5(qdm, ds)

            # Remove models with spatial resolution > 2 degrees.
            lon_res, lat_res = get_ds_res(ds)
            if lon_res > 2 or lat_res > 2:
                print(f"Model {model} has resolution > 2 degrees, skipping.")
                continue
            else:
                out_prefname = f"{outdir}/{model}"
                save_as_tiff(cds, out_prefname)
                
                good_models.append(model)
        except Exception as e:
            print(f"Error processing model {model}: {e}")
            continue

    print(good_models)

if __name__ == '__main__':
    # models = ['access_cm2', 'awi_cm_1_1_mr', 'bcc_csm2_mr', 'canesm5',
    #           'cesm2', 'cmcc_cm2_sr5', 'cmcc_esm2',
    #           'cnrm_cm6_1_hr', 'ec_earth3_veg_lr',
    #           'fgoals_f3_l', 'fgoals_g3', 'fio_esm_2_0', 'gfdl_esm4',
    #           'hadgem3_gc31_mm', 'iitm_esm',
    #           'inm_cm5_0', 'ipsl_cm6a_lr', 'kace_1_0_g',
    #           'mcm_ua_1_0', 'miroc6', 'mpi_esm1_2_lr', 'mri_esm2_0',
    #           'nesm3', 'noresm2_mm', 'taiesm1', 'ukesm1_0_ll']
    models = ['mri_esm2_0', 'cnrm_cm6_1_hr', 'cesm2', 'ukesm1_0_ll',
             'noresm2_mm', 'miroc6', 'taiesm1',
             'kace_1_0_g', 'access_cm2', 'cmcc_cm2_sr5']

    experiments = ['SSP1_26', 'SSP2_45', 'SSP5_85'] # 'Hist', 'SSP1_26', 'SSP2_45', 'SSP5_85'
    
    for experiment in experiments:
        root_dir = os.path.join(ROOT_DIR, experiment)
        out_dir = os.path.join(OUT_ROOTDIR, experiment)
        main(models, rootdir=root_dir, outdir=out_dir)