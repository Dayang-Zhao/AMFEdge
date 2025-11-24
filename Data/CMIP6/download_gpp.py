import os

import pandas as pd
import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex

import osgeo
from intake_esgf import ESGFCatalog

import matplotlib.pyplot as plt
cat = ESGFCatalog()

EXTENT = [9, -80, -21, -44]
PRE_FPATH = "F:/Research/AMFEdge/CMIP6/metaData"
SUBDIR_LUT = {'historical':'Hist','ssp126': 'SSP1_26', 'ssp245': 'SSP2_45', 'ssp585': 'SSP5_85'}
MODEL_LUT = dict(zip(
    ['MRI-ESM2-0', 'CNRM-CM6-1-HR', 'CESM2', 'UKESM1-0-LL',
        'NorESM2-MM', 'TaiESM1',
         'CMCC-CM2-SR5'],
    ['mri_esm2_0', 'cnrm_cm6_1_hr', 'cesm2', 'ukesm1_0_ll',
        'noresm2_mm', 'taiesm1','cmcc_cm2_sr5']
))
START_YEAR = 2015
END_YEAR = 2100

def download_cmip6_gpp(models, scenario):
    results = cat.search(
        project="CMIP6",
        source_id=models,
        experiment_id=scenario,
        variable_id="gpp",
        table_id="Lmon", 
        # member_id=[f"r{i}i1p1f1" for i in range(1,11)],
        file_start=f"{START_YEAR}-01",
        file_end=f"{END_YEAR}-12"
    )
    print(results)

    # Get the first member_id for each model
    search_df = results.df
    dst_models = search_df['source_id'].unique()
    for model in dst_models:
        dst_member_id = search_df[search_df['source_id'] == model]['member_id'].values[0]
        dst_result = cat.search(
            project="CMIP6",
            source_id=model,
            experiment_id=scenario,
            variable_id="gpp",
            table_id="Lmon", 
            member_id=dst_member_id,
            file_start=f"{START_YEAR}-01",
            file_end=f"{END_YEAR}-12"
        )
        print(dst_result)

        # Download the dataset
        ds_dict = cat.to_dataset_dict()
        ds = list(ds_dict.values())[0][['gpp']]
        
       # Rename dimensions.
        if 'longitude' in ds.dims:
            ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})

        # Convert longitude from [0, 360] to [-180, 180] if needed.
        ds['lon'] = ((ds['lon'] + 180) % 360) - 180
        ds = ds.sortby('lon')

        # Adjust the time dim as datetime64.
        if isinstance(ds.indexes['time'], CFTimeIndex):
            ds['time'] = pd.to_datetime(ds.indexes['time'].to_datetimeindex())
        # Clip the dataset to the specified extent
        ds_clip = ds.where(
            (ds['lat'] >= EXTENT[2]) & (ds['lat'] <= EXTENT[0]) &
            (ds['lon'] >= EXTENT[1]) & (ds['lon'] <= EXTENT[3]),
            drop=True
        )
        ds_clip = ds_clip.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))

        # Save the dataset to a NetCDF file
        rootdir = f"{PRE_FPATH}/{SUBDIR_LUT[scenario]}/"
        fname = f"{MODEL_LUT[model]}_{SUBDIR_LUT[scenario]}_gpp_2015_2100.nc"
        outpath = os.path.join(rootdir, fname)
        encoding = {var: {"zlib": True, "complevel": 9} for var in ds_clip.data_vars}
        ds_clip.to_netcdf(outpath, encoding=encoding)
        print(f"Saved clipped dataset to {outpath}")

if __name__ == "__main__":
    # models = ['mri_esm2_0', 'cnrm_cm6_1_hr', 'cesm2', 'ukesm1_0_ll',
    #         'noresm2_mm', 'miroc6', 'taiesm1',
    #         'kace_1_0_g', 'access_cm2', 'cmcc_cm2_sr5']
    models =   ['MRI-ESM2-0', 'CNRM-CM6-1-HR', 'CESM2', 'UKESM1-0-LL',
        'NorESM2-MM', 'MIROC6', 'TaiESM1', 'KACE-1-0-G', 'ACCESS-CM2',
         'CMCC-CM2-SR5']
    scenarios = ['ssp126', 'ssp245', 'ssp585']
    # scenarios = ['historical']

    for scenario in scenarios:
        download_cmip6_gpp(models, scenario)
