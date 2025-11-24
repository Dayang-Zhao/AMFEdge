import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
import pandas as pd
import osgeo
import xarray as xr
import rioxarray as rxr
import geopandas as gpd

import matplotlib.pyplot as plt
plt.ion()

import Data.clip_data as cld
import Data.convert_data as cd
import Data.save_data as sd

def rename_ds(ds:xr.Dataset, suffix:str):
    '''
    Rename the dataset variables with a prefix.
    '''
    new_ds = ds.copy()
    for var in ds.data_vars:
        new_ds = new_ds.rename({var: f"{var}_{suffix}"})
    
    return new_ds

def stat_grids(ds:xr.Dataset, shp:gpd.GeoDataFrame, ids:list, crs:str='EPSG:4326'):
    '''
    Calculate the mean of each grid cell in the dataset.
    '''
    # Add 'time' dimension to the dataset if it doesn't exist.
    if 'time' not in ds.dims:
        ds = ds.expand_dims(dim='time', axis=0)
        ds['time'] = pd.to_datetime([0])

    outdfs = []
    for id in ids:
        dst_shp = shp[shp['Id'] == id]
        # Clip.
        try:
            clipped_ds = cld.mask_ds_by_shp(ds, dst_shp, crs)
        except Exception as e:
            print(f"Skipping feature {id} due to error: {e}")
            continue

        # Calculate the mean and std.
        mean_ds = clipped_ds.mean(dim=['lon', 'lat'], skipna=True)
        std_ds = clipped_ds.std(dim=['lon', 'lat'], skipna=True)
        count_ds = clipped_ds.count(dim=['lon', 'lat'])

        # Rename and merge the mean and std datasets
        mean_df = rename_ds(mean_ds, 'mean').to_dataframe()\
            .reset_index().drop(columns=['time', 'spatial_ref'])
        std_df = rename_ds(std_ds, 'std').to_dataframe()\
            .reset_index().drop(columns=['time', 'spatial_ref'])
        count_df = rename_ds(count_ds, 'count').to_dataframe()\
            .reset_index().drop(columns=['time', 'spatial_ref'])

        outdf = pd.concat([mean_df, std_df, count_df], axis=1)
        outdf['Id'] = id
        outdfs.append(outdf)

    outdf = pd.concat(outdfs, axis=0, ignore_index=True)

    return outdf

if __name__ == '__main__':
    path = r"F:\Research\AMFEdge\Meteo\Meta\Amazon_Hist_aveMCWD.tif"
    # ds = xr.open_dataset(path)
    ds = cd.tif2ds(path).drop_vars('spatial_ref')
    nvar = list(ds.data_vars)

    shp_path = r"F:\Research\AMFEdge\Shapefile\Amazon_Grid_Final_15deg.shp"
    shp = gpd.read_file(shp_path).rename(columns={'id': 'Id'})
    ids = shp['Id'].unique()
    # ids = [470, 445, 446, 428]

    outdf = stat_grids(ds, shp, ids)

    # Save the output dataframe to a CSV file.
    outpath = r"F:\Research\AMFEdge\Meteo\Amazon_Grids_2023_histMCWD_stat.csv"
    outdf.to_csv(outpath, index=False)