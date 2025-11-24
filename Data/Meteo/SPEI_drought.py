import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
plt.ion()

import Data.clip_data as cld
import GlobVars as gv

def preprocess(ds:xr.Dataset, shp) -> xr.Dataset:
  
    ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
        .rio.write_crs("EPSG:4326")
    cds = cld.mask_ds_by_shp(ds=ds, shp=shp)

    return cds


def detect_drought(spei, threshold=-1.0, min_duration=2):

    # spei 这里是 numpy.ndarray，不再是 DataArray
    if np.all(np.isnan(spei)):
        return xr.DataArray(
            data=[np.nan, np.nan, np.nan, np.nan],
            dims=['var'],
            coords={'var': ['start', 'end', 'duration', 'min_spei']}
        )

    drought = spei < threshold
    drought_diff = np.diff(drought.astype(int))  # ✅ 改成 numpy 方式

    start_idx = np.where(drought_diff == 1)[0] + 1
    end_idx = np.where(drought_diff == -1)[0] + 1

    # 若最后一个时间点仍在干旱中
    if drought[-1]:
        end_idx = np.append(end_idx, len(drought) - 1)

    # 若一开始就是干旱
    if len(end_idx) and (len(start_idx) == 0 or start_idx[0] > end_idx[0]):
        start_idx = np.insert(start_idx, 0, 0)

    events = []
    for s, e in zip(start_idx, end_idx):
        duration = e - s + 1
        if duration >= min_duration:
            event = {
                "start": s+gv.HYDROMONTH,
                "end": e+gv.HYDROMONTH,
                "duration": duration,
                "min_spei": float(np.nanmin(spei[s:e + 1]))
            }
            events.append(event)

    if len(events) == 0:
        return xr.DataArray(
            data=[np.nan, np.nan, np.nan, np.nan],
            dims=['var'],
            coords={'var': ['start', 'end', 'duration', 'min_spei']}
        )

    # 这里只是返回一个示例结果
    return xr.DataArray(
        data=[events[0]['start'], events[0]['end'], events[0]['duration'], events[0]['min_spei']],
        dims=['var'],
        coords={'var': ['start', 'end', 'duration', 'min_spei']}
    )

def main(spei:xr.Dataset, shp, threshold=-1.0, min_duration=2):
    """
    Detect drought events based on SPEI time series.

    Parameters:
    spei (xr.DataArray): SPEI data with dimensions (time, lat, lon).
    threshold (float): SPEI threshold to define drought.
    min_duration (int): Minimum duration (in time steps) to consider a drought event.

    Returns:
    xr.DataArray: Drought events for each spatial location.
    """

    # Preprocess data
    spei = preprocess(spei, shp)
    spei = spei['spei_gamma_03']

    # Apply drought detection function across spatial dimensions
    drought_results = xr.apply_ufunc(
        detect_drought,
        spei,
        input_core_dims=[["time"]],
        output_core_dims=[["var"]],
        vectorize=True,
        kwargs={"threshold": threshold, "min_duration": min_duration},
        dask="parallelized",
        output_dtypes=[object],
    )
    outds = drought_results.to_dataset(dim='var').rename({0: 'startMonth', 1: 'endMonth', 2: 'length', 3: 'min_spei'})
    outds = outds.astype(float)

    return outds

if __name__ == "__main__":
    dst_year = 2023
    path = r"F:\Research\AMFEdge\Meteo\Processed\Amazon_CHIRPS_GLEAM_1985_2024_monthlySPEI_spei_gamma_03.nc"
    ds = xr.open_dataset(path)\
        .sel(time=slice(f'{dst_year}-{gv.HYDROMONTH}-01', f'{dst_year+1}-{gv.HYDROMONTH}-01'))
    shp_path = r"F:\Shapefile\Amazon\sum_amazonia_polygons.shp"
    shp = gpd.read_file(shp_path)
    threshold = -1.0
    min_duration = 2
    drought_ds = main(ds, shp, threshold, min_duration)

    outpath = rf"F:\Research\AMFEdge\Meteo\Processed\Amazon_CHIRPS_GLEAM_2023_drought_spei_gamma_03.nc"
    drought_ds.to_netcdf(outpath)