import os
import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import xarray as xr
import osgeo
import rioxarray as rxr
import geopandas as gpd

import Data.convert_data as cd
import Data.clip_data as cld
import GlobVars

# ------------------ S2AnoVI ------------------
# path = r"F:\Research\AMFEdge\VI\Amazon_intactForest_2023_S2AnoVI_01deg.tif"
# ds = cd.tif2ds(path)
# ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
#     .rio.write_crs("EPSG:4326")
# cds = cld.mask_ds_by_shp(ds=ds, shp=gpd.read_file(GlobVars.AMAZON_PATH))
# # cds = cds.interp_like(GlobVars.STD_DA, method='nearest')
# cds = cds.expand_dims({'time': [pd.to_datetime('2023-05-01')]})

# outpath = r"F:\Research\AMFEdge\VI\Amazon_intactForest_2023_S2AnoVI_01deg.nc"
# nvars = list(cds.data_vars)
# ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
# cds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')

# ------------------ S2AnoVI 100m with multiple tifs ------------------
# root_dir = r"F:\Research\AMFEdge\VI\Amazon_S2AnoVI_100m"
# fpaths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
# ds = cd.tifs2ds(fpaths)
# ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
#     .rio.write_crs("EPSG:4326")
# cds = cld.mask_ds_by_shp(ds=ds, shp=gpd.read_file(GlobVars.AMAZON_PATH))
# # cds = cds.interp_like(GlobVars.STD_DA, method='nearest')
# cds = cds.expand_dims({'time': [pd.to_datetime('2023-05-01')]})

# outpath = r"F:\Research\AMFEdge\VI\Amazon_2023_S2AnoVI_100m.nc"
# nvars = list(cds.data_vars)
# ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9, "dtype":"int16", 
#                             "scale_factor": 0.01, '_FillValue': 32767}]*len(nvars)))
# cds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')

# # -------------- Drought period ------------------
# path = r"F:\Research\AMFEdge\Meteo\Meta\Amazon_2023_droughtPeriod.tif"
# ds = cd.tif2ds(path)
# ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
#     .rio.write_crs("EPSG:4326")
# reprj_ds = ds.rio.reproject(
#     GlobVars.STD_DA.rio.crs, transform=GlobVars.STD_DA.rio.transform(),
#     shape=GlobVars.STD_DA.shape,resampling=5
#     ) #.rename({'x':'lon', 'y': 'lat', 1:'SCC'})
# cds = cld.mask_ds_by_shp(ds=ds, shp=gpd.read_file(GlobVars.AMAZON_PATH))
# cds = cds.expand_dims({'time': [pd.to_datetime('2023-05-01')]})

# outpath = r"F:\Research\AMFEdge\Meteo\Processed\Amazon_2023_droughtPeriod.nc"
# nvars = list(cds.data_vars)
# ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
# cds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')

# # -------------- Drought average CWD ------------------
# path = r"F:\Research\AMFEdge\Meteo\Meta\Amazon_2023_aveCWD.tif"
# ds = cd.tif2ds(path)
# ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
#     .rio.write_crs("EPSG:4326")
# reprj_ds = ds.rio.reproject(
#     GlobVars.STD_DA.rio.crs, transform=GlobVars.STD_DA.rio.transform(),
#     shape=GlobVars.STD_DA.shape,resampling=5
#     ) #.rename({'x':'lon', 'y': 'lat', 1:'SCC'})
# cds = cld.mask_ds_by_shp(ds=ds, shp=gpd.read_file(GlobVars.AMAZON_PATH))
# cds = cds.expand_dims({'time': [pd.to_datetime('2023-05-01')]})

# outpath = r"F:\Research\AMFEdge\Meteo\Processed\Amazon_2023_aveCWD.nc"
# nvars = list(cds.data_vars)
# ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
# cds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')

# # -------------- Amazon historical average MCWD ------------------
# path = r"F:\Research\AMFEdge\Meteo\Meta\Amazon_Hist_aveMCWD.tif"
# ds = cd.tif2ds(path)
# ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
#     .rio.write_crs("EPSG:4326")
# reprj_ds = ds.rio.reproject(
#     GlobVars.STD_DA.rio.crs, transform=GlobVars.STD_DA.rio.transform(),
#     shape=GlobVars.STD_DA.shape,resampling=5
#     ) #.rename({'x':'lon', 'y': 'lat', 1:'SCC'})
# cds = cld.mask_ds_by_shp(ds=ds, shp=gpd.read_file(GlobVars.AMAZON_PATH))
# cds = cds.expand_dims({'time': [pd.to_datetime('2023-05-01')]})

# outpath = r"F:\Research\AMFEdge\Meteo\Processed\Amazon_Hist_aveMCWD.nc"
# nvars = list(cds.data_vars)
# ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
# cds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')

# -------------- Drought onset offset meteo ------------------
# def ctif2nc(path, offset):
#     ds = cd.tif2ds(path)
#     ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
#         .rio.write_crs("EPSG:4326")
#     reprj_ds = ds.rio.reproject(
#         GlobVars.STD_DA.rio.crs, transform=GlobVars.STD_DA.rio.transform(),
#         shape=GlobVars.STD_DA.shape,resampling=5
#     )
#     cds = cld.mask_ds_by_shp(ds=ds, shp=gpd.read_file(GlobVars.AMAZON_PATH))
#     cds = cds.expand_dims({'time': [pd.to_datetime('2023-05-01')], 'offset': [offset]})

#     return cds

# offsets = [0, 1, 2, 3, 4, 5]
# prefpath = r"F:\Research\AMFEdge\Meteo\Meta\Amazon_2023_anoMeteo"
# ds_list = [ctif2nc(prefpath + f"{offset}M.tif", offset) for offset in offsets]
# outds = xr.concat(ds_list, dim='offset')

# outpath = r"F:\Research\AMFEdge\Meteo\Processed\Amazon_dOnset_2023_anoMeteo.nc"
# nvars = list(outds.data_vars)
# ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
# outds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')