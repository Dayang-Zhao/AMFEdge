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

# ------------------ SoilGrid ------------------
path = r"F:\Research\AMFEdge\Soil\Meta\Amazon_SoilGrid_SandSocNitrogen.tif"
ds = cd.tif2ds(path)
ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
    .rio.write_crs("EPSG:4326")
reprj_ds = ds.rio.reproject(
    GlobVars.STD_DA.rio.crs, transform=GlobVars.STD_DA.rio.transform(),
    shape=GlobVars.STD_DA.shape,resampling=5
    ).rename({'x':'lon', 'y': 'lat'})
cds = cld.mask_ds_by_shp(ds=reprj_ds, shp=gpd.read_file(GlobVars.AMAZON_PATH))

outpath = r"F:\Research\AMFEdge\Soil\Amazon_SoilGrid_SandSocNitrogen.nc"
nvars = list(cds.data_vars)
ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
cds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')

# # ------------------ Soil cation concentration ------------------
# path = r"F:\Research\Amazon2024\Soil\SoilFert\KrigeSoil_fernRcropNEW.tif"
# ds = cd.tif2ds(path)
# ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
#     .rio.write_crs("EPSG:4326")
# reprj_ds = ds.rio.reproject(
#     GlobVars.STD_DA.rio.crs, transform=GlobVars.STD_DA.rio.transform(),
#     shape=GlobVars.STD_DA.shape,resampling=5
#     ).rename({'x':'lon', 'y': 'lat', 1:'SCC'})
# cds = cld.mask_ds_by_shp(ds=reprj_ds, shp=gpd.read_file(GlobVars.AMAZON_PATH))

# outpath = r"F:\Research\AMFEdge\Environment\Amazon_KrigeSoilFern.nc"
# nvars = list(cds.data_vars)
# ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
# cds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')

# ------------------ Soil Phosphorus ------------------
# path = r"F:\Research\Amazon2023\Soil\Phosphorus\avail_p_AVG.nc"
# ds = xr.open_dataset(path).rename({'longitude':'lon', 'latitude': 'lat'})
# ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
#     .rio.write_crs("EPSG:4326")
# reprj_ds = ds.rio.reproject(
#     GlobVars.STD_DA.rio.crs, transform=GlobVars.STD_DA.rio.transform(),
#     shape=GlobVars.STD_DA.shape,resampling=5
#     ).rename({'x':'lon', 'y': 'lat'})
# cds = cld.mask_ds_by_shp(ds=reprj_ds, shp=GlobVars.SHP)

# outpath = r"F:\Research\Amazon2023\Soil\Processed\Amazon_availP.nc"
# nvars = list(cds.data_vars)
# ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
# cds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')