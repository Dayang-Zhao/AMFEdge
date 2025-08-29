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

# --------------------------- GEE Meteo datasets. -----------------------------
year = 2023
path = rf"F:\Research\AMFEdge\Meteo\Meta\Amazon_{year}_anoMeteo.tif"
ds = cd.tif2ds(path)
ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
    .rio.write_crs("EPSG:4326")
reprj_ds = ds.rio.reproject(
    GlobVars.STD_DA.rio.crs, transform=GlobVars.STD_DA.rio.transform(),
    shape=GlobVars.STD_DA.shape,resampling=5
    ).rename({'x':'lon', 'y': 'lat'})
cds = cld.mask_ds_by_shp(ds=reprj_ds, shp=gpd.read_file(GlobVars.AMAZON_PATH))
cds = cds.expand_dims({'time': [pd.to_datetime(f'{year}-05-01')]})

outpath = rf"F:\Research\AMFEdge\Meteo\Processed\Amazon_{year}_anoMeteo.nc"
nvars = list(cds.data_vars)
ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
cds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')

# # --------------------------- GEE MCWD datasets. -----------------------------
# path = r"F:\Research\AMFEdge\Meteo\Meta\Amazon_2023_rel_anoMCWD.tif"
# ds = cd.tif2ds(path)
# ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
#     .rio.write_crs("EPSG:4326")
# reprj_ds = ds.rio.reproject(
#     GlobVars.STD_DA.rio.crs, transform=GlobVars.STD_DA.rio.transform(),
#     shape=GlobVars.STD_DA.shape,resampling=5
#     ).rename({'x':'lon', 'y': 'lat'})
# cds = cld.mask_ds_by_shp(ds=reprj_ds, shp=gpd.read_file(GlobVars.AMAZON_PATH))
# cds = cds.expand_dims({'time': [pd.to_datetime('2023-05-01')]})

# outpath = r"F:\Research\AMFEdge\Meteo\Processed\Amazon_2023_rel_anoMCWD.nc"
# nvars = list(cds.data_vars)
# ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
# cds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')