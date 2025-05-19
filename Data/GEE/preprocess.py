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

# ------------------ Soil cation concentration ------------------
path = r"F:\Research\AMFEdge\Meteo\Amazon_2023_S2AnoVI.tif"
ds = cd.tif2ds(path)
ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
    .rio.write_crs("EPSG:4326")
# reprj_ds = ds.rio.reproject(
#     GlobVars.STD_DA.rio.crs, transform=GlobVars.STD_DA.rio.transform(),
#     shape=GlobVars.STD_DA.shape,resampling=5
#     ) #.rename({'x':'lon', 'y': 'lat', 1:'SCC'})
cds = cld.mask_ds_by_shp(ds=ds, shp=gpd.read_file(GlobVars.AMAZON_PATH))

outpath = r"F:\Research\AMFEdge\Meteo\Amazon_2023_S2AnoVI.nc"
nvars = list(cds.data_vars)
ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
cds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')
