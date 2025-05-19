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

years = range(2023, 2024)
outds = []
for year in years:
    path = rf"F:\Research\AMFEdge\Background\Meta\Amazon_TMFlandCover_{year}.tif"
    ds = cd.tif2ds(path)
    cds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
        .rio.write_crs("EPSG:4326")
    # reprj_ds = ds.rio.reproject(
    #     GlobVars.STD_DA.rio.crs, transform=GlobVars.STD_DA.rio.transform(),
    #     shape=GlobVars.STD_DA.shape,resampling=5
    #     ).rename({'x':'lon', 'y': 'lat'})
    # cds = cld.mask_ds_by_shp(ds=reprj_ds, shp=shp)
    cds = cds.assign_coords({'time': pd.to_datetime([f"{year}-01-01"])})
    outds.append(cds)

outds = xr.concat(outds, dim='time')
outpath = r"F:\Research\AMFEdge\Background\Post\Amazon_TMFLandCover_2023.nc"
nvars = list(outds.data_vars)
ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
outds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')