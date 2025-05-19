import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
import os
import osgeo
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import glob
import matplotlib.pyplot as plt

import GlobVars
import Data.convert_data as cd
import Data.clip_data as cld

# Step 1: Locate the TIFF files
root_dir = r"F:\Research\Amazon2024\Soil\HAND"
fname = "Copernicus_DSM_COG_*_HAND.tif"
fpaths = glob.glob(fname, root_dir=root_dir)

# Step 2: Load TIFF files as DataArrays
outds = []
os.chdir(root_dir)
for fpath in fpaths:
    print(fpath)
    ds = cd.tif2ds(fpath)
    ds = ds.rename({1:'HAND'})\
        .rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
        .rio.write_crs("EPSG:4326")
    reprj_ds = ds.rio.reproject(
        GlobVars.STD_DA.rio.crs, transform=GlobVars.STD_DA.rio.transform(),
        shape=GlobVars.STD_DA.shape,resampling=5
        ).rename({'x':'lon', 'y': 'lat'})
    outds.append(reprj_ds)

outds = xr.concat(outds, dim="band").mean(dim="band", skipna=True)

coutds = cld.mask_ds_by_shp(ds=outds, shp=gpd.read_file(GlobVars.AMAZON_PATH))
outpath = r"F:\Research\AMFEdge\Background\Amazon_CopernicusHand.nc"
nvars = list(coutds.data_vars)
ecoding = dict(zip(nvars, [{"zlib": True, "complevel": 9}]*len(nvars)))
coutds.to_netcdf(path=outpath, encoding=ecoding, engine='netcdf4')


