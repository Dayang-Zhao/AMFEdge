import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\AMFEdge')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import climate_indices.indices as indices
import climate_indices.compute as compute
from climate_indices.indices import Distribution

import Data.convert_data as cd

# ---------------- Calculate PET from ERA5 ----------------
# path = r"F:\Research\AMFEdge\Meteo\Meta\Amazon_ERA5_1985_2024_monthlyTP.tif"
# ds = cd.datetTif2ds(path)
# precip = ds['total_precipitation_sum'] * 1000  # m → mm
# temp = ds['temperature_2m'] - 273.15   # K → °C
# outds = xr.merge([precip, temp])
# outds['temperature_2m'].attrs = {'units': 'degree_celsius', 'long_name': '2m air temperature'}
# outds['total_precipitation_sum'].attrs = {'units': 'mm', 'long_name': 'total precipitation'}
# outds.to_netcdf(r"F:\Research\AMFEdge\Meteo\Processed\Amazon_ERA5_1985_2024_monthlyTP.nc")

# -------------- Align the coordinates of GLEAM PET and CHIRPS P ----------------
path = r"F:\Research\AMFEdge\Meteo\Meta\Amazon_CHIRPS_1981_2024_monthlyP.tif"
p_ds = cd.datetTif2ds(path)
path = r"F:\Research\AMFEdge\Meteo\Processed\GLEAM4_1980_2024_Amazon_Ep.nc"
e_ds = xr.open_dataset(path)
e_ds['time'] = e_ds.indexes['time'].to_period('M').to_timestamp(how='start')
e_ds = e_ds.sel(time=slice('1981-01-01', '2024-12-01'))
# p_ds2 = p_ds.interp(lon=e_ds.lon, lat=e_ds.lat, method='nearest')
p_ds2 = p_ds.interp_like(e_ds, method='nearest')
p_ds2['precipitation'].attrs = {'units': 'mm', 'long_name': 'total precipitation'}
outds = xr.merge([p_ds2, e_ds]).transpose('lat', 'lon', 'time')

outpath = r"F:\Research\AMFEdge\Meteo\Processed\Amazon_CHIRPS_GLEAM_1981_2024_monthlyPetP.nc"
encoding = {'precipitation': {"zlib": True, "complevel": 9}, 'Ep': {"zlib": True, "complevel": 9}}
outds.to_netcdf(outpath, encoding=encoding)
print(p_ds2)
