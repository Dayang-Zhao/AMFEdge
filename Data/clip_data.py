import xarray as xr
import osgeo
import rioxarray
import geopandas as gpd

def mask_ds_by_shp(ds:xr.Dataset, shp:gpd.GeoDataFrame, crs:str='EPSG:4326'):

    ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    if ds.rio.crs is None:
        ds = ds.rio.write_crs(crs)

    if shp.crs != ds.rio.crs:
        shp = shp.to_crs(ds.rio.crs)
    
    masked_ds = ds.rio.clip(shp.geometry, shp.crs)

    return masked_ds