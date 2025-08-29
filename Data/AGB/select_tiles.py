import os
import glob
import shutil

import osgeo
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import box

MIN_LON, MAX_LON, MIN_LAT, MAX_LAT = -80, -44, -21, 9
sufix_fname = '_agb.tif'

def cal_lon2lat(fname):
    num = {False: -1, True: 1}
    SN = (fname[0]=='N')
    lat = int(fname[1:3])
    ES = (fname[3]=='E')
    lon = int(fname[4:7])

    return lon*num[ES], lat*num[SN]

def check_tif_shp_intersection(da, gdf):
    """Check if tiff intersects with shp"""
    raster_bounds = da.rio.bounds()   # (minx, miny, maxx, maxy)
    raster_crs = da.rio.crs
    raster_bbox = box(*raster_bounds)

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    return gdf.intersects(raster_bbox).any()

def select_tiles(root_dir, outdir, shp_path):
    gdf = gpd.read_file(shp_path)
    fnames = glob.glob("*"+sufix_fname, root_dir=root_dir)

    for fname in fnames:
        lon, lat = cal_lon2lat(fname)

        # The extent of tile: [lon, lon+2, lat-2, lat]
        min_lon, max_lon, min_lat, max_lat = lon, lon+2, lat-2, lat
        cond = (max_lon <= MIN_LON) or (min_lon >= MAX_LON) or \
                (max_lat <= MIN_LAT) or (min_lat >= MAX_LAT)
        
        if not cond:
            try:
                fpath = os.path.join(root_dir, fname)
                da = rxr.open_rasterio(fpath, masked=True)
                sign = check_tif_shp_intersection(da, gdf)

                if sign:
                    outpath = os.path.join(outdir, fname)
                    shutil.copy2(fpath, outpath)
                    print(f"Selected tile: {fname}")
            except Exception as e:
                print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    root_dir = r"F:\Research\AMFEdge\AGB\GlobBiomass\NAmerica"
    shp_path = r"F:\Shapefile\Amazon\sum_amazonia_polygons.shp"
    outdir = r"F:\Research\AMFEdge\AGB\GlobBiomass\Amazon"
    select_tiles(root_dir, outdir, shp_path=shp_path)
    print("Tile selection completed.")