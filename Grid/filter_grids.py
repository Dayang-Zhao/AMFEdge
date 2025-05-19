import sys
sys.path.append(r"D:\ProgramData\VistualStudioCode\AMFEdge")
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np

import matplotlib.pyplot as plt
plt.ion()

import GlobVars as gv

# ------------ Filter grids by shapefile --------------
# Amazon_path = r"F:\Shapefile\Amazon\sum_amazonia_polygons.shp"
# grid_path = r"F:\Research\AMFEdge\Shapefile\TMF_Grids_JRC.shp"
# polygons_gdf = gpd.read_file(grid_path)
# filter_gdf = gpd.read_file(Amazon_path)

# filtered = polygons_gdf[polygons_gdf.intersects(filter_gdf.unary_union)]
# filtered.plot()

# outpath = r"F:\Research\AMFEdge\Shapefile\Amazon_Grids_JRC.shp"
# filtered.to_file(outpath)

# ------------ Filter grids by TreeCover --------------
grid_path = r"F:\Research\AMFEdge\Shapefile\Amazon_Grids_JRC.shp"
grids = gpd.read_file(grid_path)
grids = grids.rename(columns={'id': 'Id'})

lcc_path = r"F:\Research\AMFEdge\Background\Amazon_Grids_bg_stat.csv"
undist_treecover = pd.read_csv(lcc_path)\
    .rename(columns={'TreeCover_mean':'TreeCover'})[['Id', 'TreeCover']]

grids = grids.merge(undist_treecover, on='Id', how='inner')
vaild_grids = grids[grids['TreeCover'] > 0.2]

vaild_grids.plot(column='TreeCover', cmap='Greens', legend=True, figsize=(10, 10))

# Save dataset.
outpath = r"F:\Research\AMFEdge\Shapefile\Amazon_Grid_Final.shp"
vaild_grids.to_file(outpath, driver='ESRI Shapefile')