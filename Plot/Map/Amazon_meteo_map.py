import sys
sys.path.append(r'D:\ProgramData\VistualStudioCode\Amazon2024')
import numpy as np

import netCDF4
import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns

import GlobVars
import Data.convert_data as cd

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
matplotlib.rc('font', **font)
CRS = ccrs.PlateCarree()
EXTENT = [-81.5, -44, -22, 12.5]

def cm2inch(value):
    return value/2.54

def main(ds_array:list, dst_vars:list, grid:tuple, plot_setting: dict, 
         shp_path:str, outpath:str=None):

    nrows, ncols = grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex='col', sharey='row',
        subplot_kw={'projection': CRS, 'frameon':True}
        )
    fig.set_size_inches(cm2inch(17), cm2inch(7))
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']

    for i in range(nrows):
        for j in range(ncols):
            ds = ds_array[i*ncols+j]
            dst_var = dst_vars[i*ncols+j]
            title_num = title_nums[i*ncols+j]
            dst_da = ds[dst_var]

            # Plot setting.
            vrange = plot_setting['vranges'][i*ncols+j]
            # cbar_label = plot_setting['cbar_labels'][i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            cmap = plot_setting['cmaps'][i*ncols+j]
            level = plot_setting['levels'][i*ncols+j]
            extend = plot_setting['extend'][i*ncols+j]

            # Plot.
            ax = axes[j]
            im = dst_da.plot(ax=ax, cmap=cmap, add_colorbar=True, vmin=vrange[0], vmax=vrange[1], levels=level,
                    cbar_kwargs={'label':None,'pad':0.05, 'shrink':0.95, 'aspect':20, 'extend':extend, 'orientation':'horizontal'})

            # Add a basemap (world map)
            ax.add_feature(cfeature.COASTLINE, edgecolor='#919191')
            ax.set_extent(EXTENT, crs=CRS)

            # Plot plugins.
            ax.set_title(title_num+' '+title, loc='left')

            # Step 6: Add Shapefile overlay if provided
            if shp_path is not None:
                shapefile = gpd.read_file(shp_path)
                shapefile.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1)

            # Remove frames and ticks
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Adjust.
    fig.subplots_adjust(bottom=0.01, top=0.95, left=0.02, right=0.98, hspace=0.1, wspace=0.05)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    # Read data
    dst_year = 2023
    # Read data
    paths = [r"F:\Research\AMFEdge\Meteo\Meta\Amazon_GLEAM_2023_anoMeteo.tif",]*2 \
        + [r"F:\Research\AMFEdge\Meteo\Meta\Amazon_GLEAM_2023_MCWD.tif"]
    ds_array = [cd.tif2ds(path).drop('spatial_ref') for path in paths]
    dst_vars = ['total_precipitation_sum', 'total_evaporation_sum','MCWD']
    # ds_array[1] = ds_array[1]*-1
    ds_array[-1] = ds_array[-1]*-1

    # Plot setting.
    grid=(1,3)
    cmap1 = sns.color_palette("RdBu_r", as_cmap=True)
    cmap2 = sns.color_palette("YlOrBr", as_cmap=True)
    cmap3 = sns.color_palette("YlOrBr_r", as_cmap=True)
    cmaps = [cmap3, cmap1, cmap2]
    levels = [7, 7, 9, 5]
    vranges = [(-60, 0), (-30, 30), (0,400), (-2.5,-0.5)]
    extend = ['neither', 'neither', 'max', 'neither']
    titles = ['$\Delta P$ (%)', '$\Delta$ET (%)', 'MCWD (mm)']
    plot_setting = {'cmaps':cmaps, 'vranges':vranges, 'titles':titles, 'levels':levels, 'extend':extend}

    shp_path = r"F:\Shapefile\Amazon\sum_amazonia_polygons.shp"
    outpath = rf"E:\Thesis\AMFEdge\Figures\Description\Amazon_{dst_year}_anoMeteo.jpg"

    main(ds_array=ds_array, dst_vars=dst_vars, grid=grid, 
         plot_setting=plot_setting, shp_path=shp_path, outpath=outpath)