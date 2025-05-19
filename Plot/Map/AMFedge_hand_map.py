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

LABEL_SIZE = 12
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
matplotlib.rc('font', **font)
CRS = ccrs.PlateCarree()
EXTENT = [-80, -44, -21, 9]

def cm2inch(value):
    return value/2.54

def main(ds_array:list, dst_vars:list, grid:tuple, plot_setting: dict, 
         shp_path:str, outpath:str=None):

    nrows, ncols = grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex='col', sharey='row',
        subplot_kw={'projection': CRS, 'frameon':True}
        )
    fig.set_size_inches(cm2inch(8), cm2inch(6))
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']

    for i in range(nrows):
        for j in range(ncols):
            ds = ds_array[i*ncols+j].copy()
            dst_var = dst_vars[i*ncols+j]
            title_num = title_nums[i*ncols+j]
            dst_da = ds[dst_var]

            # Plot setting.
            vrange = plot_setting['vranges'][i*ncols+j]
            # cbar_label = plot_setting['cbar_labels'][i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            cmap = plot_setting['cmaps'][i*ncols+j]
            level = plot_setting['levels'][i*ncols+j]

            # Plot.
            ax = axes
            im = dst_da.plot(ax=ax, x='lon', y='lat', cmap=cmap, add_colorbar=True, vmin=vrange[0], 
                             vmax=vrange[1], levels=level, 
                    cbar_kwargs={'label':'Hand (m)','pad':0.02, 'shrink':0.9, 'aspect':20, 'extend':'neither',
                                 "orientation": "vertical"})

            # Adjust colorbar.
            cbar = im.colorbar
            # # Custom colorbar ticklabels
            ticks = np.linspace(vrange[0], vrange[1], 5)
            cbar.set_ticks(ticks)
            # cbar.set_label(title, fontsize=LABEL_SIZE, labelpad=0.5)

            # Add a basemap (world map)
            ax.add_feature(cfeature.COASTLINE, edgecolor='#919191')
            ax.set_extent(EXTENT, crs=CRS)

            # Plot plugins.
            # ax.set_title(title_num+' '+title, loc='left')
            ax.legend().remove()

            # Step 6: Add Shapefile overlay if provided
            if shp_path is not None:
                shapefile = gpd.read_file(shp_path)
                shapefile.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1)

    # Adjust.
    fig.subplots_adjust(bottom=0.01, top=0.96, left=0.01, right=0.98, hspace=0, wspace=0.05)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    # Read data
    paths = ["F:\Research\AMFEdge\Environment\Amazon_CopernicusHand.nc"]
    ds = xr.open_dataset(paths[0]).drop('spatial_ref')
    ds = ds.where(ds['HAND']<65)
    dst_vars = ['HAND']
    ds_array = [ds]
    
    # Plot setting.
    grid=(1,1)
    cmap = sns.color_palette("Spectral_r", as_cmap=True)

    cmaps = [cmap]
    vranges = [(0, 40)]
    levels = [9,9,9]
    titles = ['$\delta $VOD (10$^{-3}$/yr)', '$\delta $EVI (10$^{-3}$/yr)', '$\delta $SIF (10$^{-2}$/yr)']
    plot_setting = {'cmaps':cmaps, 'vranges':vranges, 'titles':titles, 'levels':levels}

    shp_path = r"F:\Shapefile\Amazon\sum_amazonia_polygons.shp"
    outpath = "E:\Thesis\AMFEdge\Figures\Description\hand_map.pdf"

    main(ds_array=ds_array, dst_vars=dst_vars, grid=grid, 
         plot_setting=plot_setting, shp_path=shp_path, outpath=outpath)