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

LABEL_SIZE = 10
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
    fig.set_size_inches(cm2inch(17), cm2inch(13))
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

            # Plot.
            ax = axes[i,j]
            im = dst_da.plot(ax=ax, cmap=cmap, add_colorbar=True, vmin=vrange[0], vmax=vrange[1], levels=level,
                    cbar_kwargs={'label':None,'pad':0.05, 'shrink':0.8, 'aspect':20, 'extend':'neither'})

            # Adjust colorbar.
            cbar = im.colorbar

            # Custom colorbar ticklabels
            if i==0:
                ticks = np.linspace(vrange[0], vrange[1], level)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(['Apr', 'Jun', 'Aug', 'Oct', 'Dec', 'Feb$^{+1}$', 'Apr$^{+1}$']) 


            # Add a basemap (world map)
            ax.add_feature(cfeature.COASTLINE, edgecolor='#919191')
            ax.set_extent(EXTENT, crs=CRS)

            # Plot plugins.
            ax.set_title(title_num+' '+title, loc='left')

            # Step 6: Add Shapefile overlay if provided
            if shp_path is not None:
                shapefile = gpd.read_file(shp_path)
                shapefile.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1)

    # Adjust.
    fig.subplots_adjust(bottom=0.02, top=0.96, left=0.02, right=0.97, hspace=0.1, wspace=0.05)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    # Read data
    dst_year = 2023
    # Read data
    paths = [r"F:\Research\AMFEdge\Meteo\Processed\Amazon_2023_droughtPeriod.nc",
             r"F:\Research\AMFEdge\Meteo\Processed\Amazon_2023_droughtPeriod.nc",
             r"F:\Research\AMFEdge\Meteo\Processed\Amazon_2023_droughtPeriod.nc",
             r"F:\Research\AMFEdge\Meteo\Processed\Amazon_2023_rel_anoMCWD.nc"
             ]
    ds_array = [xr.open_dataset(path).isel(time=0).drop('spatial_ref').drop('time') for path in paths]
    dst_vars = ['startMonth', 'endMonth', 'length', 'anoMCWD']

    # Plot setting.
    grid=(2,2)
    cmap1 = sns.color_palette("YlGn", as_cmap=True)
    cmap2 = sns.color_palette("YlOrBr", as_cmap=True)
    cmap3 = sns.color_palette("YlOrBr_r", as_cmap=True).reversed()
    cmaps = [cmap1, cmap1, cmap2, cmap3]
    levels = [7, 7, 5, 5]
    vranges = [(4, 16), (4, 17), (0,8), (0,80)]
    titles = ['Start Month', 'End Month', 'Drought Length (months)', '$\Delta$MCWD (%)']
    plot_setting = {'cmaps':cmaps, 'vranges':vranges, 'titles':titles, 'levels':levels}

    shp_path = r"F:\Shapefile\Amazon\sum_amazonia_polygons.shp"
    outpath = f"E:\Thesis\AMFEdge\Figures\Description\Amazon_{dst_year}_drought_period.jpg"

    main(ds_array=ds_array, dst_vars=dst_vars, grid=grid, 
         plot_setting=plot_setting, shp_path=shp_path, outpath=outpath)