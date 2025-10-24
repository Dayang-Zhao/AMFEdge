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
import matplotlib.colors as mcolors

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
    fig.set_size_inches(cm2inch(12), cm2inch(13))
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
            ax = axes[i, j]
            im = dst_da.plot(ax=ax, cmap=cmap, add_colorbar=True, vmin=vrange[0], vmax=vrange[1], levels=level,
                    cbar_kwargs={'label':None,'pad':0.05, 'shrink':0.9, 'aspect':20, 'extend':extend, 'orientation':'horizontal'})

            # Adjust colorbar.
            cbar = im.colorbar

            # Custom colorbar ticklabels
            if i<1:
                ticks = np.linspace(vrange[0], vrange[1], level)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(['Apr', 'Jun', 'Aug', 'Oct', 'Dec', 'Feb$^{+1}$', 'Apr$^{+1}$'],
                                    fontsize=9) 
            
            if i==1 and j==1:
                ticks = np.arange(vrange[0]+0.25, vrange[1]-0.24, 0.5)
                cbar.set_ticks(ticks)
                cbar.minorticks_off()
                cbar.set_ticklabels(['Severe', 'Medium', 'Moderate', 'Mild'], fontsize=9)


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
    fig.subplots_adjust(bottom=0.01, top=0.95, left=0.02, right=0.97, hspace=0.1, wspace=0.05)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    # Read data
    dst_year = 2023
    # Read data
    # paths = [r"F:\Research\AMFEdge\Meteo\Processed\Amazon_2023_droughtPeriod.nc",
    #          r"F:\Research\AMFEdge\Meteo\Processed\Amazon_2023_droughtPeriod.nc",
    #          r"F:\Research\AMFEdge\Meteo\Processed\Amazon_2023_droughtPeriod.nc",
    #          ]
    # ds_array = [xr.open_dataset(path).isel(time=0).drop('spatial_ref').drop('time') for path in paths]
    paths = [r"F:\Research\AMFEdge\Meteo\Meta\Amazon_2023_droughtPeriod.tif",]*3 \
        + [r"F:\Research\AMFEdge\Meteo\Meta\Amazon_2023_droughtSeverity.tif"]
    ds_array = [cd.tif2ds(path).drop('spatial_ref') for path in paths]
    ds_array[-1] = ds_array[-1].where(ds_array[-1]['anoMCWD']< -0.5).clip(min=-2.5)
    dst_vars = ['startMonth', 'endMonth', 'length', 'anoMCWD']

    # Plot setting.
    grid=(2,2)
    cmap1 = sns.color_palette("YlGn", as_cmap=True)
    cmap2 = sns.color_palette("YlOrBr", as_cmap=True)
    cmap2 =  mcolors.LinearSegmentedColormap.from_list(
        'truncated', cmap2(np.linspace(0.35, 1, 256))
    ) # Remove light colors
    cmap3 = sns.color_palette("YlOrBr_r", as_cmap=True)
    cmap3 =  mcolors.LinearSegmentedColormap.from_list(
        'truncated', cmap3(np.linspace(0, 0.65, 256))
    ) # Remove light colors
    cmaps = [cmap1, cmap1, cmap2, cmap3]
    levels = [7, 7, 5, 5]
    vranges = [(4, 16), (4, 17), (0,8), (-2.5,-0.5)]
    extend = ['neither', 'neither', 'neither', 'neither']
    titles = ['Start Month', 'End Month', 'Drought Length (months)', 'Drought severity']
    plot_setting = {'cmaps':cmaps, 'vranges':vranges, 'titles':titles, 'levels':levels, 'extend':extend}

    shp_path = r"F:\Shapefile\Amazon\sum_amazonia_polygons.shp"
    outpath = rf"E:\Thesis\AMFEdge\Figures\Description\Amazon_{dst_year}_drought_period.pdf"

    main(ds_array=ds_array, dst_vars=dst_vars, grid=grid, 
         plot_setting=plot_setting, shp_path=shp_path, outpath=outpath)