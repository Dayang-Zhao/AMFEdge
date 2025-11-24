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
    fig.set_size_inches(cm2inch(17), cm2inch(14))
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
            extend = plot_setting['extend'][i*ncols+j]

            # Plot.
            ax = axes[i,j]
            im = dst_da.plot(ax=ax, x='lon', y='lat', cmap=cmap, add_colorbar=True, vmin=vrange[0], 
                             vmax=vrange[1], levels=level, 
                    cbar_kwargs={'label':None,'pad':0.02, 'shrink':0.9, 'aspect':20, 'extend':extend,
                                 "orientation": "horizontal", 'location':'bottom'},)

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
            ax.set_title(title_num+' '+title, loc='left')
            ax.legend().remove()

            # Step 6: Add Shapefile overlay if provided
            if shp_path is not None:
                shapefile = gpd.read_file(shp_path)
                shapefile.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1)

            # Remove frames and ticks
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Adjust.
    fig.subplots_adjust(bottom=0.01, top=0.97, left=0.01, right=0.98, hspace=0.04, wspace=0.05)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    # Read data
    paths = [r"F:\Research\AMFEdge\VI\Amazon_2023_S2AnoVI.tif",
             r"F:\Research\AMFEdge\VI\Amazon_2023_HLSAnoVI.tif",
             r"F:\Research\AMFEdge\VI\Amazon_2023_ModAnoVI.tif",
             ] + [r"F:\Research\AMFEdge\Meteo\Meta\Amazon_2023_anoMeteo.tif"]*3
    ds_array = []
    for path in paths:
        if path.endswith('.nc'):
            ds = xr.open_dataset(path).isel(time=0).drop('spatial_ref').drop('time')
        else:
            ds = cd.tif2ds(path).drop('spatial_ref')
        ds_array.append(ds)
    dst_vars = ['NIRv', 'NIRv', 'NIRv', 'total_precipitation_sum', 'temperature_2m', 'surface_solar_radiation_downwards_sum']
    for i in range(0,3):
        ds_array[i][dst_vars[i]] = ds_array[i][dst_vars[i]]* -1
    
    # Plot setting.
    grid=(2,3)
    cmap1 = sns.color_palette("RdBu", as_cmap=True)
    cmap2 = sns.color_palette("YlOrBr_r", as_cmap=True)
    cmap3 = sns.color_palette("YlOrBr", as_cmap=True)

    cmaps = [cmap1]*3 + [cmap2, cmap3, cmap3]
    vranges = [(-10, 10)]*3 + [(-80,0), (0,4), (0, 20)]
    levels = [9,9,9, 9,9,9]
    extend = ['both']*3 + ['neither']*3
    titles = ['Sentinel-2 $\Delta $NIRv (%)', 
              'HLS $\Delta $NIRv (%)', 'MODIS $\Delta $NIRv (%)',  '$\Delta P$ (%)', '$\Delta T$ (°C)', '$\Delta$VPD (%)']
    # titles = ['HLS $\Delta $NIRv (%)', 
    #           'Jul-Aug $\Delta $NIRv (%)', 'Sep-Nov $\Delta $NIRv (%)', 'Dec-Feb $\Delta $NIRv (%)']
    plot_setting = {'cmaps':cmaps, 'vranges':vranges, 'titles':titles, 'levels':levels, 'extend':extend}

    shp_path = r"F:\Shapefile\Amazon\sum_amazonia_polygons.shp"
    outpath = r"E:\Thesis\AMFEdge\Figures\Description\Amazon_2023_anoVI&Meteo.jpg"

    main(ds_array=ds_array, dst_vars=dst_vars, grid=grid, 
         plot_setting=plot_setting, shp_path=shp_path, outpath=outpath)