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
import matplotlib.colors as mcolors
plt.ion()
import seaborn as sns

import GlobVars
import Data.convert_data as cd
import Data.clip_data as cld

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
matplotlib.rc('font', **font)
CRS = ccrs.PlateCarree()
EXTENT = [-81.5, -44, -22, 12.5]

def preprocess(ds:xr.Dataset, shp) -> xr.Dataset:
  
    ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')\
        .rio.write_crs("EPSG:4326")
    cds = cld.mask_ds_by_shp(ds=ds, shp=shp)
    cds['min_spei'] = (cds['min_spei']*-1).clip(max=3)
    cds['endMonth'] = cds['endMonth'].clip(max=16)

    return cds.drop('spatial_ref')

def cm2inch(value):
    return value/2.54

def main(ds_array:list, dst_vars:list, grid:tuple, plot_setting: dict, 
         shp_path:str, outpath:str=None):

    nrows, ncols = grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex='col', sharey='row',
        subplot_kw={'projection': CRS, 'frameon':True}
        )
    fig.set_size_inches(cm2inch(14), cm2inch(14))
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

            # Custom colorbar ticklabels
            if i<1:
                im = dst_da.plot(ax=ax, cmap=cmap, add_colorbar=True, vmin=vrange[0], vmax=vrange[1], levels=level,
                    cbar_kwargs={'label':None,'pad':0.05, 'shrink':0.9, 'aspect':20, 'extend':extend, 'orientation':'horizontal'})
                
                # Adjust colorbar.
                ticks = np.linspace(vrange[0], vrange[1], level)
                cbar = im.colorbar
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(['Apr', 'Jun', 'Aug', 'Oct', 'Dec', 'Feb$^{+1}$', 'Apr$^{+1}$'],
                                    fontsize=9) 
            if i==1 and j!=1:
                im = dst_da.plot(ax=ax, cmap=cmap, add_colorbar=True, vmin=vrange[0], vmax=vrange[1], levels=level,
                    cbar_kwargs={'label':None,'pad':0.05, 'shrink':0.9, 'aspect':20, 'extend':extend, 'orientation':'horizontal'})
            
            if i==1 and j==1:
                boundary = [0.5, 0.8, 1.3, 1.6, 2.0, 3.0]
                im = dst_da.plot(
                    ax=ax, x='lon', y='lat', cmap=cmap, 
                    norm=matplotlib.colors.BoundaryNorm(boundary, cmap1.N, clip=True),
                    add_colorbar=True, 
                    cbar_kwargs={
                        'pad': 0.02, 'shrink': 0.95,
                        'aspect': 20, 'extend': 'neither',
                        'orientation': 'horizontal',
                        'ticks': boundary,
                        'label':None,
                    }
                )
                cbar = im.colorbar
                ticks = [0.65, 1.05, 1.45, 1.8, 2.5]
                cbar.set_ticks(ticks)
                cbar.minorticks_off()
                cbar.set_ticklabels(['Mild', 'Moderate', 'Medium', 'Severe', 'Extreme'], fontsize=9)


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
    fig.subplots_adjust(bottom=0.01, top=0.95, left=0.02, right=0.98, hspace=0.1, wspace=0.03)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == '__main__':
    # Read data
    dst_year = 2023
    # Read data
    paths = [r"F:\Research\AMFEdge\Meteo\Processed\Amazon_ERA5_2023_drought_spei_gamma_03.nc"]*4
    ds_array = [xr.open_dataset(path).drop('spatial_ref') for path in paths]
    shp_path = r"F:\Shapefile\Amazon\sum_amazonia_polygons.shp"
    ds_array = [preprocess(ds, gpd.read_file(shp_path)) for ds in ds_array]
    dst_vars = ['startMonth', 'endMonth', 'length', 'min_spei']

    # Plot setting.
    grid=(2,2)
    cmap1 = sns.color_palette("YlGn", as_cmap=True)
    cmap2 = sns.color_palette("YlOrBr", as_cmap=True)
    cmap2 =  mcolors.LinearSegmentedColormap.from_list(
        'truncated', cmap2(np.linspace(0.35, 1, 256))
    ) # Remove light colors
    cmaps = [cmap1, cmap1, cmap2, cmap2]
    levels = [7, 7, 5, 5]
    vranges = [(4, 16), (4, 17), (0,8), (0.5, 3)]
    extend = ['neither', 'neither', 'neither', 'neither']
    titles = ['Start Month', 'End Month', 'Drought Length (months)', 'Drought severity']
    plot_setting = {'cmaps':cmaps, 'vranges':vranges, 'titles':titles, 'levels':levels, 'extend':extend}

    shp_path = r"F:\Shapefile\Amazon\sum_amazonia_polygons.shp"
    outpath = rf"E:\Thesis\AMFEdge\Figures\Meteo\Amazon_ERA5_2023_spei_gamma_03_dPeriod.jpg"

    main(ds_array=ds_array, dst_vars=dst_vars, grid=grid, 
         plot_setting=plot_setting, shp_path=shp_path, outpath=outpath)