import sys
sys.path.append(r"D:\ProgramData\VistualStudioCode\AMFEdge")

import numpy as np
import pandas as pd

import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.colors as mcolors
import seaborn as sns

import GlobVars as gv

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)
CRS = ccrs.PlateCarree()
# EXTENT = [-80, -44, -21, 10]
EXTENT = [-81.5, -44, -22, 12.5]

def cm2inch(value):
    return value/2.54

def main(dfs:list, grid:tuple, cols:list, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, subplot_kw={'projection': CRS, 'frameon':True}
        )
    fig.set_size_inches(cm2inch(17), cm2inch(7))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[j]
            df = dfs[i*ncols+j]
            col = cols[i*ncols+j]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            cmap = plot_setting['cmaps'][i*ncols+j]
            norm = plot_setting['norms'][i*ncols+j]

            extend = plot_setting['extends'][i*ncols+j]

            # Plot.
            df.plot(ax=ax, column=col, cmap=cmap, edgecolor='#919191', 
                    norm=norm, linewidth=0.3, legend=True, 
                    legend_kwds={"shrink": 0.9, 'orientation':'horizontal', 'location':'bottom', 'pad':0.05, 'extend':extend},
                    missing_kwds={'color': 'white'})

            # Plugins.
            ax.set_title(title_num+' '+title, loc='left', fontsize=LABEL_SIZE+1)

            # Add Shapefile overlay if provided
            shapefile = gpd.read_file(gv.AMAZON_PATH)
            shapefile.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, zorder=-1)
            
            # Add a basemap (world map)
            ax.add_feature(cfeature.COASTLINE, edgecolor='#919191', zorder=-1)
            ax.set_extent(EXTENT, crs=CRS)

            # Remove frames and ticks
            for spine in ax.spines.values():
                spine.set_visible(False)
    # Adjust.
    fig.subplots_adjust(bottom=0.01, top=0.9, left=0.01, right=0.98, hspace=0, wspace=0.05)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    # Read data.
    scenario = 'SSP2_45'
    gdf = gpd.read_file(gv.GRID_PATH)
    csv_path = rf"F:\Research\AMFEdge\CMIP6\Predict\Mnirv_pred_{scenario}.csv"
    df = pd.read_csv(csv_path)
    df['MCWD_mean'] = df['MCWD_mean']*-1
    ave_df = df.groupby(['Id']).mean().reset_index()

    # Merge data.
    gdf_merged = gdf.merge(ave_df, on="Id", how="left")
    dfs = [gdf_merged]*3
    cols = ['total_precipitation_sum_mean', 'total_evaporation_sum_mean', 'WD_mean']
    grid = (1, 3)

    # Plot setting.
    cmap1 = sns.color_palette(palette='YlOrBr_r', as_cmap=True)
    levels = np.arange(-30, 1, 5)
    # cmap1.set_over("#f8fc02")
    norm1 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap1.N)

    cmap2 = sns.color_palette("RdBu_r", as_cmap=True)
    # levels = np.arange(-8, 9, 2)
    levels = np.arange(-20, 21, 5)
    # cmap2.set_over("#fc0202")
    # cmap2.set_under("#fc0202")
    norm2 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap2.N)
    cmaps = [cmap1, cmap2, cmap2]
    norms = [norm1, norm2, norm2]
    extend = ['both', 'both', 'both']
    # titles = [r'MCWD (mm)', r'$M_{\Delta \mathrm{NIRv}}$ (%)']
    titles = [r'$\Delta P$ (%)', r'$\Delta ET$ (%)', r'$\Delta WD$ (%)']

    outpath = rf"E:\Thesis\AMFEdge\Figures\CMIP6\PE_{scenario}_prj_map.jpg"
    plot_setting = {'cmaps': cmaps, 'norms': norms, 'titles': titles, 'extends': extend}
    main(dfs=dfs, grid=grid, cols=cols, plot_setting=plot_setting, outpath=outpath)
