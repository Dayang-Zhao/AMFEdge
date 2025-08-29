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
    fig.set_size_inches(cm2inch(13), cm2inch(7))

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
                    legend_kwds={"shrink": 0.9, 'orientation':'horizontal', 'location':'bottom', 'pad':0.02, 'extend':extend},
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
    fig.subplots_adjust(bottom=0.01, top=0.92, left=0.01, right=0.98, hspace=0, wspace=0.05)
    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    # Read data.
    gdf = gpd.read_file(gv.GRID_PATH)
    csv_path = r"F:\Research\AMFEdge\EdgeRH\Amazon_UndistEdge_Effect_2023.csv"
    df = pd.read_csv(csv_path)
    # df = df[(df['rh98_scale']<4000)]
    df['rh98_scale'] = df['rh98_scale']/1000
    df['rh98_magnitude'] = df['rh98_magnitude']*-1
    # df['agb_scale'] = df['agb_scale']/1000
    # df['agb_magnitude'] = df['agb_magnitude']*-1

    # Merge data.
    gdf_merged = gdf.merge(df, on="Id", how="left")
    dfs = [gdf_merged]*2
    cols = ['rh98_magnitude', 'rh98_scale']
    grid = (1, 2)

    # Plot setting.
    cmap1 = sns.color_palette(palette='summer_r', as_cmap=True)
    levels = np.arange(0, 6.01, 0.5)
    # levels = np.arange(0, 0.3, 0.05)
    # cmap1.set_over("#f8fc02")
    norm1 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap1.N)

    cmap2 = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    # levels = np.arange(40, 101, 10)
    levels = np.arange(0, 9, 1)
    # cmap2.set_over("#fc0202")
    # cmap2.set_under("#fc0202")
    norm2 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap2.N)
    cmaps = [cmap2, cmap1]
    norms = [norm2, norm1]
    extend = ['both', 'max']
    titles = ['$M_{\mathrm{RH98}}$ (m)', '$S_{\mathrm{RH98}}$ (km)']

    outpath = r"E:\Thesis\AMFEdge\Figures\RH\RH98_edge_map.jpg"
    plot_setting = {'cmaps': cmaps, 'norms': norms, 'titles': titles, 'extends': extend}
    main(dfs=dfs, grid=grid, cols=cols, plot_setting=plot_setting, outpath=outpath)
