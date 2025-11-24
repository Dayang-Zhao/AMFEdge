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
    path1 = r"F:\Research\AMFEdge\EdgeRH\RH_Amazon_Edge_Effect_2023.csv"
    df1 = pd.read_csv(path1)
    path2 = r"F:\Research\AMFEdge\EdgeRH\RH_Amazon_Edge_2023.csv"
    df2 = pd.read_csv(path2)
    df2 = df2[df2['Dist']==60]

    # Merge data.
    gdf_merged1 = gdf.merge(df1, on="Id", how="left")
    gdf_merged2 = gdf.merge(df2, on="Id", how="left")
    dfs = [gdf_merged2, gdf_merged1]
    # cols = ['rh50_magnitude', 'rh50_scale']
    cols = ['rh98_mean','rh98_para3']
    grid = (1, 2)

    # Plot setting.
    cmap2 = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    levels1 = np.arange(10, 31, 2)
    levels2 = np.arange(20, 33, 2)
    norm1 = mcolors.BoundaryNorm(boundaries=levels1, ncolors=cmap2.N)
    norm2 = mcolors.BoundaryNorm(boundaries=levels2, ncolors=cmap2.N)
    cmaps = [cmap2, cmap2]
    norms = [norm1, norm2]
    extend = ['both', 'max']
    titles = ['$M_{\mathrm{RH98}}$ (m)', '$S_{\mathrm{RH98}}$ (km)']

    outpath = r"E:\Thesis\AMFEdge\Figures\EdgeRH\RH98_edge_map.pdf"
    plot_setting = {'cmaps': cmaps, 'norms': norms, 'titles': titles, 'extends': extend}
    main(dfs=dfs, grid=grid, cols=cols, plot_setting=plot_setting, outpath=outpath)
