import sys
sys.path.append(r"D:\ProgramData\VistualStudioCode\AMFEdge")

import pandas as pd

import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

import GlobVars as gv

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)
CRS = ccrs.PlateCarree()
EXTENT = [-80, -44, -21, 10]

def cm2inch(value):
    return value/2.54

def main(dfs:list, grid:tuple, cols:list, plot_setting:dict, outpath:str):
    title_nums = ['a', 'b', 'c', 'd', 'e', 'f']
    nrows, ncols = grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, subplot_kw={'projection': CRS, 'frameon':True}
        )
    fig.set_size_inches(cm2inch(15), cm2inch(6))

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

            # Plot.
            df.plot(ax=ax, column=col, cmap=cmap, edgecolor='#919191', 
                    norm=norm, linewidth=0.3, legend=True, 
                    legend_kwds={"shrink": 0.8, 'extend':'max'},missing_kwds={'color': 'white'})

            # Plugins.
            ax.set_title(title_num+' '+title, loc='left', fontsize=LABEL_SIZE+1)

            # Add Shapefile overlay if provided
            shapefile = gpd.read_file(gv.AMAZON_PATH)
            shapefile.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, zorder=-1)
            
            # Add a basemap (world map)
            ax.add_feature(cfeature.COASTLINE, edgecolor='#919191', zorder=-1)
            ax.set_extent(EXTENT, crs=CRS)
    # Adjust.
    fig.subplots_adjust(bottom=0.01, top=0.96, left=0.01, right=0.98, hspace=0, wspace=0.05)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    # Read data.
    gdf = gpd.read_file(gv.GRID_PATH)
    csv_path = r"F:\Research\AMFEdge\TreeCover\frag_Amazon_2023_dist95.csv"
    df = pd.read_csv(csv_path)
    cols = ['undistForest_dist', 'deforestLand_dist']
    df[cols] = df[cols]/1000

    # Merge data.
    gdf_merged = gdf.merge(df, on="Id", how="left")
    dfs = [gdf_merged]*2
    grid = (1, 2)

    # Plot setting.
    cmap1 = sns.color_palette(palette='summer_r', as_cmap=True)
    levels = np.arange(0, 6.1, 0.5)
    norm1 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap1.N)
    cmap2 = sns.color_palette(palette='summer_r', as_cmap=True)
    levels = np.arange(0, 3.1, 0.5)
    norm2 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap2.N)
    cmaps = [cmap1, cmap2]
    norms = [norm1, norm2]
    titles = ['$d_{95}$ (km)', '$d_{95}$ (km)']

    outpath = r"E:\Thesis\AMFEdge\Figures\Description\Amzon_forestFrag_Dist95p_map.pdf"
    plot_setting = {'cmaps': cmaps, 'norms': norms, 'titles': titles}
    main(dfs=dfs, grid=grid, cols=cols, plot_setting=plot_setting, outpath=outpath)
