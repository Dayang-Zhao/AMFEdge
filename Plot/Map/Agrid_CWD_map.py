import os
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
    fig.set_size_inches(cm2inch(15), cm2inch(11))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i,j]
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
                    legend_kwds={"shrink": 0.9, 'orientation':'horizontal', 'location':'bottom', 'pad':0.02, 'extend':'max'},
                    missing_kwds={'color': 'white'})

            # Plugins.
            ax.set_title(title_num+' '+title, loc='left', fontsize=LABEL_SIZE+1)

            # Add Shapefile overlay if provided
            shapefile = gpd.read_file(gv.AMAZON_PATH)
            shapefile.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, zorder=-1)
            
            # Add a basemap (world map)
            ax.add_feature(cfeature.COASTLINE, edgecolor='#919191', zorder=-1)
            ax.set_extent(EXTENT, crs=CRS)

            # Set axes width.
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
    # Adjust.
    fig.subplots_adjust(bottom=0.01, top=0.92, left=0.01, right=0.98, hspace=0.1, wspace=0.05)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    # Read data.
    gdf = gpd.read_file(gv.GRID_PATH)
    offsets = [1, 2, 3, 4, 5]
    root_dir = r"F:\Research\AMFEdge\EdgeOnset"
    os.chdir(root_dir)
    prefname = "anoVI_Amazon_UndistEdge_"
    paths = [prefname + str(offset) + 'M_2023.csv' for offset in offsets]
    df_list = [pd.read_csv(path) for path in paths]
    df_list = [pd.read_csv('anoVI_Amazon_UndistEdge_2023.csv')]+ df_list
    dst_dfs = [df.loc[df['Dist']==-1] for df in df_list]

    # Merge data.
    gdfs_merged = [gdf.merge(dst_df, on="Id", how="left") for dst_df in dst_dfs]
    cols = ['anoCWD_mean']*6
    grid = (2, 3)

    # Plot setting.
    cmap1 = sns.color_palette("RdBu_r", as_cmap=True) #YlOrBr
    levels = np.arange(-100, 101, 20)
    norm1 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap1.N)
    cmaps = [cmap1]*6
    norms = [norm1]*6
    titles = ['0M', '1M', '2M', '3M', '4M', '5M']

    outpath = r"E:\Thesis\AMFEdge\Figures\EdgeOnset\anoCWD_map.jpeg"
    plot_setting = {'cmaps': cmaps, 'norms': norms, 'titles': titles}
    main(dfs=gdfs_merged, grid=grid, cols=cols, plot_setting=plot_setting, outpath=outpath)
