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

LABEL_SIZE = 11
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
mpl.rc('font', **font)
CRS = ccrs.PlateCarree()
# EXTENT = [-80, -44, -21, 10]
EXTENT = [-81.5, -44, -22, 12.5]
CSIGN_PREC = 0.7

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
            col, sign_col = cols[i*ncols+j]

            # Plot setting.
            title_num = title_nums[i*ncols+j]
            title = plot_setting['titles'][i*ncols+j]
            cmap = plot_setting['cmaps'][i*ncols+j]
            norm = plot_setting['norms'][i*ncols+j]

            extend = plot_setting['extends'][i*ncols+j]

            # Plot.
            if j == 1:
                df.plot(ax=ax, column=col, cmap=cmap, edgecolor='#919191', 
                        norm=norm, linewidth=0.3, legend=True, 
                        legend_kwds={"shrink": 2.2, 'aspect':40, 'orientation':'horizontal', 
                                     'location':'bottom', 'pad':0.05,'extend':extend, 'label':r'$\Delta$MCWD difference (%)'},
                        missing_kwds={'color': 'white'})
            else:
                df.plot(ax=ax, column=col, cmap=cmap, edgecolor='#919191', 
                        norm=norm, linewidth=0.3, legend=True, 
                        legend_kwds={"shrink": 0, 'aspect':40, 'orientation':'horizontal', 
                                     'location':'bottom', 'pad':0.05,'extend':extend,
                                     'ticks':[], },
                        missing_kwds={'color': 'white'})
                
            # Add dots for grids with sign consistency above threshold
            sig = df[df[sign_col]>=0.7].copy()
            sig["x"] = sig.geometry.centroid.x
            sig["y"] = sig.geometry.centroid.y

            ax.scatter(sig["x"], sig["y"], s=2, c="grey", marker=".", transform=CRS, zorder=10, facecolors=None)

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
    scenarios = ['SSP1_26', 'SSP2_45', 'SSP5_85']
    gdf = gpd.read_file(gv.GRID_PATH)
    dfs = []

    path = r"F:\Research\AMFEdge\CMIP6\Predict\diff_2015@2090.csv"
    df = pd.read_csv(path)
    for scenario in scenarios:
        scenario_df = df[df['scenario']==scenario].copy().drop(columns=['scenario'])
        dfs.append(scenario_df)

    # Merge data.
    gdfs_merged = [gdf.merge(ave_df, on="Id", how="left") for ave_df in dfs]
    cols = [('dMCWD_mean', 'diff_sign_prec_MCWD_mean'),]*3
    grid = (1, 3)

    # Plot setting.
    cmap2 = sns.color_palette("RdBu_r", as_cmap=True)
    levels = np.arange(-30, 30.01, 5)
    # levels = np.arange(-8, 8.01, 1)
    norm2 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap2.N)
    cmaps = [cmap2]*3
    norms = [norm2]*3
    extend = ['both', 'both', 'both']
    titles = ['SSP1-2.6', 'SSP2-4.5', 'SSP5-8.5']

    outpath = rf"E:\Thesis\AMFEdge\Figures\CMIP6\dMCWD_prj_map.pdf"
    plot_setting = {'cmaps': cmaps, 'norms': norms, 'titles': titles, 'extends': extend}
    main(dfs=gdfs_merged, grid=grid, cols=cols, plot_setting=plot_setting, outpath=outpath)
