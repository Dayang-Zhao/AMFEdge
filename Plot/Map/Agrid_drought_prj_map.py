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
    fig.set_size_inches(cm2inch(17), cm2inch(12))

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
    fig.subplots_adjust(bottom=0.01, top=0.95, left=0.05, right=0.98, hspace=0.1, wspace=0.05)

    if outpath is not None:
        fig.savefig(outpath, dpi=600)

if __name__ == "__main__":
    # Read data.
    scenarios = ['SSP1_26', 'SSP2_45', 'SSP5_85']
    gdf = gpd.read_file(gv.GRID_PATH)
    ave_dfs = []
    count_dfs = []
    for scenario in scenarios:
        csv_path = rf"F:\Research\AMFEdge\CMIP6\Predict\Mnirv_pred_{scenario}.csv"
        df = pd.read_csv(csv_path)
        df = df[df['year']>=2030]
        df['MCWD_mean'] = df['MCWD_mean']*-1
        ave_df = df.groupby(['Id']).mean().reset_index()
        ave_dfs.append(ave_df)
        count_df = df.groupby(['Id']).count().reset_index()
        count_dfs.append(count_df)

    # Merge data.
    gdfs_merged = [gdf.merge(count_df, on="Id", how="left") for count_df in count_dfs] \
        + [gdf.merge(ave_df, on="Id", how="left") for ave_df in ave_dfs]
    cols = ['MCWD_mean',]*6
    grid = (2, 3)

    # Plot setting.
    cmap1 = sns.color_palette("flare", as_cmap=True)
    levels = np.arange(0, 71, 10)
    norm1= mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap1.N)

    cmap2 = sns.color_palette(palette='YlOrBr', as_cmap=True)
    levels = np.arange(0, 401, 50)
    norm2 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap2.N)

    cmaps = [cmap1]*3+[cmap2]*3
    norms = [norm1]*3+[norm2]*3
    extend = ['both']*6
    titles = ['']*6

    outpath = rf"E:\Thesis\AMFEdge\Figures\CMIP6\drought_prj_map.pdf"
    plot_setting = {'cmaps': cmaps, 'norms': norms, 'titles': titles, 'extends': extend}
    main(dfs=gdfs_merged, grid=grid, cols=cols, plot_setting=plot_setting, outpath=outpath)
