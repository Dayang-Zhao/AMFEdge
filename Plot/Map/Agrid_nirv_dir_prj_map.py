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
            if j == 1:
                df.plot(ax=ax, column=col, cmap=cmap, edgecolor='#919191', 
                        norm=norm, linewidth=0.3, legend=True, 
                        legend_kwds={"shrink": 2.2, 'aspect':40, 'orientation':'horizontal', 
                                     'location':'bottom', 'pad':0.05,'extend':extend, 'label':r' $M_{\Delta \mathrm{NIRv}}$ (%)'},
                        missing_kwds={'color': 'white'})
            else:
                df.plot(ax=ax, column=col, cmap=cmap, edgecolor='#919191', 
                        norm=norm, linewidth=0.3, legend=True, 
                        legend_kwds={"shrink": 0, 'aspect':40, 'orientation':'horizontal', 
                                     'location':'bottom', 'pad':0.05,'extend':extend,
                                     'ticks':[], },
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
    scenarios = ['SSP1_26', 'SSP2_45', 'SSP5_85']
    gdf = gpd.read_file(gv.GRID_PATH)
    dfs = []

    def cal_diff(df, dst_var):
        df_start = df[df['year'].isin(range(2021, 2031))].groupby('Id')[dst_var].mean().reset_index()
        df_end = df[df['year'].isin(range(2081, 2101))].groupby('Id')[dst_var].mean().reset_index()
        df_merged = pd.merge(df_start, df_end, on='Id', suffixes=('_start', '_end'))

        # Label the change direction.
        df_merged["label"] = 1  # 默认值
        df_merged.loc[(df_merged[f"{dst_var}_start"] > 0) & (df_merged[f"{dst_var}_end"] > 0), "label"] = 4
        df_merged.loc[(df_merged[f"{dst_var}_start"] < 0) & (df_merged[f"{dst_var}_end"] < 0), "label"] = 3
        df_merged.loc[(df_merged[f"{dst_var}_start"] > 0) & (df_merged[f"{dst_var}_end"] < 0), "label"] = 2

        return df_merged

    for scenario in scenarios:
        csv_path = rf"F:\Research\AMFEdge\CMIP6\Predict\QDM\Mnirv_Edge_pred_{scenario}.csv"
        df = pd.read_csv(csv_path)
        df2 = df.drop(columns=['model'])
        df2 = df2.groupby(['Id', 'year']).mean().reset_index()
        ave_df = cal_diff(df2, 'nirv_magnitude')
        # ave_df = df2.groupby(['Id']).mean().reset_index()
        dfs.append(ave_df)

    # Merge data.
    gdfs_merged = [gdf.merge(ave_df, on="Id", how="left") for ave_df in dfs]
    cols = ['label',]*3
    grid = (1, 3)

    # Plot setting.
    cmap2 = sns.color_palette("hls", 8, as_cmap=True)
    levels = np.arange(0, 5.01, 1)
    # levels = np.arange(-16, 17, 4)
    norm2 = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap2.N)
    cmaps = [cmap2]*3
    norms = [norm2]*3
    extend = ['neither']*3
    titles = ['RCP 2.6', 'RCP 4.5', 'RCP 8.5']

    outpath = rf"E:\Thesis\AMFEdge\Figures\CMIP6\nirv_dir_prj_map.pdf"
    plot_setting = {'cmaps': cmaps, 'norms': norms, 'titles': titles, 'extends': extend}
    main(dfs=gdfs_merged, grid=grid, cols=cols, plot_setting=plot_setting, outpath=outpath)
