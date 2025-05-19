import numpy as np

import osgeo
import xarray as xr
import rioxarray
import geopandas as gpd

import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns

LABEL_SIZE = 10
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : LABEL_SIZE}
matplotlib.rc('font', **font)
def cm2inch(value):
    return value/2.54

def main(data:xr.DataArray, cmap:str, shp_path:str, outpath:str=None):

    # Step 3: Set up the figure and axes
    fig, ax = plt.subplots(figsize=(cm2inch(12), cm2inch(7.5)))
    # fig, ax = plt.subplots(figsize=(cm2inch(12), cm2inch(9)))

    # Step 4: Plot the TIFF data
    # You may adjust the cmap as needed
    data.plot(ax=ax, cmap=cmap, add_colorbar=True, vmin=-3, vmax=3,
              cbar_kwargs={'label':None,'pad':0.05, 'shrink':0.8, 'aspect':20, 'extend':'both'})
    # data.plot(ax=ax, cmap=cmap, add_colorbar=False)

    # Step 5: Add a basemap (world map)
    # Using GeoPandas to plot world boundaries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.boundary.plot(ax=ax, color='black')

    # Optionally, adjust the limits if needed
    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])

    # Step 6: Show the plot
    # plt.title('2023 Amazon $\delta$MCWD')
    # plt.title('2023 Amazon $\Delta$NIRv')
    # plt.title('2023 Amazon intact forest')
    plt.title('2023 Amazon $\delta$Ta')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Step 6: Add Shapefile overlay if provided
    if shp_path is not None:
        shapefile = gpd.read_file(shp_path)
        shapefile.plot(ax=ax, edgecolor='#919191', facecolor='#919191', alpha=0.5, linewidth=1)

    # Adjust.
    fig.subplots_adjust(bottom=0.06, top=0.97, left=0.14, right=0.99, hspace=0, wspace=0)

    if outpath is not None:
        plt.savefig(outpath, dpi=600)


if __name__ == '__main__':
    # Read data
    path = r"F:\Research\TropicalForestEdge\Test\Amazon_2023_anoTa.tif"
    tiff_data = rioxarray.open_rasterio(path)
    band_data = tiff_data[0]
    band_data = band_data.where(band_data)
    cmap = sns.color_palette("vlag", as_cmap=True)
    # from matplotlib.colors import ListedColormap
    # cmap = ListedColormap(['none', '#4F845C'])


    shp_path = r"E:\Thesis\TropicalForestEdge\Sources\DegradationTropicalMoistForests-main\AncillaryData\FilterGrid15Final.shp"
    outpath = r'E:\Thesis\TropicalForestEdge\Figures\Test\Amazon_2023_anoTa.tif'
    main(data=band_data, cmap=cmap, shp_path=shp_path, outpath=outpath)