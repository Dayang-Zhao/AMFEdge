import osgeo
import rioxarray as rxr

import matplotlib.pyplot as plt

path = r"F:\Research\AMFEdge\CMIP6\Processed\QDM\Hist\access_cm2_WD.tif"
ds = rxr.open_rasterio(path)
print(ds)