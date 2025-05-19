# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""This module is to grid 1-dimension array into 2-dimension array.
"""

from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import xarray as xr

def round_center(
        data_lat:np.ndarray, data_lon:np.ndarray, spatial_res:tuple, decimal:int
    )->tuple:
    """Round lattitude and longitude of data to match grids.
        The latitude and longitude of data is calculated as 
        the following equations:
            grid = round((center - resolution/2)/res)*res + res/2

    Args:
        data_lat (np.ndarray): The latitude of data points.
        data_lon (np.ndarray): The longitude of data points.
        spatial_res (tuple): (res_lon, res_lat).

    Returns:
        tuple: grid_lat, grid_lon. The latitudes and longitudes of grid centers.

    Example:
    >>> data_lat = array([-10.33930583,  30.55377922,  -1.57197333,   4.20853416,
       -39.90412073,  45.67128851, -34.04700076,  66.9441159 ,
       -72.85877139,  57.06709881])
       data_lon = array([ -17.29300649,  169.93623766,  -59.31539412,   12.88335467,
         96.98479532,  -70.64994824,  -35.36556908,  -54.18073119,
        -17.13694286, -112.27658587])
        spatial_res = 0.1
       center = round_center(data_lat, data_lon, spatial_res)

    """
    # Get the spatial resolution of latitutal and longitutal axis.
    res_lon, res_lat = spatial_res

    center_lat = np.around(
        (data_lat - res_lat * 0.5) / res_lat
    ) * res_lat + res_lat*0.5
    
    center_lon = np.around(
        (data_lon - res_lon * 0.5) / res_lon
    ) * res_lon + res_lon * 0.5

    # lons = np.arange(-180, 180.05, 0.05)
    # lats = np.arange(-90, 90.05, 0.05)
    # lon_label = np.arange(-179.975, 180, 0.05)
    # lat_label = np.arange(-89.975, 90, 0.05)

    # In some cases (due to np.around(180.0)=180.05 in arrays), 
    # therefore when grid_lon>180, the grid_lon would be fixed 180.0.
    center_lon[center_lon > 180.0] = 180.0
    center_lon[center_lon < -180.0] = -180.0

    return center_lon.round(decimal), center_lat.round(decimal)

def encode_center(center_lat:np.ndarray, center_lon:np.ndarray, decimal:int)->np.ndarray:
    """Encode center according to its latitude and longitude.
    The code consists of 10 chars and is seperated into two parts:
    1. The first five chars represent latitude.
    2. The last five chars represent longitude.
    e.g., '0018501175' represent the center of 1.85 degree latitude
    and 11.75 degree longitude.

    Args:
        center_lat (n-d, numpy array): The latitude of centers.
        grid_lon (n-d, numpy array): The longitude of centers.

    Returns:
        n-d, numpy array: The code of centers.

    Notes:
        1. The shape of center_lat and center_lon must be same.
        The decimal of center_lat and center_lon can not excceed 2,
        otherwise, the excceeded parts would be ignored.
        e.g., 11.082 will be encoded as '01108'.

    Examples:
    >>> data_lat = array([-10.33930583,  30.55377922,  -1.57197333,   4.20853416,
       -39.90412073,  45.67128851, -34.04700076,  66.9441159 ,
       -72.85877139,  57.06709881])
       data_lon = array([ -17.29300649,  169.93623766,  -59.31539412,   12.88335467,
         96.98479532,  -70.64994824,  -35.36556908,  -54.18073119,
        -17.13694286, -112.27658587])
       grid_code = encode_grid(data_lat, data_lon)
    array(['001035001725', '103055116995', '000155005935', '100425101285',
       '003995109695', '104565007065', '003405003535', '106695005415',
       '007285001714', '105705011225'], dtype='<U10')
    """
    # Absolute value.
    center_lat_absolute = np.absolute(center_lat)
    center_lon_absolute = np.absolute(center_lon)

    # Encode.
    grid_code = np.char.add(
        np.char.zfill(
            (center_lat_absolute*(10**decimal) + (center_lat>0)*(10**(decimal+3)))
            .round(0).astype(np.int0).astype(str), (decimal+4)),
        np.char.zfill(
            (center_lon_absolute*(10**decimal) + (center_lon>0)*(10**(decimal+3)))
            .round(0).astype(np.int0).astype(str), (decimal+4))
        ).astype(str)

    return grid_code

def build_gcode_df(data_value:np.ndarray, data_gcode:np.ndarray)->pd.DataFrame:
    """Build dataframe with gridcode and values.

    Args:
        data_value (1d, np.ndarray): The value  of the variable.
        data_gcode (1d, np.ndarray): The gridcode  of the variable

    Returns:
        pd.DataFrame: the dataframe with gridcode and values.
    """
    """This function is to encode the grids where data points are located
    and to produce dataframe with grid code as index.

    Args:
        data_variable: 1-d numpy array
            The value array of the variable.
        data_gcode: 1-d np.ndarray
    
    Returns: 
        outputs: pandas DataFrame
            The output data with grid codes.

    Example:
    >>> data_value = array([0.50243323, 0.56395633, 0.16932973, 0.48009267, 0.79285444,
       0.7138168 , 0.41463829, 0.94008691, 0.53794628, 0.09697022])

        data_gcode = array([001035001725, 103055116995, 000155005935, 100425101285
        003995109695, 104565007065, 003405003535, 106695005415, 007285001714, 105705011225])      

        gridded_df = build_gcode_df(data_value, data_gcode)

    >>> GridCode              
        001035001725  0.502433
        103055116995  0.563956
        000155005935  0.169330
        100425101285  0.480093
        003995109695  0.792854
        104565007065  0.713817
        003405003535  0.414638
        106695005415  0.940087
        007285001714  0.537946
        105705011225  0.096970
    """

    # Build a pandas dataframe.
    output = pd.DataFrame(
        data=data_value,
        index=data_gcode
    )
    output.index.name = 'GridCode'
    
    return output

def decode_center(gridcode:np.ndarray, decimal:int)->tuple:
    """Generate the latitude and longitude based on the gridcode.

    Args:
        gridcode (1d, np.ndarray): The codes of grids.

    Returns:
        tuple (np.ndarray, np.ndarray): The latitudes and longitudes of grids.

    Example:
    >>> gridcode = np.array(
        ['100175000175', '100275000125', '000175100175', '000275100125', '100200000175']
        ) 
        grid_lat, grid_lon = decode_center(gridcode)
    >>> np.array([1.75, 2.75, -1.75, -2.75, 2.00])
        np.array([-1.75, -1.25, 1.75, 1.25, -1.75])
    """
    gridcode_num = gridcode.astype(np.int0)
    
    # lat
    lat_part = np.around(gridcode_num/(10**(decimal+4)))
    lat_np = np.around(lat_part/(10**(decimal+3)))
    lat_value = lat_part - lat_np*(10**(decimal+3))
    grid_lat = (lat_np-0.5)*2*(lat_value/(10**(decimal)))

    # lon
    lon_part = gridcode_num - lat_part*(10**(decimal+4))
    lon_np = np.around(lon_part/(10**(decimal+3)))
    lon_value = lon_part - lon_np*(10**(decimal+3))
    grid_lon = (lon_np-0.5)*2*(lon_value/(10**(decimal)))

    return grid_lat, grid_lon

# ----------------------- Main--------------------------------------
def grid_data(
        data_value:dict, data_lon:np.ndarray, data_lat:np.ndarray, geoTransform:tuple, 
        decimal:int, compfunc
        )->xr.Dataset:
    """Convert 1d data with latitude and longitude into 2d.

    Args:
        data_value (dict): The value of data points, {name:value,...}.
        data_lon (np.ndarray): The longitude of data points.
        data_lat (np.ndarray): The latitude of data points.
        geoTransform (tuple): (ul_corner_lon, br_corner_lon, res_lon, ul_corner_lat, br_corner_lat, res_lat)
        decimal (int): decimals of resulting longitude and latitude.
        compfunc (function): The function to composite value within the same grid, 
        and the variable of the function was dataframe with gcode index.

    Returns:
        xr.DataArray: 2d data array with dims=['lat','lon'].

    Examples:
    >>> nc_path = r"F:\SIF\OCO3\oco3_LtSIF_220630_B10314r_221003210913s.nc4"
        # Read data.
        dataset = xr.open_dataset(nc_path)
        data_value = {'time': dataset['TIME'].values, 'SIF_740nm':dataset['SIF_740nm'].values}
        data_lon = dataset['Longitude'].values
        data_lat = dataset['Latitude'].values

        # Set composite function.
        def csum(df:pd.DataFrame)->np.ndarray:
            cdf = df.groupby('GridCode').mean()

            return cdf
        
        geoTransform = (-180, 180, 0.1, 90, -90, 0.1)

        da = grid_data(
            data_value=data_value, data_lat=data_lat, data_lon=data_lon, 
            geoTransform=geoTransform, compfunc=csum
        )
    """
    # Encode data.
    spatial_res = (geoTransform[2], geoTransform[5])
    center_lon, center_lat = round_center(data_lat=data_lat, data_lon=data_lon, 
                                          spatial_res=spatial_res, decimal=decimal)
    gcode = encode_center(center_lat=center_lat, center_lon=center_lon, decimal=decimal)

    # Composite data according to grid code.
    df4gcode = build_gcode_df(data_value=data_value, data_gcode=gcode)
    compdf = compfunc(df4gcode)

    # If compdf is empty, return None.
    if len(compdf)==0:
        return None

    # Decode gridcode in the composite dataframe.
    cgcode = compdf.index.values
    grid_lat, grid_lon = decode_center(gridcode=cgcode, decimal=decimal)

    # Convert 1d to 2d.
    outdf = compdf.loc[:, list(data_value.keys())]
    outdf['lon'] = grid_lon
    outdf['lat'] = grid_lat
    outdf = outdf.reset_index(drop=True)
    outds = outdf.set_index(['lon', 'lat']).to_xarray()

    return outds


if __name__ == '__main__':
    import xarray as xr

    import matplotlib.pyplot as plt

    nc_path = r"G:\Data\OCO3\Raw\oco3_LtSIF_220819_B10314r_221010221558s.nc4"

    # Read data.
    dataset = xr.open_dataset(nc_path)
    data_sif = dataset['SIF_740nm'].values
    data_time = dataset['Delta_Time'].values
    data_lon = dataset['Longitude'].values
    data_lat = dataset['Latitude'].values
    data_value = {'SIF_740nm':data_sif, 'time': data_time}

    # # Convert Timestamp into datetime.
    # data_time.attrs['units'] = 'seconds since 1990-01-01'

    # dataset = xr.decode_cf(dataset)

    # Set composite function.
    def csum(df:pd.DataFrame)->np.ndarray:
        cdf = df.groupby('GridCode').mean()

        return cdf
    
    geoTransform = (-180, 180, 1, 90, -90, 1)

    da = grid_data(
        data_value=data_value, data_lat=data_lat, data_lon=data_lon, 
        geoTransform=geoTransform, compfunc=csum, decimal=1
    )

    da.plot(x='lon', y='lat')
    plt.show()
