# !/usr/bin/python3
# -*- coding: utf-8 -*-
import os
os.environ['PROJ_LIB'] = r'D:\Program Files\Python\Lib\site-packages\pyproj\proj_dir\share\proj'

import numpy as np
import pandas as pd
import netCDF4 as nc

# ---------------------------------------------------------
def save_pd_as_excel(
    data:pd.DataFrame, path:str, sheet_name:str,
    index:bool=False, add_row_or_col:str='row'
):
    """Save pd.DataFrame into excel file. When the file does not
    exist, create the excel. When the file exists, add columns or 
    rows in the end.

    Args:
        data (pd.DataFrame): The data to save.
        path (str): The excel file path.
        sheet_name (str): The name of sheet to add.
        index (bool): Whether to save index.
            Default to False.
        add_row_or_col: Whether to add columns or rows in the end
            when the sheet exists.
    """
    # Whether the excel file exists.
    do_exist = os.path.exists(path=path)
    
    # ------ If the file does not exists, -----------------
    # ------ create the excel file and save data.----------
    if not do_exist:
        # Set Excel Writer.
        excel_writer = pd.ExcelWriter(
            path=path, 
            engine='openpyxl'
            )
        data.to_excel(
            excel_writer=excel_writer,
            sheet_name=sheet_name,
            index=False
        )
    else:
    # ------ If the file exists, --------------------------
    # ------ load the existing data in the original ------- 
    # ------ excel file and append columns.----------------
        # Load existing data.
        excel_reader = pd.ExcelFile(path, engine='openpyxl')
        if sheet_name in excel_reader.sheet_names:
            existing_data = excel_reader.parse(sheet_name)
            # Concat dataframe.
            if add_row_or_col == 'col':
                concated_data = pd.concat([existing_data, data], axis=1)
            else:
                concated_data = pd.concat([existing_data, data], axis=0)
        else:
            concated_data = data

        # Set Excel Writer.
        excel_writer = pd.ExcelWriter(
            path=path, 
            engine='openpyxl',
            mode='a',
            if_sheet_exists='replace'
            )        

        # Write.
        concated_data.to_excel(
            excel_writer=excel_writer,
            sheet_name=sheet_name,
            index=index
        )

    excel_writer.close()

def save_pd_as_csv(
    data:pd.DataFrame, path:str, index:bool=False, add_row_or_col:str='col',
    compression='infer'
):
    """Save pd.DataFrame into excel file. When the file does not
    exist, create the excel. When the file exists, add columns or 
    rows in the end.

    Args:
        data (pd.DataFrame): The data to save.
        path (str): The excel file path.
        sheet_name (str): The name of sheet to add.
        index (bool): Whether to save index.
            Default to False.
        add_row_or_col: Whether to add columns or rows in the end
            when the sheet exists.
    """
    # Whether the csv file exists.
    do_exist = os.path.exists(path=path)
    
    # ------ If the file does not exists, -----------------
    # ------ create the csv file and save data.----------
    if not do_exist:
        data.to_csv(path_or_buf=path,index=False, mode='w', header=True, compression=compression)
    else:
    # ------ If the file exists, --------------------------
    # ------ load the existing data in the original ------- 
    # ------ excel file and append columns.----------------
        # Concat dataframe.
        if add_row_or_col == 'col':
            # Load existing data.
            existing_data = pd.read_csv(path)
            concated_data = pd.concat([existing_data, data], axis=1)
            concated_data.to_csv(path_or_buf=path,index=index,mode='a', compression=compression)
        else:
            existing_data = pd.read_csv(path)
            existing_columns = existing_data.columns
            data = data[existing_columns]
            data.to_csv(path_or_buf=path,index=index, mode='a', header=False, compression=compression)   

        # Write.
        


# if __name__ == '__main__':
    # # Test save_data_as_tiff.
    # data = np.random.random(size=(3, 360, 720))
    # band_names = ['Band1', 'Band2', 'Band3']
    # output_path = r'F:\SIF\TROPOMI\Ungridded\Test\test.tif'
    # prj = osr.SpatialReference()
    # import os
    # os.environ['PROJ_LIB'] = r'D:\Program Files\Python\Lib\site-packages\pyproj\proj_dir\share\proj'
    # prj.ImportFromEPSG(4326)
    # start_lat = 90
    # start_lon = -180
    # spatial_res = 0.5
    # raster = save_data_as_tiff(data, band_names, output_path, prj.ExportToWkt(), start_lat, start_lon, spatial_res)
    # print(raster)