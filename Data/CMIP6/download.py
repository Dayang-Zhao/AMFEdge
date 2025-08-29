import os
import cdsapi
import zipfile

PRE_FPATH = "F:/Research/AMFEdge/CMIP6/metaData"
SUBDIR_LUT = {'historical':'Hist','ssp1_2_6': 'SSP1_26', 'ssp2_4_5': 'SSP2_45', 'ssp5_8_5': 'SSP5_85'}
START_YEAR = 2004
END_YEAR = 2014

def unzip(rootdir, fname):
    """
    Unzip the file at the given path.
    """
    file_names = []
    with zipfile.ZipFile(os.path.join(rootdir, fname), 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(".nc"):  # 只处理 .nc 文件
                zip_ref.extract(file_name, rootdir)
                file_names.append(file_name)

    return file_names

def download_cmip6(expriments=["ssp1_2_6"], variables=["precipitation"],model="cesm2"):
    """
    Download CMIP6 data from Copernicus Climate Data Store (CDS).
    """
    for expriment in expriments:
        print(f"Experiment: {expriment}")
        for variable in variables:
            print(f"Variable: {variable}")
            dataset = "projections-cmip6"
            request = {
                "temporal_resolution": "monthly",
                "experiment": expriment,
                "variable": variable,
                "model": model,
                "month": [
                    "01", "02", "03","04", "05", "06",
                    "07", "08", "09","10", "11", "12"
                ],
                "year": [str(year) for year in range(START_YEAR, END_YEAR + 1)],
                "area": [9, -80, -21, -44]
            }

            client = cdsapi.Client()

            # Download the data
            rootdir = f"{PRE_FPATH}/{SUBDIR_LUT[expriment]}/"
            fname = f"{model}_{expriment}_{variable}_2015_2100.zip"
            outpath = os.path.join(rootdir, fname)
            try:
                client.retrieve(dataset, request, outpath)

            except Exception as e:
                print(f"Error downloading {fname}")
                continue

            # Unzip the downloaded file.
            dst_fname = unzip(rootdir, fname)[0]

            # Rename the unzipped file.
            new_fname = f"{model}_{expriment}_{variable}_2015_2100.nc"
            os.rename(os.path.join(rootdir, dst_fname), os.path.join(rootdir, new_fname))

            # Delete the zip file.
            os.remove(outpath)

            print(f"Downloaded and unzipped: {new_fname}")

if __name__ == "__main__":
    experiments = ["historical"]
    # experiments = ["ssp1_2_6"]
    variables = ['evaporation_including_sublimation_and_transpiration',
                 'precipitation', 'surface_downwelling_shortwave_radiation',
                 'near_surface_air_temperature', 'near_surface_specific_humidity',
                 'surface_air_pressure']
    # variables = ['near_surface_specific_humidity', 'surface_air_pressure']
    # compelete_models = ['access_cm2']
    models = ['access_cm2', 'awi_cm_1_1_mr', 'bcc_csm2_mr', 'canesm5', 
              'canesm5_canoe', 'cesm2', 'cmcc_cm2_sr5', 'cmcc_esm2',
              'cnrm_cm6_1', 'cnrm_cm6_1_hr', 'cnrm_esm2_1','ec_earth3_veg_lr',
              'fgoals_f3_l','fgoals_g3', 'fio_esm_2_0', 'gfdl_esm4', 
              'hadgem3_gc31_ll', 'hadgem3_gc31_mm', 'iitm_esm', 'inm_cm4_8',
              'inm_cm5_0', 'ipsl_cm5a2_inca', 'ipsl_cm6a_lr', 'kace_1_0_g',
              'mcm_ua_1_0', 'miroc6', 'miroc_es2l', 'mpi_esm1_2_lr', 'mri_esm2_0',
              'nesm3', 'noresm2_lm', 'noresm2_mm', 'taiesm1', 'ukesm1_0_ll']
    
    # Download data for each model
    for model in models:
        print(f'Downloading data for model: {model}')
        download_cmip6(experiments, variables, model)
