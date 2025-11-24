import ee

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='forestedge-432402')


def batch_del_asset(parent, id_start_with):
    # List assets in the specified parent folder or collection
    asset_paths = ee.data.listAssets({'parent': parent})['assets']

    for asset in asset_paths:
        asset_id = str(asset['name'])  # Get the asset ID as a string
        if asset_id.startswith(id_start_with):  # Check if it starts with the specified prefix
            ee.data.deleteAsset(asset_id)  # Delete the asset
            print(f"Deleted {asset_id}")

# ******** Test ********
import numpy as np

# dists = np.arange(60, 301, 60).tolist() + np.arange(420, 901, 120).tolist() \
#     + np.arange(1080, 1801, 180).tolist() + np.arange(2040, 3001, 240).tolist() \
#     + np.arange(3300, 4501, 300).tolist()+ np.arange(4860, 6301, 360).tolist()
# for dist in dists:
#     parent = f"projects/forestedge-432402/assets/AMFUndistDegdEdge2023/Edge{dist}"
#     id_start_with = f"projects/forestedge-432402/assets/AMFUndistDegdEdge2023/Edge{dist}/UndistDegdEdge{dist}"
#     batch_del_asset(parent, id_start_with)
# parent = f"projects/forestedge-432402/assets/S2/AmzS2AnoVI_GLEAM"
# id_start_with = f"projects/forestedge-432402/assets/S2/AmzS2AnoVI_GLEAM"
# batch_del_asset(parent, id_start_with)

# parent = f"projects/forestedge-432402/assets/CMIP6/SSP5_85"
# models = ['mri_esm2_0', 'cnrm_cm6_1_hr', 'cesm2', 'ukesm1_0_ll',
#           'noresm2_mm', 'miroc6', 'taiesm1',
#           'kace_1_0_g', 'access_cm2', 'cmcc_cm2_sr5']
# ids = ['temperature_2m_ano', 'WD_ano', 'DL', 'DS',
#        'vpd_ano', 'total_precipitation_sum_ano', 'surface_solar_radiation_downwards_sum_ano', 'total_evaporation_sum_ano',
#        'vpd_stdAno', 'total_precipitation_sum_stdAno', 'surface_solar_radiation_downwards_sum_stdAno', 'total_evaporation_sum_stdAno']
# for model in models:
#     for id in ids:
#         batch_del_asset(f'{parent}/{model}', f'{parent}/{model}/{id}')
