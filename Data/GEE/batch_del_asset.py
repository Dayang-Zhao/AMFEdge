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
# for dist in [60, 120, 180]:
#     parent = f"projects/forestedge-432402/assets/S2/Edge{dist}"
#     id_start_with = f"projects/forestedge-432402/assets/AmazonForestEdge2023/Edge{dist}/UndistEdge{dist}"
#     batch_del_asset(parent, id_start_with)
parent = f"projects/forestedge-432402/assets/S2/AmzS2AnoVI"
id_start_with = f"projects/forestedge-432402/assets/S2/AmzS2AnoVI/Amz_2023_"
batch_del_asset(parent, id_start_with)
