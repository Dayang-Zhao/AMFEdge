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
parent = "projects/forestedge-432402/assets/ForestEdge2023/"
id_start_with = "projects/forestedge-432402/assets/ForestEdge2023/AmeEdge"
batch_del_asset(parent, id_start_with)
