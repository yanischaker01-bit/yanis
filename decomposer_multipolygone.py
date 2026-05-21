import geopandas as gpd

# Charger le shapefile
gdf = gpd.read_file(
    r"C:\Users\ychaker\Desktop\Mesures_Compensatoires_Output\Mesures_compensatoires.shp"
)

# Décomposer les multipolygones
gdf_single = gdf.explode(ignore_index=True)

# Sauvegarde
gdf_single.to_file(
    r"C:\Users\ychaker\Desktop\Mesures_Compensatoires_Output\Mesures_compensatoires_single.shp"
)

print("Décomposition terminée")
