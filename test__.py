import geopandas as gpd

# Chemin du shapefile
input_path = r"C:\Users\ychaker\Downloads\OLD_2025\Instal_gmao_old_zs.shp"

# Charger le shapefile
gdf = gpd.read_file(input_path)

# Vérifier que la projection est bien EPSG:2154
print("CRS :", gdf.crs)
if gdf.crs.to_epsg() != 2154:
    raise ValueError("Le fichier n'est pas en EPSG:2154. Vérifiez la projection.")

# Ajouter une colonne pour la distance de buffer selon la valeur de 'old_zs'
gdf['buffer_dist'] = gdf['old_zs'].apply(lambda x: 50 if x == 'OLD' else (20 if x == 'ZS' else 0))

# Créer les buffers
gdf['geometry'] = gdf.buffer(gdf['buffer_dist'])

# Sauvegarder le nouveau shapefile avec les buffers
output_path = r"C:\Users\ychaker\Downloads\OLD_2025\Instal_gmao_old_zs_buffer___.shp"
gdf.to_file(output_path)

print("✅ Zones tampons créées avec succès dans :", output_path)
