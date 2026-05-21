import geopandas as gpd

# === Chemin du fichier en entrée ===
chemin_entree = r"C:\Users\ychaker\Downloads\COS_GMAO\partie_sud.shp"
chemin_sortie = r"C:\Users\ychaker\Downloads\COS_GMAO\partie_sud_filtré.shp"

# Charger le shapefile
gdf = gpd.read_file(chemin_entree)

# Supprimer les géométries nulles ou vides
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]

# Supprimer les entités avec une surface égale à 0
gdf = gdf[gdf.geometry.area > 0]

# Réinitialiser l'index
gdf = gdf.reset_index(drop=True)

# Sauvegarder le fichier filtré
gdf.to_file(chemin_sortie)

print("✅ Fichier exporté sans les surfaces vides :", chemin_sortie)
