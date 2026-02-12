import geopandas as gpd
from shapely.validation import explain_validity

# === Paramètres ===
chemin_entree = r"C:\Users\ychaker\Downloads\COS_GMAO\cos_manque_merge.shp"
chemin_sortie = r"C:\Users\ychaker\Downloads\COS_GMAO\cos_manque_merge_manque_sans_doublons.shp"

# === Charger la couche ===
gdf = gpd.read_file(chemin_entree)

# === Ajouter une colonne de validité géométrique ===
gdf["is_valid_geom"] = gdf.geometry.is_valid

# === Trier pour garder en priorité les géométries valides ===
gdf = gdf.sort_values(by="is_valid_geom", ascending=False)

# === Supprimer les doublons sur Code + Surface_3D ===
gdf_unique = gdf.drop_duplicates(subset=["Code", "Surface_3D"], keep="first")

# === Supprimer la colonne temporaire ===
gdf_unique = gdf_unique.drop(columns=["is_valid_geom"])

# === Sauvegarder le résultat ===
gdf_unique.to_file(chemin_sortie)

print(f"Shapefile sans doublons enregistré : {chemin_sortie}")
