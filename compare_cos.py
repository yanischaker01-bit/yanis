import geopandas as gpd
import fiona
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import snap
from shapely.errors import ShapelyError
import warnings

def lire_shapefile_sans_erreurs(path, min_area=1.0):
    features_valides = []
    with fiona.open(path) as source:
        crs = source.crs
        for feat in source:
            try:
                geom = shape(feat['geometry'])
                if geom.is_valid and not geom.is_empty and geom.area >= min_area:
                    feat['geometry'] = geom
                    features_valides.append(feat)
            except (ValueError, ShapelyError, TypeError):
                continue  # Ignore les géométries corrompues
    if not features_valides:
        print(f"❌ Aucun polygone valide trouvé dans {path}.")
        return None
    gdf = gpd.GeoDataFrame.from_features(features_valides, crs=crs)
    print(f"✔️  {path} chargé avec {len(gdf)} géométries valides.")
    return gdf

def nettoyer_geometries(gdf, surface_min=100.0):
    def valid_and_large(geom):
        if geom is None or geom.is_empty:
            return False
        if not geom.is_valid:
            return False
        if isinstance(geom, (Polygon, MultiPolygon)):
            return geom.area >= surface_min
        return True
    gdf = gdf[gdf['geometry'].apply(valid_and_large)].copy()
    return gdf

def snap_geometries(gdf_source, gdf_cible, tolerance=0.95):
    cible_union = gdf_cible.unary_union
    gdf_source['geometry'] = gdf_source['geometry'].apply(lambda geom: snap(geom, cible_union, tolerance))
    return gdf_source

def main():
    fichier_1 = r"C:\Users\ychaker\Desktop\SIG\Carlrec_cos_fusion.shp"
    fichier_2 = r"C:\Users\ychaker\Desktop\SIG\cos_prod.shp"
    sortie = r"C:\Users\ychaker\Desktop\SIG\difference_cos.shp"

    print("🔄 Chargement des fichiers...")

    gdf1 = lire_shapefile_sans_erreurs(fichier_1)
    if gdf1 is None:
        return
    gdf2 = lire_shapefile_sans_erreurs(fichier_2)
    if gdf2 is None:
        return

    print("🧹 Nettoyage des géométries...")
    gdf1 = nettoyer_geometries(gdf1, surface_min=1.0)
    gdf2 = nettoyer_geometries(gdf2, surface_min=1.0)
    print(f"✅ Fichier 1 : {len(gdf1)} géométries valides")
    print(f"✅ Fichier 2 : {len(gdf2)} géométries valides")

    print("🧲 Snapping des géométries...")
    gdf1 = snap_geometries(gdf1, gdf2, tolerance=0.5)

    print("➖ Calcul de la différence géométrique...")
    try:
        gdf_diff = gpd.overlay(gdf1, gdf2, how='difference')
        print(f"✅ Différence calculée : {len(gdf_diff)} géométries résultantes.")
    except Exception as e:
        print(f"❌ Erreur pendant l'overlay : {e}")
        return

    print(f"💾 Sauvegarde dans : {sortie}")
    gdf_diff.to_file(sortie)
    print("✅ Terminé avec succès.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
