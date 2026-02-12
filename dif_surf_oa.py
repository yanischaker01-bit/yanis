import geopandas as gpd
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from multiprocessing import Pool, cpu_count
import tqdm

# -----------------------------
FILE_OLD   = r"C:\Users\ychaker\Downloads\COS_GMAO\oa_merged.shp"
FILE_NEW   = r"C:\Users\ychaker\Downloads\COS_GMAO\surfaces_attribuees_corrige.shp"
OUTPUT_DIFF = r"C:\Users\ychaker\Downloads\COS_GMAO\surfaces_diff_multiproc.shp"

# -----------------------------
def clean_geom(geom):
    """Rend la géométrie valide pour éviter TopologyException"""
    if geom is None or geom.is_empty:
        return None
    try:
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.is_empty:
            return None
        return geom
    except:
        return None

def process_geom(args):
    """Fonction pour multiprocessing: différence d'un polygone"""
    geom_old, gdf_new = args
    geom_old = clean_geom(geom_old)
    if geom_old is None or geom_old.is_empty:
        return []

    # Index spatial pour trouver voisins
    possible_matches_idx = list(gdf_new.sindex.intersection(geom_old.bounds))
    possible_matches = gdf_new.iloc[possible_matches_idx]

    if not possible_matches.empty:
        valid_neighbors = possible_matches.geometry.apply(clean_geom)
        valid_neighbors = valid_neighbors[valid_neighbors.notna() & ~valid_neighbors.is_empty]
        if not valid_neighbors.empty:
            union_neighbors = unary_union(valid_neighbors)
            try:
                diff = geom_old.difference(union_neighbors)
            except:
                diff = geom_old.buffer(-0.01).difference(union_neighbors)
        else:
            diff = geom_old
    else:
        diff = geom_old

    if diff is None or diff.is_empty:
        return []

    # Retourner liste de polygones
    if diff.geom_type == "Polygon":
        return [diff]
    elif diff.geom_type == "MultiPolygon":
        return list(diff.geoms)
    else:
        return []

# -----------------------------
def main():
    gdf_old = gpd.read_file(FILE_OLD).to_crs(2154)
    gdf_new = gpd.read_file(FILE_NEW).to_crs(2154)

    print(f"📊 Polygones dans l'ancien fichier : {len(gdf_old)}")
    print(f"📊 Polygones dans le nouveau fichier : {len(gdf_new)}")

    # Nettoyage initial
    gdf_old["geometry"] = gdf_old.geometry.apply(clean_geom)
    gdf_old = gdf_old[gdf_old.geometry.notna() & ~gdf_old.geometry.is_empty]

    gdf_new["geometry"] = gdf_new.geometry.apply(clean_geom)
    gdf_new = gdf_new[gdf_new.geometry.notna() & ~gdf_new.geometry.is_empty]

    # Préparer arguments pour multiprocessing
    args = [(geom, gdf_new) for geom in gdf_old.geometry]

    # Pool de multiprocessing
    pool = Pool(processes=cpu_count())
    results = []

    for res in tqdm.tqdm(pool.imap(process_geom, args), total=len(args), desc="⏳ Traitement"):
        results.extend(res)

    pool.close()
    pool.join()

    print(f"❌ Polygones ou portions manquantes : {len(results)}")

    if results:
        diff_gdf = gpd.GeoDataFrame(geometry=results, crs=gdf_old.crs)
        diff_gdf.to_file(OUTPUT_DIFF, driver="ESRI Shapefile")
        print(f"💾 Diff exportée -> {OUTPUT_DIFF}")
    else:
        print("✅ Aucun polygone manquant trouvé.")

# -----------------------------
if __name__ == "__main__":
    main()
