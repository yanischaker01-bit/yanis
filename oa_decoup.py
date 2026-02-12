import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from shapely.validation import make_valid
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
SURF_FILE = r"C:\Users\ychaker\Downloads\COS_GMAO\cos_V1_V2.shp"
OA_FILE   = r"C:\Users\ychaker\Downloads\COS_GMAO\OA_V1_V2.shp"
AXE_FILE  = r"C:\Users\ychaker\Downloads\COS_GMAO\V1_V2.shp"

OUTPUT_FILE = r"C:\Users\ychaker\Downloads\COS_GMAO\surfaces_attribuees_corridor___1.shp"
OUTPUT_NON  = r"C:\Users\ychaker\Downloads\COS_GMAO\surfaces_non_attribuees_1.shp"

HALF_BUFFER = 1000.0  # demi-largeur des bandes (m)

# -----------------------------
def fix_geom(g):
    """Corrige une géométrie de manière robuste"""
    if g is None:
        return None
    try:
        if not g.is_valid:
            g = make_valid(g)
        # Pour les LineString, éviter buffer(0) qui peut les détruire
        if g.geom_type in ['LineString', 'MultiLineString']:
            return g
        g = g.buffer(0)
        if g.is_empty:
            return None
        return g
    except:
        return None

def fix_line_geom(g):
    """Corrige spécifiquement les géométries de ligne"""
    if g is None:
        return None
    try:
        if not g.is_valid:
            g = make_valid(g)
        # Pour les lignes, on évite buffer(0) qui peut les rendre vides
        return g
    except:
        return None

def get_centroid_safe(geom):
    """Obtenir le centroïde de manière sécurisée"""
    try:
        if geom is None or geom.is_empty:
            return None
        centroid = geom.centroid
        if centroid.is_empty:
            # Fallback: utiliser le centre de la bbox
            bounds = geom.bounds
            if bounds:
                return Point((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
            return None
        return centroid
    except:
        return None

def dissolve_to_single_line(gdf):
    """Fusionne les lignes en une seule ligne continue"""
    try:
        # Essayer d'abord avec union_all
        merged = linemerge(gdf.union_all())
        if merged.is_empty:
            # Fallback: utiliser la ligne la plus longue
            print("⚠️ Union_all vide, tentative avec la ligne la plus longue...")
            longest_line = None
            max_length = 0
            for geom in gdf.geometry:
                if geom and not geom.is_empty:
                    length = geom.length
                    if length > max_length:
                        max_length = length
                        longest_line = geom
            if longest_line:
                print(f"✅ Utilisation de la ligne la plus longue: {max_length:.2f}m")
                return longest_line
            else:
                print("❌ Aucune ligne valide trouvée")
                return None
        
        if merged.geom_type == "MultiLineString":
            # Prendre la ligne la plus longue du MultiLineString
            merged = max(merged.geoms, key=lambda x: x.length)
        
        print(f"✅ Axe généré - longueur: {merged.length:.2f}m")
        return merged
        
    except Exception as e:
        print(f"❌ Erreur lors de la fusion de l'axe: {e}")
        # Fallback: utiliser la première ligne valide
        for geom in gdf.geometry:
            if geom and not geom.is_empty:
                print(f"✅ Utilisation d'une ligne individuelle: {geom.length:.2f}m")
                return geom
        return None

def build_bands(axis_line, gdf_oa):
    print(f"📊 Traitement de {len(gdf_oa)} points OA")
    
    # Debug: check if geometries are valid
    valid_geoms = gdf_oa.geometry.apply(lambda g: g is not None and not g.is_empty).sum()
    print(f"📊 Geometries OA valides: {valid_geoms}/{len(gdf_oa)}")
    
    # Project points onto axis line
    gdf_oa["s"] = gdf_oa.geometry.apply(lambda p: axis_line.project(p) if p and not p.is_empty else None)
    
    # Check for None values
    none_count = gdf_oa["s"].isna().sum()
    print(f"📊 Points sans projection: {none_count}")
    
    gdf_oa = gdf_oa.dropna(subset=["s"]).sort_values("s").reset_index(drop=True)
    
    print(f"📊 Points OA après filtrage: {len(gdf_oa)}")
    
    if len(gdf_oa) < 2:
        print("❌ Pas assez de points OA pour créer des bandes (besoin d'au moins 2 points)")
        return gpd.GeoDataFrame(columns=["OA_D", "OA_F", "geometry"], crs=gdf_oa.crs)
    
    bands = []
    for i in range(len(gdf_oa) - 1):
        s0 = gdf_oa.loc[i, "s"]
        s1 = gdf_oa.loc[i + 1, "s"]
        
        if s1 <= s0:
            print(f"⚠️ Ordre inversé: s0={s0}, s1={s1}")
            continue
        
        p0 = axis_line.interpolate(s0)
        p1 = axis_line.interpolate(s1)
        
        if p0 is None or p1 is None or p0.is_empty or p1.is_empty:
            print(f"⚠️ Points d'interpolation invalides: p0={p0}, p1={p1}")
            continue
        
        line_seg = LineString([p0, p1])
        if line_seg.is_empty:
            print(f"⚠️ Segment vide entre s0={s0} et s1={s1}")
            continue
        
        band = line_seg.buffer(HALF_BUFFER, cap_style=2, join_style=2)
        if band is None or band.is_empty:
            print(f"⚠️ Bande vide entre s0={s0} et s1={s1}")
            continue
        
        bands.append({
            "OA_D": gdf_oa.loc[i, "LIBELLE"],
            "OA_F": gdf_oa.loc[i + 1, "LIBELLE"],
            "geometry": band
        })
    
    if not bands:
        print("❌ Aucune bande générée - liste vide")
        return gpd.GeoDataFrame(columns=["OA_D", "OA_F", "geometry"], crs=gdf_oa.crs)
    
    print(f"✅ {len(bands)} bandes générées avec succès")
    return gpd.GeoDataFrame(bands, crs=gdf_oa.crs)

# -----------------------------
def main():
    print("=== Découpage surfaces par OA + axe robuste ===")

    # Chargement des données
    print("📂 Chargement des fichiers...")
    gdf_surf = gpd.read_file(SURF_FILE).to_crs(2154)
    gdf_oa   = gpd.read_file(OA_FILE).to_crs(2154)
    gdf_axe  = gpd.read_file(AXE_FILE).to_crs(2154)
    
    print(f"📊 Surfaces: {len(gdf_surf)} polygones")
    print(f"📊 OA: {len(gdf_oa)} entités - type: {gdf_oa.geometry.type.iloc[0]}")
    print(f"📊 Axe: {len(gdf_axe)} entités - type: {gdf_axe.geometry.type.iloc[0]}")
    
    # Debug: afficher les premières géométries de l'axe
    print("🔍 Inspection de l'axe avant nettoyage:")
    for i, geom in enumerate(gdf_axe.geometry.head()):
        print(f"  Ligne {i}: valide={geom.is_valid}, vide={geom.is_empty}, longueur={geom.length:.2f}m")

    # Nettoyage des géométries des surfaces
    print("🧹 Nettoyage des géométries surfaces...")
    gdf_surf["geometry"] = gdf_surf.geometry.apply(fix_geom)
    gdf_surf = gdf_surf[gdf_surf.geometry.notna() & ~gdf_surf.geometry.is_empty]
    print(f"📊 Surfaces valides après nettoyage: {len(gdf_surf)}")

    # Nettoyage et préparation des OA
    print("🧹 Nettoyage des géométries OA...")
    gdf_oa["geometry"] = gdf_oa.geometry.apply(fix_geom)
    gdf_oa = gdf_oa[gdf_oa.geometry.notna() & ~gdf_oa.geometry.is_empty]
    
    print(f"📊 OA valides après nettoyage initial: {len(gdf_oa)}")
    
    if len(gdf_oa) == 0:
        print("❌ Aucun OA valide après nettoyage")
        return
    
    # Conversion des polygones OA en points (centroïdes)
    print("📌 Conversion des OA en centroïdes...")
    gdf_oa["centroid"] = gdf_oa.geometry.apply(get_centroid_safe)
    gdf_oa = gdf_oa[gdf_oa.centroid.notna()]
    gdf_oa["geometry"] = gdf_oa["centroid"]
    gdf_oa = gdf_oa.drop(columns=["centroid"])
    print(f"📊 OA après conversion centroïdes: {len(gdf_oa)}")

    # Traitement spécial pour l'axe - éviter de le détruire avec buffer(0)
    print("🧹 Traitement de l'axe (méthode spéciale)...")
    
    # Méthode 1: Essayer sans nettoyage agressif
    gdf_axe_clean = gdf_axe.copy()
    gdf_axe_clean["geometry"] = gdf_axe_clean.geometry.apply(fix_line_geom)
    gdf_axe_clean = gdf_axe_clean[gdf_axe_clean.geometry.notna() & ~gdf_axe_clean.geometry.is_empty]
    
    if len(gdf_axe_clean) == 0:
        print("⚠️ Méthode 1 échouée, tentative méthode 2: utilisation directe")
        # Méthode 2: Utiliser les géométries originales si valides
        gdf_axe_clean = gdf_axe[gdf_axe.geometry.notna() & gdf_axe.geometry.is_valid & ~gdf_axe.geometry.is_empty]
    
    if len(gdf_axe_clean) == 0:
        print("⚠️ Méthode 2 échouée, tentative méthode 3: réparation manuelle")
        # Méthode 3: Réparer manuellement chaque ligne
        valid_geoms = []
        for geom in gdf_axe.geometry:
            if geom and not geom.is_empty:
                try:
                    if not geom.is_valid:
                        geom = make_valid(geom)
                    valid_geoms.append(geom)
                except:
                    continue
        if valid_geoms:
            gdf_axe_clean = gpd.GeoDataFrame({'geometry': valid_geoms}, crs=gdf_axe.crs)
    
    print(f"📊 Segments d'axe valides après traitement: {len(gdf_axe_clean)}")
    
    if len(gdf_axe_clean) == 0:
        print("❌ Impossible de récupérer des segments d'axe valides")
        return

    # Création de l'axe
    print("🛣️  Création de l'axe...")
    axis_line = dissolve_to_single_line(gdf_axe_clean)
    if axis_line is None or axis_line.is_empty:
        print("❌ Axe vide ou invalide")
        return

    # Vérifier que les points OA peuvent être projetés sur l'axe
    test_point = gdf_oa.geometry.iloc[0]
    try:
        test_projection = axis_line.project(test_point)
        print(f"✅ Test de projection réussi: s={test_projection:.2f}")
    except Exception as e:
        print(f"❌ Erreur de projection: {e}")
        return

    # Création des bandes
    print("🎯 Création des bandes...")
    bands = build_bands(axis_line, gdf_oa)
    if bands.empty:
        print("❌ Aucune bande générée")
        return

    # Découpage des surfaces - Gestion des conflits de colonnes
    print("✂️  Découpage des surfaces...")
    try:
        # Identifier les colonnes en conflit
        surf_cols = set(gdf_surf.columns)
        bands_cols = set(bands.columns)
        common_cols = surf_cols.intersection(bands_cols) - {'geometry'}
        
        print(f"📊 Colonnes en commun (hors geometry): {common_cols}")
        
        # Renommer les colonnes en conflit dans bands pour éviter les erreurs
        bands_renamed = bands.rename(columns={col: f"band_{col}" for col in common_cols})
        
        # Utiliser overlay avec les colonnes renommées
        attribues = gpd.overlay(gdf_surf, bands_renamed, how="intersection")
        
        # Calculer la surface
        attribues["surface_m2"] = attribues.geometry.area

        # Flag découpage - méthode plus robuste
        def get_decoupe_flag(row):
            try:
                original_area = gdf_surf.loc[row.name].geometry.area
                return "Oui" if row["surface_m2"] < original_area else "Non"
            except:
                return "Inconnu"
        
        attribues["decoupe"] = attribues.apply(get_decoupe_flag, axis=1)

        # Surfaces non attribuées
        attrib_ids = attribues.index.unique()
        non_attribues = gdf_surf.loc[~gdf_surf.index.isin(attrib_ids)].copy()

        # Export
        print("💾 Export des résultats...")
        attribues.to_file(OUTPUT_FILE, driver="ESRI Shapefile")
        non_attribues.to_file(OUTPUT_NON, driver="ESRI Shapefile")

        print(f"✅ Surfaces attribuées : {len(attribues)} -> {OUTPUT_FILE}")
        print(f"❌ Surfaces non attribuées : {len(non_attribues)} -> {OUTPUT_NON}")
        print("🎉 Traitement terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors du découpage: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()