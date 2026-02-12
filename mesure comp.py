import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime

def decomposer_multipolygones(gdf):
    """Décompose les multipolygones en polygones simples tout en conservant les attributs"""
    gdf = gdf.reset_index(drop=True)
    gdf['original_id'] = gdf.index
    gdf_exploded = gdf.explode(index_parts=True).reset_index(drop=True)
    
    # Pour suivre les parties d'un même multipolygone original
    gdf_exploded['part_id'] = gdf_exploded.groupby('original_id').cumcount() + 1
    gdf_exploded['is_multipart'] = gdf_exploded['original_id'].duplicated(keep=False)
    
    return gdf_exploded

def main():
    # === 1. Chemins des fichiers ===
    poly_path = r"C:\Users\ychaker\Downloads\l-mesure-compensatoire-juillet-2025-shp\m_geomce.l_mesure_compensatoire_s_000.shp"
    pk_path = r"C:\Users\ychaker\Downloads\shp\LRS_PK_1M.shp"
    emprise_path = r"C:\Users\ychaker\Downloads\kmz\emprise\EMP_cosea.shp"
    
    # === 2. Configuration des sorties ===
    output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Mesures_Compensatoires_Output")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    excel_path = os.path.join(output_dir, f"mesures_compensatoires_PK_{timestamp}.xlsx")
    shp_path = os.path.join(output_dir, f"mesures_compensatoires_PK_{timestamp}.shp")
    
    # === 3. Chargement des données ===
    print("🔍 Chargement des données...")
    gdf_poly = gpd.read_file(poly_path)
    gdf_pk = gpd.read_file(pk_path)
    emprise = gpd.read_file(emprise_path)
    
    # === 4. Reprojection en Lambert 93 ===
    print("🔄 Reprojection en cours...")
    target_crs = "EPSG:2154"
    gdf_poly = gdf_poly.to_crs(target_crs)
    gdf_pk = gdf_pk.to_crs(target_crs)
    emprise = emprise.to_crs(target_crs)
    
    # === 5. Décomposition rigoureuse des MultiPolygones ===
    print("🧩 Décomposition des MultiPolygones...")
    gdf_poly = decomposer_multipolygones(gdf_poly)
    
    # === 6. Filtrage par emprise (buffer 50m) ===
    print("🌐 Filtrage spatial avec buffer 50m...")
    emprise_buffer = emprise.copy()
    emprise_buffer["geometry"] = emprise_buffer.buffer(50)
    emprise_union = emprise_buffer.geometry.unary_union
    gdf_poly = gdf_poly[gdf_poly.intersects(emprise_union)].copy()
    
    if gdf_poly.empty:
        raise ValueError("❌ Aucun polygone dans le périmètre étendu de l'emprise COSEA.")
    
    # === 7. Création d'un index spatial sur les PK ===
    print("⚡ Création de l'index spatial des PK...")
    pk_sindex = gdf_pk.sindex
    
    # === 8. Traitement des polygones ===
    print("🚀 Traitement des polygones...")
    
    buffer_distance = 80
    results = []
    
    # Création d'un index explicite pour la fusion
    gdf_poly = gdf_poly.reset_index(drop=True)
    gdf_poly['merge_index'] = gdf_poly.index
    
    for idx, row in tqdm(gdf_poly.iterrows(), total=len(gdf_poly), desc="Traitement des polygones"):
        geom = row.geometry
        bounds = geom.bounds
        
        # Recherche PK intersectant la bbox du polygone
        candidate_pk_idx = list(pk_sindex.intersection(bounds))
        if not candidate_pk_idx:
            results.append({
                'merge_index': idx,
                'PK_D': None,
                'PK_F': None,
                'Voie': None,
                'Distance_D': None,
                'Distance_F': None
            })
            continue
        
        candidate_pk = gdf_pk.iloc[candidate_pk_idx]
        intersecting_pk = candidate_pk[candidate_pk.intersects(geom)]
        
        if not intersecting_pk.empty:
            intersecting_pk = intersecting_pk.sort_values('pk')
            pk_d = intersecting_pk.iloc[0]['pk']
            pk_f = intersecting_pk.iloc[-1]['pk']
            voie = intersecting_pk.iloc[0].get('voie', None)
            dist_d = 0
            dist_f = 0
        else:
            buffered_geom = geom.buffer(buffer_distance)
            nearby_idx = list(pk_sindex.intersection(buffered_geom.bounds))
            if not nearby_idx:
                results.append({
                    'merge_index': idx,
                    'PK_D': None,
                    'PK_F': None,
                    'Voie': None,
                    'Distance_D': None,
                    'Distance_F': None
                })
                continue
            
            nearby_pk = gdf_pk.iloc[nearby_idx]
            nearby_pk = nearby_pk[nearby_pk.geometry.within(buffered_geom)]
            
            if nearby_pk.empty:
                results.append({
                    'merge_index': idx,
                    'PK_D': None,
                    'PK_F': None,
                    'Voie': None,
                    'Distance_D': None,
                    'Distance_F': None
                })
                continue
            
            nearby_pk = nearby_pk.assign(distance=nearby_pk.geometry.apply(lambda x: geom.distance(x)))
            nearby_pk = nearby_pk.sort_values(['distance', 'pk'])
            
            pk_d = nearby_pk.iloc[0]['pk']
            pk_f = nearby_pk.iloc[-1]['pk'] if len(nearby_pk) > 1 else pk_d
            dist_d = nearby_pk.iloc[0]['distance']
            dist_f = nearby_pk.iloc[-1]['distance'] if len(nearby_pk) > 1 else dist_d
            voie = nearby_pk.iloc[0].get('voie', None)
        
        results.append({
            'merge_index': idx,
            'PK_D': pk_d,
            'PK_F': pk_f,
            'Voie': voie,
            'Distance_D': round(dist_d, 2) if dist_d is not None else None,
            'Distance_F': round(dist_f, 2) if dist_f is not None else None
        })
    
    # === 9. Fusion des résultats ===
    print("📊 Fusion des résultats...")
    results_df = pd.DataFrame(results)
    
    # Fusion sur l'index explicite
    final_gdf = gdf_poly.merge(results_df, on='merge_index', how='left')
    
    # === 10. Export des résultats ===
    print("💾 Export des fichiers...")
    
    # Export Excel (sans la géométrie et l'index de fusion)
    excel_cols = [col for col in final_gdf.columns if col not in ['geometry', 'merge_index']]
    final_gdf[excel_cols].to_excel(excel_path, index=False)
    
    # Export Shapefile (avec tous les champs)
    final_gdf.to_file(shp_path, encoding='UTF-8')
    
    # === 11. Résumé des résultats ===
    print("\n✅ Traitement terminé avec succès!")
    print(f"📂 Dossier de sortie : {output_dir}")
    print(f"📝 Fichier Excel : {os.path.basename(excel_path)}")
    print(f"🗺️ Fichier Shapefile : {os.path.basename(shp_path)}")
    print(f"📌 Polygones traités : {len(final_gdf)}")
    
    # Statistiques supplémentaires
    with_pk = final_gdf[final_gdf['PK_D'].notna()]
    print(f"🔢 Polygones avec PK associés : {len(with_pk)} ({len(with_pk)/len(final_gdf)*100:.1f}%)")
    if not with_pk.empty:
        print(f"📏 Distance moyenne : {with_pk['Distance_D'].mean():.1f}m (max: {with_pk['Distance_D'].max():.1f}m)")

if __name__ == "__main__":
    main()