import geopandas as gpd
import pandas as pd
import numpy as np
import os
from datetime import datetime
from shapely.geometry import Point
import warnings

warnings.filterwarnings('ignore')

def trouver_points_extremes(polygone):
    """
    Trouve les points extrêmes nord et sud d'un polygone.
    Retourne: (point_nord, point_sud)
    """
    try:
        # Obtenir tous les points du contour extérieur
        coords = list(polygone.exterior.coords)
        
        # Initialiser avec les premières coordonnées
        x_nord, y_nord = coords[0]
        x_sud, y_sud = coords[0]
        
        # Trouver les points extrêmes
        for x, y in coords:
            # Point le plus au NORD (Y maximum)
            if y > y_nord:
                y_nord = y
                x_nord = x
            
            # Point le plus au SUD (Y minimum)
            if y < y_sud:
                y_sud = y
                x_sud = x
        
        point_nord = Point(x_nord, y_nord)
        point_sud = Point(x_sud, y_sud)
        
        return point_nord, point_sud
        
    except Exception as e:
        # En cas d'erreur, utiliser le centroïde pour les deux
        centroid = polygone.centroid
        return centroid, centroid

def trouver_pk_le_plus_proche(point, gdf_pk_voie, max_distance=200):
    """
    Trouve le PK le plus proche d'un point donné.
    Retourne: (pk_value, distance) ou (None, None) si pas trouvé
    """
    if gdf_pk_voie.empty:
        return None, None
    
    try:
        # Calculer les distances
        distances = gdf_pk_voie.geometry.distance(point)
        
        if distances.empty:
            return None, None
        
        # Trouver l'index de la distance minimale
        min_idx = distances.idxmin()
        min_distance = distances[min_idx]
        
        # Vérifier si dans la distance max
        if min_distance <= max_distance:
            pk_value = gdf_pk_voie.loc[min_idx, 'PK_NUM']
            return float(pk_value), float(min_distance)
        else:
            return None, None
            
    except:
        return None, None

def recalculer_pk_deb_fin(shp_path):
    """
    Recalcule PK_DEB (point nord) et PK_FIN (point sud) pour tous les polygones
    avec une règle de 200m
    """
    print(f"🔍 Chargement du shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)
    
    # Vérifier les champs nécessaires
    print(f"\n📊 ANALYSE DU FICHIER:")
    print(f"   Total polygones: {len(gdf)}")
    print(f"   Colonnes disponibles: {list(gdf.columns)}")
    
    # Vérifier les champs PK existants
    pk_deb_existe = 'PK_DEB' in gdf.columns
    pk_fin_existe = 'PK_FIN' in gdf.columns
    code_axe_existe = 'CODE_AXE' in gdf.columns
    
    if not code_axe_existe:
        print(f"❌ Champ CODE_AXE non trouvé!")
        # Chercher un champ similaire
        for col in gdf.columns:
            if 'VOIE' in col.upper() or 'AXE' in col.upper():
                print(f"   Utilisation de {col} comme CODE_AXE")
                gdf['CODE_AXE'] = gdf[col]
                code_axe_existe = True
                break
    
    if not code_axe_existe:
        print(f"❌ Impossible de trouver un champ voie/axe")
        return
    
    # Charger les données PK originales
    pk_path = r"C:\Users\ychaker\Downloads\shp\LRS_PK_1M.shp"
    print(f"\n📂 Chargement des données PK...")
    
    try:
        gdf_pk = gpd.read_file(pk_path)
        
        # Reprojection si nécessaire
        target_crs = "EPSG:2154"
        if gdf_pk.crs != target_crs:
            gdf_pk = gdf_pk.to_crs(target_crs)
        
        # Identifier les champs
        voie_field = None
        for field in ['voie', 'VOIE', 'CODE_AXE', 'code_axe']:
            if field in gdf_pk.columns:
                voie_field = field
                break
        
        if not voie_field:
            print(f"❌ Champ voie non trouvé dans les PK!")
            return
        
        pk_field = None
        for field in ['pk', 'PK', 'POINT_KM']:
            if field in gdf_pk.columns:
                pk_field = field
                break
        
        if not pk_field:
            print(f"❌ Champ PK non trouvé dans les PK!")
            return
        
        print(f"   ✅ PK chargés: {len(gdf_pk)} points")
        print(f"   ✅ Champ voie PK: {voie_field}")
        print(f"   ✅ Champ PK: {pk_field}")
        
        # Convertir PK en numérique
        gdf_pk['PK_NUM'] = pd.to_numeric(gdf_pk[pk_field], errors='coerce')
        gdf_pk_clean = gdf_pk.dropna(subset=['PK_NUM', voie_field]).copy()
        
        # Grouper les PK par voie pour optimisation
        pk_by_voie = {}
        for voie in gdf_pk_clean[voie_field].unique():
            pk_by_voie[voie] = gdf_pk_clean[gdf_pk_clean[voie_field] == voie]
        
        print(f"   ✅ {len(pk_by_voie)} voies préparées")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des PK: {e}")
        return
    
    # TRAITEMENT PRINCIPAL
    print(f"\n🚀 TRAITEMENT DES POLYGONES...")
    print(f"   PK_DEB = point le plus au NORD (haut)")
    print(f"   PK_FIN = point le plus au SUD (bas)")
    print(f"   RECHERCHE DANS UN RAYON DE 200M")
    
    # Listes pour stocker les résultats
    nouveaux_pk_deb = []
    nouveaux_pk_fin = []
    dist_pk_deb = []
    dist_pk_fin = []
    points_nord = []
    points_sud = []
    
    # Compteurs
    succes_complet = 0
    succes_partiel = 0
    echec = 0
    
    for idx, row in gdf.iterrows():
        polygone = row.geometry
        code_axe = row['CODE_AXE'] if code_axe_existe else None
        
        # Vérifier la géométrie
        if polygone is None or polygone.is_empty or code_axe is None:
            nouveaux_pk_deb.append(None)
            nouveaux_pk_fin.append(None)
            dist_pk_deb.append(None)
            dist_pk_fin.append(None)
            points_nord.append(None)
            points_sud.append(None)
            echec += 1
            continue
        
        # 1. Trouver les points extrêmes nord et sud
        point_nord, point_sud = trouver_points_extremes(polygone)
        points_nord.append(point_nord)
        points_sud.append(point_sud)
        
        # 2. Chercher les PK correspondants DANS UN RAYON DE 200M
        pk_deb_val, dist_deb = None, None
        pk_fin_val, dist_fin = None, None
        
        if code_axe in pk_by_voie:
            pk_voie = pk_by_voie[code_axe]
            
            # PK_DEB = point NORD (recherche dans 200m)
            pk_deb_val, dist_deb = trouver_pk_le_plus_proche(point_nord, pk_voie, 200)
            
            # PK_FIN = point SUD (recherche dans 200m)
            pk_fin_val, dist_fin = trouver_pk_le_plus_proche(point_sud, pk_voie, 200)
        
        # Stocker les résultats
        nouveaux_pk_deb.append(pk_deb_val)
        nouveaux_pk_fin.append(pk_fin_val)
        dist_pk_deb.append(dist_deb)
        dist_pk_fin.append(dist_fin)
        
        # Compter les succès
        if pk_deb_val is not None and pk_fin_val is not None:
            succes_complet += 1
        elif pk_deb_val is not None or pk_fin_val is not None:
            succes_partiel += 1
        else:
            echec += 1
    
    # APPLIQUER LES NOUVEAUX PK
    print(f"\n📊 RÉSULTATS DU TRAITEMENT:")
    print(f"   Succès complet (PK_DEB + PK_FIN): {succes_complet}")
    print(f"   Succès partiel (un seul PK): {succes_partiel}")
    print(f"   Échec (aucun PK): {echec}")
    print(f"   Total: {len(gdf)}")
    
    # Ajouter les nouveaux champs
    gdf['PK_DEB_NEW'] = nouveaux_pk_deb
    gdf['PK_FIN_NEW'] = nouveaux_pk_fin
    gdf['DIST_PK_DEB'] = dist_pk_deb
    gdf['DIST_PK_FIN'] = dist_pk_fin
    
    # Conserver les anciennes valeurs pour comparaison
    if pk_deb_existe:
        gdf['PK_DEB_OLD'] = gdf['PK_DEB']
    if pk_fin_existe:
        gdf['PK_FIN_OLD'] = gdf['PK_FIN']
    
    # Remplacer par les nouvelles valeurs
    gdf['PK_DEB'] = gdf['PK_DEB_NEW']
    gdf['PK_FIN'] = gdf['PK_FIN_NEW']
    
    # ANALYSE DES CHANGEMENTS
    print(f"\n🔍 ANALYSE DES CHANGEMENTS:")
    
    if pk_deb_existe:
        changements_deb = 0
        for old, new in zip(gdf['PK_DEB_OLD'], gdf['PK_DEB']):
            if pd.notna(old) and pd.notna(new) and abs(old - new) > 0.001:
                changements_deb += 1
            elif pd.isna(old) and pd.notna(new):
                changements_deb += 1
        print(f"   PK_DEB modifiés: {changements_deb}")
    
    if pk_fin_existe:
        changements_fin = 0
        for old, new in zip(gdf['PK_FIN_OLD'], gdf['PK_FIN']):
            if pd.notna(old) and pd.notna(new) and abs(old - new) > 0.001:
                changements_fin += 1
            elif pd.isna(old) and pd.notna(new):
                changements_fin += 1
        print(f"   PK_FIN modifiés: {changements_fin}")
    
    # STATISTIQUES DES DISTANCES
    print(f"\n📏 STATISTIQUES DES DISTANCES (rayon 200m):")
    
    dist_deb_valid = [d for d in dist_pk_deb if d is not None]
    dist_fin_valid = [d for d in dist_pk_fin if d is not None]
    
    if dist_deb_valid:
        print(f"   PK_DEB - Distance moyenne: {np.mean(dist_deb_valid):.2f} m")
        print(f"   PK_DEB - Distance médiane: {np.median(dist_deb_valid):.2f} m")
        print(f"   PK_DEB - Distance max: {np.max(dist_deb_valid):.2f} m")
        print(f"   PK_DEB - Nombre < 10m: {len([d for d in dist_deb_valid if d <= 10])}")
        print(f"   PK_DEB - Nombre < 50m: {len([d for d in dist_deb_valid if d <= 50])}")
        print(f"   PK_DEB - Nombre < 100m: {len([d for d in dist_deb_valid if d <= 100])}")
        print(f"   PK_DEB - Nombre < 200m: {len([d for d in dist_deb_valid if d <= 200])}")
    
    if dist_fin_valid:
        print(f"   PK_FIN - Distance moyenne: {np.mean(dist_fin_valid):.2f} m")
        print(f"   PK_FIN - Distance médiane: {np.median(dist_fin_valid):.2f} m")
        print(f"   PK_FIN - Distance max: {np.max(dist_fin_valid):.2f} m")
        print(f"   PK_FIN - Nombre < 10m: {len([d for d in dist_fin_valid if d <= 10])}")
        print(f"   PK_FIN - Nombre < 50m: {len([d for d in dist_fin_valid if d <= 50])}")
        print(f"   PK_FIN - Nombre < 100m: {len([d for d in dist_fin_valid if d <= 100])}")
        print(f"   PK_FIN - Nombre < 200m: {len([d for d in dist_fin_valid if d <= 200])}")
    
    # CALCULER LA LONGUEUR
    print(f"\n📐 CALCUL DE LA LONGUEUR:")
    
    gdf['LONGUEUR_PK'] = gdf.apply(
        lambda row: abs(row['PK_FIN'] - row['PK_DEB']) 
        if pd.notna(row['PK_DEB']) and pd.notna(row['PK_FIN']) 
        else None,
        axis=1
    )
    
    longueurs_valid = gdf['LONGUEUR_PK'].dropna()
    if not longueurs_valid.empty:
        print(f"   Longueur moyenne: {longueurs_valid.mean():.3f} km")
        print(f"   Longueur min: {longueurs_valid.min():.3f} km")
        print(f"   Longueur max: {longueurs_valid.max():.3f} km")
        
        # Distribution des longueurs
        print(f"\n📈 DISTRIBUTION DES LONGUEURS:")
        bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 5, 10]
        for i in range(len(bins)-1):
            count = len([l for l in longueurs_valid if bins[i] <= l < bins[i+1]])
            if count > 0:
                pourcentage = count / len(longueurs_valid) * 100
                print(f"   {bins[i]:5.2f}-{bins[i+1]:5.2f} km: {count:>4} ({pourcentage:5.1f}%)")
    
    # FORMATER LES PK POUR AFFICHAGE
    print(f"\n🎨 FORMATAGE DES PK...")
    
    def format_pk_str(pk_value):
        if pd.isna(pk_value):
            return None
        try:
            pk_float = float(pk_value)
            km = int(pk_float)
            meters = int(round((pk_float - km) * 1000))
            return f"{km:02d}+{meters:03d}"
        except:
            return f"{pk_value:.3f}"
    
    gdf['PK_DEB_STR'] = gdf['PK_DEB'].apply(format_pk_str)
    gdf['PK_FIN_STR'] = gdf['PK_FIN'].apply(format_pk_str)
    
    # Ajouter des informations sur la méthode
    gdf['METHODE_PK_DEB'] = 'POINT_NORD_200M'
    gdf['METHODE_PK_FIN'] = 'POINT_SUD_200M'
    
    # EXPORT
    print(f"\n💾 EXPORT DES FICHIERS...")
    
    base_dir = os.path.dirname(shp_path)
    base_name = os.path.basename(shp_path)
    name_without_ext = os.path.splitext(base_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Shapefile - version propre
    new_shp_name = f"{name_without_ext}_PK_NORD_SUD_200M_{timestamp}.shp"
    new_shp_path = os.path.join(base_dir, new_shp_name)
    
    # Colonnes à conserver
    cols_to_keep = []
    for col in gdf.columns:
        if col not in ['PK_DEB_NEW', 'PK_FIN_NEW', 'PK_DEB_OLD', 'PK_FIN_OLD']:
            cols_to_keep.append(col)
    
    gdf_final = gdf[cols_to_keep]
    gdf_final.to_file(new_shp_path, encoding='UTF-8')
    print(f"   ✅ Shapefile principal: {new_shp_path}")
    
    # Excel complet avec toutes les colonnes
    excel_path = os.path.join(base_dir, f"{name_without_ext}_PK_NORD_SUD_200M_{timestamp}.xlsx")
    excel_cols = [col for col in gdf.columns if col != 'geometry']
    gdf[excel_cols].to_excel(excel_path, index=False)
    print(f"   ✅ Excel complet: {excel_path}")
    
    # EXEMPLES DE RÉSULTATS
    print(f"\n📋 EXEMPLES DE RÉSULTATS (5 premiers avec succès complet):")
    
    succes_mask = (gdf_final['PK_DEB'].notna()) & (gdf_final['PK_FIN'].notna())
    if succes_mask.any():
        exemples = gdf_final[succes_mask].head(5)
        
        for i, (_, row) in enumerate(exemples.iterrows(), 1):
            diff_nord_sud = row['PK_FIN'] - row['PK_DEB'] if pd.notna(row['PK_FIN']) and pd.notna(row['PK_DEB']) else None
            
            print(f"\n   Polygone {i}:")
            print(f"     Voie: {row['CODE_AXE']}")
            print(f"     PK_DEB (nord): {row['PK_DEB_STR']} ({row['PK_DEB']:.3f} km)")
            print(f"     PK_FIN (sud): {row['PK_FIN_STR']} ({row['PK_FIN']:.3f} km)")
            
            if diff_nord_sud is not None:
                print(f"     Différence PK_FIN - PK_DEB: {diff_nord_sud:.3f} km")
            
            if 'DIST_PK_DEB' in row and pd.notna(row['DIST_PK_DEB']):
                print(f"     Distance PK_DEB: {row['DIST_PK_DEB']:.1f} m")
            
            if 'DIST_PK_FIN' in row and pd.notna(row['DIST_PK_FIN']):
                print(f"     Distance PK_FIN: {row['DIST_PK_FIN']:.1f} m")
    
    # RAPPORT FINAL
    print(f"\n" + "="*60)
    print("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    print(f"\n📁 FICHIERS CRÉÉS:")
    print(f"   🗺️  Shapefile: {new_shp_path}")
    print(f"   📝 Excel: {excel_path}")
    
    print(f"\n🎯 MÉTHODE APPLIQUÉE:")
    print(f"   • PK_DEB = point le plus au NORD du polygone")
    print(f"   • PK_FIN = point le plus au SUD du polygone")
    print(f"   • Recherche du PK le plus proche dans un rayon de 200m")

def main():
    """Fonction principale avec interface utilisateur"""
    print("="*70)
    print("🔄 RECALCUL DES PK_DEB ET PK_FIN - RÈGLE 200m")
    print("   PK_DEB = point NORD | PK_FIN = point SUD")
    print("   Recherche dans un rayon de 200m")
    print("="*70)
    
    # Chemin par défaut
    shp_path = r"C:\Users\ychaker\Desktop\Boisement_Enrichi_RAPIDE\Boisement_PK_20260113_1531.shp"
    
    # Vérifier existence
    if not os.path.exists(shp_path):
        print(f"\n❌ Fichier par défaut non trouvé: {shp_path}")
        
        # Chercher d'autres fichiers
        search_dir = r"C:\Users\ychaker\Desktop"
        fichiers_trouves = []
        
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.shp') and 'Boisement' in file and 'PK' in file:
                    full_path = os.path.join(root, file)
                    fichiers_trouves.append((file, full_path))
        
        if fichiers_trouves:
            print(f"\n📁 FICHIERS BOISEMENT AVEC PK TROUVÉS:")
            for i, (nom_fichier, chemin) in enumerate(fichiers_trouves[:10], 1):
                print(f"   {i}. {nom_fichier}")
                print(f"      {chemin}")
            
            try:
                choix = int(input(f"\n📝 Sélection (1-{len(fichiers_trouves)}): ")) - 1
                if 0 <= choix < len(fichiers_trouves):
                    shp_path = fichiers_trouves[choix][1]
                else:
                    print("❌ Choix invalide")
                    return
            except:
                # Utiliser le premier trouvé
                shp_path = fichiers_trouves[0][1]
                print(f"   Utilisation du premier fichier: {os.path.basename(shp_path)}")
        else:
            # Demander le chemin manuellement
            shp_path = input("\n📝 Entrez le chemin complet du shapefile: ").strip()
            
            if not os.path.exists(shp_path):
                print(f"❌ Fichier non trouvé: {shp_path}")
                return
    
    print(f"\n✅ FICHIER SÉLECTIONNÉ:")
    print(f"   {shp_path}")
    
    # Afficher les informations du fichier
    try:
        gdf_test = gpd.read_file(shp_path)
        print(f"   Nombre de polygones: {len(gdf_test)}")
        print(f"   Colonnes PK trouvées: {[c for c in gdf_test.columns if 'PK' in c]}")
    except:
        print("   (Impossible de lire les informations du fichier)")
    
    # Confirmation
    print(f"\n" + "-"*50)
    print("⚠️  Ce script utilisera une règle de 200m pour la recherche")
    print("    des PK (au lieu de 100m)")
    print("-"*50)
    
    confirm = input("Confirmer le traitement? (o/n): ").strip().lower()
    
    if confirm in ['o', 'oui', 'y', 'yes']:
        print(f"\n" + "="*50)
        recalculer_pk_deb_fin(shp_path)
    else:
        print(f"\n❌ Traitement annulé")

if __name__ == "__main__":
    main()