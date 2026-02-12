import geopandas as gpd
import pandas as pd
import numpy as np
import os
from datetime import datetime
from shapely.geometry import Point, Polygon
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

def identifier_voie_proche(polygone, gdf_pk_all, max_distance=200):
    """
    Identifie la voie la plus proche d'un polygone
    en cherchant les PK à proximité dans un rayon de 200m
    """
    try:
        # Prendre le centroïde du polygone
        centroid = polygone.centroid
        
        # Chercher les PK dans un rayon de 200m
        gdf_proche = gdf_pk_all[gdf_pk_all.geometry.distance(centroid) <= max_distance]
        
        if gdf_proche.empty:
            # Élargir la recherche à 400m si pas trouvé à 200m
            gdf_proche = gdf_pk_all[gdf_pk_all.geometry.distance(centroid) <= max_distance * 2]
        
        if not gdf_proche.empty:
            # Retourner la voie la plus fréquente parmi les PK proches
            voies_counts = gdf_proche['VOIE'].value_counts()
            if not voies_counts.empty:
                voie_majoritaire = voies_counts.index[0]
                
                # Vérifier le nombre de PK trouvés pour cette voie
                nb_pk_voie = voies_counts.iloc[0]
                distance_moyenne = gdf_proche[gdf_proche['VOIE'] == voie_majoritaire].geometry.distance(centroid).mean()
                
                print(f"      ✓ Voie {voie_majoritaire}: {nb_pk_voie} PK trouvés (distance moyenne: {distance_moyenne:.1f}m)")
                return voie_majoritaire
            else:
                print(f"      ✗ Aucune voie majoritaire trouvée")
                return None
        else:
            print(f"      ✗ Aucun PK trouvé dans un rayon de {max_distance}m")
            return None
        
    except Exception as e:
        print(f"      ✗ Erreur lors de la recherche de voie: {e}")
        return None

def calculer_pk_approximatif(point, gdf_pk_voie, max_distance=200):
    """
    Calcule un PK approximatif par interpolation linéaire
    si le point n'est pas exactement sur un PK
    """
    if gdf_pk_voie.empty or len(gdf_pk_voie) < 2:
        return None, None
    
    try:
        # Calculer les distances à tous les PK de la voie
        distances = gdf_pk_voie.geometry.distance(point)
        gdf_pk_voie['DISTANCE'] = distances
        
        # Filtrer les PK dans le rayon de recherche
        gdf_proche = gdf_pk_voie[distances <= max_distance].copy()
        
        if gdf_proche.empty:
            return None, None
        
        # Si un seul PK dans le rayon, le retourner
        if len(gdf_proche) == 1:
            pk_val = gdf_proche.iloc[0]['PK_NUM']
            dist = gdf_proche.iloc[0]['DISTANCE']
            return float(pk_val), float(dist)
        
        # Pour plusieurs PK, prendre les 2 plus proches pour l'interpolation
        gdf_sorted = gdf_proche.sort_values('DISTANCE')
        pk1 = gdf_sorted.iloc[0]
        pk2 = gdf_sorted.iloc[1]
        
        # Interpolation linéaire
        d1 = pk1['DISTANCE']
        d2 = pk2['DISTANCE']
        pk1_val = pk1['PK_NUM']
        pk2_val = pk2['PK_NUM']
        
        # Calcul du PK approximatif
        if d1 == 0:
            return pk1_val, 0
        elif d2 == 0:
            return pk2_val, 0
        else:
            # Poids inversement proportionnel aux distances
            total_dist = d1 + d2
            w1 = d2 / total_dist  # Plus proche de pk1 si d1 petit
            w2 = d1 / total_dist  # Plus proche de pk2 si d2 petit
            
            pk_approx = pk1_val * w1 + pk2_val * w2
            distance_min = min(d1, d2)
            
            return pk_approx, distance_min
            
    except:
        return None, None

def creer_vue_ensemble(gdf_polygones, gdf_pk, buffer_distance=20):
    """
    Crée une vue d'ensemble des polygones et PK pour vérification
    """
    try:
        # Créer des géométries simplifiées
        gdf_poly_simple = gdf_polygones.copy()
        gdf_poly_simple['geometry'] = gdf_poly_simple.geometry.centroid.buffer(buffer_distance)
        
        # Sélectionner un échantillon de PK
        gdf_pk_sample = gdf_pk.copy()
        if len(gdf_pk_sample) > 1000:
            gdf_pk_sample = gdf_pk_sample.sample(1000)
        
        # Combiner les deux datasets
        gdf_poly_simple['TYPE'] = 'POLYGONE'
        gdf_pk_sample['TYPE'] = 'PK'
        
        # Sélectionner les colonnes communes
        cols = ['TYPE', 'geometry']
        
        # Ajouter des informations supplémentaires
        if 'VOIE' in gdf_pk_sample.columns:
            gdf_pk_sample['LABEL'] = gdf_pk_sample['VOIE'] + '_' + gdf_pk_sample['PK_NUM'].astype(str)
            cols.append('LABEL')
        
        if 'CODE_AXE' in gdf_poly_simple.columns:
            gdf_poly_simple['LABEL'] = gdf_poly_simple['CODE_AXE'].astype(str)
            cols.append('LABEL')
        
        # Combiner
        vue_ensemble = pd.concat([
            gdf_poly_simple[cols],
            gdf_pk_sample[cols]
        ], ignore_index=True)
        
        return vue_ensemble
        
    except Exception as e:
        print(f"⚠️ Impossible de créer la vue d'ensemble: {e}")
        return None

def assigner_pk_aux_polygones(shp_path):
    """
    Assigner PK_DEB et PK_FIN aux polygones en fonction de leur position nord-sud
    avec une règle de 200m par rapport à la voie
    """
    print(f"🔍 Chargement du shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)
    
    # Vérifier les champs nécessaires
    print(f"\n📊 ANALYSE DU FICHIER:")
    print(f"   Total polygones: {len(gdf)}")
    print(f"   Colonnes disponibles: {list(gdf.columns)}")
    
    # Charger les données PK
    pk_path = r"C:\Users\ychaker\Downloads\shp\LRS_PK_1M.shp"
    print(f"\n📂 Chargement des données PK...")
    
    try:
        gdf_pk = gpd.read_file(pk_path)
        
        # Reprojection si nécessaire
        target_crs = "EPSG:2154"
        if gdf_pk.crs != target_crs:
            gdf_pk = gdf_pk.to_crs(target_crs)
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        
        # Identifier les champs dans les PK
        voie_field = None
        for field in ['voie', 'VOIE', 'CODE_AXE', 'code_axe', 'NOM_VOIE', 'NOM']:
            if field in gdf_pk.columns:
                voie_field = field
                break
        
        if not voie_field:
            print(f"❌ Champ voie non trouvé dans les PK!")
            # Chercher par nom
            for col in gdf_pk.columns:
                if 'voie' in col.lower() or 'route' in col.lower() or 'nom' in col.lower():
                    voie_field = col
                    print(f"   Utilisation de {col} comme champ voie")
                    break
        
        pk_field = None
        for field in ['pk', 'PK', 'POINT_KM', 'KILOMETRAG', 'CHAINE', 'CHAINAGE']:
            if field in gdf_pk.columns:
                pk_field = field
                break
        
        if not pk_field:
            print(f"❌ Champ PK non trouvé dans les PK!")
            return
        
        print(f"   ✅ PK chargés: {len(gdf_pk)} points")
        print(f"   ✅ Champ voie PK: {voie_field}")
        print(f"   ✅ Champ PK: {pk_field}")
        
        # Nettoyer et préparer les PK
        gdf_pk['PK_NUM'] = pd.to_numeric(gdf_pk[pk_field], errors='coerce')
        gdf_pk['VOIE'] = gdf_pk[voie_field].astype(str)
        gdf_pk_clean = gdf_pk.dropna(subset=['PK_NUM', 'VOIE']).copy()
        
        # Grouper les PK par voie pour optimisation
        pk_by_voie = {}
        for voie in gdf_pk_clean['VOIE'].unique():
            pk_by_voie[voie] = gdf_pk_clean[gdf_pk_clean['VOIE'] == voie]
        
        print(f"   ✅ {len(pk_by_voie)} voies préparées")
        
        # Afficher les voies disponibles
        print(f"\n📋 VOIES DISPONIBLES DANS LES PK:")
        voies_liste = list(pk_by_voie.keys())
        for i, voie in enumerate(voies_liste[:20]):
            nb_pk = len(pk_by_voie[voie])
            print(f"   {i+1:2d}. {voie:20s} ({nb_pk:3d} PK)")
        if len(voies_liste) > 20:
            print(f"   ... et {len(voies_liste) - 20} autres voies")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des PK: {e}")
        return
    
    # ÉTAPE 1: IDENTIFIER LA VOIE POUR CHAQUE POLYGONE (RAYON 200m)
    print(f"\n🔎 IDENTIFICATION DES VOIES POUR CHAQUE POLYGONE (rayon 200m)...")
    
    voies_polygones = []
    methodes_detection = []
    distances_moyennes = []
    
    for idx, row in gdf.iterrows():
        polygone = row.geometry
        
        if polygone is None or polygone.is_empty:
            voies_polygones.append(None)
            methodes_detection.append('ERREUR_GEOMETRIE')
            distances_moyennes.append(None)
            continue
        
        print(f"   Polygone {idx+1:4d}/{len(gdf)}: ", end="")
        
        # Méthode 1: Chercher les PK à proximité dans un rayon de 200m
        voie_trouvee = identifier_voie_proche(polygone, gdf_pk_clean, 200)
        methode = 'PK_200M'
        
        # Méthode 2: Si pas trouvé, utiliser un buffer plus large (400m)
        if not voie_trouvee:
            voie_trouvee = identifier_voie_proche(polygone, gdf_pk_clean, 400)
            methode = 'PK_400M'
        
        # Méthode 3: Si toujours pas trouvé, chercher dans les attributs du polygone
        if not voie_trouvee:
            for col in gdf.columns:
                if pd.notna(row[col]):
                    valeur = str(row[col]).strip()
                    # Chercher une correspondance dans les voies PK
                    for voie_pk in voies_liste:
                        if valeur in voie_pk or voie_pk in valeur:
                            voie_trouvee = voie_pk
                            methode = 'ATTRIBUT_CORRESPONDANCE'
                            print(f"      ✓ Trouvé par attribut: {col} = {valeur}")
                            break
                    if voie_trouvee:
                        break
        
        if voie_trouvee:
            # Calculer la distance moyenne aux PK de cette voie
            pk_voie = pk_by_voie.get(voie_trouvee)
            if pk_voie is not None and not pk_voie.empty:
                centroid = polygone.centroid
                distances = pk_voie.geometry.distance(centroid)
                distance_moy = distances.mean()
                distances_moyennes.append(distance_moy)
                print(f"      Distance moyenne: {distance_moy:.1f}m")
            else:
                distances_moyennes.append(None)
        else:
            distances_moyennes.append(None)
            print(f"      ✗ Aucune voie identifiée")
        
        voies_polygones.append(voie_trouvee)
        methodes_detection.append(methode if voie_trouvee else 'NON_TROUVE')
    
    gdf['CODE_AXE'] = voies_polygones
    gdf['METHODE_VOIE'] = methodes_detection
    gdf['DISTANCE_MOY_VOIE'] = distances_moyennes
    
    # Statistiques de détection
    print(f"\n📊 STATISTIQUES D'IDENTIFICATION DES VOIES:")
    methode_counts = pd.Series(methodes_detection).value_counts()
    for methode, count in methode_counts.items():
        pourcentage = count / len(gdf) * 100
        print(f"   {methode:25s}: {count:4d} ({pourcentage:5.1f}%)")
    
    # Statistiques des distances
    dist_valid = [d for d in distances_moyennes if d is not None]
    if dist_valid:
        print(f"\n📏 DISTANCES MOYENNES AUX VOIES:")
        print(f"   Distance moyenne: {np.mean(dist_valid):.1f} m")
        print(f"   Distance médiane: {np.median(dist_valid):.1f} m")
        print(f"   Distance min: {np.min(dist_valid):.1f} m")
        print(f"   Distance max: {np.max(dist_valid):.1f} m")
        
        # Distribution des distances
        print(f"\n📈 DISTRIBUTION DES DISTANCES:")
        bins = [0, 10, 25, 50, 100, 150, 200, 300, 400]
        for i in range(len(bins)-1):
            count = len([d for d in dist_valid if bins[i] <= d < bins[i+1]])
            if count > 0:
                pourcentage = count / len(dist_valid) * 100
                print(f"   {bins[i]:4d}-{bins[i+1]:4d} m: {count:>4} ({pourcentage:5.1f}%)")
    
    # ÉTAPE 2: CALCULER PK_DEB ET PK_FIN (RAYON 200m)
    print(f"\n🚀 CALCUL DES PK_DEB ET PK_FIN (rayon 200m)...")
    print(f"   PK_DEB = point le plus au NORD (haut)")
    print(f"   PK_FIN = point le plus au SUD (bas)")
    print(f"   Recherche dans un rayon de 200m")
    
    # Listes pour stocker les résultats
    pk_deb_list = []
    pk_fin_list = []
    dist_deb_list = []
    dist_fin_list = []
    points_nord_list = []
    points_sud_list = []
    methodes_pk_deb = []
    methodes_pk_fin = []
    
    # Compteurs
    succes_complet = 0
    succes_partiel = 0
    echec = 0
    
    for idx, row in gdf.iterrows():
        polygone = row.geometry
        voie = row['CODE_AXE']
        
        # Vérifier la géométrie et la voie
        if polygone is None or polygone.is_empty or voie is None:
            pk_deb_list.append(None)
            pk_fin_list.append(None)
            dist_deb_list.append(None)
            dist_fin_list.append(None)
            points_nord_list.append(None)
            points_sud_list.append(None)
            methodes_pk_deb.append('ERREUR')
            methodes_pk_fin.append('ERREUR')
            echec += 1
            continue
        
        # 1. Trouver les points extrêmes nord et sud
        point_nord, point_sud = trouver_points_extremes(polygone)
        points_nord_list.append(point_nord)
        points_sud_list.append(point_sud)
        
        # 2. Chercher les PK correspondants dans un rayon de 200m
        pk_deb_val, dist_deb = None, None
        pk_fin_val, dist_fin = None, None
        methode_deb = 'NON_TROUVE'
        methode_fin = 'NON_TROUVE'
        
        if voie in pk_by_voie:
            pk_voie = pk_by_voie[voie]
            
            if not pk_voie.empty:
                # PK_DEB = point NORD (recherche dans 200m)
                pk_deb_val, dist_deb = trouver_pk_le_plus_proche(point_nord, pk_voie, 200)
                if pk_deb_val is not None:
                    methode_deb = 'PK_EXACT_200M'
                else:
                    # Essayer l'interpolation
                    pk_deb_val, dist_deb = calculer_pk_approximatif(point_nord, pk_voie, 200)
                    if pk_deb_val is not None:
                        methode_deb = 'PK_INTERPOLE_200M'
                
                # PK_FIN = point SUD (recherche dans 200m)
                pk_fin_val, dist_fin = trouver_pk_le_plus_proche(point_sud, pk_voie, 200)
                if pk_fin_val is not None:
                    methode_fin = 'PK_EXACT_200M'
                else:
                    # Essayer l'interpolation
                    pk_fin_val, dist_fin = calculer_pk_approximatif(point_sud, pk_voie, 200)
                    if pk_fin_val is not None:
                        methode_fin = 'PK_INTERPOLE_200M'
        
        # Stocker les résultats
        pk_deb_list.append(pk_deb_val)
        pk_fin_list.append(pk_fin_val)
        dist_deb_list.append(dist_deb)
        dist_fin_list.append(dist_fin)
        methodes_pk_deb.append(methode_deb)
        methodes_pk_fin.append(methode_fin)
        
        # Compter les succès
        if pk_deb_val is not None and pk_fin_val is not None:
            succes_complet += 1
        elif pk_deb_val is not None or pk_fin_val is not None:
            succes_partiel += 1
        else:
            echec += 1
    
    # APPLIQUER LES RÉSULTATS
    print(f"\n📊 RÉSULTATS DU CALCUL DES PK:")
    print(f"   Succès complet (PK_DEB + PK_FIN): {succes_complet}")
    print(f"   Succès partiel (un seul PK): {succes_partiel}")
    print(f"   Échec (aucun PK): {echec}")
    print(f"   Total: {len(gdf)}")
    
    # Ajouter les nouveaux champs
    gdf['PK_DEB'] = pk_deb_list
    gdf['PK_FIN'] = pk_fin_list
    gdf['DIST_PK_DEB'] = dist_deb_list
    gdf['DIST_PK_FIN'] = dist_fin_list
    gdf['METHODE_PK_DEB'] = methodes_pk_deb
    gdf['METHODE_PK_FIN'] = methodes_pk_fin
    
    # ANALYSE DES MÉTHODES
    print(f"\n🔍 ANALYSE DES MÉTHODES UTILISÉES:")
    
    print(f"\n   PK_DEB:")
    meth_deb_counts = pd.Series(methodes_pk_deb).value_counts()
    for meth, count in meth_deb_counts.items():
        pourcentage = count / len(gdf) * 100
        print(f"     {meth:20s}: {count:4d} ({pourcentage:5.1f}%)")
    
    print(f"\n   PK_FIN:")
    meth_fin_counts = pd.Series(methodes_pk_fin).value_counts()
    for meth, count in meth_fin_counts.items():
        pourcentage = count / len(gdf) * 100
        print(f"     {meth:20s}: {count:4d} ({pourcentage:5.1f}%)")
    
    # STATISTIQUES DES DISTANCES
    print(f"\n📏 STATISTIQUES DES DISTANCES AUX PK:")
    
    dist_deb_valid = [d for d in dist_deb_list if d is not None]
    dist_fin_valid = [d for d in dist_fin_list if d is not None]
    
    if dist_deb_valid:
        print(f"   PK_DEB - Distance moyenne: {np.mean(dist_deb_valid):.2f} m")
        print(f"   PK_DEB - Distance médiane: {np.median(dist_deb_valid):.2f} m")
        print(f"   PK_DEB - Distance max: {np.max(dist_deb_valid):.2f} m")
        print(f"   PK_DEB - Nombre < 10m: {len([d for d in dist_deb_valid if d <= 10])}")
        print(f"   PK_DEB - Nombre < 50m: {len([d for d in dist_deb_valid if d <= 50])}")
        print(f"   PK_DEB - Nombre < 100m: {len([d for d in dist_deb_valid if d <= 100])}")
    
    if dist_fin_valid:
        print(f"   PK_FIN - Distance moyenne: {np.mean(dist_fin_valid):.2f} m")
        print(f"   PK_FIN - Distance médiane: {np.median(dist_fin_valid):.2f} m")
        print(f"   PK_FIN - Distance max: {np.max(dist_fin_valid):.2f} m")
        print(f"   PK_FIN - Nombre < 10m: {len([d for d in dist_fin_valid if d <= 10])}")
        print(f"   PK_FIN - Nombre < 50m: {len([d for d in dist_fin_valid if d <= 50])}")
        print(f"   PK_FIN - Nombre < 100m: {len([d for d in dist_fin_valid if d <= 100])}")
    
    # CALCULER LA LONGUEUR
    print(f"\n📐 CALCUL DE LA LONGUEUR EN KM:")
    
    gdf['LONGUEUR_KM'] = gdf.apply(
        lambda row: abs(row['PK_FIN'] - row['PK_DEB']) 
        if pd.notna(row['PK_DEB']) and pd.notna(row['PK_FIN']) 
        else None,
        axis=1
    )
    
    longueurs_valid = gdf['LONGUEUR_KM'].dropna()
    if not longueurs_valid.empty:
        print(f"   Longueur moyenne: {longueurs_valid.mean():.3f} km")
        print(f"   Longueur min: {longueurs_valid.min():.3f} km")
        print(f"   Longueur max: {longueurs_valid.max():.3f} km")
        
        print(f"\n📈 DISTRIBUTION DES LONGUEURS:")
        bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
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
            return f"{km:03d}+{meters:03d}"
        except:
            return f"{pk_value:.3f}"
    
    gdf['PK_DEB_STR'] = gdf['PK_DEB'].apply(format_pk_str)
    gdf['PK_FIN_STR'] = gdf['PK_FIN'].apply(format_pk_str)
    
    # EXPORT DES FICHIERS
    print(f"\n💾 EXPORT DES FICHIERS...")
    
    base_dir = os.path.dirname(shp_path)
    base_name = os.path.basename(shp_path)
    name_without_ext = os.path.splitext(base_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Shapefile principal
    new_shp_name = f"{name_without_ext}_PK_200M_{timestamp}.shp"
    new_shp_path = os.path.join(base_dir, new_shp_name)
    
    # Sélectionner les colonnes pour le shapefile
    cols_essentielles = [col for col in gdf.columns if col != 'geometry']
    cols_essentielles = ['geometry'] + [c for c in cols_essentielles if c != 'geometry']
    
    gdf_final = gdf[cols_essentielles].copy()
    
    # Exporter
    gdf_final.to_file(new_shp_path, encoding='UTF-8')
    print(f"   ✅ Shapefile principal: {new_shp_path}")
    print(f"   Nombre d'enregistrements: {len(gdf_final)}")
    
    # Excel avec toutes les informations
    excel_path = os.path.join(base_dir, f"{name_without_ext}_PK_200M_{timestamp}.xlsx")
    
    # Préparer les données pour Excel
    gdf_excel = gdf.copy()
    
    # Convertir les géométries en WKT pour Excel
    gdf_excel['GEOMETRY_WKT'] = gdf_excel.geometry.apply(lambda g: g.wkt if g else None)
    
    # Enlever la colonne geometry pour Excel
    cols_excel = [col for col in gdf_excel.columns if col != 'geometry']
    
    # Trier les colonnes
    cols_prioritaires = ['CODE_AXE', 'PK_DEB', 'PK_FIN', 'PK_DEB_STR', 'PK_FIN_STR', 
                         'LONGUEUR_KM', 'DIST_PK_DEB', 'DIST_PK_FIN', 'DISTANCE_MOY_VOIE',
                         'METHODE_VOIE', 'METHODE_PK_DEB', 'METHODE_PK_FIN']
    
    # Réorganiser les colonnes
    autres_cols = [col for col in cols_excel if col not in cols_prioritaires]
    cols_ordonnees = cols_prioritaires + autres_cols
    
    gdf_excel[cols_ordonnees].to_excel(excel_path, index=False)
    print(f"   ✅ Excel complet: {excel_path}")
    
    # RAPPORT DÉTAILLÉ
    print(f"\n📋 RAPPORT DÉTAILLÉ:")
    
    # Exemples de résultats
    print(f"\n📝 EXEMPLES DE RÉSULTATS (5 polygones avec succès complet):")
    
    succes_mask = (gdf_final['PK_DEB'].notna()) & (gdf_final['PK_FIN'].notna())
    if succes_mask.any():
        exemples = gdf_final[succes_mask].head(5)
        
        for i, (_, row) in enumerate(exemples.iterrows(), 1):
            print(f"\n   Polygone {i}:")
            print(f"     Voie: {row['CODE_AXE']}")
            print(f"     PK_DEB (nord): {row['PK_DEB_STR']} ({row['PK_DEB']:.3f} km)")
            print(f"     PK_FIN (sud): {row['PK_FIN_STR']} ({row['PK_FIN']:.3f} km)")
            if 'LONGUEUR_KM' in row and pd.notna(row['LONGUEUR_KM']):
                print(f"     Longueur: {row['LONGUEUR_KM']:.3f} km")
            if 'DIST_PK_DEB' in row and pd.notna(row['DIST_PK_DEB']):
                print(f"     Distance PK_DEB: {row['DIST_PK_DEB']:.1f} m")
            if 'DIST_PK_FIN' in row and pd.notna(row['DIST_PK_FIN']):
                print(f"     Distance PK_FIN: {row['DIST_PK_FIN']:.1f} m")
            print(f"     Méthode voie: {row['METHODE_VOIE']}")
            print(f"     Méthode PK_DEB: {row['METHODE_PK_DEB']}")
            print(f"     Méthode PK_FIN: {row['METHODE_PK_FIN']}")
    
    # Avertissements pour les problèmes
    print(f"\n⚠️  POLYGONES SANS VOIE IDENTIFIÉE:")
    sans_voie = gdf_final[gdf_final['CODE_AXE'].isna()]
    if len(sans_voie) > 0:
        print(f"   Nombre: {len(sans_voie)}")
        print(f"   Méthodes d'identification: {list(sans_voie['METHODE_VOIE'].unique())}")
    else:
        print(f"   Aucun ✓")
    
    print(f"\n⚠️  POLYGONES SANS PK:")
    sans_pk = gdf_final[(gdf_final['PK_DEB'].isna()) & (gdf_final['PK_FIN'].isna())]
    if len(sans_pk) > 0:
        print(f"   Nombre: {len(sans_pk)}")
    else:
        print(f"   Aucun ✓")
    
    # RAPPORT FINAL
    print(f"\n" + "="*70)
    print("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*70)
    print(f"\n📁 FICHIERS CRÉÉS DANS: {base_dir}")
    print(f"   🗺️  Shapefile principal: {os.path.basename(new_shp_path)}")
    print(f"   📝 Excel complet: {os.path.basename(excel_path)}")
    
    print(f"\n📊 RÉCAPITULATIF:")
    print(f"   • Polygones traités: {len(gdf_final)}")
    print(f"   • Voies identifiées: {gdf_final['CODE_AXE'].notna().sum()}")
    print(f"   • PK_DEB attribués: {gdf_final['PK_DEB'].notna().sum()}")
    print(f"   • PK_FIN attribués: {gdf_final['PK_FIN'].notna().sum()}")
    print(f"   • Longueurs calculées: {gdf_final['LONGUEUR_KM'].notna().sum()}")
    
    print(f"\n🎯 RÈGLES APPLIQUÉES:")
    print(f"   • Identification voie: rayon de 200m autour du polygone")
    print(f"   • Recherche PK: rayon de 200m autour des points nord/sud")
    print(f"   • Interpolation si PK non trouvé exactement")

def main():
    """Fonction principale"""
    print("="*70)
    print("🎯 ASSIGNATION DES PK AUX POLYGONES - RÈGLE 200m")
    print("   Identification automatique des voies + calcul PK nord/sud")
    print("   Rayon de recherche: 200m")
    print("="*70)
    
    # Chemin du fichier
    shp_path = r"C:\Users\ychaker\Desktop\Mesures_Compensatoires_Output\Mesures_compensatoires_single.shp"
    
    # Vérifier existence
    if not os.path.exists(shp_path):
        print(f"\n❌ Fichier non trouvé: {shp_path}")
        
        # Chercher d'autres fichiers
        search_dir = r"C:\Users\ychaker\Desktop"
        fichiers_trouves = []
        
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.shp') and ('mesure' in file.lower() or 'compensatoire' in file.lower()):
                    full_path = os.path.join(root, file)
                    fichiers_trouves.append((file, full_path))
        
        if fichiers_trouves:
            print(f"\n📁 FICHIERS MESURES COMPENSATOIRES TROUVÉS:")
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
        print(f"   Système de coordonnées: {gdf_test.crs}")
        
        # Chercher des champs potentiels pour l'identification
        champs_potentiels = []
        for col in gdf_test.columns:
            if 'voie' in col.lower() or 'route' in col.lower() or 'axe' in col.lower() or 'nom' in col.lower():
                champs_potentiels.append(col)
        
        if champs_potentiels:
            print(f"   Champs potentiels pour identification: {champs_potentiels}")
        
    except Exception as e:
        print(f"   Erreur lors de la lecture: {e}")
    
    # Confirmation
    print(f"\n" + "-"*50)
    print("⚠️  ATTENTION: Ce script va utiliser une règle de 200m:")
    print("   1. Identifier les voies dans un rayon de 200m autour des polygones")
    print("   2. Calculer PK_DEB/PK_FIN dans un rayon de 200m des points nord/sud")
    print("   3. Interpoler si pas de PK exact")
    print("-"*50)
    
    confirm = input("\nConfirmer le traitement? (o/n): ").strip().lower()
    
    if confirm in ['o', 'oui', 'y', 'yes']:
        print(f"\n" + "="*50)
        assigner_pk_aux_polygones(shp_path)
    else:
        print(f"\n❌ Traitement annulé")

if __name__ == "__main__":
    main()