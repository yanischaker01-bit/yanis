import os
from datetime import date, datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, box
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.validation import make_valid

from pystac_client import Client
from odc.stac import load
import rioxarray

# === CONFIGURATION ===
BASSINS_FILE = r"C:/Users/ychaker/Downloads/Bassin/Bassins_inspection.shp"
OUTPUT_CSV = r"C:/Users/ychaker/Desktop/NDWI/ndwi_historique_bassins.csv"
OUTPUT_XLSX = r"C:/Users/ychaker/Desktop/NDWI/ndwi_historique_bassins.xlsx"
OUTPUT_CENTROIDS = r"C:/Users/ychaker/Desktop/NDWI/bassins_centroids.geojson"

# API Microsoft Planetary Computer (plus fiable et rapide)
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
CLOUD_COVER_MAX = 20  # Un peu plus élevé pour avoir plus d'images
DATE_START = "2025-03-01"
DATE_END = date.today().strftime("%Y-%m-%d")
DATE_RANGE = f"{DATE_START}/{DATE_END}"
ABS_MIN_EAU = 0.1
MAX_WORKERS = 6

# === FONCTION POUR RECHERCHER LES IMAGES ===
def search_images_for_bassin(geometry_wgs84, bassin_id):
    """Recherche toutes les images disponibles pour un bassin"""
    client = Client.open(STAC_URL)
    
    # Convertir la géométrie en bbox pour la recherche
    bbox = geometry_wgs84.bounds
    
    try:
        search = client.search(
            collections=[COLLECTION],
            bbox=bbox,  # Utiliser bbox au lieu de intersects pour plus de fiabilité
            datetime=DATE_RANGE,
            query={"eo:cloud_cover": {"lt": CLOUD_COVER_MAX}}
        )
        
        items = list(search.get_items())
        print(f"    📡 {len(items)} images trouvées pour {bassin_id}")
        return items
        
    except Exception as e:
        print(f"    ⚠️ Erreur recherche pour {bassin_id}: {e}")
        return []

# === FONCTION POUR TRAITER UNE IMAGE ===
def process_single_image(bassin_info, item):
    """Traite une seule image pour un bassin"""
    try:
        bassin_id = bassin_info["id"]
        geometry = bassin_info["geometry"]
        geom = mapping(geometry)
        
        # Récupérer la date
        if item.datetime:
            acquisition_date = str(item.datetime.date())
        elif item.properties.get("datetime"):
            acquisition_date = item.properties["datetime"][:10]
        else:
            acquisition_date = "Date_inconnue"
        
        # Vérifier la couverture nuageuse
        cloud_cover = item.properties.get("eo:cloud_cover", 100)
        if cloud_cover > CLOUD_COVER_MAX:
            raise ValueError(f"Couverture nuageuse trop élevée: {cloud_cover}%")
        
        print(f"      📅 Traitement {acquisition_date} ({cloud_cover}% nuages)")
        
        # Charger les données
        data = load(
            [item],
            bands=["green", "nir"],  # Noms standardisés pour Planetary Computer
            geopolygon=geom,
            groupby="solar_day",
            chunks={},
            resolution=10,  # 20m pour plus de stabilité
            fail_on_error=False
        )
        
        if data is None or len(data.time) == 0:
            # Essayer avec les anciens noms de bandes
            data = load(
                [item],
                bands=["B03", "B08"],
                geopolygon=geom,
                groupby="solar_day",
                chunks={},
                resolution=20,
                fail_on_error=False
            )
            
            if data is None or len(data.time) == 0:
                raise ValueError("Données non chargées")
            
            scale = 0.0001
            green = data.B03 * scale
            nir = data.B08 * scale
        else:
            # Les bandes sont déjà dans la bonne échelle pour Planetary Computer
            green = data.green
            nir = data.nir
        
        # Calcul NDWI
        ndwi = (green - nir) / (green + nir + 1e-10)
        ndwi_clean = ndwi.where((ndwi > -1.0) & (ndwi < 1.0)).astype("float32")
        
        # Récupérer le CRS
        try:
            if 'green' in data:
                crs = data.green.rio.crs
            else:
                crs = data.B03.rio.crs
        except:
            crs = "EPSG:32631"  # UTM par défaut
        
        ndwi_clean.rio.write_crs(crs, inplace=True)
        
        # Projeter la géométrie du bassin
        bassin_gdf = gpd.GeoDataFrame({"geometry": [geometry]}, crs="EPSG:4326")
        bassin_proj = bassin_gdf.to_crs(crs)
        mask_geom = mapping(bassin_proj.geometry.iloc[0])
        
        # Clipper
        clipped = ndwi_clean.rio.clip([mask_geom], crs, drop=True, all_touched=True)
        
        # Extraire les valeurs
        arr = clipped.values.flatten()
        arr = arr[~np.isnan(arr)]
        
        if len(arr) == 0:
            raise ValueError("Zone vide après clipping")
        
        # Statistiques
        ndwi_min = float(np.min(arr))
        ndwi_max = float(np.max(arr))
        ndwi_mean = float(np.mean(arr))
        ndwi_std = float(np.std(arr))
        
        # Détection de l'eau avec seuil adaptatif
        # Utiliser le percentile 75 comme seuil minimum
        seuil_adaptatif = max(np.percentile(arr, 75) * 0.3, ABS_MIN_EAU)
        
        nb_total = len(arr)
        nb_eau = len(arr[arr > seuil_adaptatif])
        nb_sec = nb_total - nb_eau
        pct_eau = (nb_eau / nb_total) * 100 if nb_total > 0 else 0
        
        # Classification améliorée
        if nb_eau == 0:
            presence = "À sec"
            niveau = "aucun"
        elif pct_eau > 40:
            presence = "Présence d'eau"
            niveau = "très élevé"
        elif pct_eau > 25:
            presence = "Présence d'eau"
            niveau = "élevé"
        elif pct_eau > 15:
            presence = "Présence d'eau"
            niveau = "moyen"
        elif pct_eau > 5:
            presence = "Présence d'eau"
            niveau = "faible"
        else:
            presence = "Présence d'eau"
            niveau = "très faible"
        
        commentaire = f"{nb_eau}/{nb_total} pixels eau (seuil={seuil_adaptatif:.3f}, NDWI_moy={ndwi_mean:.3f})"
        
        # Créer le résultat
        result = {
            "ID_BASSIN": bassin_id,
            "Nom_du_bassin": bassin_info.get("LIBELLE_OB", bassin_info.get("NOM", bassin_id)),
            "Date_acquisition": acquisition_date,
            "Presence_eau": presence,
            "Niveau_eau": niveau,
            "NDWI_min": ndwi_min,
            "NDWI_max": ndwi_max,
            "NDWI_mean": ndwi_mean,
            "NDWI_std": ndwi_std,
            "Pourcentage_pixels_eau": pct_eau,
            "Nombre_pixels_eau": nb_eau,
            "Nombre_pixels_sec": nb_sec,
            "Total_pixels": nb_total,
            "Seuil_eau_utilise": seuil_adaptatif,
            "Couverture_nuageuse": cloud_cover,
            "Commentaire": commentaire
        }
        
        # Ajouter tous les autres champs du bassin
        for key, value in bassin_info.items():
            if key not in ["id", "geometry"] and key not in result:
                # Nettoyer les noms de colonnes pour Excel
                clean_key = key.replace(" ", "_").replace("-", "_").replace(".", "_")
                result[clean_key] = value
        
        return result
        
    except Exception as e:
        print(f"      ❌ Erreur traitement image: {e}")
        # En cas d'erreur, retourner un résultat minimal
        result = {
            "ID_BASSIN": bassin_info["id"],
            "Nom_du_bassin": bassin_info.get("LIBELLE_OB", bassin_info.get("NOM", bassin_info["id"])),
            "Date_acquisition": acquisition_date if 'acquisition_date' in locals() else "N/A",
            "Presence_eau": "Donnée manquante",
            "Niveau_eau": "N/A",
            "NDWI_min": np.nan,
            "NDWI_max": np.nan,
            "NDWI_mean": np.nan,
            "NDWI_std": np.nan,
            "Pourcentage_pixels_eau": np.nan,
            "Nombre_pixels_eau": 0,
            "Nombre_pixels_sec": 0,
            "Total_pixels": 0,
            "Seuil_eau_utilise": ABS_MIN_EAU,
            "Couverture_nuageuse": cloud_cover if 'cloud_cover' in locals() else np.nan,
            "Commentaire": f"Erreur: {str(e)[:150]}"
        }
        
        # Ajouter tous les autres champs du bassin
        for key, value in bassin_info.items():
            if key not in ["id", "geometry"] and key not in result:
                clean_key = key.replace(" ", "_").replace("-", "_").replace(".", "_")
                result[clean_key] = value
        
        return result

# === FONCTION PRINCIPALE POUR UN BASSIN ===
def process_bassin_historique(row_dict):
    """Traite l'historique complet d'un bassin depuis mars 2025"""
    idx, row = row_dict
    bassin_id = row.get("ID", row.get("LIBELLE_OB", f"Bassin_{idx}"))
    
    print(f"\n🔍 Traitement du bassin: {bassin_id}")
    
    # Préparer les informations du bassin
    bassin_info = {
        "id": bassin_id,
        "geometry": row["geometry"]
    }
    
    # Ajouter tous les attributs du bassin
    for key, value in row.items():
        if key != "geometry":
            bassin_info[key] = value
    
    results = []
    
    try:
        # 1. Rechercher toutes les images disponibles
        all_items = search_images_for_bassin(row["geometry"], bassin_id)
        
        if not all_items:
            print(f"  ⚠️  Aucune image trouvée pour {bassin_id}")
            return create_empty_result(bassin_info)
        
        # 2. Grouper par mois pour éviter la redondance
        items_by_month = {}
        for item in all_items:
            if item.datetime:
                date_obj = item.datetime.date()
                month_key = f"{date_obj.year}-{date_obj.month:02d}"
                
                # Garder l'image avec la plus faible couverture nuageuse par mois
                cloud_cover = item.properties.get("eo:cloud_cover", 100)
                
                if month_key not in items_by_month or cloud_cover < items_by_month[month_key][1]:
                    items_by_month[month_key] = (item, cloud_cover)
        
        # 3. Sélectionner les images à traiter (max 1 par mois)
        selected_items = []
        for month_key, (item, cloud_cover) in sorted(items_by_month.items()):
            if cloud_cover <= CLOUD_COVER_MAX:
                selected_items.append(item)
            if len(selected_items) >= 8:  # Maximum 8 mois d'historique
                break
        
        # 4. Ajouter les 3 dernières images disponibles (même si même mois)
        # Pour avoir une meilleure temporalité
        recent_items = sorted(all_items, 
                             key=lambda x: x.datetime if x.datetime else datetime.min, 
                             reverse=True)[:3]
        
        # Combiner en évitant les doublons par date
        final_items = []
        dates_seen = set()
        
        for item in recent_items + selected_items:
            if item.datetime:
                date_str = str(item.datetime.date())
                if date_str not in dates_seen:
                    final_items.append(item)
                    dates_seen.add(date_str)
        
        # Limiter à 10 images maximum
        final_items = final_items[:10]
        
        print(f"  📊 {len(final_items)} images sélectionnées pour traitement")
        
        # 5. Traiter chaque image
        for i, item in enumerate(final_items, 1):
            print(f"  [{i}/{len(final_items)}] Traitement en cours...")
            result = process_single_image(bassin_info, item)
            results.append(result)
        
        print(f"  ✅ {len(results)} dates traitées pour {bassin_id}")
        
    except Exception as e:
        print(f"  ❌ Erreur générale pour {bassin_id}: {e}")
        results = [create_empty_result(bassin_info, error_msg=str(e))]
    
    return results

def create_empty_result(bassin_info, error_msg="Aucune image disponible"):
    """Crée un résultat vide pour un bassin"""
    bassin_id = bassin_info["id"]
    
    result = {
        "ID_BASSIN": bassin_id,
        "Nom_du_bassin": bassin_info.get("LIBELLE_OB", bassin_info.get("NOM", bassin_id)),
        "Date_acquisition": "N/A",
        "Presence_eau": "Donnée manquante",
        "Niveau_eau": "N/A",
        "NDWI_min": np.nan,
        "NDWI_max": np.nan,
        "NDWI_mean": np.nan,
        "NDWI_std": np.nan,
        "Pourcentage_pixels_eau": np.nan,
        "Nombre_pixels_eau": 0,
        "Nombre_pixels_sec": 0,
        "Total_pixels": 0,
        "Seuil_eau_utilise": ABS_MIN_EAU,
        "Couverture_nuageuse": np.nan,
        "Commentaire": error_msg
    }
    
    # Ajouter tous les autres champs du bassin
    for key, value in bassin_info.items():
        if key not in ["id", "geometry"] and key not in result:
            clean_key = key.replace(" ", "_").replace("-", "_").replace(".", "_")
            result[clean_key] = value
    
    return result

# === MAIN ===
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 DÉMARRAGE DU TRAITEMENT NDWI HISTORIQUE")
    print("=" * 70)
    print(f"📅 Période: {DATE_START} au {DATE_END}")
    print(f"🌐 API utilisée: Microsoft Planetary Computer")
    print(f"📊 Collection: {COLLECTION}")
    print(f"☁️  Couverture nuageuse max: {CLOUD_COVER_MAX}%")
    print("=" * 70)
    
    # 1. Chargement des bassins
    print("\n📂 CHARGEMENT DES BASSINS...")
    bassins = gpd.read_file(BASSINS_FILE)
    
    # Vérifier et définir le CRS
    if bassins.crs is None:
        bassins = bassins.set_crs(epsg=4326)
        print("⚠️  CRS défini à WGS84 (EPSG:4326)")
    elif bassins.crs != "EPSG:4326":
        print(f"🔄 Conversion du CRS {bassins.crs} vers WGS84")
        bassins = bassins.to_crs(epsg=4326)
    
    # Simplifier légèrement les géométries
    print("🔧 Simplification des géométries...")
    bassins["geometry"] = bassins["geometry"].apply(
        lambda g: g.simplify(0.0001, preserve_topology=True) if g else g
    )
    
    # Créer un ID si nécessaire
    if "ID" not in bassins.columns:
        bassins["ID"] = [f"Bassin_{i:04d}" for i in range(len(bassins))]
    
    print(f"✅ {len(bassins)} bassins chargés")
    print(f"📋 Colonnes: {list(bassins.columns)}")
    
    # 2. Préparer les données pour le traitement
    print("\n⚡ PRÉPARATION DU TRAITEMENT PARALLÈLE...")
    rows = []
    for idx in bassins.index:
        row_dict = bassins.loc[idx].to_dict()
        row_dict["geometry"] = bassins.loc[idx, "geometry"]
        rows.append((idx, row_dict))
    
    # 3. Traitement parallèle
    print(f"\n⚡ TRAITEMENT PARALLÈLE SUR {MAX_WORKERS} WORKERS...")
    all_results = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_bassin_historique, row): row for row in rows}
        
        for i, future in enumerate(as_completed(futures), 1):
            try:
                results = future.result()
                all_results.extend(results)
                
                # Récupérer l'ID du bassin
                idx, row_dict = futures[future]
                bassin_id = row_dict.get("ID", f"Bassin_{idx}")
                
                dates_count = len([r for r in results if r.get("Date_acquisition", "N/A") != "N/A"])
                print(f"✅ [{i:3d}/{len(bassins)}] {bassin_id:20} → {dates_count:2d} dates traitées")
                
            except Exception as e:
                print(f"❌ [{i:3d}/{len(bassins)}] Erreur: {e}")
                # Ajouter un résultat d'erreur
                idx, row_dict = futures[future]
                bassin_id = row_dict.get("ID", f"Bassin_{idx}")
                all_results.append(create_empty_result(
                    {"id": bassin_id, **row_dict}, 
                    error_msg=f"Erreur traitement: {str(e)[:100]}"
                ))
    
    # 4. Création du DataFrame
    print("\n💾 CRÉATION DU DATAFRAME...")
    df_results = pd.DataFrame(all_results)
    
    # Vérifier qu'on a des données
    if len(df_results) == 0:
        print("❌ AUCUNE DONNÉE GÉNÉRÉE!")
        exit(1)
    
    print(f"✅ {len(df_results)} observations générées")
    
    # Trier par bassin puis date
    if "Date_acquisition" in df_results.columns and "ID_BASSIN" in df_results.columns:
        print("📅 Tri par date...")
        df_results["Date_tri"] = pd.to_datetime(df_results["Date_acquisition"], errors='coerce')
        df_results = df_results.sort_values(["ID_BASSIN", "Date_tri"], ascending=[True, False])
        df_results = df_results.drop(columns=["Date_tri"])
    
    # 5. Sauvegarde CSV
    print(f"\n📄 SAUVEGARDE CSV: {OUTPUT_CSV}")
    df_results.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Fichier CSV créé ({os.path.getsize(OUTPUT_CSV)/1024:.1f} KB)")
    
    # 6. Statistiques détaillées
    print("\n📊 STATISTIQUES DÉTAILLÉES")
    print("-" * 50)
    
    # Dates couvertes
    valid_dates = df_results[df_results["Date_acquisition"] != "N/A"]["Date_acquisition"]
    if len(valid_dates) > 0:
        unique_dates = valid_dates.unique()
        print(f"• Dates uniques: {len(unique_dates)}")
        print(f"• Période: {min(unique_dates)} à {max(unique_dates)}")
    
    # Présence d'eau
    if "Presence_eau" in df_results.columns:
        water_stats = df_results["Presence_eau"].value_counts()
        print(f"• Présence d'eau: {dict(water_stats)}")
    
    # Niveaux d'eau
    if "Niveau_eau" in df_results.columns:
        level_stats = df_results["Niveau_eau"].value_counts()
        print(f"• Niveaux d'eau: {dict(level_stats)}")
    
    # Par bassin
    bassins_with_data = df_results[df_results["Date_acquisition"] != "N/A"]["ID_BASSIN"].nunique()
    print(f"• Bassins avec données: {bassins_with_data}/{len(bassins)}")
    
    # 7. Sauvegarde Excel avec onglets détaillés
    try:
        print(f"\n📊 CRÉATION DU FICHIER EXCEL: {OUTPUT_XLSX}")
        
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            # Onglet 1: Toutes les données
            df_results.to_excel(writer, sheet_name="Historique_Complet", index=False)
            print("  ✅ Onglet 'Historique_Complet' créé")
            
            # Onglet 2: Situation actuelle (dernière observation par bassin)
            latest_data = df_results.drop_duplicates(subset=["ID_BASSIN"], keep="first")
            latest_data.to_excel(writer, sheet_name="Situation_Actuelle", index=False)
            print(f"  ✅ Onglet 'Situation_Actuelle' créé ({len(latest_data)} bassins)")
            
            # Onglet 3: Résumé mensuel
            if "Date_acquisition" in df_results.columns:
                df_temp = df_results.copy()
                df_temp["Date"] = pd.to_datetime(df_temp["Date_acquisition"], errors='coerce')
                df_temp["Mois"] = df_temp["Date"].dt.to_period('M').astype(str)
                
                monthly_summary = df_temp.groupby(["ID_BASSIN", "Mois"]).agg({
                    "Pourcentage_pixels_eau": "mean",
                    "NDWI_mean": "mean",
                    "Presence_eau": "last"
                }).reset_index()
                
                monthly_summary.to_excel(writer, sheet_name="Resume_Mensuel", index=False)
                print("  ✅ Onglet 'Resume_Mensuel' créé")
            
            # Onglet 4: Statistiques par bassin
            summary_stats = []
            for bassin in df_results["ID_BASSIN"].unique():
                bassin_data = df_results[df_results["ID_BASSIN"] == bassin]
                valid_data = bassin_data[bassin_data["Date_acquisition"] != "N/A"]
                
                if len(valid_data) > 0:
                    summary_stats.append({
                        "ID_BASSIN": bassin,
                        "Nom": valid_data.iloc[0].get("Nom_du_bassin", ""),
                        "Premiere_date": valid_data["Date_acquisition"].min(),
                        "Derniere_date": valid_data["Date_acquisition"].max(),
                        "Nb_dates": len(valid_data),
                        "Pct_eau_moyen": valid_data["Pourcentage_pixels_eau"].mean(),
                        "Pct_eau_max": valid_data["Pourcentage_pixels_eau"].max(),
                        "NDWI_moyen": valid_data["NDWI_mean"].mean(),
                        "Statut_actuel": valid_data.iloc[0].get("Presence_eau", ""),
                        "Niveau_actuel": valid_data.iloc[0].get("Niveau_eau", "")
                    })
            
            if summary_stats:
                summary_df = pd.DataFrame(summary_stats)
                summary_df.to_excel(writer, sheet_name="Statistiques_Bassins", index=False)
                print(f"  ✅ Onglet 'Statistiques_Bassins' créé ({len(summary_stats)} bassins)")
            
        print(f"✅ Fichier Excel créé ({os.path.getsize(OUTPUT_XLSX)/1024:.1f} KB)")
        
    except Exception as e:
        print(f"⚠️ Erreur création Excel: {e}")
    
    # 8. Sauvegarde des centroïdes
    try:
        print(f"\n🗺️  SAUVEGARDE DES CENTROÏDES: {OUTPUT_CENTROIDS}")
        
        centroids_data = []
        for idx, row in bassins.iterrows():
            if hasattr(row.geometry, 'centroid'):
                centroid = row.geometry.centroid
                centroids_data.append({
                    "ID": row.get("ID", f"Bassin_{idx}"),
                    "Nom": row.get("LIBELLE_OB", row.get("NOM", f"Bassin_{idx}")),
                    "Longitude": centroid.x,
                    "Latitude": centroid.y,
                    "geometry": centroid
                })
        
        centroids_gdf = gpd.GeoDataFrame(centroids_data, crs="EPSG:4326")
        centroids_gdf.to_file(OUTPUT_CENTROIDS, driver="GeoJSON")
        print(f"✅ Fichier GeoJSON créé ({len(centroids_data)} centroïdes)")
        
    except Exception as e:
        print(f"⚠️ Erreur création centroïdes: {e}")
    
    # 9. Rapport final
    print("\n" + "=" * 70)
    print("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS!")
    print("=" * 70)
    print(f"📁 Fichier CSV: {OUTPUT_CSV}")
    print(f"📁 Fichier Excel: {OUTPUT_XLSX}")
    print(f"🗺️  Fichier GeoJSON: {OUTPUT_CENTROIDS}")
    print(f"📊 Total observations: {len(df_results)}")
    print(f"📈 Bassins traités: {df_results['ID_BASSIN'].nunique()}")
    
    # Dates disponibles
    valid_dates = df_results[df_results["Date_acquisition"] != "N/A"]["Date_acquisition"]
    if len(valid_dates) > 0:
        print(f"📅 Dates disponibles: {len(valid_dates.unique())}")
        print(f"📅 Période couverte: {min(valid_dates)} à {max(valid_dates)}")
    
    print("=" * 70)