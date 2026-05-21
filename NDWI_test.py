import os
from datetime import date, datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point
from concurrent.futures import ProcessPoolExecutor, as_completed

from pystac_client import Client
from odc.stac import load
import rioxarray

# === CONFIGURATION ===
BASSINS_FILE = r"C:/Users/ychaker/Downloads/Bassin/Bassins_inspection.shp"
OUTPUT_CSV = r"C:/Users/ychaker/Desktop/NDWI/ndwi_historique_bassins.csv"
OUTPUT_XLSX = r"C:/Users/ychaker/Desktop/NDWI/ndwi_historique_bassins.xlsx"
OUTPUT_CENTROIDS = r"C:/Users/ychaker/Desktop/NDWI/bassins_centroids.geojson"

COLLECTION = "sentinel-2-l2a"
STAC_URL = "https://earth-search.aws.element84.com/v1"
CLOUD_COVER_MAX = 10
DATE_RANGE = f"2025-01-01/{date.today()}"
ABS_MIN_EAU = 0.1
MAX_WORKERS = 6

# === FONCTION POUR CALCULER LES CENTROÏDES ===
def calculate_centroids(bassins_gdf):
    centroids = {}
    if bassins_gdf.crs is None:
        bassins_gdf = bassins_gdf.set_crs(epsg=4326)
    for idx, row in bassins_gdf.iterrows():
        bassin_id = row.get("ID", row.get("LIBELLE_OB", f"Bassin_{idx}"))
        try:
            if hasattr(row.geometry, 'centroid'):
                centroid = row.geometry.centroid
                x_coord = centroid.x
                y_coord = centroid.y
                centroids[bassin_id] = {
                    "X_WGS84": x_coord,
                    "Y_WGS84": y_coord,
                    "X_LAMBERT": None,
                    "Y_LAMBERT": None
                }
                try:
                    bassins_lambert = bassins_gdf.to_crs(epsg=2154)
                    row_lambert = bassins_lambert.loc[idx]
                    if hasattr(row_lambert.geometry, 'centroid'):
                        centroid_lambert = row_lambert.geometry.centroid
                        centroids[bassin_id]["X_LAMBERT"] = centroid_lambert.x
                        centroids[bassin_id]["Y_LAMBERT"] = centroid_lambert.y
                except:
                    pass
        except Exception as e:
            print(f"⚠️ Erreur calcul centroïde pour {bassin_id}: {e}")
            centroids[bassin_id] = {"X_WGS84": 0.0, "Y_WGS84": 0.0, "X_LAMBERT": None, "Y_LAMBERT": None}
    return centroids

# === FONCTION POUR TRAITER UN BASSIN ET UNE IMAGE ===
def process_bassin_image(bassin_info, item, centroids_dict):
    try:
        bassin_id = bassin_info["id"]
        geometry = bassin_info["geometry"]
        geom = mapping(geometry)
        acquisition_date = str(item.datetime.date()) if item.datetime else item.properties.get("datetime", "Date_inconnue")[:10]

        data = load([item], bands=["B03", "B08"], geopolygon=geom, groupby="solar_day", chunks={})
        if data is None or len(data.time) == 0:
            raise ValueError("Données non chargées")

        scale = 0.0001
        green = data.B03 * scale
        nir = data.B08 * scale
        ndwi = (green - nir) / (green + nir + 1e-10)
        ndwi_clean = ndwi.where((ndwi > -1.0) & (ndwi < 1.0)).astype("float32")
        crs = data.B03.rio.crs
        ndwi_clean.rio.write_crs(crs, inplace=True)

        bassin_proj = gpd.GeoSeries([geometry], crs=4326).to_crs(crs)
        mask_geom = mapping(bassin_proj.iloc[0])
        clipped = ndwi_clean.rio.clip([mask_geom], crs, drop=True)
        arr = clipped.values[0]
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            raise ValueError("Zone vide")

        eau = arr[arr > ABS_MIN_EAU]
        nb_eau = len(eau)
        nb_total = len(arr)
        nb_sec = nb_total - nb_eau
        pct_eau = (nb_eau / nb_total) * 100
        ndwi_max = float(np.max(arr))
        ndwi_min = float(np.min(arr))
        ndwi_mean = float(np.mean(arr))

        if nb_eau == 0:
            presence, niveau, commentaire = "À sec", "aucun", "Aucun pixel en eau"
        else:
            presence = "Présence d'eau"
            ratio = nb_eau / nb_total
            niveau = (
                "élevé" if ratio >= 1/4 else
                "moyen" if ratio >= 1/5 else
                "faible" if ratio >= 1/6 else
                "très faible"
            )
            commentaire = f"{nb_eau} pixels en eau (NDWI max = {ndwi_max:.2f})"

        centroid_data = centroids_dict.get(bassin_id, {})

        result = {
            "ID_BASSIN": bassin_id,
            "Nom_du_bassin": bassin_info.get("LIBELLE_OB", bassin_id),
            "Date_acquisition": acquisition_date,
            "Presence_eau": presence,
            "Niveau_eau": niveau,
            "NDWI_min": ndwi_min,
            "NDWI_max": ndwi_max,
            "NDWI_mean": ndwi_mean,
            "Pourcentage_pixels_eau": pct_eau,
            "Nombre_pixels_eau": nb_eau,
            "Nombre_pixels_sec": nb_sec,
            "Total_pixels": nb_total,
            "Commentaire": commentaire,
            "Seuil_eau_utilise": ABS_MIN_EAU,
            "X_WGS84": centroid_data.get("X_WGS84", 0.0),
            "Y_WGS84": centroid_data.get("Y_WGS84", 0.0),
            "X_LAMBERT": centroid_data.get("X_LAMBERT"),
            "Y_LAMBERT": centroid_data.get("Y_LAMBERT"),
            "Annee": acquisition_date[:4] if acquisition_date != "Date_inconnue" else "N/A",
            "Mois": acquisition_date[5:7] if acquisition_date != "Date_inconnue" else "N/A",
            "Jour": acquisition_date[8:10] if acquisition_date != "Date_inconnue" else "N/A",
            "Date_complete": pd.to_datetime(acquisition_date, errors='coerce') if acquisition_date != "Date_inconnue" else pd.NaT
        }

        for key, value in bassin_info.items():
            if key not in ["id", "geometry"] and key not in result:
                clean_key = key.replace(" ", "_").replace("-", "_").replace(".", "_")
                result[clean_key] = value

        return result

    except Exception as e:
        centroid_data = centroids_dict.get(bassin_info["id"], {})
        result = {
            "ID_BASSIN": bassin_info["id"],
            "Nom_du_bassin": bassin_info.get("LIBELLE_OB", bassin_info["id"]),
            "Date_acquisition": acquisition_date if 'acquisition_date' in locals() else "N/A",
            "Presence_eau": "Donnée manquante",
            "Niveau_eau": "N/A",
            "NDWI_min": np.nan,
            "NDWI_max": np.nan,
            "NDWI_mean": np.nan,
            "Pourcentage_pixels_eau": np.nan,
            "Nombre_pixels_eau": 0,
            "Nombre_pixels_sec": 0,
            "Total_pixels": 0,
            "Commentaire": f"Erreur : {str(e)[:100]}",
            "Seuil_eau_utilise": ABS_MIN_EAU,
            "X_WGS84": centroid_data.get("X_WGS84", 0.0),
            "Y_WGS84": centroid_data.get("Y_WGS84", 0.0),
            "X_LAMBERT": centroid_data.get("X_LAMBERT"),
            "Y_LAMBERT": centroid_data.get("Y_LAMBERT"),
            "Annee": "N/A",
            "Mois": "N/A",
            "Jour": "N/A",
            "Date_complete": pd.NaT
        }

        for key, value in bassin_info.items():
            if key not in ["id", "geometry"] and key not in result:
                clean_key = key.replace(" ", "_").replace("-", "_").replace(".", "_")
                result[clean_key] = value

        return result

# === FONCTION POUR TRAITER TOUTES LES IMAGES D'UN BASSIN ===
def process_bassin_all_dates(row_dict, centroids_dict):
    idx, row = row_dict
    bassin_id = row.get("ID", row.get("LIBELLE_OB", f"Bassin_{idx}"))
    print(f"  🔍 Traitement du bassin: {bassin_id}")

    bassin_info = {"id": bassin_id, "geometry": row["geometry"]}
    for key, value in row.items():
        if key != "geometry":
            bassin_info[key] = value

    results = []
    try:
        client = Client.open(STAC_URL)
        geom = mapping(row["geometry"])
        search = client.search(
            collections=[COLLECTION],
            intersects=geom,
            datetime=DATE_RANGE,
            query={"eo:cloud_cover": {"lt": CLOUD_COVER_MAX}},
            max_items=500
        )
        items = list(search.get_items())
        if not items:
            print(f"  ⚠️  Aucune image pour {bassin_id}")
            result = bassin_info.copy()
            centroid_data = centroids_dict.get(bassin_id, {})
            result.update({
                "ID_BASSIN": bassin_id,
                "Nom_du_bassin": bassin_info.get("LIBELLE_OB", bassin_id),
                "Date_acquisition": "N/A",
                "Presence_eau": "Donnée manquante",
                "Niveau_eau": "N/A",
                "NDWI_min": np.nan,
                "NDWI_max": np.nan,
                "NDWI_mean": np.nan,
                "Pourcentage_pixels_eau": np.nan,
                "Nombre_pixels_eau": 0,
                "Nombre_pixels_sec": 0,
                "Total_pixels": 0,
                "Commentaire": "Aucune image disponible",
                "Seuil_eau_utilise": ABS_MIN_EAU,
                "X_WGS84": centroid_data.get("X_WGS84", 0.0),
                "Y_WGS84": centroid_data.get("Y_WGS84", 0.0),
                "X_LAMBERT": centroid_data.get("X_LAMBERT"),
                "Y_LAMBERT": centroid_data.get("Y_LAMBERT"),
                "Annee": "N/A",
                "Mois": "N/A",
                "Jour": "N/A",
                "Date_complete": pd.NaT
            })
            result.pop("geometry", None)
            result.pop("id", None)
            for key, value in bassin_info.items():
                if key not in ["id", "geometry"] and key not in result:
                    clean_key = key.replace(" ", "_").replace("-", "_").replace(".", "_")
                    result[clean_key] = value
            return [result]

        # Trier toutes les images par date décroissante
        items = sorted(items, key=lambda x: x.datetime if x.datetime else datetime.min, reverse=True)

        # Traiter toutes les images disponibles
        for i, item in enumerate(items):
            print(f"    📊 Traitement image {i+1}/{len(items)}")
            result = process_bassin_image(bassin_info, item, centroids_dict)
            results.append(result)

    except Exception as e:
        print(f"  ❌ Erreur générale pour {bassin_id}: {e}")
        result = bassin_info.copy()
        centroid_data = centroids_dict.get(bassin_id, {})
        result.update({
            "ID_BASSIN": bassin_id,
            "Nom_du_bassin": bassin_info.get("LIBELLE_OB", bassin_id),
            "Date_acquisition": "N/A",
            "Presence_eau": "Erreur",
            "Niveau_eau": "N/A",
            "NDWI_min": np.nan,
            "NDWI_max": np.nan,
            "NDWI_mean": np.nan,
            "Pourcentage_pixels_eau": np.nan,
            "Nombre_pixels_eau": 0,
            "Nombre_pixels_sec": 0,
            "Total_pixels": 0,
            "Commentaire": f"Erreur traitement: {str(e)[:100]}",
            "Seuil_eau_utilise": ABS_MIN_EAU,
            "X_WGS84": centroid_data.get("X_WGS84", 0.0),
            "Y_WGS84": centroid_data.get("Y_WGS84", 0.0),
            "X_LAMBERT": centroid_data.get("X_LAMBERT"),
            "Y_LAMBERT": centroid_data.get("Y_LAMBERT"),
            "Annee": "N/A",
            "Mois": "N/A",
            "Jour": "N/A",
            "Date_complete": pd.NaT
        })
        result.pop("geometry", None)
        result.pop("id", None)
        for key, value in bassin_info.items():
            if key not in ["id", "geometry"] and key not in result:
                clean_key = key.replace(" ", "_").replace("-", "_").replace(".", "_")
                result[clean_key] = value
        return [result]

    return results

# === MAIN ===
if __name__ == "__main__":
    print("🚀 Démarrage du traitement NDWI historique...")
    print(f"📅 Période: {DATE_RANGE}")

    bassins = gpd.read_file(BASSINS_FILE)
    if bassins.crs is None:
        bassins = bassins.set_crs(epsg=4326)
    else:
        bassins = bassins.to_crs(epsg=4326)

    def clean_geometry(geom):
        try:
            return geom.simplify(tolerance=0.000001, preserve_topology=True)
        except:
            return geom
    bassins["geometry"] = bassins["geometry"].apply(clean_geometry)

    if "ID" not in bassins.columns:
        bassins["ID"] = [f"Bassin_{i:04d}" for i in range(len(bassins))]

    print("📍 Calcul des centroïdes pour Power BI...")
    centroids_dict = calculate_centroids(bassins)
    print(f"✅ Centroïdes calculés pour {len(centroids_dict)} bassins")

    rows = [(idx, bassins.loc[idx].to_dict()) for idx in bassins.index]
    for i, (idx, row) in enumerate(rows):
        row["geometry"] = bassins.loc[idx, "geometry"]

    print(f"\n⚡ Traitement parallèle sur {MAX_WORKERS} workers...")
    all_results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_bassin_all_dates, row, centroids_dict): row for row in rows}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                results = future.result()
                all_results.extend(results)
                idx, row_dict = futures[future]
                bassin_id = row_dict.get("ID", f"Bassin_{idx}")
                dates_count = len([r for r in results if r.get("Date_acquisition", "N/A") != "N/A"])
                print(f"  ✅ {i:3d}/{len(bassins)} - {bassin_id:20} → {dates_count:2d} dates traitées")
            except Exception as e:
                idx, row_dict = futures[future]
                bassin_id = row_dict.get("ID", f"Bassin_{idx}")
                centroid_data = centroids_dict.get(bassin_id, {})
                result = {
                    "ID_BASSIN": bassin_id,
                    "Nom_du_bassin": row_dict.get("LIBELLE_OB", bassin_id),
                    "Date_acquisition": "N/A",
                    "Presence_eau": "Erreur traitement",
                    "Niveau_eau": "N/A",
                    "NDWI_min": np.nan,
                    "NDWI_max": np.nan,
                    "NDWI_mean": np.nan,
                    "Pourcentage_pixels_eau": np.nan,
                    "Nombre_pixels_eau": 0,
                    "Nombre_pixels_sec": 0,
                    "Total_pixels": 0,
                    "Commentaire": f"Erreur majeure: {str(e)[:100]}",
                    "Seuil_eau_utilise": ABS_MIN_EAU,
                    "X_WGS84": centroid_data.get("X_WGS84", 0.0),
                    "Y_WGS84": centroid_data.get("Y_WGS84", 0.0),
                    "X_LAMBERT": centroid_data.get("X_LAMBERT"),
                    "Y_LAMBERT": centroid_data.get("Y_LAMBERT"),
                    "Annee": "N/A",
                    "Mois": "N/A",
                    "Jour": "N/A",
                    "Date_complete": pd.NaT
                }
                for key, value in row_dict.items():
                    if key != "geometry" and key not in result:
                        clean_key = key.replace(" ", "_").replace("-", "_").replace(".", "_")
                        result[clean_key] = value
                all_results.append(result)

    print("\n💾 Création du DataFrame...")
    df_results = pd.DataFrame(all_results)

    # Réorganiser colonnes pour Power BI
    powerbi_columns = [
        "ID_BASSIN", "Nom_du_bassin", "Date_acquisition",
        "Presence_eau", "Niveau_eau",
        "NDWI_min", "NDWI_max", "NDWI_mean",
        "Pourcentage_pixels_eau",
        "Nombre_pixels_eau", "Nombre_pixels_sec", "Total_pixels",
        "Commentaire", "Seuil_eau_utilise",
        "X_WGS84", "Y_WGS84", "X_LAMBERT", "Y_LAMBERT",
        "Annee", "Mois", "Jour", "Date_complete"
    ]
    powerbi_columns = [col for col in powerbi_columns if col in df_results.columns]
    other_cols = [col for col in df_results.columns if col not in powerbi_columns]
    df_results = df_results[powerbi_columns + other_cols]

    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ CSV sauvegardé: {OUTPUT_CSV}")

    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="NDWI_historique", index=False)
    print(f"✅ Excel sauvegardé: {OUTPUT_XLSX}")

    print("🌐 Sauvegarde des centroïdes en GeoJSON...")
    centroids_gdf = gpd.GeoDataFrame(
        df_results[["ID_BASSIN", "Nom_du_bassin", "X_WGS84", "Y_WGS84", "X_LAMBERT", "Y_LAMBERT"]].drop_duplicates(),
        geometry=gpd.points_from_xy(df_results["X_WGS84"], df_results["Y_WGS84"]),
        crs="EPSG:4326"
    )
    centroids_gdf.to_file(OUTPUT_CENTROIDS, driver="GeoJSON")
    print(f"✅ GeoJSON sauvegardé: {OUTPUT_CENTROIDS}")