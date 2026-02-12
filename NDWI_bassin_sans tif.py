from pystac_client import Client
from odc.stac import load
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
import pandas as pd
import rioxarray
import warnings

warnings.filterwarnings("ignore")

# === CONFIGURATION ===
input_file = "C:/Users/ychaker/Downloads/kmz/Emprise COSEA.geojson"
bassins_file = "C:/Users/ychaker/Downloads/Bassin/Bassins.shp"
output_csv = "C:/Users/ychaker/Desktop/NDWI/ndwi_bassins.csv"
date_range = "2025-01-01/2025-02-01"
collection = "sentinel-2-l2a"
client = Client.open("https://earth-search.aws.element84.com/v1")

# === 1. CHARGER LES DONNÉES GÉOGRAPHIQUES ===
aoi = gpd.read_file(input_file).to_crs(4326)
bassins = gpd.read_file(bassins_file).to_crs(4326)

def clean_geometry(geom):
    try:
        return geom.simplify(tolerance=0.0000001, preserve_topology=True)
    except:
        return geom

bassins["geometry"] = bassins["geometry"].apply(clean_geometry)

results = []

# === 2. TRAITEMENT PAR BASSIN ===
for idx, row in bassins.iterrows():
    geom = mapping(row.geometry)

    search = client.search(
        collections=[collection],
        intersects=geom,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 10}}
    )
    items = list(search.items())

    if not items:
        results.append([
            row.get("LIBELLE_OB", f"Bassin_{idx}"),
            "Donnée manquante", "N/A", np.nan, np.nan,
            0, 0, 0, "Image non disponible", "N/A"
        ])
        continue

    item = sorted(items, key=lambda x: x.datetime, reverse=True)[0]
    acquisition_date = str(item.datetime.date())

    data = load(
        [item],
        bands=["B03", "B08"],
        geopolygon=geom,
        groupby="solar_day",
        chunks={}
    )

    scale = 0.0001
    green = data.B03 * scale
    nir = data.B08 * scale
    ndwi = (green - nir) / (green + nir)
    ndwi_clean = ndwi.where((ndwi > -1.0) & (ndwi < 1.0)).astype("float32")

    # ✅ Écriture du CRS depuis la bande B03
    crs = data.B03.rio.crs
    ndwi_clean.rio.write_crs(crs, inplace=True)

    try:
        bassin_proj = gpd.GeoSeries([row.geometry], crs=4326).to_crs(crs)
        mask_geom = mapping(bassin_proj.iloc[0])

        clipped = ndwi_clean.rio.clip([mask_geom], crs, drop=True)
        arr = clipped.values[0]
        arr = arr[~np.isnan(arr)]

        if len(arr) == 0:
            raise ValueError("Zone vide")

        eau = arr[arr > 0.1]
        nb_eau = len(eau)
        nb_total = len(arr)
        nb_sec = nb_total - nb_eau
        pct_eau = (nb_eau / nb_total) * 100
        ndwi_max = float(np.max(arr))
        ndwi_min = float(np.min(arr))

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

    except Exception as e:
        presence, niveau, commentaire = "Donnée manquante", "N/A", f"Erreur : {str(e)}"
        ndwi_min = ndwi_max = pct_eau = nb_eau = nb_sec = 0

    results.append([
        row.get("LIBELLE_OB", f"Bassin_{idx}"),
        presence,
        niveau,
        ndwi_min,
        ndwi_max,
        pct_eau,
        nb_eau,
        nb_sec,
        commentaire,
        acquisition_date
    ])

# === 3. EXPORT CSV FINAL ===
df = pd.DataFrame(results, columns=[
    "Nom du bassin",
    "Présence d'eau",
    "Niveau d'eau",
    "NDWI min",
    "NDWI max",
    "Pourcentage de pixels aquatiques (%)",
    "Nombre de pixels en eau",
    "Nombre de pixels à sec",
    "Commentaire",
    "Date d'acquisition"
])

df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"✅ Fichier exporté : {output_csv}")
