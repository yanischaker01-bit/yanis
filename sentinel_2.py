from pystac_client import Client
from odc.stac import load
import geopandas as gpd
from shapely.geometry import mapping
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
from odc.geo.xr import write_cog
import os
import warnings

warnings.filterwarnings("ignore")

# === CONFIGURATION ===
input_file = "C:/Users/ychaker/Downloads/kmz/Emprise COSEA.geojson"  # Emprise AOI
bassins_file = "C:/Users/ychaker/Downloads/Bassin/Bassins.shp"          # Fichier des bassins
output_csv = "C:/Users/ychaker/Desktop/NDWI/ndwi_bassins.csv"
temp_dir = "C:/Users/ychaker/Desktop/NDWI/temp_ndwi"
os.makedirs(temp_dir, exist_ok=True)

# === 1. CHARGER LES DONNÉES GÉOGRAPHIQUES ===
aoi = gpd.read_file(input_file).to_crs(4326)
bassins = gpd.read_file(bassins_file).to_crs(4326)

# Nettoyage des géométries
def clean_geometry(geom):
    try:
        return geom.simplify(tolerance=0.0000001, preserve_topology=True)
    except:
        return geom

bassins["geometry"] = bassins["geometry"].apply(clean_geometry)

# === 2. ANALYSE PAR BASSIN ===
def analyze_ndwi_for_bassin(bassin_geom, raster_path):
    with rasterio.open(raster_path) as src:
        bassin_proj = gpd.GeoSeries([bassin_geom], crs=4326).to_crs(src.crs)
        try:
            out_image, _ = mask(src, bassin_proj.geometry, crop=True)
            ndwi_data = np.squeeze(out_image)
            ndwi_data = ndwi_data[ndwi_data != src.nodata]

            if len(ndwi_data) == 0:
                return None

            eau = ndwi_data[ndwi_data > 0.1]
            return {
                "min": float(np.min(ndwi_data)),
                "max": float(np.max(ndwi_data)),
                "count": len(ndwi_data),
                "pct_eau": len(eau) / len(ndwi_data) * 100,
                "nb_eau": len(eau),
                "nb_sec": len(ndwi_data) - len(eau)
            }
        except:
            return None

# === 3. STAC SEARCH CLIENT ===
client = Client.open("https://earth-search.aws.element84.com/v1")
collection = "sentinel-2-l2a"
date_range = "2025-02-01/2025-03-17"
results = []

# === 4. PARCOURS DES BASSINS ===
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

    raster_path = f"{temp_dir}/ndwi_bassin_{idx}.tif"
    write_cog(ndwi_clean, fname=raster_path, overwrite=True)

    stats = analyze_ndwi_for_bassin(row.geometry, raster_path)

    if stats is None:
        presence, niveau, commentaire = "Donnée manquante", "N/A", "Image non exploitable"
        ndwi_min = ndwi_max = pct_eau = nb_eau = nb_sec = 0
    else:
        ndwi_max = stats['max']
        ndwi_min = stats['min']
        pct_eau = stats['pct_eau']
        nb_eau = stats['nb_eau']
        nb_sec = stats['nb_sec']
        total = stats['count']

        if nb_eau == 0:
            presence, niveau, commentaire = "À sec", "aucun", "Aucun pixel en eau"
        else:
            presence = "Présence d'eau"
            ratio = nb_eau / total
            niveau = (
                "élevé" if ratio >= 1/4 else
                "moyen" if ratio >= 1/5 else
                "faible" if ratio >= 1/6 else
                "très faible"
            )
            commentaire = f"{nb_eau} pixels en eau (NDWI max = {ndwi_max:.2f})"

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

# === 5. EXPORT CSV FINAL ===
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

df.to_csv(output_csv, index=False , encoding='utf-8-sig' )
