import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from branca.colormap import linear

# === CONFIGURATION ===
csv_path = "C:/Users/ychaker/Desktop/NDWI/ndwi_bassins.csv"
shp_path = "C:/Users/ychaker/Downloads/Bassin/Bassins.shp"

# === CHARGEMENT DES DONNÉES ===
df = pd.read_csv(csv_path)
gdf = gpd.read_file(shp_path).to_crs(epsg=4326)

# Fusion des données attributaires
gdf = gdf.merge(df, left_on="LIBELLE_OB", right_on="Nom du bassin", how="left")

# === INTERFACE STREAMLIT ===
st.set_page_config(layout="wide")
st.title("🔍 Suivi de l'état des bassins - NDWI Sentinel-2")

# Alerte si présence d'eau
if (df["Présence d'eau"] == "Présence d'eau").any():
    st.error("🚨 **Alerte : présence d’eau détectée dans au moins un bassin !**")
else:
    st.success("✅ Aucun bassin en eau détecté.")

# === FILTRES ===
st.sidebar.header("🎛️ Filtres")

# Filtre niveau d'eau
niveau_eau = st.sidebar.multiselect(
    "Niveau d'eau",
    options=sorted(df["Niveau d'eau"].dropna().unique()),
    default=sorted(df["Niveau d'eau"].dropna().unique())
)

# Filtrage DataFrame et GeoDataFrame
df_filtered = df[df["Niveau d'eau"].isin(niveau_eau)]
gdf_filtered = gdf[gdf["Niveau d'eau"].isin(niveau_eau)]

# === CARTE ===
st.subheader("🗺️ Carte des bassins")

# ✅ Correction : reprojection temporaire pour calcul de centroïdes précis
gdf_projected = gdf.to_crs(epsg=3857)
centroid_y = gdf_projected.geometry.centroid.y.mean()
centroid_x = gdf_projected.geometry.centroid.x.mean()

# Revenir en coordonnées GPS pour la carte
center = [gpd.GeoSeries([gpd.points_from_xy([centroid_x], [centroid_y])[0]], crs=3857).to_crs(4326)[0].y,
          gpd.GeoSeries([gpd.points_from_xy([centroid_x], [centroid_y])[0]], crs=3857).to_crs(4326)[0].x]

m = folium.Map(location=center, zoom_start=10)

# Couleurs personnalisées par niveau
color_scale = {
    "élevé": "#08306b",
    "moyen": "#2171b5",
    "faible": "#6baed6",
    "très faible": "#c6dbef",
    "aucun": "lightgray",
    "N/A": "gray"
}

# Ajout des bassins sur la carte
for _, row in gdf_filtered.iterrows():
    libelle = row['LIBELLE_OB']
    niveau = row["Niveau d'eau"]
    tooltip_text = f"{libelle} ({niveau})"
    color = color_scale.get(niveau, "gray")
    
    geo_json = folium.GeoJson(
        row["geometry"],
        tooltip=folium.Tooltip(tooltip_text),
        style_function=lambda feature, col=color: {
            'fillColor': col,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        }
    )
    geo_json.add_to(m)

# Affichage carte dans Streamlit
st_data = st_folium(m, width=1000, height=600)

# === TABLEAU ===
st.subheader("📊 Tableau récapitulatif")

# Formatage
styled_df = df_filtered.style.format({
    "NDWI min": "{:.2f}",
    "NDWI max": "{:.2f}",
    "Pourcentage de pixels aquatiques (%)": "{:.1f}",
    "Nombre de pixels en eau": "{:,.0f}",
    "Nombre de pixels à sec": "{:,.0f}"
})

st.dataframe(styled_df, use_container_width=True)
