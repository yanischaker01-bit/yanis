from __future__ import annotations

import io
import math
import os
import time
from datetime import date, timedelta

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="LGV SEA - Pluie et FIRMS", page_icon="🌧️", layout="wide")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FIRMS_AREA_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{key}/{source}/{area}/{day_range}/{date}"
)
FIRMS_SOURCES = ("VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT")
FIRMS_BBOX = "-1.15,44.55,1.15,47.55"  # ouest,sud,est,nord
DEFAULT_FIRMS_RADIUS_KM = 0.5

# Tracé simplifié Tours-Bordeaux. Pour un filtrage précis au PK, remplacez-le
# par les coordonnées du tracé réel dans LGV_ROUTE ou chargez un CSV ci-dessous.
LGV_ROUTE = [
    (47.3941, 0.6848), (47.1700, 0.7000), (46.8130, 0.5450),
    (46.5800, 0.3400), (46.3300, 0.2500), (46.0200, 0.1350),
    (45.7600, 0.1600), (45.4500, 0.1550), (45.1800, 0.0200),
    (44.8378, -0.5792),
]

COMMUNES = {
    "Tours": (47.3941, 0.6848),
    "Châtellerault": (46.8179, 0.5461),
    "Poitiers": (46.5802, 0.3404),
    "Ruffec": (46.0282, 0.1987),
    "Angoulême": (45.6484, 0.1560),
    "Libourne": (44.9153, -0.2439),
    "Bordeaux": (44.8378, -0.5792),
}

# -----------------------------------------------------------------------------
# Outils généraux
# -----------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def build_polyline(points: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    output: list[tuple[float, float, float]] = []
    cumulative = 0.0
    for index, (lat, lon) in enumerate(points):
        if index:
            previous_lat, previous_lon = points[index - 1]
            cumulative += haversine_km(previous_lat, previous_lon, lat, lon)
        output.append((lat, lon, cumulative))
    return output


def pk_and_distance(lat: float, lon: float, polyline: list[tuple[float, float, float]]) -> tuple[float | None, float | None]:
    if len(polyline) < 2:
        return None, None

    best_distance_sq = float("inf")
    best_pk = None
    mean_lat = math.radians(lat)
    km_lat = 111.32
    km_lon = 111.32 * max(math.cos(mean_lat), 0.01)

    for index in range(len(polyline) - 1):
        lat1, lon1, pk1 = polyline[index]
        lat2, lon2, pk2 = polyline[index + 1]
        ax = (lon1 - lon) * km_lon
        ay = (lat1 - lat) * km_lat
        bx = (lon2 - lon) * km_lon
        by = (lat2 - lat) * km_lat
        vx, vy = bx - ax, by - ay
        denominator = vx * vx + vy * vy
        t = 0.0 if denominator == 0 else max(0.0, min(1.0, -(ax * vx + ay * vy) / denominator))
        px, py = ax + t * vx, ay + t * vy
        distance_sq = px * px + py * py
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_pk = pk1 + t * (pk2 - pk1)

    return best_pk, math.sqrt(best_distance_sq)


def get_firms_map_key() -> str | None:
    try:
        key = st.secrets.get("FIRMS_MAP_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return (os.getenv("FIRMS_MAP_KEY") or os.getenv("FIRMS_KEY") or "").strip() or None


def load_route_from_csv(uploaded_file) -> list[tuple[float, float]]:
    if uploaded_file is None:
        return LGV_ROUTE
    try:
        frame = pd.read_csv(uploaded_file)
        lower = {str(column).lower().strip(): column for column in frame.columns}
        lat_column = lower.get("latitude") or lower.get("lat")
        lon_column = lower.get("longitude") or lower.get("lon") or lower.get("lng")
        if not lat_column or not lon_column:
            raise ValueError("colonnes latitude/longitude absentes")
        frame[lat_column] = pd.to_numeric(frame[lat_column], errors="coerce")
        frame[lon_column] = pd.to_numeric(frame[lon_column], errors="coerce")
        frame = frame.dropna(subset=[lat_column, lon_column])
        if len(frame) < 2:
            raise ValueError("au moins deux points sont nécessaires")
        return list(zip(frame[lat_column].astype(float), frame[lon_column].astype(float)))
    except Exception as exc:
        st.warning(f"Tracé CSV ignoré : {exc}. Le tracé simplifié est utilisé.")
        return LGV_ROUTE

# -----------------------------------------------------------------------------
# Open-Meteo
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_forecast(lat: float, lon: float) -> pd.DataFrame:
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max",
            "timezone": "Europe/Paris",
            "forecast_days": 7,
        },
        timeout=(5, 20),
    )
    response.raise_for_status()
    daily = response.json().get("daily", {})
    return pd.DataFrame({
        "Date": daily.get("time", []),
        "Pluie (mm)": daily.get("precipitation_sum", []),
        "Probabilité (%)": daily.get("precipitation_probability_max", []),
        "T. max (°C)": daily.get("temperature_2m_max", []),
    })

# -----------------------------------------------------------------------------
# NASA FIRMS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_firms_source(key: str, source: str, day_range: int, start_date: str) -> pd.DataFrame:
    url = FIRMS_AREA_URL.format(
        key=key, source=source, area=FIRMS_BBOX,
        day_range=day_range, date=start_date,
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(5, 25))
            response.raise_for_status()
            text = response.text.strip()
            beginning = text[:400].lower()
            if "invalid" in beginning and "map" in beginning:
                raise ValueError("invalid_key")
            if not text or "<html" in beginning:
                raise RuntimeError("Réponse FIRMS inattendue")
            frame = pd.read_csv(io.StringIO(text))
            if frame.empty:
                return frame
            required = {"latitude", "longitude"}
            if not required.issubset(frame.columns):
                raise RuntimeError("Colonnes FIRMS latitude/longitude absentes")
            frame["source"] = source
            return frame
        except ValueError:
            raise
        except (requests.RequestException, RuntimeError, pd.errors.ParserError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(str(last_error) if last_error else "Échec FIRMS")


def load_firms_filtered(
    polyline: list[tuple[float, float, float]],
    day_range: int,
    end_date: date,
    radius_km: float,
) -> tuple[pd.DataFrame, str | None, int]:
    key = get_firms_map_key()
    if not key:
        return pd.DataFrame(), "missing_key", 0

    # Avec DATE, FIRMS renvoie DATE à DATE + DAY_RANGE - 1.
    start_date = end_date - timedelta(days=day_range - 1)
    frames: list[pd.DataFrame] = []
    successful_sources = 0
    invalid_key = False

    for source in FIRMS_SOURCES:
        try:
            frame = fetch_firms_source(key, source, day_range, start_date.isoformat())
            successful_sources += 1
            if not frame.empty:
                frames.append(frame)
        except ValueError as exc:
            if str(exc) == "invalid_key":
                invalid_key = True
                break
        except Exception:
            continue

    if invalid_key:
        return pd.DataFrame(), "invalid_key", successful_sources
    if successful_sources == 0:
        return pd.DataFrame(), "fetch_failed", 0
    if not frames:
        return pd.DataFrame(), None, successful_sources

    frame = pd.concat(frames, ignore_index=True)
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame = frame.dropna(subset=["latitude", "longitude"])

    positions = frame.apply(
        lambda row: pk_and_distance(float(row["latitude"]), float(row["longitude"]), polyline),
        axis=1,
    )
    frame["PK (km)"] = [position[0] for position in positions]
    frame["Distance LGV (km)"] = [position[1] for position in positions]
    frame = frame[frame["Distance LGV (km)"] <= radius_km].copy()

    if frame.empty:
        return frame, None, successful_sources

    frame["Distance LGV (m)"] = (frame["Distance LGV (km)"] * 1000).round(0).astype(int)
    frame["PK (km)"] = frame["PK (km)"].round(1)
    frame["Date"] = frame.get("acq_date", "")
    frame["Heure UTC"] = frame.get("acq_time", "").astype(str).str.zfill(4).str.replace(r"^(\d{2})(\d{2})$", r"\1:\2", regex=True)
    frame["Confiance"] = frame.get("confidence", "")
    frame["FRP (MW)"] = pd.to_numeric(frame.get("frp"), errors="coerce").round(1)
    frame["Satellite"] = frame.get("satellite", frame["source"])
    frame = frame.sort_values(["Distance LGV (km)", "Date"])
    return frame, None, successful_sources

# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------
st.title("🌧️ LGV SEA - Pluviométrie et détections FIRMS")
st.caption("Version simple : météo 7 jours, carte classique/satellite et détections NASA FIRMS filtrées selon leur distance à la LGV.")

with st.sidebar:
    st.header("Paramètres")
    selected_commune = st.selectbox("Commune météo", list(COMMUNES))
    firms_days = st.slider("Période FIRMS (jours)", 1, 5, 1)
    firms_end_date = st.date_input("Fin de période FIRMS", value=date.today(), max_value=date.today())
    firms_radius_m = st.slider("Distance maximale à la LGV", 100, 5000, 500, step=100)
    route_file = st.file_uploader(
        "Tracé LGV CSV facultatif",
        type=["csv"],
        help="Colonnes attendues : latitude et longitude, ordonnées du nord vers le sud.",
    )
    if st.button("Actualiser les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

route = load_route_from_csv(route_file)
polyline = build_polyline(route)
latitude, longitude = COMMUNES[selected_commune]

left, right = st.columns([1, 2])
with left:
    st.subheader(f"Prévision à {selected_commune}")
    try:
        forecast = load_forecast(latitude, longitude)
        total_rain = pd.to_numeric(forecast["Pluie (mm)"], errors="coerce").sum()
        max_rain = pd.to_numeric(forecast["Pluie (mm)"], errors="coerce").max()
        metric1, metric2 = st.columns(2)
        metric1.metric("Cumul 7 jours", f"{total_rain:.1f} mm")
        metric2.metric("Maximum journalier", f"{max_rain:.1f} mm")
        st.bar_chart(forecast.set_index("Date")["Pluie (mm)"], use_container_width=True)
        st.dataframe(forecast, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Open-Meteo indisponible : {exc}")

with st.spinner("Interrogation de NASA FIRMS..."):
    firms, firms_error, successful_sources = load_firms_filtered(
        polyline=polyline,
        day_range=firms_days,
        end_date=firms_end_date,
        radius_km=firms_radius_m / 1000,
    )

with right:
    st.subheader("Carte LGV SEA")
    map_object = folium.Map(location=[46.15, 0.05], zoom_start=7, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Plan OpenStreetMap", show=True, control=True).add_to(map_object)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite Esri",
        overlay=False,
        control=True,
        show=False,
        max_zoom=19,
    ).add_to(map_object)

    folium.PolyLine(route, color="#2563eb", weight=5, opacity=0.9, tooltip="LGV SEA").add_to(map_object)
    for commune, (commune_lat, commune_lon) in COMMUNES.items():
        folium.CircleMarker(
            [commune_lat, commune_lon], radius=5,
            color="#0f172a", fill=True, fill_color="#ffffff", fill_opacity=1,
            tooltip=commune,
        ).add_to(map_object)

    if not firms.empty:
        firms_group = folium.FeatureGroup(name=f"Détections FIRMS ({len(firms)})", show=True)
        for _, hotspot in firms.iterrows():
            confidence = str(hotspot.get("Confiance", ""))
            frp = hotspot.get("FRP (MW)", "")
            popup = (
                f"<b>Détection FIRMS</b><br>PK estimé : {hotspot['PK (km)']} km"
                f"<br>Distance : {hotspot['Distance LGV (m)']} m"
                f"<br>Date : {hotspot.get('Date', '')} {hotspot.get('Heure UTC', '')} UTC"
                f"<br>Confiance : {confidence}<br>FRP : {frp} MW"
                f"<br>Satellite : {hotspot.get('Satellite', '')}"
            )
            folium.CircleMarker(
                [float(hotspot["latitude"]), float(hotspot["longitude"])],
                radius=8, color="#7f1d1d", weight=2,
                fill=True, fill_color="#ef4444", fill_opacity=0.85,
                popup=folium.Popup(popup, max_width=320), tooltip="Détection FIRMS",
            ).add_to(firms_group)
        firms_group.add_to(map_object)

    folium.LayerControl(position="topright", collapsed=False).add_to(map_object)
    st_folium(map_object, use_container_width=True, height=570, returned_objects=[])

st.subheader("🔥 Filtre NASA FIRMS")
if firms_error == "missing_key":
    st.warning(
        "Clé FIRMS manquante. Ajoute `FIRMS_MAP_KEY = \"ta_cle\"` dans "
        "`.streamlit/secrets.toml` ou configure la variable d'environnement `FIRMS_MAP_KEY`."
    )
elif firms_error == "invalid_key":
    st.error("La clé FIRMS est invalide. Vérifie la valeur de `FIRMS_MAP_KEY`.")
elif firms_error == "fetch_failed":
    st.warning("FIRMS est actuellement injoignable. Le statut incendie n'a pas pu être vérifié.")
elif firms.empty:
    st.success(
        f"Aucune détection FIRMS à moins de {firms_radius_m} m du tracé "
        f"sur la période sélectionnée. Sources interrogées avec succès : {successful_sources}."
    )
else:
    st.error(f"{len(firms)} détection(s) FIRMS trouvée(s) à moins de {firms_radius_m} m de la LGV.")
    columns = ["PK (km)", "Distance LGV (m)", "Date", "Heure UTC", "Confiance", "FRP (MW)", "Satellite"]
    st.dataframe(firms[columns], use_container_width=True, hide_index=True)

st.caption(
    "Le PK affiché est une estimation calculée sur le tracé chargé. Pour un résultat métier précis, "
    "charge un CSV issu du tracé SIG réel de la LGV. Une détection satellite FIRMS est un point chaud, "
    "pas une confirmation automatique d'incendie sur l'infrastructure."
)
