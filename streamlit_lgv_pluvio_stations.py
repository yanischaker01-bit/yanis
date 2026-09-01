from __future__ import annotations

import io
import math
import os
import time
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import folium
from folium import plugins
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

# =============================================================================
# CONFIGURATION
# =============================================================================
SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
VIGICRUES_URLS = [
    "https://www.vigicrues.gouv.fr/services/1/InfoVigiCru.geojson/",
    "https://www.vigicrues.gouv.fr/services/1/InfoVigiCru.geojson",
]
FIRMS_AREA_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{key}/{source}/{area}/{day_range}/{start_date}"
)
FIRMS_SOURCES = ["VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT"]
FIRMS_BBOX = "-0.7,44.75,1.0,47.5"
FIRMS_RADIUS_KM = 0.5
FIRMS_MAX_DAY_RANGE = 5
FIRMS_MAX_LOOKBACK_DAYS = 60

HEADERS = {
    "User-Agent": "LGV-PluvioStations/2.0 (Streamlit)",
    "Accept": "application/json, application/geo+json;q=0.9, text/csv;q=0.8, */*;q=0.1",
}

DEPS = {
    "37": {"nom": "Indre-et-Loire", "lat": 47.38, "lon": 0.69},
    "86": {"nom": "Vienne", "lat": 46.58, "lon": 0.34},
    "79": {"nom": "Deux-Sèvres", "lat": 46.32, "lon": -0.46},
    "16": {"nom": "Charente", "lat": 45.65, "lon": 0.16},
    "17": {"nom": "Charente-Maritime", "lat": 45.75, "lon": -0.63},
    "33": {"nom": "Gironde", "lat": 44.84, "lon": -0.58},
}

LEVEL_COLOR = {
    "ROUGE": "#dc2626", "ORANGE": "#ea580c", "JAUNE": "#eab308",
    "VERT": "#16a34a", "INFO": "#3b82f6",
}
LEVEL_RANK = {"ROUGE": 4, "ORANGE": 3, "JAUNE": 2, "VERT": 1, "INFO": 0}
LEVEL_LABEL = {"ROUGE": "Rouge", "ORANGE": "Orange", "JAUNE": "Jaune", "VERT": "Vert", "INFO": "Info"}
RIVERS_RAW = [
    "vienne", "clain", "charente", "boutonne", "seugne", "touvre",
    "dronne", "isle", "dordogne", "garonne", "thouet", "sevre",
    "indre", "cher", "creuse", "ciron", "jalles", "estey",
]

# =============================================================================
# OUTILS
# =============================================================================
def normalize(value: object) -> str:
    text = "" if value is None else str(value).lower()
    return "".join(
        character for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return default if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default


def request_json(url: str, params: dict | None = None, timeout: int = 30) -> dict:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                url, params=params, headers=HEADERS,
                timeout=(8, timeout), allow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Réponse JSON inattendue")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(last_error))


def rain_risk(maximum_mm: float) -> tuple[str, str, str]:
    if maximum_mm >= 60:
        return "ROUGE", "#dc2626", "🔴"
    if maximum_mm >= 30:
        return "ORANGE", "#ea580c", "🟠"
    if maximum_mm >= 10:
        return "JAUNE", "#eab308", "🟡"
    return "VERT", "#16a34a", "🟢"


def rain_color(mm: float) -> str:
    return rain_risk(to_float(mm))[1]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(min(1.0, math.sqrt(value)))


def show_chart(fig: go.Figure, height: int = 380, hovermode: str = "x unified") -> None:
    fig.update_layout(
        height=height,
        hovermode=hovermode,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=55, b=50, l=55, r=85),
        font=dict(family="Arial, sans-serif", color="#334155"),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#cbd5e1", automargin=True)
    fig.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", automargin=True)
    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

# =============================================================================
# SNAPSHOT, MÉTÉO ET HISTORIQUE
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_snapshot() -> dict:
    return request_json(SNAPSHOT_URL, timeout=40)


@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast(latitude: float, longitude: float) -> pd.DataFrame:
    try:
        daily = request_json(
            FORECAST_URL,
            params={
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
                "daily": (
                    "precipitation_sum,precipitation_probability_max,"
                    "temperature_2m_max,weather_code,wind_speed_10m_max"
                ),
                "forecast_days": 7,
                "timezone": "Europe/Paris",
            },
        ).get("daily", {})
        frame = pd.DataFrame({
            "date": daily.get("time", []),
            "rain": daily.get("precipitation_sum", []),
            "probability": daily.get("precipitation_probability_max", []),
            "temperature": daily.get("temperature_2m_max", []),
            "weather_code": daily.get("weather_code", []),
            "wind": daily.get("wind_speed_10m_max", []),
        })
        for column in ["rain", "probability", "temperature", "weather_code", "wind"]:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        return frame
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(latitude: float, longitude: float, start_date: date, end_date: date) -> pd.DataFrame:
    try:
        daily = request_json(
            ARCHIVE_URL,
            params={
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
                "start_date": str(start_date),
                "end_date": str(end_date),
                "daily": "precipitation_sum",
                "timezone": "Europe/Paris",
            },
            timeout=60,
        ).get("daily", {})
        frame = pd.DataFrame({
            "date": daily.get("time", []),
            "rain": daily.get("precipitation_sum", []),
        })
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"])
            frame["rain"] = pd.to_numeric(frame["rain"], errors="coerce").fillna(0)
        return frame
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def compare_communes(sectors: pd.DataFrame) -> pd.DataFrame:
    required = {"commune_name", "latitude", "longitude"}
    if sectors.empty or not required.issubset(sectors.columns):
        return pd.DataFrame()
    coordinates = (
        sectors.dropna(subset=list(required))
        .groupby("commune_name")[["latitude", "longitude"]]
        .mean()
    )
    rows = []
    for commune, point in coordinates.iterrows():
        forecast = load_forecast(float(point["latitude"]), float(point["longitude"]))
        if forecast.empty:
            continue
        rains = forecast["rain"].tolist()
        max_three_days = max(
            (sum(rains[index:index + 3]) for index in range(max(1, len(rains) - 2))),
            default=sum(rains),
        )
        peak_index = forecast["rain"].idxmax()
        rows.append({
            "commune": commune,
            "rain_7d": round(sum(rains), 1),
            "rain_3d": round(max_three_days, 1),
            "peak": round(max(rains), 1),
            "peak_date": str(forecast.loc[peak_index, "date"]),
            "latitude": float(point["latitude"]),
            "longitude": float(point["longitude"]),
        })
    return pd.DataFrame(rows).sort_values("rain_7d", ascending=False) if rows else pd.DataFrame()

# =============================================================================
# VIGICRUES
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_vigicrues() -> tuple[list[dict], bool]:
    payload = None
    for url in VIGICRUES_URLS:
        try:
            candidate = request_json(url, timeout=45)
            if isinstance(candidate.get("features"), list):
                payload = candidate
                break
        except Exception:
            continue
    if payload is None:
        return [], False

    mapping = {0: "VERT", 1: "VERT", 2: "JAUNE", 3: "ORANGE", 4: "ROUGE"}
    results = []
    for feature in payload["features"]:
        properties = feature.get("properties", {}) if isinstance(feature, dict) else {}
        name = str(
            properties.get("NomEntVigiCru")
            or properties.get("LbEntVigiCru")
            or properties.get("lbentcru")
            or ""
        ).strip()
        if not name or not any(river in normalize(name) for river in RIVERS_RAW):
            continue
        raw_level = properties.get(
            "NivSituVigiCruEnt",
            properties.get("NivInfViCr", properties.get("NivVigiCru")),
        )
        try:
            level = mapping.get(int(float(raw_level)))
        except (TypeError, ValueError):
            level = {
                "vert": "VERT", "jaune": "JAUNE",
                "orange": "ORANGE", "rouge": "ROUGE",
            }.get(normalize(raw_level))
        if level:
            results.append({"river": name, "level": level})
    unique = {(item["river"], item["level"]): item for item in results}
    return list(unique.values()), True

# =============================================================================
# NASA FIRMS
# =============================================================================
def get_firms_key() -> str | None:
    try:
        key = st.secrets.get("FIRMS_MAP_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass
    key = os.getenv("FIRMS_MAP_KEY") or os.getenv("FIRMS_KEY")
    return key.strip() if key else None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_firms_source(
    key: str,
    source: str,
    start_date: date,
    days: int,
) -> tuple[pd.DataFrame, str | None]:
    url = FIRMS_AREA_URL.format(
        key=key, source=source, area=FIRMS_BBOX,
        day_range=days, start_date=start_date,
    )
    try:
        response = requests.get(url, headers=HEADERS, timeout=(8, 40))
        response.raise_for_status()
        text = response.text.strip()
        preview = text[:300].lower()
        if "invalid" in preview:
            return pd.DataFrame(), "invalid_key"
        if not text or "<html" in preview:
            return pd.DataFrame(), "unexpected_response"
        frame = pd.read_csv(io.StringIO(text))
        if frame.empty:
            return frame, None
        if not {"latitude", "longitude"}.issubset(frame.columns):
            return pd.DataFrame(), "unexpected_response"
        frame["source"] = source
        return frame, None
    except requests.Timeout:
        return pd.DataFrame(), "timeout"
    except Exception:
        return pd.DataFrame(), "fetch_failed"


def load_firms(
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, str | None]:
    key = get_firms_key()
    if not key:
        return pd.DataFrame(), "missing_key"
    requested_days = (end_date - start_date).days + 1
    if requested_days < 1 or requested_days > FIRMS_MAX_DAY_RANGE:
        return pd.DataFrame(), "invalid_period"

    frames = []
    success_count = 0
    for source in FIRMS_SOURCES:
        frame, error = fetch_firms_source(key, source, start_date, requested_days)
        if error == "invalid_key":
            return pd.DataFrame(), error
        if error is None:
            success_count += 1
            if not frame.empty:
                frames.append(frame)

    if success_count == 0:
        return pd.DataFrame(), "fetch_failed"
    if not frames:
        return pd.DataFrame(), None

    output = pd.concat(frames, ignore_index=True)
    subset = [
        column for column in
        ["latitude", "longitude", "acq_date", "acq_time", "satellite", "source"]
        if column in output.columns
    ]
    if subset:
        output = output.drop_duplicates(subset=subset)
    return output, None


@st.cache_data(ttl=3600, show_spinner=False)
def build_lgv_polyline(lines: object) -> list[tuple[float, float, float]]:
    if not isinstance(lines, list) or not lines or not isinstance(lines[0], list):
        return []
    points = [
        (to_float(point.get("lat")), to_float(point.get("lon")))
        for point in lines[0]
        if isinstance(point, dict)
        and point.get("lat") is not None
        and point.get("lon") is not None
    ]
    if len(points) < 2:
        return []
    output = [(points[0][0], points[0][1], 0.0)]
    cumulative = 0.0
    for first, second in zip(points, points[1:]):
        cumulative += haversine_km(first[0], first[1], second[0], second[1])
        output.append((second[0], second[1], cumulative))
    return output


def pk_and_distance(
    latitude: float,
    longitude: float,
    polyline: list[tuple[float, float, float]],
) -> tuple[float | None, float | None]:
    if len(polyline) < 2:
        return None, None
    best_distance = None
    best_pk = None
    for (lat1, lon1, pk1), (lat2, lon2, pk2) in zip(polyline, polyline[1:]):
        middle_latitude = (lat1 + lat2) / 2
        kx = 111.32 * math.cos(math.radians(middle_latitude))
        ky = 111.32
        x1, y1 = lon1 * kx, lat1 * ky
        x2, y2 = lon2 * kx, lat2 * ky
        xp, yp = longitude * kx, latitude * ky
        dx, dy = x2 - x1, y2 - y1
        denominator = dx * dx + dy * dy
        ratio = 0 if denominator == 0 else ((xp - x1) * dx + (yp - y1) * dy) / denominator
        ratio = max(0.0, min(1.0, ratio))
        nearest_x, nearest_y = x1 + ratio * dx, y1 + ratio * dy
        distance = math.hypot(xp - nearest_x, yp - nearest_y)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_pk = pk1 + ratio * (pk2 - pk1)
    return best_pk, best_distance


def filter_firms_near_lgv(
    frame: pd.DataFrame,
    polyline: list[tuple[float, float, float]],
    radius_km: float,
) -> list[dict]:
    alerts = []
    if frame.empty or not polyline:
        return alerts
    for _, row in frame.iterrows():
        latitude = to_float(row.get("latitude"), float("nan"))
        longitude = to_float(row.get("longitude"), float("nan"))
        if math.isnan(latitude) or math.isnan(longitude):
            continue
        pk, distance = pk_and_distance(latitude, longitude, polyline)
        if pk is None or distance is None or distance > radius_km:
            continue
        acquisition_time = str(row.get("acq_time", "")).zfill(4)
        alerts.append({
            "latitude": latitude,
            "longitude": longitude,
            "pk": round(pk, 1),
            "distance_m": round(distance * 1000),
            "date": str(row.get("acq_date", "")),
            "time": f"{acquisition_time[:2]}:{acquisition_time[2:]}",
            "confidence": str(row.get("confidence", "")),
            "satellite": str(row.get("satellite") or row.get("source") or ""),
            "frp": row.get("frp"),
        })
    return sorted(alerts, key=lambda item: item["distance_m"])

# =============================================================================
# INTERFACE
# =============================================================================
st.set_page_config(
    page_title="LGV SEA - Pluviométrie",
    page_icon="🌧️",
    layout="wide",
)
st.title("🌧️ LGV SEA - Pluviométrie et surveillance")
header_left, header_right = st.columns([6, 1])
header_left.caption(
    "Météo Open-Meteo · Crues Vigicrues · Détections thermiques NASA FIRMS · "
    "Carte satellite Esri"
)
if header_right.button("🔄 Rafraîchir", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

try:
    snapshot = load_snapshot()
except Exception as exc:
    st.error(f"Snapshot indisponible : {exc}")
    st.stop()

sector_payload = snapshot.get("sectors", {})
sector_records = sector_payload.get("sectors", []) if isinstance(sector_payload, dict) else []
sectors = pd.DataFrame(sector_records)
for column in ["latitude", "longitude", "pk_km"]:
    if column in sectors:
        sectors[column] = pd.to_numeric(sectors[column], errors="coerce")

required = {"commune_name", "latitude", "longitude"}
if sectors.empty or not required.issubset(sectors.columns):
    st.error("Le snapshot ne contient pas les colonnes nécessaires.")
    st.stop()

communes = sorted(sectors["commune_name"].dropna().astype(str).unique())
coordinates = (
    sectors.dropna(subset=list(required))
    .groupby("commune_name")[["latitude", "longitude"]]
    .mean()
)

with st.sidebar:
    st.header("Paramètres")
    selected_commune = st.selectbox("Commune principale", ["Toutes"] + communes)
    comparison_selection = st.multiselect(
        "Communes à comparer",
        communes,
        default=communes[: min(8, len(communes))],
    )
    st.subheader("Historique")
    history_start = st.date_input(
        "Date de début", date(2021, 1, 1),
        min_value=date(2021, 1, 1), max_value=date.today(),
    )
    history_end = st.date_input(
        "Date de fin", date.today() - timedelta(days=5),
        min_value=date(2021, 1, 1), max_value=date.today(),
    )
    history_group = st.radio("Regroupement", ["Mensuel", "Annuel"], horizontal=True)

    st.subheader("NASA FIRMS")
    today_utc = datetime.now(timezone.utc).date()
    firms_days = st.slider("Période FIRMS", 1, FIRMS_MAX_DAY_RANGE, 1)
    firms_end = st.date_input(
        "Date de fin FIRMS", today_utc,
        min_value=today_utc - timedelta(days=FIRMS_MAX_LOOKBACK_DAYS),
        max_value=today_utc,
    )
    firms_start = firms_end - timedelta(days=firms_days - 1)

# Comparaison de toutes les communes
st.subheader("Comparaison des communes sur les 7 prochains jours")
with st.spinner("Chargement des prévisions communales..."):
    comparison = compare_communes(sectors)
if comparison.empty:
    st.warning("Comparaison indisponible.")
else:
    displayed_comparison = (
        comparison[comparison["commune"].isin(comparison_selection)]
        if comparison_selection else comparison.head(15)
    )
    figure = go.Figure()
    figure.add_bar(
        x=displayed_comparison["commune"],
        y=displayed_comparison["rain_7d"],
        name="Cumul prévu 7 jours",
        marker_color="#60a5fa",
    )
    figure.add_scatter(
        x=displayed_comparison["commune"],
        y=displayed_comparison["rain_3d"],
        name="Maximum glissant 3 jours",
        mode="markers+text",
        marker=dict(
            color=[rain_color(value) for value in displayed_comparison["rain_3d"]],
            size=12,
        ),
        text=[f"{value:.1f}" if value >= 20 else "" for value in displayed_comparison["rain_3d"]],
        textposition="top center",
        customdata=displayed_comparison[["peak", "peak_date"]],
        hovertemplate=(
            "%{x}<br>Maximum 3 jours : %{y:.1f} mm"
            "<br>Pic journalier : %{customdata[0]:.1f} mm"
            "<br>Date du pic : %{customdata[1]}<extra></extra>"
        ),
    )
    figure.update_layout(yaxis_title="Pluie (mm)", xaxis_tickangle=-40)
    show_chart(figure, 500, "closest")
    st.dataframe(
        comparison.rename(columns={
            "commune": "Commune", "rain_7d": "Cumul 7 j (mm)",
            "rain_3d": "Maximum 3 j (mm)", "peak": "Pic journalier (mm)",
            "peak_date": "Date du pic",
        })[["Commune", "Cumul 7 j (mm)", "Maximum 3 j (mm)", "Pic journalier (mm)", "Date du pic"]],
        use_container_width=True,
        hide_index=True,
    )

# Centre et sélection de la carte
if selected_commune == "Toutes":
    selected_sectors = sectors.copy()
else:
    selected_sectors = sectors[sectors["commune_name"] == selected_commune].copy()
map_points = selected_sectors.dropna(subset=["latitude", "longitude"])
center_latitude = float(map_points["latitude"].mean()) if not map_points.empty else 46.2
center_longitude = float(map_points["longitude"].mean()) if not map_points.empty else 0.2

# Prévision détaillée
location_label = "LGV SEA" if selected_commune == "Toutes" else selected_commune
st.subheader(f"Prévisions détaillées : {location_label}")
forecast = load_forecast(center_latitude, center_longitude)
if forecast.empty:
    st.info("Prévisions indisponibles.")
else:
    forecast_figure = go.Figure()
    forecast_figure.add_bar(
        x=forecast["date"], y=forecast["rain"],
        name="Pluie prévue (mm)",
        marker_color=[rain_color(value) for value in forecast["rain"]],
        text=[f"{value:.1f}" for value in forecast["rain"]],
        textposition="outside",
    )
    forecast_figure.add_scatter(
        x=forecast["date"], y=forecast["probability"],
        name="Probabilité (%)", yaxis="y2", mode="lines+markers",
        line=dict(color="#6366f1", dash="dot"),
    )
    forecast_figure.add_scatter(
        x=forecast["date"], y=forecast["temperature"],
        name="Température max. (°C)", yaxis="y3", mode="lines+markers",
        line=dict(color="#f97316"),
    )
    forecast_figure.update_layout(
        yaxis=dict(title="Pluie (mm)"),
        yaxis2=dict(title="Probabilité (%)", overlaying="y", side="right", range=[0, 100], showgrid=False),
        yaxis3=dict(title="Température (°C)", overlaying="y", side="right", anchor="free", position=0.92, showgrid=False),
    )
    show_chart(forecast_figure)

# Historique
st.subheader(f"Historique pluviométrique depuis 2021 : {location_label}")
if history_start > history_end:
    st.error("La date de début doit précéder la date de fin.")
else:
    history = load_history(center_latitude, center_longitude, history_start, history_end)
    if history.empty:
        st.info("Historique indisponible.")
    else:
        history["period"] = (
            history["date"].dt.to_period("M").astype(str)
            if history_group == "Mensuel"
            else history["date"].dt.year.astype(str)
        )
        grouped = history.groupby("period", as_index=False)["rain"].sum()
        peak = history.loc[history["rain"].idxmax()]
        history_figure = go.Figure(go.Bar(
            x=grouped["period"], y=grouped["rain"],
            marker_color=["#dc2626" if value == grouped["rain"].max() else "#2563eb" for value in grouped["rain"]],
            text=[f"{value:.0f}" for value in grouped["rain"]],
            textposition="outside",
        ))
        history_figure.update_layout(yaxis_title="Cumul pluie (mm)")
        show_chart(history_figure, 430, "closest")
        metrics = st.columns(3)
        metrics[0].metric("Cumul", f"{history['rain'].sum():.1f} mm")
        metrics[1].metric("Pic journalier", f"{peak['rain']:.1f} mm")
        metrics[2].metric("Date du pic", peak["date"].strftime("%d/%m/%Y"))

# Vigicrues
st.subheader("Vigicrues")
river_alerts, river_ok = load_vigicrues()
active_rivers = [item for item in river_alerts if item["level"] in {"JAUNE", "ORANGE", "ROUGE"}]
if not river_ok:
    st.warning("Vigicrues injoignable : statut des crues non vérifié.")
elif not active_rivers:
    st.success("Vigicrues vérifié : aucune vigilance active sur les cours d'eau suivis.")
else:
    for alert in active_rivers:
        st.warning(f"{alert['river']} : vigilance {alert['level'].lower()}")

# FIRMS
polyline = build_lgv_polyline(snapshot.get("lgv_lines", []))
firms_frame, firms_error = load_firms(firms_start, firms_end)
firms_alerts = filter_firms_near_lgv(firms_frame, polyline, FIRMS_RADIUS_KM)

st.subheader("Carte satellite des secteurs LGV SEA")
map_object = folium.Map(
    location=[center_latitude, center_longitude],
    zoom_start=8 if selected_commune == "Toutes" else 12,
    tiles=None,
    control_scale=True,
    prefer_canvas=True,
)

# Fond satellite affiché par défaut
folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    name="Satellite Esri",
    attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    overlay=False,
    control=True,
    show=True,
    max_zoom=22,
).add_to(map_object)

# Noms et limites par-dessus le satellite
folium.TileLayer(
    tiles=(
        "https://services.arcgisonline.com/ArcGIS/rest/services/"
        "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
    ),
    name="Noms et limites",
    attr="Esri",
    overlay=True,
    control=True,
    show=True,
    max_zoom=20,
).add_to(map_object)

# Fond clair alternatif
folium.TileLayer(
    tiles="CartoDB positron",
    name="Carte claire",
    overlay=False,
    control=True,
    show=False,
).add_to(map_object)

# Tracé LGV avec halo noir pour une bonne visibilité sur l'imagerie
if polyline:
    coordinates_line = [(item[0], item[1]) for item in polyline]
    folium.PolyLine(
        coordinates_line,
        color="#111827",
        weight=8,
        opacity=0.75,
        tooltip="LGV SEA",
    ).add_to(map_object)
    folium.PolyLine(
        coordinates_line,
        color="#facc15",
        weight=4,
        opacity=1.0,
        tooltip="LGV SEA",
    ).add_to(map_object)

# Secteurs
sector_group = folium.FeatureGroup(name="Secteurs LGV", show=True)
comparison_lookup = comparison.set_index("commune").to_dict("index") if not comparison.empty else {}
for _, row in map_points.iterrows():
    commune = str(row.get("commune_name", "Secteur"))
    data = comparison_lookup.get(commune, {})
    rain_7d = to_float(data.get("rain_7d"))
    peak = to_float(data.get("peak"))
    color = rain_color(rain_7d)
    pk_value = row.get("pk_km")
    pk_label = f"{to_float(pk_value):.3f}" if pd.notna(pk_value) else "N/D"
    popup = folium.Popup(
        f"<b>{commune}</b><br>PK : {pk_label}<br>"
        f"Cumul prévu 7 j : <b>{rain_7d:.1f} mm</b><br>"
        f"Pic journalier : {peak:.1f} mm",
        max_width=300,
    )
    folium.CircleMarker(
        location=[float(row["latitude"]), float(row["longitude"])],
        radius=5 if selected_commune == "Toutes" else 8,
        color="#ffffff",
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=0.92,
        tooltip=f"{commune} | PK {pk_label} | prévu 7 j {rain_7d:.1f} mm",
        popup=popup,
    ).add_to(sector_group)
sector_group.add_to(map_object)

# Détections FIRMS
if firms_alerts:
    firms_group = folium.FeatureGroup(name="Détections FIRMS", show=True)
    for alert in firms_alerts:
        folium.CircleMarker(
            location=[alert["latitude"], alert["longitude"]],
            radius=10,
            color="#ffffff",
            weight=2,
            fill=True,
            fill_color="#dc2626",
            fill_opacity=0.95,
            tooltip=(
                f"FIRMS | PK {alert['pk']} | {alert['distance_m']} m de la LGV | "
                f"{alert['date']} {alert['time']} UTC"
            ),
            popup=folium.Popup(
                f"<b>Détection thermique FIRMS</b><br>"
                f"PK approximatif : {alert['pk']} km<br>"
                f"Distance à la LGV : {alert['distance_m']} m<br>"
                f"Date : {alert['date']} {alert['time']} UTC<br>"
                f"Confiance : {alert['confidence']}<br>"
                f"Satellite : {alert['satellite']}",
                max_width=320,
            ),
        ).add_to(firms_group)
    firms_group.add_to(map_object)

# Légende fixe
legend_html = """
<div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
background: rgba(255,255,255,0.92); padding: 10px 12px; border-radius: 8px;
box-shadow: 0 1px 6px rgba(0,0,0,.35); font-size: 12px;">
<b>Prévision pluie 7 jours</b><br>
<span style="color:#16a34a">●</span> Faible &lt; 10 mm<br>
<span style="color:#eab308">●</span> Modérée 10 à 29 mm<br>
<span style="color:#ea580c">●</span> Forte 30 à 59 mm<br>
<span style="color:#dc2626">●</span> Très forte ≥ 60 mm<br>
<span style="color:#facc15">━</span> Tracé LGV SEA
</div>
"""
map_object.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl(collapsed=False, position="topright").add_to(map_object)
plugins.Fullscreen(
    position="topleft",
    title="Plein écran",
    title_cancel="Quitter le plein écran",
    force_separate_button=True,
).add_to(map_object)
plugins.MeasureControl(
    position="topleft",
    primary_length_unit="meters",
    secondary_length_unit="kilometers",
).add_to(map_object)

st_folium(
    map_object,
    use_container_width=True,
    height=650,
    returned_objects=[],
)
st.caption(
    "Le fond Satellite Esri est affiché par défaut. Utilise le sélecteur en haut à droite "
    "pour passer à la carte claire ou masquer les noms et limites."
)

if firms_error == "missing_key":
    st.info("NASA FIRMS non activé : ajoute FIRMS_MAP_KEY dans les secrets Streamlit.")
elif firms_error == "invalid_key":
    st.error("Clé NASA FIRMS invalide.")
elif firms_error == "invalid_period":
    st.error("La période FIRMS doit être comprise entre 1 et 5 jours.")
elif firms_error:
    st.warning("NASA FIRMS n'a pas pu être vérifié pour cette période.")
elif firms_alerts:
    st.error(f"{len(firms_alerts)} détection(s) thermique(s) à moins de {FIRMS_RADIUS_KM * 1000:.0f} m de la LGV.")
    st.dataframe(pd.DataFrame(firms_alerts), use_container_width=True, hide_index=True)
else:
    st.success("NASA FIRMS vérifié : aucune détection thermique proche de la LGV.")

st.subheader("Secteurs LGV SEA")
table_columns = [column for column in ["commune_name", "pk_km", "latitude", "longitude"] if column in selected_sectors]
display_table = selected_sectors[table_columns].rename(columns={
    "commune_name": "Commune", "pk_km": "PK (km)",
    "latitude": "Latitude", "longitude": "Longitude",
})
if not display_table.empty:
    if "PK (km)" in display_table:
        display_table = display_table.sort_values("PK (km)")
    st.dataframe(display_table, use_container_width=True, hide_index=True, height=340)

st.caption(
    "Sources : Open-Meteo, Vigicrues, NASA FIRMS, Esri World Imagery et snapshot LGV SEA. "
    "Les indicateurs sont informatifs et ne remplacent pas les procédures de maintenance ou de sécurité."
)
