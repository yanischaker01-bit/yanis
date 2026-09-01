from __future__ import annotations

import io
import math
import os
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone

import folium
from folium import plugins
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VIGICRUES_URLS = [
    "https://www.vigicrues.gouv.fr/services/1/InfoVigiCru.geojson/",
    "https://www.vigicrues.gouv.fr/services/1/InfoVigiCru.geojson",
]
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{days}/{end_date}"
FIRMS_SOURCES = ["VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"]
FIRMS_BBOX = "-0.7,44.75,1.0,47.5"
FIRMS_RADIUS_KM = 0.5

DEPS = {
    "37": {"nom": "Indre-et-Loire", "lat": 47.38, "lon": 0.69},
    "86": {"nom": "Vienne", "lat": 46.58, "lon": 0.34},
    "79": {"nom": "Deux-Sèvres", "lat": 46.32, "lon": -0.46},
    "16": {"nom": "Charente", "lat": 45.65, "lon": 0.16},
    "17": {"nom": "Charente-Maritime", "lat": 45.75, "lon": -0.63},
    "33": {"nom": "Gironde", "lat": 44.84, "lon": -0.58},
}
LEVEL_RANK = {"ROUGE": 4, "ORANGE": 3, "JAUNE": 2, "VERT": 1, "INFO": 0}
LEVEL_COLOR = {"ROUGE": "#dc2626", "ORANGE": "#ea580c", "JAUNE": "#eab308", "VERT": "#16a34a", "INFO": "#64748b"}
RIVERS = ["vienne", "clain", "charente", "boutonne", "seugne", "touvre", "dronne", "isle", "dordogne", "garonne", "thouet", "sevre", "indre", "cher", "creuse", "ciron", "jalles", "estey"]
HEADERS = {"User-Agent": "Mozilla/5.0 LGV-PluvioStations/3.0", "Accept": "application/json,application/geo+json,*/*"}


def normalize(value: object) -> str:
    text = "" if value is None else str(value).lower()
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def number(value: object, default: float = 0.0) -> float:
    try:
        return default if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default


def get_json(url: str, params: dict | None = None, timeout: int = 30) -> dict:
    error = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=(8, timeout))
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("Réponse JSON invalide")
            return data
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(error))


def rain_color(value: float) -> str:
    if value >= 60:
        return "#dc2626"
    if value >= 30:
        return "#ea580c"
    if value >= 10:
        return "#2563eb"
    return "#93c5fd"


def style_chart(fig: go.Figure, height: int = 390) -> None:
    fig.update_layout(
        height=height, plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified", margin=dict(t=70, b=55, l=55, r=75),
        font=dict(family="Arial", color="#334155"),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#cbd5e1")
    fig.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


@st.cache_data(ttl=900, show_spinner=False)
def load_snapshot() -> dict:
    return get_json(SNAPSHOT_URL)


@st.cache_data(ttl=1800, show_spinner=False)
def forecast_at(lat: float, lon: float) -> pd.DataFrame:
    try:
        daily = get_json(FORECAST_URL, {
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max,weather_code,wind_speed_10m_max",
            "forecast_days": 7, "timezone": "Europe/Paris",
        }).get("daily", {})
        return pd.DataFrame({
            "date": daily.get("time", []), "pluie_mm": daily.get("precipitation_sum", []),
            "proba": daily.get("precipitation_probability_max", []), "tmax": daily.get("temperature_2m_max", []),
            "code": daily.get("weather_code", []), "vent": daily.get("wind_speed_10m_max", []),
        })
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def weekly_all_communes(sectors: pd.DataFrame) -> pd.DataFrame:
    required = {"commune_name", "latitude", "longitude"}
    if sectors.empty or not required.issubset(sectors.columns):
        return pd.DataFrame()
    coords = sectors.dropna(subset=list(required)).groupby("commune_name")[["latitude", "longitude"]].mean()
    rows = []
    for commune, coord in coords.iterrows():
        frame = forecast_at(float(coord.latitude), float(coord.longitude))
        if frame.empty:
            continue
        rain = pd.to_numeric(frame.pluie_mm, errors="coerce").fillna(0)
        peak_index = rain.idxmax()
        rows.append({
            "commune": commune, "cumul_7j": round(float(rain.sum()), 1),
            "pic_mm": round(float(rain.max()), 1),
            "date_pic": str(frame.loc[peak_index, "date"]),
            "latitude": float(coord.latitude), "longitude": float(coord.longitude),
        })
    return pd.DataFrame(rows).sort_values(["cumul_7j", "pic_mm"], ascending=False) if rows else pd.DataFrame()


@st.cache_data(ttl=21600, show_spinner=False)
def historical_daily(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    try:
        daily = get_json(ARCHIVE_URL, {
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "start_date": str(start), "end_date": str(end),
            "daily": "precipitation_sum", "timezone": "Europe/Paris",
        }, timeout=60).get("daily", {})
        frame = pd.DataFrame({"date": daily.get("time", []), "pluie_mm": daily.get("precipitation_sum", [])})
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame.date)
        frame["pluie_mm"] = pd.to_numeric(frame.pluie_mm, errors="coerce").fillna(0)
        return frame
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def vigicrues() -> tuple[list[dict], bool]:
    payload = None
    for url in VIGICRUES_URLS:
        try:
            data = get_json(url, timeout=45)
            if isinstance(data.get("features"), list):
                payload = data
                break
        except Exception:
            continue
    if payload is None:
        return [], False
    alerts = []
    level_map = {0: "VERT", 1: "VERT", 2: "JAUNE", 3: "ORANGE", 4: "ROUGE"}
    for feature in payload["features"]:
        props = feature.get("properties", {}) if isinstance(feature, dict) else {}
        name = str(props.get("NomEntVigiCru") or props.get("LbEntVigiCru") or props.get("lbentcru") or "").strip()
        if not name or not any(river in normalize(name) for river in RIVERS):
            continue
        raw = props.get("NivSituVigiCruEnt", props.get("NivInfViCr", props.get("NivVigiCru")))
        try:
            level = level_map.get(int(float(raw)))
        except (TypeError, ValueError):
            level = {"vert": "VERT", "jaune": "JAUNE", "orange": "ORANGE", "rouge": "ROUGE"}.get(normalize(raw))
        if level:
            alerts.append({"name": name, "level": level, "message": f"{name} : vigilance {level.lower()}"})
    unique = {(a["name"], a["level"]): a for a in alerts}
    return sorted(unique.values(), key=lambda a: (-LEVEL_RANK[a["level"]], a["name"])), True


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * radius * math.asin(min(1, math.sqrt(a)))


def polyline_from_snapshot(lines: object) -> list[tuple[float, float, float]]:
    if not isinstance(lines, list) or not lines or not isinstance(lines[0], list):
        return []
    pts = [(number(p.get("lat")), number(p.get("lon"))) for p in lines[0] if isinstance(p, dict) and p.get("lat") is not None and p.get("lon") is not None]
    if len(pts) < 2:
        return []
    out = [(pts[0][0], pts[0][1], 0.0)]
    distance = 0.0
    for first, second in zip(pts, pts[1:]):
        distance += haversine(*first, *second)
        out.append((second[0], second[1], distance))
    return out


def nearest_pk(lat: float, lon: float, line: list[tuple[float, float, float]]) -> tuple[float | None, float | None]:
    best = None
    best_pk = None
    for (lat1, lon1, pk1), (lat2, lon2, pk2) in zip(line, line[1:]):
        mid = (lat1 + lat2) / 2
        kx, ky = 111.32 * math.cos(math.radians(mid)), 111.32
        x1, y1, x2, y2, xp, yp = lon1*kx, lat1*ky, lon2*kx, lat2*ky, lon*kx, lat*ky
        dx, dy = x2-x1, y2-y1
        den = dx*dx + dy*dy
        t = 0 if den == 0 else max(0, min(1, ((xp-x1)*dx + (yp-y1)*dy)/den))
        dist2 = (xp-(x1+t*dx))**2 + (yp-(y1+t*dy))**2
        if best is None or dist2 < best:
            best, best_pk = dist2, pk1 + t*(pk2-pk1)
    return best_pk, math.sqrt(best) if best is not None else None


def firms_key() -> str | None:
    try:
        value = st.secrets.get("FIRMS_MAP_KEY")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv("FIRMS_MAP_KEY")


@st.cache_data(ttl=300, show_spinner=False)
def firms_hotspots(key: str, end_date: date, days: int) -> tuple[pd.DataFrame, str | None]:
    frames, successes = [], 0
    for source in FIRMS_SOURCES:
        try:
            url = FIRMS_URL.format(key=key, source=source, area=FIRMS_BBOX, days=days, end_date=end_date)
            response = requests.get(url, headers=HEADERS, timeout=(8, 30))
            response.raise_for_status()
            text = response.text.strip()
            if "invalid" in text[:200].lower():
                return pd.DataFrame(), "invalid_key"
            frame = pd.read_csv(io.StringIO(text))
            successes += 1
            if not frame.empty:
                frame["source"] = source
                frames.append(frame)
        except Exception:
            continue
    if not successes:
        return pd.DataFrame(), "fetch_failed"
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), None


st.set_page_config(page_title="LGV SEA - Pluviométrie", page_icon="🌧️", layout="wide")
st.title("🌧️ LGV SEA - Pluviométrie, crues et incendies")
head1, head2 = st.columns([5, 1])
head1.caption("Open-Meteo · Vigicrues · NASA FIRMS · historique consultable depuis le 1er janvier 2021")
if head2.button("🔄 Actualiser", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

try:
    snapshot = load_snapshot()
except Exception as exc:
    st.error(f"Snapshot LGV indisponible : {exc}")
    st.stop()
records = snapshot.get("sectors", {}).get("sectors", []) if isinstance(snapshot.get("sectors"), dict) else []
sectors = pd.DataFrame(records)
for col in ["latitude", "longitude", "pk_km"]:
    if col in sectors:
        sectors[col] = pd.to_numeric(sectors[col], errors="coerce")
communes = sorted(sectors.commune_name.dropna().astype(str).unique()) if "commune_name" in sectors else []
coords = sectors.dropna(subset=["commune_name", "latitude", "longitude"]).groupby("commune_name")[["latitude", "longitude"]].mean() if communes else pd.DataFrame()

with st.sidebar:
    st.header("Paramètres")
    selected = st.selectbox("Commune principale", communes if communes else ["Indisponible"])
    st.subheader("Historique")
    history_start = st.date_input("Début", value=date(2021, 1, 1), min_value=date(2021, 1, 1), max_value=date.today())
    history_end = st.date_input("Fin", value=date.today() - timedelta(days=5), min_value=date(2021, 1, 1), max_value=date.today())
    aggregation = st.radio("Regroupement", ["Mensuel", "Annuel"], horizontal=True)
    st.subheader("NASA FIRMS")
    firms_days = st.slider("Période FIRMS", 1, 10, 1)

st.subheader("🌧️ Toutes les communes : prévision cumulée sur une semaine")
with st.spinner("Calcul des prévisions pour toutes les communes..."):
    weekly = weekly_all_communes(sectors)
if weekly.empty:
    st.warning("Les prévisions communales ne sont pas disponibles.")
else:
    peak_threshold = float(weekly.pic_mm.quantile(0.90))
    colors = ["#dc2626" if value >= peak_threshold else "#2563eb" for value in weekly.pic_mm]
    sizes = [14 if value >= peak_threshold else 8 for value in weekly.pic_mm]
    fig = go.Figure()
    fig.add_bar(x=weekly.commune, y=weekly.cumul_7j, name="Cumul 7 jours", marker_color="#93c5fd", hovertemplate="%{x}<br>Cumul : %{y:.1f} mm<extra></extra>")
    fig.add_scatter(
        x=weekly.commune, y=weekly.pic_mm, name="Pic journalier", mode="markers+text",
        marker=dict(color=colors, size=sizes, line=dict(color="white", width=1)),
        text=[f"{value:.1f}" if value >= peak_threshold else "" for value in weekly.pic_mm],
        textposition="top center",
        customdata=weekly[["date_pic"]],
        hovertemplate="%{x}<br>Pic : %{y:.1f} mm<br>Date : %{customdata[0]}<extra></extra>",
    )
    fig.update_layout(yaxis_title="Pluie (mm)", xaxis_title=None)
    style_chart(fig, 520)
    st.caption(f"Les points rouges et leurs valeurs correspondent aux 10 % de pics journaliers les plus élevés, seuil actuel : {peak_threshold:.1f} mm.")
    st.dataframe(weekly.rename(columns={"commune": "Commune", "cumul_7j": "Cumul 7 j (mm)", "pic_mm": "Pic journalier (mm)", "date_pic": "Date du pic"})[["Commune", "Cumul 7 j (mm)", "Pic journalier (mm)", "Date du pic"]], use_container_width=True, hide_index=True)

st.divider()
st.subheader(f"📅 Historique pluviométrique de {selected} depuis 2021")
if history_start > history_end:
    st.error("La date de début doit précéder la date de fin.")
elif selected in coords.index:
    point = coords.loc[selected]
    with st.spinner("Chargement de l'historique..."):
        history = historical_daily(float(point.latitude), float(point.longitude), history_start, history_end)
    if history.empty:
        st.warning("Historique indisponible pour cette période.")
    else:
        if aggregation == "Mensuel":
            history["periode"] = history.date.dt.to_period("M").astype(str)
        else:
            history["periode"] = history.date.dt.year.astype(str)
        grouped = history.groupby("periode", as_index=False).pluie_mm.sum()
        peak_day = history.loc[history.pluie_mm.idxmax()]
        peak_period = str(peak_day.date.to_period("M")) if aggregation == "Mensuel" else str(peak_day.date.year)
        fig_history = go.Figure(go.Bar(x=grouped.periode, y=grouped.pluie_mm, marker_color=["#dc2626" if p == peak_period else "#2563eb" for p in grouped.periode], text=[f"{v:.0f}" for v in grouped.pluie_mm], textposition="outside"))
        fig_history.update_layout(yaxis_title="Cumul de pluie (mm)", xaxis_title="Période")
        style_chart(fig_history, 450)
        c1, c2, c3 = st.columns(3)
        c1.metric("Cumul période", f"{history.pluie_mm.sum():.1f} mm")
        c2.metric("Pic journalier", f"{peak_day.pluie_mm:.1f} mm")
        c3.metric("Date du pic", peak_day.date.strftime("%d/%m/%Y"))
else:
    st.info("Coordonnées indisponibles pour la commune sélectionnée.")

st.divider()
st.subheader("🏞️ Vigicrues")
river_alerts, river_ok = vigicrues()
active_rivers = [a for a in river_alerts if a["level"] in {"JAUNE", "ORANGE", "ROUGE"}]
if not river_ok:
    st.warning("Vigicrues est temporairement injoignable. Le statut des crues n'a pas pu être vérifié.")
elif not active_rivers:
    st.success("Vigicrues vérifié : aucune vigilance active sur les cours d'eau sélectionnés.")
else:
    for alert in active_rivers:
        st.markdown(f"<div style='border-left:5px solid {LEVEL_COLOR[alert['level']]};padding:10px;margin:7px 0;background:#f8fafc'><b>{alert['message']}</b></div>", unsafe_allow_html=True)

st.divider()
st.subheader("🗺️ Carte LGV avec fond satellite")
if selected in coords.index:
    point = coords.loc[selected]
    center = [float(point.latitude), float(point.longitude)]
else:
    center = [46.2, 0.2]
map_object = folium.Map(location=center, zoom_start=10, tiles=None, control_scale=True)
folium.TileLayer("CartoDB positron", name="Carte claire", show=True).add_to(map_object)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    name="Satellite", attr="Esri, Maxar, Earthstar Geographics, GIS User Community", show=False, max_zoom=22,
).add_to(map_object)
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    name="Noms et limites", attr="Esri", overlay=True, show=False, max_zoom=20,
).add_to(map_object)
for _, row in sectors.dropna(subset=["latitude", "longitude"]).iterrows():
    label = str(row.get("commune_name", "Secteur"))
    pk = number(row.get("pk_km"), float("nan"))
    folium.CircleMarker([float(row.latitude), float(row.longitude)], radius=4, color="#ef4444", fill=True, fill_opacity=.85, tooltip=f"{label} · PK {pk:.3f}" if not math.isnan(pk) else label).add_to(map_object)
line = polyline_from_snapshot(snapshot.get("lgv_lines", []))
if line:
    folium.PolyLine([(p[0], p[1]) for p in line], color="#facc15", weight=4, opacity=.9, tooltip="LGV SEA").add_to(map_object)
key = firms_key()
firms_frame, firms_error = (firms_hotspots(key, date.today(), firms_days) if key else (pd.DataFrame(), "missing_key"))
near_fires = []
if not firms_frame.empty and line:
    for _, row in firms_frame.iterrows():
        pk, distance = nearest_pk(number(row.get("latitude")), number(row.get("longitude")), line)
        if distance is not None and distance <= FIRMS_RADIUS_KM:
            near_fires.append((row, pk, distance))
            folium.CircleMarker([number(row.latitude), number(row.longitude)], radius=9, color="#7f1d1d", fill=True, fill_color="#dc2626", fill_opacity=.9, tooltip=f"FIRMS · PK {pk:.3f} · {distance*1000:.0f} m").add_to(map_object)
folium.LayerControl(collapsed=False).add_to(map_object)
plugins.Fullscreen(position="topleft", title="Plein écran", title_cancel="Quitter").add_to(map_object)
st_folium(map_object, use_container_width=True, height=600, returned_objects=[])
st.caption("Dans le sélecteur en haut à droite, choisis Satellite puis active Noms et limites pour le mode hybride.")
if firms_error == "missing_key":
    st.info("NASA FIRMS non activé : ajoute FIRMS_MAP_KEY dans les secrets Streamlit.")
elif firms_error:
    st.warning("NASA FIRMS n'a pas pu être vérifié.")
elif near_fires:
    st.error(f"{len(near_fires)} détection(s) FIRMS à moins de {FIRMS_RADIUS_KM*1000:.0f} m de la LGV.")
else:
    st.success("NASA FIRMS vérifié : aucune détection proche de la LGV.")

st.caption("Sources : Open-Meteo, Vigicrues, NASA FIRMS, Esri World Imagery et snapshot LGV SEA. Les indicateurs ne remplacent pas les consignes officielles.")
