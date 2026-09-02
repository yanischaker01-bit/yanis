from __future__ import annotations

import math
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

# =============================================================================
# CONFIGURATION
# =============================================================================
SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MF_URL = "https://public.opendatasoft.com/api/records/1.0/search/"
MF_DATASET = "weatherref-france-vigilance-meteo-departement"
VIGICRUES_GEOJSON_URL = "https://www.vigicrues.gouv.fr/services/1/InfoVigiCru.geojson"

DEPARTMENTS = ["33", "16", "17", "86", "79", "37"]
DEPARTMENT_INFO = {
    "33": {"name": "Gironde", "latitude": 44.8378, "longitude": -0.5792},
    "16": {"name": "Charente", "latitude": 45.6484, "longitude": 0.1560},
    "17": {"name": "Charente-Maritime", "latitude": 45.7470, "longitude": -0.6340},
    "86": {"name": "Vienne", "latitude": 46.5802, "longitude": 0.3404},
    "79": {"name": "Deux-Sevres", "latitude": 46.3237, "longitude": -0.4588},
    "37": {"name": "Indre-et-Loire", "latitude": 47.3941, "longitude": 0.6848},
}

RIVERS = [
    "sevre niortaise", "sevre nantaise", "dordogne", "charente", "boutonne",
    "garonne", "vienne", "seugne", "touvre", "dronne", "thouet", "sevre",
    "indre", "creuse", "jalles", "clain", "isle", "cher", "ciron",
]
RIVER_LABELS = {
    "sevre niortaise": "Sevre Niortaise", "sevre nantaise": "Sevre Nantaise",
    "dordogne": "Dordogne", "charente": "Charente", "boutonne": "Boutonne",
    "garonne": "Garonne", "vienne": "Vienne", "seugne": "Seugne",
    "touvre": "Touvre", "dronne": "Dronne", "thouet": "Thouet",
    "sevre": "Sevre", "indre": "Indre", "creuse": "Creuse",
    "jalles": "Jalles", "clain": "Clain", "isle": "Isle", "cher": "Cher",
    "ciron": "Ciron",
}

LEVEL_RANK = {"INDETERMINE": -1, "VERT": 0, "JAUNE": 1, "ORANGE": 2, "ROUGE": 3}
LEVEL_COLOR = {"INDETERMINE": "#64748b", "VERT": "#16a34a", "JAUNE": "#eab308", "ORANGE": "#ea580c", "ROUGE": "#dc2626"}
LEVEL_ACTION = {
    "INDETERMINE": "Donnees insuffisantes, controle manuel necessaire",
    "VERT": "Surveillance courante",
    "JAUNE": "Surveillance renforcee et controle de la fraicheur des donnees",
    "ORANGE": "Inspection ciblee et controle du drainage",
    "ROUGE": "Controle prioritaire et application des consignes metier",
}

WEATHER_CODE_LABELS = {
    0: "Ciel degage", 1: "Principalement degage", 2: "Partiellement nuageux", 3: "Couvert",
    45: "Brouillard", 48: "Brouillard givrant", 51: "Bruine faible", 53: "Bruine moderee",
    55: "Bruine forte", 56: "Bruine verglacante faible", 57: "Bruine verglacante forte",
    61: "Pluie faible", 63: "Pluie moderee", 65: "Pluie forte", 66: "Pluie verglacante faible",
    67: "Pluie verglacante forte", 71: "Neige faible", 73: "Neige moderee", 75: "Neige forte",
    77: "Grains de neige", 80: "Averses faibles", 81: "Averses moderees", 82: "Averses violentes",
    85: "Averses de neige faibles", 86: "Averses de neige fortes", 95: "Orage",
    96: "Orage avec grele faible", 99: "Orage avec grele forte",
}

st.set_page_config(page_title="LGV SEA - Alertes meteo et Vigicrues", page_icon="⚠️", layout="wide")

# =============================================================================
# OUTILS
# =============================================================================
def norm(value: Any) -> str:
    value = unicodedata.normalize("NFD", str(value or "").lower())
    return "".join(c for c in value if unicodedata.category(c) != "Mn")


def safe_float(value: Any, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_json(url: str, params=None, timeout=(5, 30), headers=None) -> dict:
    response = requests.get(url, params=params, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.json()


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1, math.sqrt(a)))


def build_polyline(lines) -> list[tuple[float, float, float]]:
    candidates = []
    for segment in lines or []:
        points = []
        for p in segment if isinstance(segment, list) else []:
            if isinstance(p, dict) and "lat" in p and "lon" in p:
                points.append((float(p["lat"]), float(p["lon"])))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                points.append((float(p[0]), float(p[1])))
        if len(points) >= 2:
            candidates.append(points)
    if not candidates:
        return []
    points = max(candidates, key=len)
    output = [(points[0][0], points[0][1], 0.0)]
    distance = 0.0
    for a, b in zip(points, points[1:]):
        distance += haversine_km(*a, *b)
        output.append((b[0], b[1], distance))
    return output


def pk_distance(lat: float, lon: float, line) -> tuple[float | None, float | None]:
    if len(line) < 2:
        return None, None
    best_d2, best_pk = None, None
    for (lat1, lon1, pk1), (lat2, lon2, pk2) in zip(line, line[1:]):
        lat_mid = (lat1 + lat2) / 2
        kx, ky = 111.320 * math.cos(math.radians(lat_mid)), 111.320
        x1, y1, x2, y2, xp, yp = lon1 * kx, lat1 * ky, lon2 * kx, lat2 * ky, lon * kx, lat * ky
        dx, dy = x2 - x1, y2 - y1
        den = dx * dx + dy * dy
        t = 0 if den == 0 else max(0, min(1, ((xp - x1) * dx + (yp - y1) * dy) / den))
        cx, cy = x1 + t * dx, y1 + t * dy
        d2 = (xp - cx) ** 2 + (yp - cy) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2, best_pk = d2, pk1 + t * (pk2 - pk1)
    return best_pk, math.sqrt(best_d2) if best_d2 is not None else None


def weather_label(code):
    try:
        return WEATHER_CODE_LABELS.get(int(code), "Situation inconnue")
    except (TypeError, ValueError):
        return "Situation inconnue"


def strongest_level(levels):
    values = list(levels)
    return max(values, key=lambda x: LEVEL_RANK.get(x, -1)) if values else "INDETERMINE"

# =============================================================================
# SNAPSHOT LGV
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_snapshot():
    return get_json(SNAPSHOT_URL)


def make_line_signature(line, max_points=800):
    if not line:
        return tuple()
    step = max(1, len(line) // max_points)
    simplified = [(round(lat, 6), round(lon, 6), round(pk, 3)) for lat, lon, pk in line[::step]]
    last = (round(line[-1][0], 6), round(line[-1][1], 6), round(line[-1][2], 3))
    if simplified[-1] != last:
        simplified.append(last)
    return tuple(simplified)

# =============================================================================
# METEO PAR DEPARTEMENT
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_department_forecast(dep_code):
    info = DEPARTMENT_INFO[dep_code]
    return get_json(FORECAST_URL, params={
        "latitude": info["latitude"], "longitude": info["longitude"],
        "hourly": "precipitation,precipitation_probability,wind_gusts_10m,soil_moisture_0_to_7cm",
        "daily": "weather_code,precipitation_sum,precipitation_probability_max,wind_gusts_10m_max,temperature_2m_min,temperature_2m_max",
        "forecast_days": 7, "timezone": "Europe/Paris",
    })


def department_forecast_dataframe(dep_code, payload):
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        return pd.DataFrame()
    n = len(dates)
    def values(name):
        data = daily.get(name, [])
        return data if len(data) == n else [np.nan] * n
    df = pd.DataFrame({
        "date": pd.to_datetime(dates, errors="coerce"),
        "weather_code": values("weather_code"), "rain_mm": values("precipitation_sum"),
        "rain_probability": values("precipitation_probability_max"), "gust_kmh": values("wind_gusts_10m_max"),
        "temperature_min": values("temperature_2m_min"), "temperature_max": values("temperature_2m_max"),
    })
    for col in ["weather_code", "rain_mm", "rain_probability", "gust_kmh", "temperature_min", "temperature_max"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["department"] = dep_code
    df["department_name"] = DEPARTMENT_INFO[dep_code]["name"]
    df["weather"] = df["weather_code"].apply(weather_label)
    return df.dropna(subset=["date"]).sort_values("date")


@st.cache_data(ttl=1800, show_spinner=False)
def load_all_department_forecasts():
    frames, errors = [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(load_department_forecast, dep): dep for dep in DEPARTMENTS}
        for future in as_completed(jobs):
            dep = jobs[future]
            try:
                frame = department_forecast_dataframe(dep, future.result())
                frames.append(frame) if not frame.empty else errors.append(dep)
            except Exception:
                errors.append(dep)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), errors


def forecast_warning_level(rain_mm, probability, gust_kmh, weather_code):
    rain_mm = safe_float(rain_mm, 0.0)
    probability = safe_float(probability, 0.0)
    gust_kmh = safe_float(gust_kmh, 0.0)
    try:
        weather_code = int(weather_code)
    except (TypeError, ValueError):
        weather_code = -1
    score, reasons = 0, []
    if rain_mm >= 60: score = max(score, 3); reasons.append(f"pluie tres forte {rain_mm:.0f} mm")
    elif rain_mm >= 35: score = max(score, 2); reasons.append(f"forte pluie {rain_mm:.0f} mm")
    elif rain_mm >= 15: score = max(score, 1); reasons.append(f"pluie notable {rain_mm:.0f} mm")
    if probability >= 80 and rain_mm >= 10: score = max(score, 1); reasons.append(f"probabilite {probability:.0f} %")
    if gust_kmh >= 110: score = max(score, 3); reasons.append(f"rafales {gust_kmh:.0f} km/h")
    elif gust_kmh >= 90: score = max(score, 2); reasons.append(f"rafales {gust_kmh:.0f} km/h")
    elif gust_kmh >= 70: score = max(score, 1); reasons.append(f"rafales {gust_kmh:.0f} km/h")
    if weather_code in [95, 96, 99]: score = max(score, 1); reasons.append("risque orageux")
    return {0: "VERT", 1: "JAUNE", 2: "ORANGE", 3: "ROUGE"}[score], ", ".join(reasons) or "Aucun signal notable"

# =============================================================================
# VIGILANCE METEO-FRANCE
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_mf_alerts():
    alerts, successful = [], 0
    for dep in DEPARTMENTS:
        try:
            payload = get_json(MF_URL, params={"dataset": MF_DATASET, "rows": 100, "refine.domain_id": dep})
            successful += 1
            for record in payload.get("records", []):
                f = record.get("fields", {})
                raw = f.get("color") or f.get("couleur") or f.get("vigilance_color") or f.get("niveau")
                level = {"vert":"VERT", "green":"VERT", "1":"VERT", "jaune":"JAUNE", "yellow":"JAUNE", "2":"JAUNE", "orange":"ORANGE", "3":"ORANGE", "rouge":"ROUGE", "red":"ROUGE", "4":"ROUGE"}.get(norm(raw), "INDETERMINE")
                alerts.append({
                    "dep": dep, "department_name": DEPARTMENT_INFO[dep]["name"], "level": level,
                    "phenomenon": str(f.get("phenomenon") or f.get("phenomene") or f.get("risque") or "Tous phenomenes"),
                    "day": str(f.get("echeance") or f.get("day") or f.get("date") or ""),
                })
        except Exception:
            continue
    return alerts, successful > 0

# =============================================================================
# VIGICRUES SPATIAL
# =============================================================================
def iter_geojson_coordinates(geometry):
    output = []
    def walk(value):
        if not isinstance(value, (list, tuple)): return
        if len(value) >= 2 and isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
            output.append((float(value[1]), float(value[0])))
        else:
            for child in value: walk(child)
    if isinstance(geometry, dict): walk(geometry.get("coordinates", []))
    return output


def identify_followed_river(name):
    text = norm(name)
    for river in sorted(RIVERS, key=len, reverse=True):
        if norm(river) in text:
            return RIVER_LABELS.get(river, river.title())
    return None


def distance_geometry_to_lgv(geometry, line):
    coords = iter_geojson_coordinates(geometry)
    if not coords or not line: return None, None
    step = max(1, len(coords) // 250)
    best_pk, best_distance = None, None
    for lat, lon in coords[::step]:
        pk, distance = pk_distance(lat, lon, line)
        if distance is not None and (best_distance is None or distance < best_distance):
            best_pk, best_distance = pk, distance
    return best_pk, best_distance


@st.cache_data(ttl=900, show_spinner=False)
def load_vigicrues_spatial(line_signature, max_distance_km=50.0):
    line = [tuple(item) for item in line_signature]
    try:
        payload = get_json(VIGICRUES_GEOJSON_URL, timeout=(5, 45), headers={"Accept":"application/geo+json, application/json", "User-Agent":"LGV-SEA-Monitoring/4.0"})
    except Exception:
        return [], False, None
    results = []
    publication = payload.get("DtHrInfoVigiCru") or payload.get("date") or payload.get("updated")
    for feature in payload.get("features", []):
        props, geometry = feature.get("properties", {}), feature.get("geometry", {})
        name = props.get("NomEntVigiCru") or props.get("LbEntVigiCru") or props.get("name") or props.get("nom") or ""
        river = identify_followed_river(name)
        if not river: continue
        raw = props.get("NivSituVigiCruEnt") or props.get("NivVigiCru") or props.get("niveau") or props.get("level")
        level = {"1":"VERT", "2":"JAUNE", "3":"ORANGE", "4":"ROUGE", "vert":"VERT", "jaune":"JAUNE", "orange":"ORANGE", "rouge":"ROUGE"}.get(norm(raw), "INDETERMINE")
        pk, distance = distance_geometry_to_lgv(geometry, line)
        if distance is None or distance > max_distance_km: continue
        results.append({"code": props.get("CdEntVigiCru", ""), "name": str(name), "river": river, "level": level, "pk_km": pk, "distance_km": distance, "geometry": geometry})
    unique = {}
    for row in results:
        key = row["code"] or row["name"]
        if key not in unique or LEVEL_RANK[row["level"]] > LEVEL_RANK[unique[key]["level"]]: unique[key] = row
    ordered = sorted(unique.values(), key=lambda r: (-LEVEL_RANK.get(r["level"], -1), r["distance_km"], r["name"]))
    return ordered, True, publication

# =============================================================================
# APPLICATION
# =============================================================================
st.title("⚠️ Alertes meteo par departement et Vigicrues spatial LGV")
st.caption("Departements suivis : 33, 16, 17, 86, 79 et 37. Les previsions sont indicatives et ne remplacent pas la vigilance officielle.")

try:
    snapshot = load_snapshot()
    line = build_polyline(snapshot.get("lgv_lines"))
except Exception as exc:
    st.error(f"Snapshot LGV indisponible : {exc}")
    st.stop()
if not line:
    st.error("Le trace LGV est absent du snapshot.")
    st.stop()
line_signature = make_line_signature(line)
lat_c = float(np.mean([p[0] for p in line]))
lon_c = float(np.mean([p[1] for p in line]))

with st.sidebar:
    st.header("Pilotage")
    vigicrues_radius = st.slider("Distance maximale cours d'eau / LGV", 5, 100, 50, 5, format="%d km")
    selected_department = st.selectbox("Departement a detailler", DEPARTMENTS, format_func=lambda dep: f"{dep} - {DEPARTMENT_INFO[dep]['name']}")
    show_only_alerts = st.checkbox("Afficher seulement les alertes non vertes", False)
    if st.button("🔄 Actualiser les donnees"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Chargement des vigilances, previsions et troncons Vigicrues..."):
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_mf = pool.submit(load_mf_alerts)
        f_fc = pool.submit(load_all_department_forecasts)
        f_vc = pool.submit(load_vigicrues_spatial, line_signature, float(vigicrues_radius))
        mf_alerts, mf_ok = f_mf.result()
        forecasts, forecast_errors = f_fc.result()
        vigicrues_rows, vigicrues_ok, vigicrues_date = f_vc.result()

# Synthese departementale
summary_rows = []
for dep in DEPARTMENTS:
    df = forecasts[forecasts["department"] == dep].copy() if not forecasts.empty else pd.DataFrame()
    official = [x["level"] for x in mf_alerts if x["dep"] == dep and x["level"] != "INDETERMINE"]
    official_level = strongest_level(official) if official else ("VERT" if mf_ok else "INDETERMINE")
    forecast_level, reason = "INDETERMINE", "Prevision indisponible"
    rain_24h = rain_3d = rain_7d = gust = np.nan
    if not df.empty:
        d0 = df.iloc[0]
        forecast_level, reason = forecast_warning_level(d0["rain_mm"], d0["rain_probability"], d0["gust_kmh"], d0["weather_code"])
        rain_24h = safe_float(d0["rain_mm"])
        rain_3d = float(df.head(3)["rain_mm"].fillna(0).sum())
        rain_7d = float(df.head(7)["rain_mm"].fillna(0).sum())
        gust = float(df["gust_kmh"].max())
    global_level = strongest_level([official_level, forecast_level])
    summary_rows.append({"Departement": f"{dep} - {DEPARTMENT_INFO[dep]['name']}", "Niveau global": global_level, "Vigilance officielle": official_level, "Signal previsionnel": forecast_level, "Pluie 24 h (mm)": rain_24h, "Pluie 3 j (mm)": rain_3d, "Pluie 7 j (mm)": rain_7d, "Rafale max (km/h)": gust, "Motif": reason})
summary = pd.DataFrame(summary_rows).sort_values("Niveau global", key=lambda s: s.map(LEVEL_RANK), ascending=False)

non_green_rivers = [r for r in vigicrues_rows if r["level"] in ["JAUNE", "ORANGE", "ROUGE"]]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Departements", len(DEPARTMENTS))
c2.metric("Rouge", int((summary["Niveau global"] == "ROUGE").sum()))
c3.metric("Orange", int((summary["Niveau global"] == "ORANGE").sum()))
c4.metric("Jaune", int((summary["Niveau global"] == "JAUNE").sum()))
c5.metric("Troncons crues en alerte", len(non_green_rivers))

st.subheader("1. Synthese meteo par departement")
st.dataframe(summary, use_container_width=True, hide_index=True, column_config={
    "Pluie 24 h (mm)": st.column_config.NumberColumn(format="%.1f"), "Pluie 3 j (mm)": st.column_config.NumberColumn(format="%.1f"),
    "Pluie 7 j (mm)": st.column_config.NumberColumn(format="%.1f"), "Rafale max (km/h)": st.column_config.NumberColumn(format="%.0f"),
})
if not mf_ok: st.warning("Vigilance Meteo-France non verifiee. Le signal previsionnel ne remplace pas la vigilance officielle.")
if forecast_errors: st.warning("Previsions indisponibles pour : " + ", ".join(forecast_errors))

st.subheader(f"2. Previsions a 7 jours : {selected_department} - {DEPARTMENT_INFO[selected_department]['name']}")
dep_forecast = forecasts[forecasts["department"] == selected_department].copy() if not forecasts.empty else pd.DataFrame()
if dep_forecast.empty:
    st.warning("Prevision departementale indisponible.")
else:
    warnings = dep_forecast.apply(lambda r: forecast_warning_level(r["rain_mm"], r["rain_probability"], r["gust_kmh"], r["weather_code"]), axis=1)
    dep_forecast["Niveau"] = [x[0] for x in warnings]
    dep_forecast["Motif"] = [x[1] for x in warnings]
    d0 = dep_forecast.iloc[0]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Pluie aujourd'hui", f"{d0['rain_mm']:.1f} mm")
    p2.metric("Probabilite maximale", f"{d0['rain_probability']:.0f} %")
    p3.metric("Rafale maximale", f"{d0['gust_kmh']:.0f} km/h")
    p4.metric("Cumul sur 7 jours", f"{dep_forecast['rain_mm'].fillna(0).sum():.1f} mm")
    display = dep_forecast[["date", "Niveau", "weather", "rain_mm", "rain_probability", "gust_kmh", "temperature_min", "temperature_max", "Motif"]].rename(columns={"date":"Date", "weather":"Situation", "rain_mm":"Pluie (mm)", "rain_probability":"Probabilite pluie (%)", "gust_kmh":"Rafales (km/h)", "temperature_min":"Temperature min. (C)", "temperature_max":"Temperature max. (C)"})
    st.dataframe(display, use_container_width=True, hide_index=True)
    fig = go.Figure()
    fig.add_bar(x=dep_forecast["date"], y=dep_forecast["rain_mm"], name="Pluie prevue", marker_color="#3b82f6")
    fig.add_scatter(x=dep_forecast["date"], y=dep_forecast["gust_kmh"], name="Rafales", mode="lines+markers", line=dict(color="#f97316", width=3), yaxis="y2")
    fig.update_layout(height=420, hovermode="x unified", yaxis=dict(title="Precipitations (mm)"), yaxis2=dict(title="Rafales (km/h)", overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.subheader("3. Vigicrues spatial LGV")
if not vigicrues_ok:
    st.warning("Flux Vigicrues injoignable. Reessayer dans quelques minutes.")
elif not vigicrues_rows:
    st.success(f"Aucun cours d'eau suivi trouve a moins de {vigicrues_radius} km de la LGV.")
else:
    if vigicrues_date: st.caption(f"Date du flux Vigicrues : {vigicrues_date}")
    vc_df = pd.DataFrame(vigicrues_rows)
    vc_display = vc_df[["river", "name", "level", "pk_km", "distance_km", "code"]].rename(columns={"river":"Cours d'eau", "name":"Troncon Vigicrues", "level":"Niveau", "pk_km":"PK LGV le plus proche", "distance_km":"Distance LGV (km)", "code":"Code troncon"})
    if show_only_alerts: vc_display = vc_display[vc_display["Niveau"] != "VERT"]
    st.dataframe(vc_display, use_container_width=True, hide_index=True)

    m = folium.Map(location=[lat_c, lon_c], zoom_start=7, tiles=None, control_scale=True)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri, Maxar, Earthstar Geographics, GIS User Community", name="Satellite", max_zoom=19).add_to(m)
    folium.TileLayer(tiles="OpenStreetMap", name="Plan").add_to(m)
    for segment in snapshot.get("lgv_lines") or []:
        points = [[float(p["lat"]), float(p["lon"])] for p in segment if isinstance(p, dict) and "lat" in p and "lon" in p]
        if points: folium.PolyLine(points, color="#ef4444", weight=4, opacity=0.95, tooltip="LGV SEA").add_to(m)
    rows_to_map = [r for r in vigicrues_rows if not show_only_alerts or r["level"] != "VERT"]
    for row in rows_to_map:
        color = LEVEL_COLOR.get(row["level"], LEVEL_COLOR["INDETERMINE"])
        tooltip = f"{row['river']} | {row['name']} | {row['level']} | distance LGV {row['distance_km']:.1f} km | PK {row['pk_km']:.1f}"
        folium.GeoJson({"type":"Feature", "properties":{}, "geometry":row["geometry"]}, tooltip=tooltip, style_function=lambda feature, c=color: {"color":c, "weight":5, "opacity":0.9, "fillColor":c, "fillOpacity":0.25}).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, use_container_width=True, height=620, returned_objects=[])

st.subheader("4. Priorites de surveillance")
priorities = []
for row in summary_rows:
    if row["Niveau global"] not in ["VERT", "INDETERMINE"]:
        priorities.append({"Type":"Meteo", "Element":row["Departement"], "Niveau":row["Niveau global"], "Action":LEVEL_ACTION[row["Niveau global"]]})
for row in vigicrues_rows:
    if row["level"] not in ["VERT", "INDETERMINE"]:
        priorities.append({"Type":"Vigicrues", "Element":f"{row['river']} - {row['name']} | PK {row['pk_km']:.1f}", "Niveau":row["level"], "Action":LEVEL_ACTION[row["level"]]})
if priorities:
    priority_df = pd.DataFrame(priorities).sort_values("Niveau", key=lambda s: s.map(LEVEL_RANK), ascending=False)
    st.dataframe(priority_df, use_container_width=True, hide_index=True)
else:
    st.success("Aucune priorite particuliere detectee. Maintenir la surveillance courante.")

st.caption("Sources : vigilance meteorologique, Open-Meteo, Vigicrues et snapshot LGV. Les seuils previsionnels sont des indicateurs internes d'aide a la surveillance.")
