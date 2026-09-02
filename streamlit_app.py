from __future__ import annotations

import math
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
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
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PIEZO_STATIONS_URL = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations"
PIEZO_TR_URL = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques_tr"
PIEZO_HISTORY_URL = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques"
VIGICRUES_URL = "https://www.vigicrues.gouv.fr/services/v1.1/TerEntVigiCru.json"
VIGICRUES_GEOJSON_URL = "https://www.vigicrues.gouv.fr/services/vigicrues.geojson/"
HYDRO_STATIONS_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations"
# Points centraux indicatifs, utilises uniquement pour une prevision synthetique par departement.
DEPARTMENT_INFO = {
    "33": {"name": "Gironde", "lat": 44.8378, "lon": -0.5792},
    "16": {"name": "Charente", "lat": 45.6484, "lon": 0.1560},
    "17": {"name": "Charente-Maritime", "lat": 45.7467, "lon": -0.6333},
    "86": {"name": "Vienne", "lat": 46.5802, "lon": 0.3404},
    "79": {"name": "Deux-Sevres", "lat": 46.3237, "lon": -0.4588},
    "37": {"name": "Indre-et-Loire", "lat": 47.3941, "lon": 0.6848},
}
MF_URL = "https://public.opendatasoft.com/api/records/1.0/search/"
MF_DATASET = "weatherref-france-vigilance-meteo-departement"
DEPARTMENTS = ["37", "86", "79", "16", "17", "33"]
RIVERS = ["vienne", "clain", "charente", "boutonne", "seugne", "touvre", "dronne", "isle", "dordogne", "garonne", "thouet", "sevre", "indre", "cher", "creuse", "ciron", "jalles"]

LEVEL_RANK = {"INDETERMINE": -1, "VERT": 0, "JAUNE": 1, "ORANGE": 2, "ROUGE": 3}
LEVEL_COLOR = {"INDETERMINE": "#64748b", "VERT": "#16a34a", "JAUNE": "#eab308", "ORANGE": "#ea580c", "ROUGE": "#dc2626"}
LEVEL_ACTION = {
    "INDETERMINE": "Données insuffisantes, contrôle manuel nécessaire",
    "VERT": "Surveillance courante",
    "JAUNE": "Surveillance renforcée et contrôle de la fraîcheur des données",
    "ORANGE": "Inspection ciblée et contrôle du drainage",
    "ROUGE": "Contrôle prioritaire et application des consignes métier",
}

st.set_page_config(page_title="LGV SEA - Surveillance optimisée", page_icon="⚠️", layout="wide")

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
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*radius*math.asin(min(1, math.sqrt(a)))


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
    output, distance = [(points[0][0], points[0][1], 0.0)], 0.0
    for a, b in zip(points, points[1:]):
        distance += haversine_km(*a, *b)
        output.append((b[0], b[1], distance))
    return output


def pk_distance(lat: float, lon: float, line) -> tuple[float | None, float | None]:
    if len(line) < 2:
        return None, None
    best_d2, best_pk = None, None
    for (lat1, lon1, pk1), (lat2, lon2, pk2) in zip(line, line[1:]):
        lat_mid = (lat1+lat2)/2
        kx, ky = 111.320*math.cos(math.radians(lat_mid)), 111.320
        x1, y1, x2, y2, xp, yp = lon1*kx, lat1*ky, lon2*kx, lat2*ky, lon*kx, lat*ky
        dx, dy = x2-x1, y2-y1
        den = dx*dx+dy*dy
        t = 0 if den == 0 else max(0, min(1, ((xp-x1)*dx+(yp-y1)*dy)/den))
        cx, cy = x1+t*dx, y1+t*dy
        d2 = (xp-cx)**2+(yp-cy)**2
        if best_d2 is None or d2 < best_d2:
            best_d2, best_pk = d2, pk1+t*(pk2-pk1)
    return best_pk, math.sqrt(best_d2) if best_d2 is not None else None


def risk_level(score: float) -> str:
    if pd.isna(score): return "INDETERMINE"
    if score >= 75: return "ROUGE"
    if score >= 50: return "ORANGE"
    if score >= 25: return "JAUNE"
    return "VERT"


def scale(value, low, high):
    if pd.isna(value): return np.nan
    return float(np.clip((value-low)/(high-low), 0, 1))

# =============================================================================
# SNAPSHOT ET SECTEURS
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_snapshot():
    return get_json(SNAPSHOT_URL)


def load_points(snapshot):
    payload = snapshot.get("sectors", {})
    df = pd.DataFrame(payload.get("sectors", []) if isinstance(payload, dict) else [])
    for col in ["latitude", "longitude", "pk_km", "ai_pred_probability", "ai_soil_fragility", "score"]:
        if col in df: df[col] = pd.to_numeric(df[col], errors="coerce")
    if "commune_name" not in df: df["commune_name"] = "Commune inconnue"
    return df.dropna(subset=["latitude", "longitude", "pk_km"])


def make_sectors(points, width=10):
    work = points.copy()
    work["pk_start"] = (work["pk_km"]//width).astype(int)*width
    rows = []
    for pk_start, group in work.groupby("pk_start"):
        ai = pd.to_numeric(group.get("ai_pred_probability"), errors="coerce").dropna()
        soil = pd.to_numeric(group.get("ai_soil_fragility"), errors="coerce").dropna()
        measured = pd.to_numeric(group.get("score"), errors="coerce").dropna()
        base = float(ai.max()) if not ai.empty else 0.40
        fragility = float(soil.mean()) if not soil.empty else 0.40
        signal = min(1.0, float(measured.max())/4) if not measured.empty else 0.20
        quick_score = round(100*(0.55*base+0.30*fragility+0.15*signal), 1)
        rows.append({
            "sector_id": f"PK_{pk_start:03d}_{pk_start+width:03d}",
            "name": f"PK {pk_start:03d}-{pk_start+width:03d}",
            "pk_start": float(pk_start), "pk_end": float(pk_start+width),
            "latitude": float(group["latitude"].mean()), "longitude": float(group["longitude"].mean()),
            "communes": ", ".join(sorted(group["commune_name"].astype(str).unique())),
            "static_score": quick_score, "static_level": risk_level(quick_score),
            "susceptibility": base, "soil_fragility": fragility, "signal": signal,
        })
    return pd.DataFrame(rows).sort_values("pk_start")

# =============================================================================
# METEO ET PLUIE
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast(lat, lon):
    return get_json(FORECAST_URL, params={
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "hourly": "precipitation,soil_moisture_0_to_7cm,wind_gusts_10m",
        "daily": "precipitation_sum,precipitation_probability_max,wind_gusts_10m_max,weather_code",
        "forecast_days": 7, "timezone": "Europe/Paris",
    })


def forecast_summary(payload):
    hourly = payload.get("hourly", {})
    daily = payload.get("daily", {})
    rain_h = pd.to_numeric(pd.Series(hourly.get("precipitation", [])), errors="coerce").fillna(0)
    soil = pd.to_numeric(pd.Series(hourly.get("soil_moisture_0_to_7cm", [])), errors="coerce").dropna()
    rain_d = pd.to_numeric(pd.Series(daily.get("precipitation_sum", [])), errors="coerce").fillna(0)
    return {
        "rain_6h": float(rain_h.iloc[:6].sum()), "rain_24h": float(rain_h.iloc[:24].sum()),
        "rain_72h": float(rain_h.iloc[:72].sum()), "rain_7d": float(rain_d.sum()),
        "soil": float(soil.iloc[0]) if not soil.empty else np.nan, "daily": daily,
    }


@st.cache_data(ttl=21600, show_spinner=False)
def load_daily_rain(lat, lon, start_date, end_date):
    payload = get_json(ARCHIVE_URL, params={
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
        "daily": "precipitation_sum", "timezone": "Europe/Paris",
    })
    daily = payload.get("daily", {})
    df = pd.DataFrame({"date": daily.get("time", []), "rain_mm": daily.get("precipitation_sum", [])})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce")
    return df.dropna().sort_values("date")


def add_rain_features(df):
    df = df.sort_values("date").copy()
    for days in [3, 7, 15, 30]:
        df[f"rain_{days}d"] = df["rain_mm"].rolling(days, min_periods=1).sum()
    return df

# =============================================================================
# PIEZOMETRIE ACTIVE
# =============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def load_piezo_referential():
    frames = []
    for dep in DEPARTMENTS:
        try:
            rows = get_json(PIEZO_STATIONS_URL, params={"code_departement": dep, "size": 20000}).get("data", [])
            if rows: frames.append(pd.DataFrame(rows))
        except Exception:
            pass
    if not frames: return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("code_bss") if "code_bss" in df else df


@st.cache_data(ttl=900, show_spinner=False)
def load_piezo_tr(code_bss, days=8):
    start = (datetime.now(timezone.utc)-timedelta(days=days)).strftime("%Y-%m-%d")
    rows = get_json(PIEZO_TR_URL, params={"code_bss": code_bss, "date_debut_mesure": start, "size": 20000}).get("data", [])
    df = pd.DataFrame(rows)
    if df.empty: return df
    date_col = next((c for c in ["date_mesure", "date_mesure_utc", "date"] if c in df), None)
    level_col = next((c for c in ["niveau_nappe_eau", "niveau_eau_ngf", "niveau"] if c in df), None)
    if not date_col: return pd.DataFrame()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["level"] = pd.to_numeric(df[level_col], errors="coerce") if level_col else np.nan
    return df.dropna(subset=["date"]).sort_values("date")


@st.cache_data(ttl=3600, show_spinner=False)
def load_piezo_history(code_bss, start_date, end_date):
    rows = get_json(PIEZO_HISTORY_URL, params={
        "code_bss": code_bss, "date_debut_mesure": start_date.isoformat(),
        "date_fin_mesure": end_date.isoformat(), "size": 20000,
    }, timeout=(5, 45)).get("data", [])
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["date"] = pd.to_datetime(df.get("date_mesure"), errors="coerce")
    df["level"] = pd.to_numeric(df.get("niveau_nappe_eau"), errors="coerce")
    return df.dropna(subset=["date", "level"]).sort_values("date").drop_duplicates("date")


def piezo_status(df):
    if df.empty: return None
    last = df.iloc[-1]
    age = (pd.Timestamp.now(tz="UTC")-last["date"]).total_seconds()/3600
    status = "ACTIF" if age <= 48 else "RETARD" if age <= 168 else "HORS_LIGNE"
    valid = df.dropna(subset=["level"])
    current = float(valid.iloc[-1]["level"]) if not valid.empty else np.nan
    def delta(hours):
        before = valid[valid["date"] <= valid.iloc[-1]["date"]-pd.Timedelta(hours=hours)] if not valid.empty else pd.DataFrame()
        return current-float(before.iloc[-1]["level"]) if not before.empty else np.nan
    return {"status": status, "last_date": last["date"], "age_h": age, "level": current,
            "delta_24h": delta(24), "delta_7d": delta(168)}


def active_piezometers(line, radius, limit, pk_min=None, pk_max=None):
    stations = load_piezo_referential()
    if stations.empty: return pd.DataFrame()
    lat_col = next((c for c in ["latitude", "y"] if c in stations), None)
    lon_col = next((c for c in ["longitude", "x"] if c in stations), None)
    if not lat_col or not lon_col: return pd.DataFrame()
    stations = stations.copy()
    stations["latitude"] = pd.to_numeric(stations[lat_col], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations[lon_col], errors="coerce")
    stations = stations.dropna(subset=["latitude", "longitude", "code_bss"])
    projection = stations.apply(lambda r: pk_distance(r["latitude"], r["longitude"], line), axis=1)
    stations[["pk_km", "distance_km"]] = pd.DataFrame(projection.tolist(), index=stations.index)
    stations = stations[stations["distance_km"] <= radius]
    if pk_min is not None: stations = stations[stations["pk_km"] >= pk_min-10]
    if pk_max is not None: stations = stations[stations["pk_km"] <= pk_max+10]
    stations = stations.sort_values("distance_km").head(limit)
    if stations.empty: return pd.DataFrame()
    rows = []
    with ThreadPoolExecutor(max_workers=min(10, len(stations))) as pool:
        jobs = {pool.submit(load_piezo_tr, str(r.code_bss)): r for r in stations.itertuples(index=False)}
        for future in as_completed(jobs):
            try:
                info = piezo_status(future.result())
                if info and info["status"] in ["ACTIF", "RETARD"]:
                    row = jobs[future]._asdict(); row.update(info); rows.append(row)
            except Exception:
                pass
    return pd.DataFrame(rows)

# =============================================================================
# VIGILANCES
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_mf_alerts():
    """Charge les vigilances departementales depuis le miroir OpenDataSoft.

    Le flux officiel Météo-France nécessite une clé API. Ce flux public sert de
    secours sans secret. L'interface signale clairement si la source ne répond pas.
    """
    try:
        query = " OR ".join(f"domain_id:{d}" for d in DEPARTMENTS)
        records = get_json(MF_URL, params={
            "dataset": MF_DATASET, "q": query, "rows": 500
        }).get("records", [])
        alerts = []
        for rec in records:
            f = rec.get("fields", {})
            dep = str(f.get("domain_id", f.get("department", ""))).zfill(2)
            if dep not in DEPARTMENTS:
                continue
            raw_color = f.get("color", f.get("couleur", f.get("level", "")))
            level = {"vert":"VERT", "jaune":"JAUNE", "orange":"ORANGE", "rouge":"ROUGE"}.get(norm(raw_color))
            if level:
                alerts.append({
                    "level": level,
                    "dep": dep,
                    "phenomenon": f.get("phenomenon", f.get("phenomene", "Tous phenomenes")),
                    "day": f.get("echeance", f.get("validity", "En cours")),
                })
        return alerts, True
    except Exception:
        return [], False


def _iter_geo_points(obj):
    """Extrait recursivement les paires XY d'une geometrie GeoJSON."""
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and all(isinstance(v, (int, float)) for v in obj[:2]):
            yield float(obj[0]), float(obj[1])
        else:
            for part in obj:
                yield from _iter_geo_points(part)


def _geojson_points_wgs84(geometry):
    points = list(_iter_geo_points((geometry or {}).get("coordinates", [])))
    if not points:
        return []
    # Le flux historique Vigicrues est annonce en Lambert-93. Certains miroirs
    # livrent deja du WGS84. On detecte le cas a partir de l'ordre de grandeur.
    projected = abs(points[0][0]) > 180 or abs(points[0][1]) > 90
    if not projected:
        return [(y, x) for x, y in points]
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
        converted = []
        for x, y in points:
            lon, lat = transformer.transform(x, y)
            converted.append((lat, lon))
        return converted
    except Exception:
        return []


def _nearest_line_location(points, line):
    """Retourne PK et distance minimale entre des points de troncon et la LGV."""
    best_pk, best_dist = None, None
    if not points:
        return best_pk, best_dist
    # Echantillonnage borne pour conserver un chargement rapide.
    step = max(1, len(points) // 250)
    sample = points[::step]
    if points[-1] not in sample:
        sample.append(points[-1])
    for lat, lon in sample:
        pk, dist = pk_distance(lat, lon, line)
        if dist is not None and (best_dist is None or dist < best_dist):
            best_pk, best_dist = pk, dist
    return best_pk, best_dist


@st.cache_data(ttl=900, show_spinner=False)
def load_vigicrues_near_lgv(line_signature, radius_km=5.0):
    """Tous les troncons Vigicrues croisant ou proches de la LGV.

    Important : Vigicrues ne couvre pas chaque petit cours d'eau. Cette fonction
    selectionne automatiquement tous les troncons du reseau réglementaire
    Vigicrues dont la geometrie est a moins de radius_km du trace LGV.
    """
    line = [tuple(x) for x in line_signature]
    headers = {"Accept": "application/geo+json,application/json", "User-Agent": "LGV-SEA-Monitoring/4.0"}
    try:
        payload = get_json(VIGICRUES_GEOJSON_URL, headers=headers, timeout=(8, 45))
        features = payload.get("features", [])
    except Exception:
        return [], False
    rows = []
    for feature in features:
        props = feature.get("properties", {}) or {}
        points = _geojson_points_wgs84(feature.get("geometry", {}))
        pk, distance = _nearest_line_location(points, line)
        if distance is None or distance > float(radius_km):
            continue
        raw = props.get("NivSituVigiCruEnt", props.get("NivVigiCru", props.get("niveau", 0)))
        level = {"1":"VERT", "2":"JAUNE", "3":"ORANGE", "4":"ROUGE",
                 "vert":"VERT", "jaune":"JAUNE", "orange":"ORANGE", "rouge":"ROUGE"}.get(norm(raw), "INDETERMINE")
        rows.append({
            "code": str(props.get("CdEntVigiCru", props.get("code", ""))),
            "name": str(props.get("NomEntVigiCru", props.get("name", "Cours d'eau surveille"))),
            "level": level,
            "pk_km": pk,
            "distance_km": distance,
        })
    unique = {(r["code"], r["name"]): r for r in rows}
    return sorted(unique.values(), key=lambda r: (r["distance_km"], r["pk_km"] or 9999)), True


@st.cache_data(ttl=1800, show_spinner=False)
def load_department_forecasts():
    """Previsions 7 jours pour les six departements, chargees en parallele."""
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {
            pool.submit(load_forecast, info["lat"], info["lon"]): (dep, info)
            for dep, info in DEPARTMENT_INFO.items()
        }
        for future in as_completed(jobs):
            dep, info = jobs[future]
            try:
                summary = forecast_summary(future.result())
                rows.append({"dep": dep, "department": info["name"], **summary, "ok": True})
            except Exception:
                rows.append({"dep": dep, "department": info["name"], "ok": False})
    return sorted(rows, key=lambda r: DEPARTMENTS.index(r["dep"]))


# =============================================================================
# RISQUE
# =============================================================================
def historical_piezo_context(history, target):
    if history.empty: return None
    past = history[history["date"] <= target]
    if past.empty: return None
    current = float(past.iloc[-1]["level"])
    ref = history[history["date"].dt.month == target.month]["level"]
    if len(ref) < 20: ref = history["level"]
    percentile = float((ref <= current).mean()) if len(ref) else np.nan
    before = past[past["date"] <= target-pd.Timedelta(days=1)]
    delta = current-float(before.iloc[-1]["level"]) if not before.empty else np.nan
    return {"percentile":percentile, "delta_24h":delta}


def calculate_risk(rain, sector, piezo=None, forecast=None, mf="VERT", vc="VERT", historical=False):
    c = {
        "Pluie jour":scale(rain.get("rain_mm"),5,60), "Pluie 3 j":scale(rain.get("rain_3d"),15,100),
        "Pluie 7 j":scale(rain.get("rain_7d"),25,150), "Pluie 30 j":scale(rain.get("rain_30d"),60,300),
        "Sensibilité":safe_float(sector.get("susceptibility"),.4), "Sol":safe_float(sector.get("soil_fragility"),.4),
        "Piézo niveau":scale(piezo.get("percentile"),.60,.99) if piezo else np.nan,
        "Piézo tendance":scale(piezo.get("delta_24h"),.02,.50) if piezo else np.nan,
        "Prévision 72 h":scale(forecast.get("rain_72h"),10,100) if forecast and not historical else np.nan,
        "Humidité sol":scale(forecast.get("soil"),.20,.50) if forecast and not historical else np.nan,
        "Vigilance météo":LEVEL_RANK.get(mf,0)/3 if not historical else np.nan,
        "Vigicrues":LEVEL_RANK.get(vc,0)/3 if not historical else np.nan,
    }
    weights = {"Pluie jour":.07,"Pluie 3 j":.11,"Pluie 7 j":.16,"Pluie 30 j":.08,"Sensibilité":.13,"Sol":.08,
               "Piézo niveau":.14,"Piézo tendance":.10,"Prévision 72 h":.07,"Humidité sol":.03,"Vigilance météo":.02,"Vigicrues":.01}
    valid = {k:v for k,v in c.items() if not pd.isna(v)}
    valid_weight = sum(weights[k] for k in valid)
    score = 100*sum(valid[k]*weights[k] for k in valid)/valid_weight if valid_weight else np.nan
    factors = [k for k,v in sorted(valid.items(), key=lambda x:x[1], reverse=True)[:4] if v >= .5]
    return {"score":round(score,1),"level":risk_level(score),"confidence":round(100*valid_weight),"factors":factors}

# =============================================================================
# APPLICATION LAZY LOAD
# =============================================================================
st.title("⚠️ LGV SEA - Surveillance optimisée des risques de glissement")
st.caption("Chargement rapide : tous les secteurs sont affichés immédiatement. Les API lourdes sont appelées uniquement dans le module choisi.")

try:
    snapshot = load_snapshot(); points = load_points(snapshot); line = build_polyline(snapshot.get("lgv_lines"))
except Exception as exc:
    st.error(f"Snapshot indisponible : {exc}"); st.stop()
if points.empty or not line:
    st.error("Tracé LGV ou secteurs absents."); st.stop()
sectors = make_sectors(points)

with st.sidebar:
    st.header("Pilotage")
    sector_name = st.selectbox("Secteur surveillé", ["Tous les secteurs"]+sectors["name"].tolist(), index=0)
    module = st.radio("Module", ["Vue rapide", "Alertes et prévisions", "Pluie historique", "Piézomètres actifs", "Risque historique", "Carte satellite"], index=0)
    radius = st.slider("Rayon piézomètres", 1, 20, 8, format="%d km")
    river_radius = st.slider("Proximité cours d'eau / LGV", 0.5, 15.0, 5.0, 0.5, format="%.1f km")
    station_limit = st.slider("Stations à tester", 10, 60, 30, 10)
    if st.button("🔄 Actualiser"):
        st.cache_data.clear(); st.rerun()

all_selected = sector_name == "Tous les secteurs"
selected_sectors = sectors if all_selected else sectors[sectors["name"] == sector_name]
if all_selected:
    selected_points = points
else:
    s = selected_sectors.iloc[0]
    selected_points = points[(points["pk_km"] >= s["pk_start"]) & (points["pk_km"] <= s["pk_end"])]
lat_c, lon_c = float(selected_points["latitude"].mean()), float(selected_points["longitude"].mean())

if module == "Vue rapide":
    st.subheader("Vue rapide de tous les secteurs")
    quick = sectors[["name","pk_start","pk_end","communes","static_score","static_level"]].rename(columns={
        "name":"Secteur","pk_start":"PK début","pk_end":"PK fin","communes":"Communes","static_score":"Indice disponible","static_level":"Niveau"})
    quick = quick.sort_values("Niveau", key=lambda x:x.map(LEVEL_RANK), ascending=False)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Secteurs",len(quick)); c2.metric("Rouges",int((quick["Niveau"]=="ROUGE").sum())); c3.metric("Orange",int((quick["Niveau"]=="ORANGE").sum())); c4.metric("Jaune",int((quick["Niveau"]=="JAUNE").sum()))
    st.dataframe(quick,use_container_width=True,hide_index=True,height=550)
    st.info("Vue instantanée depuis le snapshot. Sélectionne un secteur et un module pour charger les données détaillées.")

elif module == "Alertes et prévisions":
    st.subheader("Alertes météo par département et Vigicrues spatial LGV")
    st.caption("Départements suivis : 33, 16, 17, 86, 79 et 37. Les prévisions sont calculées au point central indicatif de chaque département.")
    line_signature = tuple((round(a, 6), round(b, 6), round(pk, 3)) for a, b, pk in line)
    with st.spinner("Chargement parallèle : vigilance météo, prévisions et tronçons Vigicrues proches de la LGV..."):
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_mf = pool.submit(load_mf_alerts)
            f_fc = pool.submit(load_department_forecasts)
            f_vc = pool.submit(load_vigicrues_near_lgv, line_signature, river_radius)
            mf_alerts, mf_ok = f_mf.result()
            dept_forecasts = f_fc.result()
            vc_alerts, vc_ok = f_vc.result()

    st.markdown("### 1. Vigilances et prévisions par département")
    alert_by_dep = {}
    for item in mf_alerts:
        alert_by_dep.setdefault(item["dep"], []).append(item)
    forecast_rows = []
    for item in dept_forecasts:
        dep = item["dep"]
        dep_alerts = alert_by_dep.get(dep, [])
        max_level = max((a["level"] for a in dep_alerts), key=lambda x: LEVEL_RANK.get(x, -1), default="VERT" if mf_ok else "INDETERMINE")
        phenomena = ", ".join(sorted({str(a["phenomenon"]) for a in dep_alerts if a.get("phenomenon")})) or "Aucune vigilance non verte"
        forecast_rows.append({
            "Département": f"{dep} - {item['department']}",
            "Vigilance": max_level,
            "Phénomène": phenomena,
            "Pluie 6 h (mm)": round(item.get("rain_6h", np.nan), 1),
            "Pluie 24 h (mm)": round(item.get("rain_24h", np.nan), 1),
            "Pluie 72 h (mm)": round(item.get("rain_72h", np.nan), 1),
            "Pluie 7 j (mm)": round(item.get("rain_7d", np.nan), 1),
            "Source prévision": "Disponible" if item.get("ok") else "Indisponible",
        })
    forecast_df = pd.DataFrame(forecast_rows)
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    if not mf_ok:
        st.warning("Le flux de vigilance départementale n'a pas répondu. Les prévisions restent affichées, mais la vigilance officielle est à vérifier sur Météo-France.")

    st.markdown("### 2. Tronçons Vigicrues croisant la LGV ou à proximité")
    st.caption(f"Sélection géographique automatique dans un rayon de {river_radius:.1f} km autour du tracé LGV. Le PK indiqué est le point du tracé le plus proche.")
    if not vc_ok:
        st.warning("Le flux géographique Vigicrues n'a pas répondu.")
    elif not vc_alerts:
        st.info("Aucun tronçon du réseau réglementaire Vigicrues trouvé dans ce rayon. Cela ne signifie pas qu'il n'existe aucun petit cours d'eau.")
    else:
        vc_df = pd.DataFrame(vc_alerts)
        vc_df = vc_df.rename(columns={"code":"Code", "name":"Cours d'eau / tronçon", "level":"Vigilance", "pk_km":"PK LGV proche", "distance_km":"Distance LGV (km)"})
        vc_df["PK LGV proche"] = vc_df["PK LGV proche"].round(1)
        vc_df["Distance LGV (km)"] = vc_df["Distance LGV (km)"].round(2)
        vc_df = vc_df.sort_values("Vigilance", key=lambda col: col.map(LEVEL_RANK), ascending=False)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tronçons proches", len(vc_df))
        c2.metric("Jaune", int((vc_df["Vigilance"] == "JAUNE").sum()))
        c3.metric("Orange", int((vc_df["Vigilance"] == "ORANGE").sum()))
        c4.metric("Rouge", int((vc_df["Vigilance"] == "ROUGE").sum()))
        st.dataframe(vc_df, use_container_width=True, hide_index=True, height=430)
        st.download_button("Télécharger les tronçons proches (CSV)", vc_df.to_csv(index=False).encode("utf-8-sig"), "vigicrues_proche_lgv.csv", "text/csv")
    st.info("Limite métier : Vigicrues surveille les principaux tronçons réglementaires, pas tous les fossés ni tous les petits cours d'eau. Pour un inventaire exhaustif des franchissements, il faut croiser le tracé LGV avec le référentiel hydrographique national ou votre couche interne d'ouvrages hydrauliques.")

elif module == "Pluie historique":
    st.subheader(f"Pluviométrie journalière - {sector_name}")
    start_date=st.date_input("Début",date.today()-timedelta(days=365),max_value=date.today())
    end_date=st.date_input("Fin",date.today()-timedelta(days=1),min_value=start_date,max_value=date.today())
    coords=selected_points.groupby("commune_name")[["latitude","longitude"]].mean().reset_index()
    names=sorted(coords["commune_name"].tolist())
    chosen_names=st.multiselect("Communes",names,default=names[:min(6,len(names))])
    coords=coords[coords["commune_name"].isin(chosen_names)]
    frames=[]
    with st.spinner(f"Chargement parallèle de {len(coords)} commune(s)..."):
        with ThreadPoolExecutor(max_workers=min(8,max(1,len(coords)))) as pool:
            jobs={pool.submit(load_daily_rain,r.latitude,r.longitude,start_date,end_date):r.commune_name for r in coords.itertuples(index=False)}
            for future in as_completed(jobs):
                try:
                    df=future.result(); df["commune"]=jobs[future]; frames.append(df)
                except Exception: pass
    rain=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if rain.empty: st.warning("Données indisponibles")
    else:
        fig=go.Figure()
        for name,g in rain.groupby("commune"): fig.add_scatter(x=g["date"],y=g["rain_mm"],mode="lines",name=name)
        fig.update_layout(height=430,yaxis_title="Pluie journalière (mm)",hovermode="x unified")
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        maxima=rain.loc[rain.groupby("commune")["rain_mm"].idxmax(),["commune","date","rain_mm"]]
        st.dataframe(maxima.rename(columns={"commune":"Commune","date":"Date du maximum","rain_mm":"Maximum (mm)"}),use_container_width=True,hide_index=True)

elif module == "Piézomètres actifs":
    st.subheader(f"Piézomètres qui émettent - {sector_name}")
    pk_min=selected_sectors["pk_start"].min() if not all_selected else None; pk_max=selected_sectors["pk_end"].max() if not all_selected else None
    with st.spinner("Vérification parallèle des émissions..."):
        stations=active_piezometers(line,radius,station_limit,pk_min,pk_max)
    if stations.empty: st.warning("Aucun piézomètre actif ou en retard léger trouvé.")
    else:
        cols=["code_bss","nom_commune","pk_km","distance_km","status","last_date","level","delta_24h","delta_7d"]
        display=stations[[c for c in cols if c in stations]].rename(columns={"code_bss":"Code BSS","nom_commune":"Commune","pk_km":"PK (km)","distance_km":"Distance LGV (km)","status":"Emission","last_date":"Dernière mesure","level":"Niveau NGF","delta_24h":"Variation 24 h","delta_7d":"Variation 7 j"})
        st.dataframe(display.sort_values("PK (km)"),use_container_width=True,hide_index=True)

elif module == "Risque historique":
    if all_selected:
        st.warning("Sélectionne un secteur précis pour calculer son risque historique."); st.stop()
    st.subheader(f"Risque historique - {sector_name}")
    data_start=st.date_input("Début des données",date(2021,1,1),max_value=date.today())
    data_end=st.date_input("Fin des données",date.today()-timedelta(days=1),min_value=data_start,max_value=date.today())
    mode=st.radio("Consultation",["Un jour donné","Une période"],horizontal=True)
    if mode=="Un jour donné":
        target=st.date_input("Jour",data_end,min_value=data_start,max_value=data_end); period_start=period_end=target
    else:
        chosen_period=st.date_input("Période",value=(max(data_start,data_end-timedelta(days=90)),data_end),min_value=data_start,max_value=data_end)
        if not isinstance(chosen_period,(tuple,list)) or len(chosen_period)!=2: st.info("Choisis deux dates"); st.stop()
        period_start,period_end=chosen_period
    sector=selected_sectors.iloc[0]
    coords=selected_points.groupby("commune_name")[["latitude","longitude"]].mean().reset_index()
    frames=[]
    with st.spinner("Chargement parallèle des pluies du secteur..."):
        with ThreadPoolExecutor(max_workers=min(8,max(1,len(coords)))) as pool:
            jobs=[pool.submit(load_daily_rain,r.latitude,r.longitude,data_start,data_end) for r in coords.itertuples(index=False)]
            for future in as_completed(jobs):
                try: frames.append(future.result())
                except Exception: pass
    if not frames: st.error("Pluie indisponible"); st.stop()
    rain=add_rain_features(pd.concat(frames).groupby("date",as_index=False)["rain_mm"].max())
    piezo_history=pd.DataFrame()
    with st.spinner("Recherche rapide d'un piézomètre actif..."):
        stations=active_piezometers(line,radius,min(20,station_limit),sector["pk_start"],sector["pk_end"])
    if not stations.empty:
        station=stations.sort_values("distance_km").iloc[0]
        try:
            piezo_history=load_piezo_history(str(station["code_bss"]),data_start,data_end)
            st.caption(f"Piézomètre utilisé : {station['code_bss']} | PK {station['pk_km']:.1f}")
        except Exception: pass
    period=rain[(rain["date"].dt.date>=period_start)&(rain["date"].dt.date<=period_end)]
    rows=[]
    for _,rr in period.iterrows():
        ctx=historical_piezo_context(piezo_history,rr["date"]) if not piezo_history.empty else None
        result=calculate_risk(rr,sector,ctx,historical=True)
        rows.append({"Date":rr["date"],"Indice":result["score"],"Niveau":result["level"],"Confiance (%)":result["confidence"],"Pluie jour":rr["rain_mm"],"Pluie 7 j":rr["rain_7d"],"Pluie 30 j":rr["rain_30d"],"Facteurs":", ".join(result["factors"])})
    history=pd.DataFrame(rows)
    if history.empty: st.info("Aucune donnée")
    else:
        fig=go.Figure(); fig.add_scatter(x=history["Date"],y=history["Indice"],mode="lines+markers",name="Indice")
        for y,label,color in [(25,"Jaune","#eab308"),(50,"Orange","#ea580c"),(75,"Rouge","#dc2626")]: fig.add_hline(y=y,line_dash="dash",line_color=color,annotation_text=label)
        fig.update_layout(height=430,yaxis=dict(title="Indice / 100",range=[0,100]),hovermode="x unified")
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        worst=history.loc[history["Indice"].idxmax()]
        c1,c2,c3,c4=st.columns(4); c1.metric("Niveau maximal",worst["Niveau"]); c2.metric("Indice maximal",f"{worst['Indice']:.1f}/100"); c3.metric("Date",worst["Date"].strftime("%d/%m/%Y")); c4.metric("Confiance",f"{worst['Confiance (%)']} %")
        st.dataframe(history.sort_values("Date",ascending=False),use_container_width=True,hide_index=True)
        st.download_button("⬇️ Export CSV",history.to_csv(index=False).encode("utf-8-sig"),f"risque_{sector['sector_id']}.csv","text/csv")
        st.caption("Pour les dates passées, les anciennes prévisions et vigilances ne sont pas inventées. La confiance est réduite lorsqu'elles ne sont pas archivées.")

elif module == "Carte satellite":
    st.subheader(f"Carte satellite - {sector_name}")
    m=folium.Map(location=[lat_c,lon_c],zoom_start=8 if all_selected else 11,tiles=None,control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",attr="Esri, Maxar, Earthstar Geographics, GIS User Community",name="Satellite",max_zoom=19).add_to(m)
    for segment in snapshot.get("lgv_lines") or []:
        pts=[[p["lat"],p["lon"]] for p in segment if isinstance(p,dict) and "lat" in p and "lon" in p]
        if pts: folium.PolyLine(pts,color="#ef4444",weight=3,opacity=.9).add_to(m)
    for _,sector in selected_sectors.iterrows():
        folium.CircleMarker([sector["latitude"],sector["longitude"]],radius=8,color=LEVEL_COLOR[sector["static_level"]],fill=True,fill_opacity=.85,tooltip=f"{sector['name']} | {sector['static_level']} | {sector['static_score']:.0f}/100").add_to(m)
    st_folium(m,use_container_width=True,height=560,returned_objects=[])

st.caption("Optimisations : vue par défaut sans API lourde, chargement à la demande, requêtes parallèles limitées et cache par source.")
