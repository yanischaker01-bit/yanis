from __future__ import annotations

import io
import math
import unicodedata
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
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
HUBEAU_STATIONS = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations"
HUBEAU_CHRONIQUES = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques"
HUBEAU_CHRONIQUES_TR = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques_tr"
VIGICRUES_URL = "https://www.vigicrues.gouv.fr/services/v1.1/TerEntVigiCru.json"
MF_VIGILANCE_URL = "https://public.opendatasoft.com/api/records/1.0/search/"
MF_VIGILANCE_DATASET = "weatherref-france-vigilance-meteo-departement"
DEPARTEMENTS = ["37", "86", "79", "16", "17", "33"]

LEVELS = ["VERT", "JAUNE", "ORANGE", "ROUGE"]
LEVEL_RANK = {"INDETERMINE": -1, "VERT": 0, "JAUNE": 1, "ORANGE": 2, "ROUGE": 3}
LEVEL_COLOR = {"INDETERMINE": "#64748b", "VERT": "#16a34a", "JAUNE": "#eab308", "ORANGE": "#ea580c", "ROUGE": "#dc2626"}
LEVEL_ACTION = {
    "INDETERMINE": "Données insuffisantes : contrôle manuel nécessaire",
    "VERT": "Surveillance normale",
    "JAUNE": "Surveillance renforcée et contrôle de la fraîcheur des mesures",
    "ORANGE": "Inspection ciblée, contrôle du drainage et suivi rapproché",
    "ROUGE": "Contrôle prioritaire du secteur et application des consignes métier",
}
RIVER_NAMES = ["vienne", "clain", "charente", "boutonne", "seugne", "touvre", "dronne", "isle", "dordogne", "garonne", "thouet", "sevre", "indre", "cher", "creuse", "ciron", "jalles"]

st.set_page_config(page_title="LGV SEA - Risque de glissement", page_icon="⚠️", layout="wide")

# =============================================================================
# UTILITAIRES
# =============================================================================
def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def safe_float(value: Any, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def request_json(url: str, params=None, timeout=(5, 35), headers=None) -> dict:
    response = requests.get(url, params=params, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.json()


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1, math.sqrt(a)))


def build_lgv_polyline(lines) -> list[tuple[float, float, float]]:
    candidates = []
    for segment in lines or []:
        pts = []
        if isinstance(segment, list):
            for point in segment:
                if isinstance(point, dict) and point.get("lat") is not None and point.get("lon") is not None:
                    pts.append((float(point["lat"]), float(point["lon"])))
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    pts.append((float(point[0]), float(point[1])))
        if len(pts) >= 2:
            candidates.append(pts)
    if not candidates:
        return []
    pts = max(candidates, key=len)
    output = [(pts[0][0], pts[0][1], 0.0)]
    distance = 0.0
    for p1, p2 in zip(pts, pts[1:]):
        distance += haversine_km(*p1, *p2)
        output.append((p2[0], p2[1], distance))
    return output


def pk_and_distance(lat: float, lon: float, polyline) -> tuple[float | None, float | None]:
    if len(polyline) < 2:
        return None, None
    best_dist2, best_pk = None, None
    for (lat1, lon1, pk1), (lat2, lon2, pk2) in zip(polyline, polyline[1:]):
        lat_mid = (lat1 + lat2) / 2
        kx, ky = 111.320 * math.cos(math.radians(lat_mid)), 111.320
        x1, y1, x2, y2, xp, yp = lon1*kx, lat1*ky, lon2*kx, lat2*ky, lon*kx, lat*ky
        dx, dy = x2-x1, y2-y1
        denominator = dx*dx + dy*dy
        t = 0 if denominator == 0 else max(0, min(1, ((xp-x1)*dx + (yp-y1)*dy)/denominator))
        cx, cy = x1+t*dx, y1+t*dy
        dist2 = (xp-cx)**2 + (yp-cy)**2
        if best_dist2 is None or dist2 < best_dist2:
            best_dist2, best_pk = dist2, pk1+t*(pk2-pk1)
    return best_pk, math.sqrt(best_dist2) if best_dist2 is not None else None


def level_from_score(score: float) -> str:
    if pd.isna(score): return "INDETERMINE"
    if score >= 75: return "ROUGE"
    if score >= 50: return "ORANGE"
    if score >= 25: return "JAUNE"
    return "VERT"


def zone_label(pk: float, width: int = 10) -> str:
    start = int(pk // width) * width
    return f"PK {start:03d}-{start+width:03d}"

# =============================================================================
# CHARGEMENT LGV ET COMMUNES
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_snapshot() -> dict:
    return request_json(SNAPSHOT_URL)


def sectors_from_snapshot(snapshot: dict) -> pd.DataFrame:
    raw = snapshot.get("sectors", {})
    df = pd.DataFrame(raw.get("sectors", []) if isinstance(raw, dict) else [])
    for col in ["latitude", "longitude", "pk_km"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "commune_name" not in df.columns:
        df["commune_name"] = "Commune inconnue"
    return df.dropna(subset=["latitude", "longitude", "pk_km"])


def automatic_glissement_sectors(sectors: pd.DataFrame, width=10) -> pd.DataFrame:
    df = sectors.copy()
    df["zone"] = df["pk_km"].apply(lambda x: zone_label(x, width))
    rows = []
    for name, group in df.groupby("zone"):
        pmin = int(group["pk_km"].min() // width) * width
        rows.append({
            "sector_id": name.replace(" ", "_"), "name": name,
            "pk_start": float(pmin), "pk_end": float(pmin + width),
            "latitude": float(group["latitude"].mean()), "longitude": float(group["longitude"].mean()),
            "communes": ", ".join(sorted(group["commune_name"].dropna().astype(str).unique())),
            "susceptibility": 0.40, "slope_score": 0.40, "clay_score": 0.40,
            "drainage_score": 0.40, "history_score": 0.20,
        })
    return pd.DataFrame(rows).sort_values("pk_start")


def load_uploaded_sectors(upload, fallback: pd.DataFrame) -> pd.DataFrame:
    if upload is None:
        return fallback
    try:
        df = pd.read_csv(upload, sep=None, engine="python")
        required = {"sector_id", "name", "pk_start", "pk_end"}
        if not required.issubset(df.columns):
            st.sidebar.error("CSV secteurs : colonnes obligatoires sector_id, name, pk_start, pk_end.")
            return fallback
        for col in ["pk_start", "pk_end", "susceptibility", "slope_score", "clay_score", "drainage_score", "history_score"]:
            if col not in df: df[col] = 0.4
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.4).clip(0, 1)
        if "latitude" not in df: df["latitude"] = np.nan
        if "longitude" not in df: df["longitude"] = np.nan
        if "communes" not in df: df["communes"] = ""
        return df
    except Exception as exc:
        st.sidebar.error(f"CSV secteurs illisible : {exc}")
        return fallback

# =============================================================================
# PLUVIOMETRIE HISTORIQUE ET PREVISIONS
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_daily_rain(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    payload = request_json(OPEN_METEO_ARCHIVE, params={
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "precipitation_sum", "timezone": "Europe/Paris",
    })
    daily = payload.get("daily", {})
    df = pd.DataFrame({"date": daily.get("time", []), "rain_mm": daily.get("precipitation_sum", [])})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce")
    return df.dropna(subset=["date", "rain_mm"]).sort_values("date")


@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast(lat: float, lon: float) -> dict:
    return request_json(OPEN_METEO_FORECAST, params={
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "hourly": "precipitation,soil_moisture_0_to_7cm,wind_gusts_10m",
        "daily": "precipitation_sum,precipitation_probability_max,wind_gusts_10m_max,weather_code",
        "forecast_days": 7, "timezone": "Europe/Paris",
    })


def rain_features(rain: pd.DataFrame) -> pd.DataFrame:
    df = rain.copy().sort_values("date")
    for days in [3, 7, 15, 30]:
        df[f"rain_{days}d"] = df["rain_mm"].rolling(days, min_periods=1).sum()
    df["wet_days_7d"] = (df["rain_mm"] > 1).rolling(7, min_periods=1).sum()
    return df


def forecast_summary(payload: dict) -> dict:
    daily = payload.get("daily", {})
    hourly = payload.get("hourly", {})
    rain_h = pd.to_numeric(pd.Series(hourly.get("precipitation", [])), errors="coerce").fillna(0)
    soil = pd.to_numeric(pd.Series(hourly.get("soil_moisture_0_to_7cm", [])), errors="coerce").dropna()
    gust = pd.to_numeric(pd.Series(hourly.get("wind_gusts_10m", [])), errors="coerce").fillna(0)
    rain_d = pd.to_numeric(pd.Series(daily.get("precipitation_sum", [])), errors="coerce").fillna(0)
    return {
        "forecast_6h": float(rain_h.iloc[:6].sum()), "forecast_24h": float(rain_h.iloc[:24].sum()),
        "forecast_72h": float(rain_h.iloc[:72].sum()), "forecast_7d": float(rain_d.sum()),
        "max_daily": float(rain_d.max()) if len(rain_d) else 0,
        "soil": float(soil.iloc[0]) if len(soil) else np.nan,
        "gust": float(gust.max()) if len(gust) else 0,
        "daily": daily,
    }

# =============================================================================
# PIEZOMETRES TELETRANSMIS UNIQUEMENT
# =============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def load_station_referential() -> pd.DataFrame:
    frames = []
    for dep in DEPARTEMENTS:
        try:
            data = request_json(HUBEAU_STATIONS, params={"code_departement": dep, "size": 20000}).get("data", [])
            if data: frames.append(pd.DataFrame(data))
        except Exception:
            continue
    if not frames: return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("code_bss") if "code_bss" in df else df


@st.cache_data(ttl=900, show_spinner=False)
def load_piezo_realtime(code_bss: str, days=8) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    params = {"code_bss": code_bss, "date_debut_mesure": start.strftime("%Y-%m-%d"), "size": 20000}
    data = request_json(HUBEAU_CHRONIQUES_TR, params=params, timeout=(5, 30)).get("data", [])
    df = pd.DataFrame(data)
    if df.empty: return df
    date_col = next((c for c in ["date_mesure", "date_mesure_utc", "date"] if c in df), None)
    level_col = next((c for c in ["niveau_nappe_eau", "niveau_eau_ngf", "niveau"] if c in df), None)
    depth_col = next((c for c in ["profondeur_nappe", "profondeur_nappe_eau"] if c in df), None)
    if not date_col: return pd.DataFrame()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["niveau_ngf"] = pd.to_numeric(df[level_col], errors="coerce") if level_col else np.nan
    df["profondeur_m"] = pd.to_numeric(df[depth_col], errors="coerce") if depth_col else np.nan
    return df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")


@st.cache_data(ttl=3600, show_spinner=False)
def load_piezo_history(code_bss: str, start: date, end: date) -> pd.DataFrame:
    data = request_json(HUBEAU_CHRONIQUES, params={
        "code_bss": code_bss, "date_debut_mesure": start.isoformat(),
        "date_fin_mesure": end.isoformat(), "size": 20000,
    }, timeout=(5, 45)).get("data", [])
    df = pd.DataFrame(data)
    if df.empty: return df
    df["date"] = pd.to_datetime(df.get("date_mesure"), errors="coerce")
    df["niveau_ngf"] = pd.to_numeric(df.get("niveau_nappe_eau"), errors="coerce")
    return df.dropna(subset=["date", "niveau_ngf"]).sort_values("date").drop_duplicates("date")


def realtime_station_status(df: pd.DataFrame) -> dict:
    if df.empty: return {"status": "SANS_EMISSION"}
    last = df.iloc[-1]
    age_h = (pd.Timestamp.now(tz="UTC") - last["date"]).total_seconds()/3600
    status = "ACTIF" if age_h <= 48 else "RETARD" if age_h <= 168 else "HORS_LIGNE"
    series = df.dropna(subset=["niveau_ngf"])
    current = float(series["niveau_ngf"].iloc[-1]) if not series.empty else np.nan
    def delta(hours):
        if series.empty: return np.nan
        target = series["date"].iloc[-1] - pd.Timedelta(hours=hours)
        past = series[series["date"] <= target]
        return current - float(past["niveau_ngf"].iloc[-1]) if not past.empty else np.nan
    return {"status": status, "last_date": last["date"], "age_h": age_h, "level": current,
            "delta_6h": delta(6), "delta_24h": delta(24), "delta_7d": delta(168)}


def discover_active_piezometers(stations: pd.DataFrame, polyline, radius_km: float, max_candidates=80) -> pd.DataFrame:
    if stations.empty: return pd.DataFrame()
    lat_col = next((c for c in ["latitude", "y"] if c in stations), None)
    lon_col = next((c for c in ["longitude", "x"] if c in stations), None)
    if not lat_col or not lon_col: return pd.DataFrame()
    work = stations.copy()
    work["latitude"] = pd.to_numeric(work[lat_col], errors="coerce")
    work["longitude"] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.dropna(subset=["latitude", "longitude", "code_bss"])
    projections = work.apply(lambda r: pk_and_distance(r["latitude"], r["longitude"], polyline), axis=1)
    work[["pk_km", "distance_km"]] = pd.DataFrame(projections.tolist(), index=work.index)
    work = work[work["distance_km"] <= radius_km].sort_values("distance_km").head(max_candidates)
    rows = []
    progress = st.progress(0, text="Vérification des émissions piézométriques...")
    for i, (_, station) in enumerate(work.iterrows(), start=1):
        try:
            realtime = load_piezo_realtime(str(station["code_bss"]))
            status = realtime_station_status(realtime)
            if status.get("status") in ["ACTIF", "RETARD"]:
                row = station.to_dict(); row.update(status); rows.append(row)
        except Exception:
            pass
        progress.progress(i/max(1, len(work)), text=f"Piézomètres vérifiés : {i}/{len(work)}")
    progress.empty()
    return pd.DataFrame(rows)

# =============================================================================
# VIGILANCE METEO ET VIGICRUES
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_mf_vigilance() -> tuple[list, bool]:
    try:
        query = " OR ".join(f"domain_id:{d}" for d in DEPARTEMENTS)
        records = request_json(MF_VIGILANCE_URL, params={"dataset": MF_VIGILANCE_DATASET, "q": query, "rows": 100}).get("records", [])
        alerts = []
        for record in records:
            fields = record.get("fields", {})
            color = normalize_text(fields.get("color"))
            level = {"vert":"VERT", "jaune":"JAUNE", "orange":"ORANGE", "rouge":"ROUGE"}.get(color)
            if level and level != "VERT":
                alerts.append({"dep": str(fields.get("domain_id", "")), "level": level,
                               "phenomenon": fields.get("phenomenon", ""), "echeance": fields.get("echeance", "")})
        return alerts, True
    except Exception:
        return [], False


@st.cache_data(ttl=1800, show_spinner=False)
def load_vigicrues() -> tuple[list, bool]:
    headers = {"Accept": "application/json", "User-Agent": "LGV-SEA-Monitoring/2.0"}
    try:
        root = request_json(VIGICRUES_URL, headers=headers)
    except Exception:
        return [], False
    territories = root.get("ListEntVigiCru", [])
    alerts = []
    for territory in territories if isinstance(territories, list) else []:
        code = territory.get("CdEntVigiCru")
        if not code: continue
        try:
            payload = request_json(VIGICRUES_URL, params={"CdEntVigiCru": code, "TypEntVigiCru": territory.get("TypEntVigiCru", "5")}, headers=headers)
        except Exception:
            continue
        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                name = next((item.get(k) for k in ["LbEntVigiCru", "LibEntVigiCru", "LibTroncon", "NomTroncon", "NomCoursDeau", "Nom"] if item.get(k)), None)
                raw = next((item.get(k) for k in ["NivVigiCru", "NiveauVigilance", "CdCouleur", "Couleur"] if item.get(k) is not None), None)
                if name and any(r in normalize_text(name) for r in RIVER_NAMES):
                    norm = normalize_text(raw)
                    level = {"1":"VERT", "2":"JAUNE", "3":"ORANGE", "4":"ROUGE", "vert":"VERT", "jaune":"JAUNE", "orange":"ORANGE", "rouge":"ROUGE"}.get(norm)
                    if level: alerts.append({"name": str(name), "level": level})
                stack.extend(item.values())
            elif isinstance(item, list): stack.extend(item)
    unique = {(a["name"], a["level"]): a for a in alerts}
    return sorted(unique.values(), key=lambda x: -LEVEL_RANK[x["level"]]), True

# =============================================================================
# MODELE EXPLICABLE DE RISQUE
# =============================================================================
def scale(value, low, high) -> float:
    if pd.isna(value): return np.nan
    if high <= low: return 0
    return float(np.clip((value-low)/(high-low), 0, 1))


def static_sector_score(sector: pd.Series) -> float:
    values = [safe_float(sector.get(c), .4) for c in ["susceptibility", "slope_score", "clay_score", "drainage_score", "history_score"]]
    weights = [.30, .20, .20, .15, .15]
    return sum(v*w for v, w in zip(values, weights))


def risk_calculation(rain_row: pd.Series, sector: pd.Series, piezo=None, forecast=None,
                     mf_level="VERT", vigicrue_level="VERT", historical=False) -> dict:
    components, missing, factors = {}, [], []
    components["Pluie journalière"] = scale(rain_row.get("rain_mm"), 5, 60)
    components["Pluie 3 jours"] = scale(rain_row.get("rain_3d"), 15, 100)
    components["Pluie 7 jours"] = scale(rain_row.get("rain_7d"), 25, 150)
    components["Pluie 30 jours"] = scale(rain_row.get("rain_30d"), 60, 300)
    components["Sensibilité secteur"] = static_sector_score(sector)
    if piezo and not pd.isna(piezo.get("percentile", np.nan)):
        components["Niveau piézométrique"] = scale(piezo["percentile"], .60, .99)
        components["Tendance piézométrique"] = scale(piezo.get("delta_24h", np.nan), .02, .50)
    else:
        components["Niveau piézométrique"] = np.nan; components["Tendance piézométrique"] = np.nan
        missing.append("piézométrie")
    if not historical and forecast:
        components["Prévision 72 h"] = scale(forecast.get("forecast_72h", 0), 10, 100)
        components["Humidité du sol"] = scale(forecast.get("soil", np.nan), .20, .50)
        components["Vigilance météo"] = LEVEL_RANK.get(mf_level, 0)/3
        components["Vigicrues"] = LEVEL_RANK.get(vigicrue_level, 0)/3
    else:
        components["Prévision 72 h"] = np.nan; components["Humidité du sol"] = np.nan
        components["Vigilance météo"] = np.nan; components["Vigicrues"] = np.nan
        if historical: missing.append("archives vigilance/prévisions")
    weights = {"Pluie journalière":.08, "Pluie 3 jours":.12, "Pluie 7 jours":.15, "Pluie 30 jours":.08,
               "Sensibilité secteur":.18, "Niveau piézométrique":.15, "Tendance piézométrique":.10,
               "Prévision 72 h":.07, "Humidité du sol":.04, "Vigilance météo":.02, "Vigicrues":.01}
    valid = {k:v for k,v in components.items() if not pd.isna(v)}
    valid_weight = sum(weights[k] for k in valid)
    score = 100*sum(valid[k]*weights[k] for k in valid)/valid_weight if valid_weight else np.nan
    confidence = round(100*valid_weight/sum(weights.values()))
    for name, value in sorted(valid.items(), key=lambda x:x[1], reverse=True)[:4]:
        if value >= .5: factors.append(name)
    return {"score": round(score, 1), "level": level_from_score(score), "confidence": confidence,
            "factors": factors, "missing": missing, "components": components}


def historical_piezo_context(history: pd.DataFrame, target: pd.Timestamp) -> dict | None:
    if history.empty: return None
    past = history[history["date"] <= target].copy()
    if past.empty: return None
    current = float(past.iloc[-1]["niveau_ngf"])
    reference = history[history["date"].dt.month == target.month]["niveau_ngf"].dropna()
    if len(reference) < 20: reference = history["niveau_ngf"].dropna()
    percentile = float((reference <= current).mean()) if len(reference) else np.nan
    before = past[past["date"] <= target-pd.Timedelta(days=1)]
    delta = current-float(before.iloc[-1]["niveau_ngf"]) if not before.empty else np.nan
    return {"level": current, "percentile": percentile, "delta_24h": delta}

# =============================================================================
# APPLICATION
# =============================================================================
st.title("⚠️ LGV SEA - Surveillance hydrométéorologique et risque de glissement")
st.caption("Indice explicable d'aide à la surveillance. Ce résultat n'est ni une expertise géotechnique ni une garantie de survenue ou d'absence de glissement.")

try:
    snapshot = load_snapshot()
    lgv_polyline = build_lgv_polyline(snapshot.get("lgv_lines"))
    communes_df = sectors_from_snapshot(snapshot)
except Exception as exc:
    st.error(f"Chargement LGV impossible : {exc}"); st.stop()
if not lgv_polyline or communes_df.empty:
    st.error("Le snapshot ne contient pas le tracé ou les communes nécessaires."); st.stop()

auto_sectors = automatic_glissement_sectors(communes_df)
with st.sidebar:
    st.header("Configuration")
    sector_file = st.file_uploader("Référentiel secteurs sensibles CSV", type=["csv"])
    st.caption("Facultatif. Sans fichier, zones automatiques de 10 km avec sensibilité neutre.")
sectors_df = load_uploaded_sectors(sector_file, auto_sectors)
# Compléter les coordonnées depuis les points LGV compris dans chaque secteur.
for idx, s in sectors_df.iterrows():
    if pd.isna(s.get("latitude")) or pd.isna(s.get("longitude")):
        matching = communes_df[(communes_df["pk_km"] >= s["pk_start"]) & (communes_df["pk_km"] <= s["pk_end"])]
        if not matching.empty:
            sectors_df.at[idx, "latitude"] = matching["latitude"].mean(); sectors_df.at[idx, "longitude"] = matching["longitude"].mean()
sectors_df = sectors_df.dropna(subset=["latitude", "longitude"])

with st.sidebar:
    sector_name = st.selectbox("Secteur surveillé", sectors_df["name"].tolist())
    radius = st.slider("Rayon piézomètres", 1, 20, 8, format="%d km")
    max_candidates = st.slider("Stations candidates à tester", 10, 120, 50, 10)
    history_start = st.date_input("Début historique", date(2021, 1, 1), max_value=date.today())
    history_end = st.date_input("Fin historique", date.today()-timedelta(days=1), min_value=history_start, max_value=date.today())
    mode = st.radio("Analyse temporelle", ["Un jour donné", "Une période"])
    if mode == "Un jour donné":
        selected_day = st.date_input("Jour analysé", min(history_end, date.today()-timedelta(days=1)), min_value=history_start, max_value=history_end)
        selected_period = (selected_day, selected_day)
    else:
        selected_period = st.date_input("Période de risque", value=(max(history_start, history_end-timedelta(days=90)), history_end), min_value=history_start, max_value=history_end)
        if not isinstance(selected_period, (tuple, list)) or len(selected_period) != 2:
            selected_period = (history_start, history_end)
    if st.button("🔄 Vider le cache et actualiser"):
        st.cache_data.clear(); st.rerun()

sector = sectors_df[sectors_df["name"] == sector_name].iloc[0]
sector_communes = communes_df[(communes_df["pk_km"] >= sector["pk_start"]) & (communes_df["pk_km"] <= sector["pk_end"])]
lat_c, lon_c = float(sector["latitude"]), float(sector["longitude"])

# Sources temps réel
mf_alerts, mf_ok = load_mf_vigilance()
vc_alerts, vc_ok = load_vigicrues()
mf_worst = max([a["level"] for a in mf_alerts], key=LEVEL_RANK.get) if mf_alerts else "VERT"
vc_worst = max([a["level"] for a in vc_alerts], key=LEVEL_RANK.get) if vc_alerts else "VERT"
try:
    forecast = forecast_summary(load_forecast(lat_c, lon_c)); forecast_ok = True
except Exception:
    forecast, forecast_ok = {}, False

# Piézomètres actifs
st.subheader(f"📍 {sector_name} | PK {sector['pk_start']:.1f} à {sector['pk_end']:.1f}")
with st.spinner("Recherche des piézomètres qui émettent réellement..."):
    referential = load_station_referential()
    active_piezos = discover_active_piezometers(referential, lgv_polyline, radius, max_candidates)
if not active_piezos.empty:
    active_piezos = active_piezos[(active_piezos["pk_km"] >= sector["pk_start"]-10) & (active_piezos["pk_km"] <= sector["pk_end"]+10)].copy()

# Pluie historique de chaque commune
commune_coords = sector_communes.groupby("commune_name")[["latitude", "longitude"]].mean().reset_index()
rain_frames = []
with st.spinner("Chargement de la pluie journalière historique des communes..."):
    for _, commune in commune_coords.iterrows():
        try:
            df = load_daily_rain(commune["latitude"], commune["longitude"], history_start, history_end)
            df["commune"] = commune["commune_name"]; rain_frames.append(df)
        except Exception:
            pass
all_rain = pd.concat(rain_frames, ignore_index=True) if rain_frames else pd.DataFrame()
if all_rain.empty:
    st.error("Pluviométrie historique indisponible pour ce secteur."); st.stop()
zone_rain = all_rain.groupby("date", as_index=False)["rain_mm"].max()
zone_rain = rain_features(zone_rain)

# Station choisie et historique piézo
piezo_history = pd.DataFrame(); piezo_choice = None
if not active_piezos.empty:
    active_piezos["label"] = active_piezos.apply(lambda r: f"{r.get('nom_commune','Station')} | {r['code_bss']} | PK {r['pk_km']:.1f} | {r['status']}", axis=1)
    piezo_choice = st.selectbox("Piézomètre télétransmis utilisé pour le secteur", active_piezos.sort_values("distance_km")["label"].tolist())
    piezo_station = active_piezos[active_piezos["label"] == piezo_choice].iloc[0]
    try:
        piezo_history = load_piezo_history(str(piezo_station["code_bss"]), history_start, history_end)
    except Exception:
        piezo_history = pd.DataFrame()

# Risque courant
today_rain = zone_rain.iloc[-1]
current_piezo = None
if not active_piezos.empty:
    ps = active_piezos.iloc[0]
    try:
        ref_hist = load_piezo_history(str(ps["code_bss"]), date.today()-timedelta(days=365*8), date.today())
        current_piezo = historical_piezo_context(ref_hist, pd.Timestamp.now())
        if current_piezo: current_piezo["delta_24h"] = ps.get("delta_24h", current_piezo.get("delta_24h"))
    except Exception: pass
current_risk = risk_calculation(today_rain, sector, current_piezo, forecast if forecast_ok else None, mf_worst, vc_worst)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Risque actuel", current_risk["level"])
c2.metric("Indice", f"{current_risk['score']:.0f}/100")
c3.metric("Confiance", f"{current_risk['confidence']} %")
c4.metric("Pluie 7 j", f"{today_rain['rain_7d']:.1f} mm")
c5.metric("Prévision 72 h", f"{forecast.get('forecast_72h', np.nan):.1f} mm" if forecast_ok else "Non vérifiée")
st.markdown(f"**Action proposée :** {LEVEL_ACTION[current_risk['level']]}")
if current_risk["factors"]: st.info("Facteurs dominants : " + ", ".join(current_risk["factors"]))
if current_risk["missing"]: st.warning("Confiance réduite, données absentes : " + ", ".join(current_risk["missing"]))

# Alertes et prévisions
st.subheader("🌦️ Alertes météo, prévisions et Vigicrues")
a,b,c = st.columns(3)
with a:
    st.markdown("**Vigilance Météo-France**")
    if not mf_ok: st.warning("Non vérifiée")
    elif not mf_alerts: st.success("Aucune vigilance non verte détectée")
    else:
        for x in mf_alerts: st.write(f"{x['level']} | Dép. {x['dep']} | {x['phenomenon']} | {x['echeance']}")
with b:
    st.markdown("**Prévision du secteur**")
    if not forecast_ok: st.warning("Non vérifiée")
    else:
        st.write(f"6 h : {forecast['forecast_6h']:.1f} mm")
        st.write(f"24 h : {forecast['forecast_24h']:.1f} mm")
        st.write(f"72 h : {forecast['forecast_72h']:.1f} mm")
        st.write(f"7 jours : {forecast['forecast_7d']:.1f} mm")
with c:
    st.markdown("**Vigicrues**")
    if not vc_ok: st.warning("Non vérifié")
    elif not [x for x in vc_alerts if x['level'] != 'VERT']: st.success("Aucune vigilance non verte détectée")
    else:
        for x in vc_alerts[:12]: st.write(f"{x['level']} | {x['name']}")

# Piézomètres qui émettent
st.subheader("💧 Piézomètres télétransmis proches")
if active_piezos.empty:
    st.warning("Aucun piézomètre avec émission récente n'a été trouvé dans le rayon et parmi les candidats testés.")
else:
    cols = ["code_bss", "nom_commune", "pk_km", "distance_km", "status", "last_date", "level", "delta_24h", "delta_7d"]
    display = active_piezos[[x for x in cols if x in active_piezos]].copy()
    display = display.rename(columns={"code_bss":"Code BSS", "nom_commune":"Commune", "pk_km":"PK (km)", "distance_km":"Distance LGV (km)", "status":"Emission", "last_date":"Dernière mesure", "level":"Niveau NGF", "delta_24h":"Variation 24 h", "delta_7d":"Variation 7 j"})
    st.dataframe(display.sort_values("PK (km)"), use_container_width=True, hide_index=True)

# Pluie historique par commune
st.subheader("🌧️ Pluviométrie journalière historique par commune")
selected_communes = st.multiselect("Communes affichées", sorted(all_rain["commune"].unique()), default=sorted(all_rain["commune"].unique())[:6])
rain_view = all_rain[all_rain["commune"].isin(selected_communes)]
fig_rain = go.Figure()
for commune, group in rain_view.groupby("commune"):
    fig_rain.add_scatter(x=group["date"], y=group["rain_mm"], mode="lines", name=commune)
fig_rain.update_layout(height=380, yaxis_title="Pluie journalière (mm)", hovermode="x unified", margin=dict(l=20,r=20,t=20,b=20))
st.plotly_chart(fig_rain, use_container_width=True, config={"displayModeBar":False})
maxima = rain_view.loc[rain_view.groupby("commune")["rain_mm"].idxmax(), ["commune","date","rain_mm"]] if not rain_view.empty else pd.DataFrame()
if not maxima.empty:
    st.dataframe(maxima.rename(columns={"commune":"Commune", "date":"Date du maximum", "rain_mm":"Maximum journalier (mm)"}), use_container_width=True, hide_index=True)

# Reconstitution historique du risque
st.subheader("🕒 Niveau de risque à une date ou sur une période passée")
period_start, period_end = selected_period
risk_rows = []
period_rain = zone_rain[(zone_rain["date"].dt.date >= period_start) & (zone_rain["date"].dt.date <= period_end)]
for _, rr in period_rain.iterrows():
    pctx = historical_piezo_context(piezo_history, rr["date"]) if not piezo_history.empty else None
    result = risk_calculation(rr, sector, pctx, historical=True)
    risk_rows.append({"date":rr["date"], "score":result["score"], "niveau":result["level"], "confiance":result["confidence"],
                      "pluie_jour":rr["rain_mm"], "pluie_3j":rr["rain_3d"], "pluie_7j":rr["rain_7d"], "pluie_30j":rr["rain_30d"],
                      "facteurs":", ".join(result["factors"])})
risk_history = pd.DataFrame(risk_rows)
if risk_history.empty:
    st.info("Aucune donnée sur la période sélectionnée.")
else:
    fig_risk = go.Figure()
    fig_risk.add_scatter(x=risk_history["date"], y=risk_history["score"], mode="lines+markers", name="Indice de risque", line=dict(color="#dc2626", width=2))
    for y,label,color in [(25,"Jaune","#eab308"),(50,"Orange","#ea580c"),(75,"Rouge","#dc2626")]:
        fig_risk.add_hline(y=y, line_dash="dash", line_color=color, annotation_text=label)
    fig_risk.update_layout(height=420, yaxis=dict(title="Indice / 100", range=[0,100]), hovermode="x unified", margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar":False})
    worst = risk_history.loc[risk_history["score"].idxmax()]
    x1,x2,x3,x4 = st.columns(4)
    x1.metric("Risque maximal", worst["niveau"])
    x2.metric("Indice maximal", f"{worst['score']:.1f}/100")
    x3.metric("Date du maximum", worst["date"].strftime("%d/%m/%Y"))
    x4.metric("Confiance historique", f"{worst['confiance']} %")
    st.caption("Le risque historique est recalculé avec les pluies, la sensibilité du secteur et la piézométrie disponible. Les anciennes prévisions, vigilances Météo-France et Vigicrues ne sont pas réinventées : elles sont exclues et la confiance est réduite.")
    st.dataframe(risk_history.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Exporter l'historique du risque", risk_history.to_csv(index=False).encode("utf-8-sig"), f"risque_{sector['sector_id']}_{period_start}_{period_end}.csv", "text/csv")

# Historique piézométrique
if not piezo_history.empty:
    st.subheader("📈 Historique piézométrique de la station sélectionnée")
    fig_p = go.Figure(go.Scatter(x=piezo_history["date"], y=piezo_history["niveau_ngf"], mode="lines", name="Niveau NGF"))
    max_idx = piezo_history["niveau_ngf"].idxmax()
    fig_p.add_scatter(x=[piezo_history.loc[max_idx,"date"]], y=[piezo_history.loc[max_idx,"niveau_ngf"]], mode="markers+text", text=["Maximum"], textposition="top center", marker=dict(color="#dc2626", size=11))
    fig_p.update_layout(height=380, yaxis_title="m NGF", hovermode="x unified", margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar":False})

# Carte satellite
st.subheader("🗺️ Carte satellite de surveillance")
m = folium.Map(location=[lat_c,lon_c], zoom_start=11, tiles=None, control_scale=True)
folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri, Maxar, Earthstar Geographics, GIS User Community", name="Satellite Esri", max_zoom=19).add_to(m)
for segment in snapshot.get("lgv_lines") or []:
    pts = [[p["lat"],p["lon"]] for p in segment if isinstance(p,dict) and "lat" in p and "lon" in p]
    if pts: folium.PolyLine(pts,color="#ef4444",weight=3,opacity=.9,tooltip="LGV SEA").add_to(m)
folium.Circle([lat_c,lon_c], radius=max(500,(sector["pk_end"]-sector["pk_start"])*400), color=LEVEL_COLOR[current_risk["level"]], fill=True, fill_opacity=.18,
              tooltip=f"{sector_name} | {current_risk['level']} | {current_risk['score']:.0f}/100").add_to(m)
if not active_piezos.empty:
    for _,p in active_piezos.iterrows():
        folium.CircleMarker([p["latitude"],p["longitude"]],radius=6,color="#38bdf8",fill=True,fill_opacity=.9,
            tooltip=f"Piézo {p['code_bss']} | PK {p['pk_km']:.1f} | {p['status']}",
            popup=folium.Popup(f"<b>{p.get('nom_commune','Piézomètre')}</b><br>Code BSS : {p['code_bss']}<br>PK : {p['pk_km']:.1f}<br>Distance LGV : {p['distance_km']*1000:.0f} m<br>Emission : {p['status']}",max_width=320)).add_to(m)
folium.LayerControl().add_to(m)
st_folium(m,use_container_width=True,height=520,returned_objects=[])

st.caption("Sources : Open-Meteo, Hub'Eau/ADES, Météo-France open data, Vigicrues et référentiel LGV. Les pondérations sont configurables et doivent être calibrées avec les événements et seuils métier réels.")
