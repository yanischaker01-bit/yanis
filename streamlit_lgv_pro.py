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
MF_URL = "https://public.opendatasoft.com/api/records/1.0/search/"
MF_DATASET = "weatherref-france-vigilance-meteo-departement"

DEPARTMENTS = ["37", "86", "79", "16", "17", "33"]
RIVERS = [
    "vienne", "clain", "charente", "boutonne", "seugne", "touvre",
    "dronne", "isle", "dordogne", "garonne", "thouet", "sevre",
    "indre", "cher", "creuse", "ciron", "jalles",
]
THUNDERSTORM_CODES = {95, 96, 99}
SECTOR_WIDTH_KM = 10

LEVEL_RANK = {"INDETERMINE": -1, "VERT": 0, "JAUNE": 1, "ORANGE": 2, "ROUGE": 3}
LEVEL_COLOR = {
    "INDETERMINE": "#64748b", "VERT": "#16a34a", "JAUNE": "#eab308",
    "ORANGE": "#ea580c", "ROUGE": "#dc2626",
}
LEVEL_ACTION = {
    "INDETERMINE": "Données insuffisantes, contrôle manuel nécessaire",
    "VERT": "Surveillance courante",
    "JAUNE": "Surveillance renforcée et contrôle de la fraîcheur des données",
    "ORANGE": "Inspection ciblée et contrôle du drainage",
    "ROUGE": "Contrôle prioritaire et application des consignes métier",
}

st.set_page_config(
    page_title="LGV SEA - Surveillance optimisée",
    page_icon="⚠️",
    layout="wide",
)

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


def clean_department(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().replace(".0", "")
    digits = "".join(c for c in text if c.isdigit())
    return digits.zfill(2) if digits else ""


def get_json(url: str, params=None, timeout=(5, 30), headers=None) -> dict:
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
        kx = 111.320 * math.cos(math.radians(lat_mid))
        ky = 111.320
        x1, y1 = lon1 * kx, lat1 * ky
        x2, y2 = lon2 * kx, lat2 * ky
        xp, yp = lon * kx, lat * ky
        dx, dy = x2 - x1, y2 - y1
        den = dx * dx + dy * dy
        t = 0 if den == 0 else max(0, min(1, ((xp - x1) * dx + (yp - y1) * dy) / den))
        cx, cy = x1 + t * dx, y1 + t * dy
        d2 = (xp - cx) ** 2 + (yp - cy) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_pk = pk1 + t * (pk2 - pk1)

    return best_pk, math.sqrt(best_d2) if best_d2 is not None else None


def risk_level(score: float) -> str:
    if pd.isna(score):
        return "INDETERMINE"
    if score >= 75:
        return "ROUGE"
    if score >= 50:
        return "ORANGE"
    if score >= 25:
        return "JAUNE"
    return "VERT"


def scale(value, low, high):
    if pd.isna(value):
        return np.nan
    return float(np.clip((value - low) / (high - low), 0, 1))


def highest_level(items, default="VERT") -> str:
    levels = [x.get("level", default) for x in items or []]
    return max(levels, key=lambda x: LEVEL_RANK.get(x, -1)) if levels else default


def wmo_label(code: Any) -> str:
    code = int(code) if not pd.isna(code) else -1
    labels = {
        0: "Ciel dégagé", 1: "Peu nuageux", 2: "Partiellement nuageux",
        3: "Couvert", 45: "Brouillard", 48: "Brouillard givrant",
        51: "Bruine faible", 53: "Bruine", 55: "Bruine forte",
        61: "Pluie faible", 63: "Pluie", 65: "Pluie forte",
        80: "Averses faibles", 81: "Averses", 82: "Averses fortes",
        95: "Orage", 96: "Orage avec grêle possible", 99: "Orage fort avec grêle possible",
    }
    return labels.get(code, f"Code météo {code}")

# =============================================================================
# SNAPSHOT ET SECTEURS DE 10 KM
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_snapshot():
    return get_json(SNAPSHOT_URL)


def load_points(snapshot):
    payload = snapshot.get("sectors", {})
    df = pd.DataFrame(payload.get("sectors", []) if isinstance(payload, dict) else [])

    for col in ["latitude", "longitude", "pk_km", "ai_pred_probability", "ai_soil_fragility", "score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "commune_name" not in df.columns:
        df["commune_name"] = "Commune inconnue"

    department_candidates = [
        "department_code", "departement_code", "code_departement",
        "department", "departement", "dep",
    ]
    dep_col = next((c for c in department_candidates if c in df.columns), None)
    df["department_code"] = df[dep_col].map(clean_department) if dep_col else ""

    return df.dropna(subset=["latitude", "longitude", "pk_km"]).copy()


def make_sectors(points, width=SECTOR_WIDTH_KM):
    work = points.copy()
    work["pk_start"] = (np.floor(work["pk_km"] / width).astype(int) * width)
    rows = []

    for pk_start, group in work.groupby("pk_start"):
        ai = pd.to_numeric(group.get("ai_pred_probability"), errors="coerce").dropna()
        soil = pd.to_numeric(group.get("ai_soil_fragility"), errors="coerce").dropna()
        measured = pd.to_numeric(group.get("score"), errors="coerce").dropna()

        susceptibility = float(ai.max()) if not ai.empty else 0.40
        fragility = float(soil.mean()) if not soil.empty else 0.40
        signal = min(1.0, float(measured.max()) / 4) if not measured.empty else 0.20
        quick_score = round(100 * (0.55 * susceptibility + 0.30 * fragility + 0.15 * signal), 1)

        deps = sorted({clean_department(v) for v in group["department_code"] if clean_department(v)})
        rows.append({
            "sector_id": f"PK_{pk_start:03d}_{pk_start + width:03d}",
            "name": f"PK {pk_start:03d}-{pk_start + width:03d}",
            "pk_start": float(pk_start),
            "pk_end": float(pk_start + width),
            "latitude": float(group["latitude"].mean()),
            "longitude": float(group["longitude"].mean()),
            "communes": ", ".join(sorted(group["commune_name"].astype(str).unique())),
            "departments": ", ".join(deps),
            "static_score": quick_score,
            "static_level": risk_level(quick_score),
            "susceptibility": susceptibility,
            "soil_fragility": fragility,
            "signal": signal,
        })

    return pd.DataFrame(rows).sort_values("pk_start").reset_index(drop=True)

# =============================================================================
# METEO, PLUIE ET ORAGES COMMUNAUX
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast(lat, lon):
    return get_json(FORECAST_URL, params={
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "hourly": "precipitation,precipitation_probability,weather_code,soil_moisture_0_to_7cm,wind_gusts_10m",
        "daily": "weather_code,precipitation_sum,precipitation_probability_max,wind_gusts_10m_max",
        "forecast_days": 7,
        "timezone": "Europe/Paris",
    })


def forecast_summary(payload):
    hourly = payload.get("hourly", {})
    daily = payload.get("daily", {})
    rain_h = pd.to_numeric(pd.Series(hourly.get("precipitation", [])), errors="coerce").fillna(0)
    soil = pd.to_numeric(pd.Series(hourly.get("soil_moisture_0_to_7cm", [])), errors="coerce").dropna()
    rain_d = pd.to_numeric(pd.Series(daily.get("precipitation_sum", [])), errors="coerce").fillna(0)
    codes = pd.to_numeric(pd.Series(daily.get("weather_code", [])), errors="coerce").dropna().astype(int)
    return {
        "rain_6h": float(rain_h.iloc[:6].sum()),
        "rain_24h": float(rain_h.iloc[:24].sum()),
        "rain_72h": float(rain_h.iloc[:72].sum()),
        "rain_7d": float(rain_d.sum()),
        "soil": float(soil.iloc[0]) if not soil.empty else np.nan,
        "thunderstorm": bool(codes.isin(THUNDERSTORM_CODES).any()),
        "daily": daily,
    }


def analyse_commune_forecast(commune, department, lat, lon, pk_min, pk_max):
    payload = load_forecast(lat, lon)
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    codes = daily.get("weather_code", [])
    rain = daily.get("precipitation_sum", [])
    probability = daily.get("precipitation_probability_max", [])
    gusts = daily.get("wind_gusts_10m_max", [])
    rows = []

    for index, forecast_date in enumerate(dates):
        code = safe_float(codes[index]) if index < len(codes) else np.nan
        rows.append({
            "Commune": commune,
            "Département": department or "Non renseigné",
            "Date": pd.to_datetime(forecast_date),
            "PK min": safe_float(pk_min),
            "PK max": safe_float(pk_max),
            "Code météo": int(code) if not pd.isna(code) else np.nan,
            "Phénomène": wmo_label(code),
            "Orage": int(code) in THUNDERSTORM_CODES if not pd.isna(code) else False,
            "Probabilité pluie (%)": safe_float(probability[index]) if index < len(probability) else np.nan,
            "Pluie prévue (mm)": safe_float(rain[index]) if index < len(rain) else np.nan,
            "Rafales max (km/h)": safe_float(gusts[index]) if index < len(gusts) else np.nan,
        })
    return rows


def load_commune_forecasts(coords):
    records = list(coords.itertuples(index=False))
    if not records:
        return pd.DataFrame()

    output = []
    with ThreadPoolExecutor(max_workers=min(8, len(records))) as pool:
        jobs = {
            pool.submit(
                analyse_commune_forecast,
                r.commune_name, r.department_code, r.latitude, r.longitude, r.pk_min, r.pk_max,
            ): r.commune_name
            for r in records
        }
        for future in as_completed(jobs):
            try:
                output.extend(future.result())
            except Exception:
                pass
    return pd.DataFrame(output)


@st.cache_data(ttl=21600, show_spinner=False)
def load_daily_rain(lat, lon, start_date, end_date):
    payload = get_json(ARCHIVE_URL, params={
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "precipitation_sum",
        "timezone": "Europe/Paris",
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
# PIEZOMETRIE
# =============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def load_piezo_referential():
    frames = []
    for dep in DEPARTMENTS:
        try:
            rows = get_json(PIEZO_STATIONS_URL, params={"code_departement": dep, "size": 20000}).get("data", [])
            if rows:
                frames.append(pd.DataFrame(rows))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("code_bss") if "code_bss" in df.columns else df


@st.cache_data(ttl=900, show_spinner=False)
def load_piezo_tr(code_bss, days=8):
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = get_json(PIEZO_TR_URL, params={
        "code_bss": code_bss,
        "date_debut_mesure": start,
        "size": 20000,
    }).get("data", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    date_col = next((c for c in ["date_mesure", "date_mesure_utc", "date"] if c in df.columns), None)
    level_col = next((c for c in ["niveau_nappe_eau", "niveau_eau_ngf", "niveau"] if c in df.columns), None)
    if not date_col:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["level"] = pd.to_numeric(df[level_col], errors="coerce") if level_col else np.nan
    return df.dropna(subset=["date"]).sort_values("date")


@st.cache_data(ttl=3600, show_spinner=False)
def load_piezo_history(code_bss, start_date, end_date):
    rows = get_json(PIEZO_HISTORY_URL, params={
        "code_bss": code_bss,
        "date_debut_mesure": start_date.isoformat(),
        "date_fin_mesure": end_date.isoformat(),
        "size": 20000,
    }, timeout=(5, 45)).get("data", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df.get("date_mesure"), errors="coerce")
    df["level"] = pd.to_numeric(df.get("niveau_nappe_eau"), errors="coerce")
    return df.dropna(subset=["date", "level"]).sort_values("date").drop_duplicates("date")


def piezo_status(df):
    if df.empty:
        return None
    last = df.iloc[-1]
    age = (pd.Timestamp.now(tz="UTC") - last["date"]).total_seconds() / 3600
    status = "ACTIF" if age <= 48 else "RETARD" if age <= 168 else "HORS_LIGNE"
    valid = df.dropna(subset=["level"])
    current = float(valid.iloc[-1]["level"]) if not valid.empty else np.nan

    def delta(hours):
        if valid.empty:
            return np.nan
        before = valid[valid["date"] <= valid.iloc[-1]["date"] - pd.Timedelta(hours=hours)]
        return current - float(before.iloc[-1]["level"]) if not before.empty else np.nan

    return {
        "status": status, "last_date": last["date"], "age_h": age,
        "level": current, "delta_24h": delta(24), "delta_7d": delta(168),
    }


def active_piezometers(line, radius, limit, pk_min=None, pk_max=None):
    stations = load_piezo_referential()
    if stations.empty:
        return pd.DataFrame()

    lat_col = next((c for c in ["latitude", "y"] if c in stations.columns), None)
    lon_col = next((c for c in ["longitude", "x"] if c in stations.columns), None)
    if not lat_col or not lon_col or "code_bss" not in stations.columns:
        return pd.DataFrame()

    stations = stations.copy()
    stations["latitude"] = pd.to_numeric(stations[lat_col], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations[lon_col], errors="coerce")
    stations = stations.dropna(subset=["latitude", "longitude", "code_bss"])
    projection = stations.apply(lambda r: pk_distance(r["latitude"], r["longitude"], line), axis=1)
    stations[["pk_km", "distance_km"]] = pd.DataFrame(projection.tolist(), index=stations.index)
    stations = stations.dropna(subset=["pk_km", "distance_km"])
    stations = stations[stations["distance_km"] <= radius]

    if pk_min is not None:
        stations = stations[stations["pk_km"] >= pk_min - 10]
    if pk_max is not None:
        stations = stations[stations["pk_km"] <= pk_max + 10]

    stations = stations.sort_values("distance_km").head(limit)
    if stations.empty:
        return pd.DataFrame()

    rows = []
    with ThreadPoolExecutor(max_workers=min(10, len(stations))) as pool:
        jobs = {pool.submit(load_piezo_tr, str(r.code_bss)): r for r in stations.itertuples(index=False)}
        for future in as_completed(jobs):
            try:
                info = piezo_status(future.result())
                if info and info["status"] in ["ACTIF", "RETARD"]:
                    row = jobs[future]._asdict()
                    row.update(info)
                    rows.append(row)
            except Exception:
                pass
    return pd.DataFrame(rows)

# =============================================================================
# VIGILANCES
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_mf_alerts():
    try:
        query = " OR ".join(f"domain_id:{d}" for d in DEPARTMENTS)
        records = get_json(MF_URL, params={
            "dataset": MF_DATASET,
            "q": query,
            "rows": 200,
        }).get("records", [])

        alerts = []
        for rec in records:
            fields = rec.get("fields", {})
            dep = clean_department(fields.get("domain_id"))
            level = {
                "vert": "VERT", "jaune": "JAUNE", "orange": "ORANGE", "rouge": "ROUGE",
            }.get(norm(fields.get("color")))
            if dep in DEPARTMENTS and level and level != "VERT":
                alerts.append({
                    "level": level,
                    "dep": dep,
                    "phenomenon": fields.get("phenomenon", "Phénomène non précisé"),
                    "day": fields.get("echeance", ""),
                })
        return alerts, True
    except Exception:
        return [], False


@st.cache_data(ttl=1800, show_spinner=False)
def load_vigicrues():
    headers = {"Accept": "application/json", "User-Agent": "LGV-SEA-Monitoring/4.0"}
    try:
        root = get_json(VIGICRUES_URL, headers=headers)
    except Exception:
        return [], False

    results = []
    territories = root.get("ListEntVigiCru", [])
    for territory in territories if isinstance(territories, list) else []:
        code = territory.get("CdEntVigiCru")
        if not code:
            continue
        try:
            payload = get_json(VIGICRUES_URL, params={
                "CdEntVigiCru": code,
                "TypEntVigiCru": territory.get("TypEntVigiCru", "5"),
            }, headers=headers)
        except Exception:
            continue

        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                name = next((item.get(k) for k in [
                    "LbEntVigiCru", "LibEntVigiCru", "LibTroncon", "NomTroncon",
                    "NomCoursDeau", "Nom",
                ] if item.get(k)), None)
                raw = next((item.get(k) for k in [
                    "NivVigiCru", "NiveauVigilance", "CdCouleur", "Couleur",
                ] if item.get(k) is not None), None)
                level = {
                    "1": "VERT", "2": "JAUNE", "3": "ORANGE", "4": "ROUGE",
                    "vert": "VERT", "jaune": "JAUNE", "orange": "ORANGE", "rouge": "ROUGE",
                }.get(norm(raw))
                if name and level and any(r in norm(name) for r in RIVERS):
                    results.append({"name": str(name), "level": level})
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)

    return list({(x["name"], x["level"]): x for x in results}.values()), True

# =============================================================================
# RISQUE
# =============================================================================
def historical_piezo_context(history, target):
    if history.empty:
        return None
    target = pd.Timestamp(target)
    past = history[history["date"] <= target]
    if past.empty:
        return None
    current = float(past.iloc[-1]["level"])
    ref = history[history["date"].dt.month == target.month]["level"]
    if len(ref) < 20:
        ref = history["level"]
    percentile = float((ref <= current).mean()) if len(ref) else np.nan
    before = past[past["date"] <= target - pd.Timedelta(days=1)]
    delta = current - float(before.iloc[-1]["level"]) if not before.empty else np.nan
    return {"percentile": percentile, "delta_24h": delta}


def live_piezo_context(station_row):
    if station_row is None:
        return None
    return {
        "percentile": np.nan,
        "delta_24h": safe_float(station_row.get("delta_24h")),
    }


def calculate_risk(rain, sector, piezo=None, forecast=None, mf="VERT", vc="VERT", historical=False):
    components = {
        "Pluie jour": scale(rain.get("rain_mm"), 5, 60),
        "Pluie 3 j": scale(rain.get("rain_3d"), 15, 100),
        "Pluie 7 j": scale(rain.get("rain_7d"), 25, 150),
        "Pluie 30 j": scale(rain.get("rain_30d"), 60, 300),
        "Sensibilité": safe_float(sector.get("susceptibility"), 0.4),
        "Sol": safe_float(sector.get("soil_fragility"), 0.4),
        "Piézo niveau": scale(piezo.get("percentile"), 0.60, 0.99) if piezo else np.nan,
        "Piézo tendance": scale(piezo.get("delta_24h"), 0.02, 0.50) if piezo else np.nan,
        "Prévision 72 h": scale(forecast.get("rain_72h"), 10, 100) if forecast and not historical else np.nan,
        "Humidité sol": scale(forecast.get("soil"), 0.20, 0.50) if forecast and not historical else np.nan,
        "Vigilance météo": LEVEL_RANK.get(mf, 0) / 3 if not historical else np.nan,
        "Vigicrues": LEVEL_RANK.get(vc, 0) / 3 if not historical else np.nan,
    }
    weights = {
        "Pluie jour": 0.07, "Pluie 3 j": 0.11, "Pluie 7 j": 0.16,
        "Pluie 30 j": 0.08, "Sensibilité": 0.13, "Sol": 0.08,
        "Piézo niveau": 0.14, "Piézo tendance": 0.10,
        "Prévision 72 h": 0.07, "Humidité sol": 0.03,
        "Vigilance météo": 0.02, "Vigicrues": 0.01,
    }
    valid = {k: v for k, v in components.items() if not pd.isna(v)}
    valid_weight = sum(weights[k] for k in valid)
    score = 100 * sum(valid[k] * weights[k] for k in valid) / valid_weight if valid_weight else np.nan
    factors = [k for k, v in sorted(valid.items(), key=lambda x: x[1], reverse=True)[:4] if v >= 0.5]
    return {
        "score": round(score, 1) if not pd.isna(score) else np.nan,
        "level": risk_level(score),
        "confidence": round(100 * valid_weight),
        "factors": factors,
    }

# =============================================================================
# APPLICATION
# =============================================================================
st.title("⚠️ LGV SEA - Surveillance des risques de glissement")
st.caption(
    "Découpage fixe par secteurs de 10 km. Les données lourdes sont chargées uniquement dans le module sélectionné."
)

try:
    snapshot = load_snapshot()
    points = load_points(snapshot)
    line = build_polyline(snapshot.get("lgv_lines"))
except Exception as exc:
    st.error(f"Snapshot indisponible : {exc}")
    st.stop()

if points.empty or not line:
    st.error("Tracé LGV ou secteurs absents dans le snapshot.")
    st.stop()

sectors = make_sectors(points)

with st.sidebar:
    st.header("Pilotage")
    sector_name = st.selectbox("Secteur surveillé", ["Tous les secteurs"] + sectors["name"].tolist())
    module = st.radio("Module", [
        "Vue rapide", "Alertes et prévisions", "Pluie historique",
        "Piézomètres actifs", "Risque historique", "Carte satellite",
    ])
    radius = st.slider("Rayon piézomètres", 1, 20, 8, format="%d km")
    station_limit = st.slider("Stations à tester", 10, 60, 30, 10)
    if st.button("🔄 Actualiser"):
        st.cache_data.clear()
        st.rerun()

all_selected = sector_name == "Tous les secteurs"
selected_sectors = sectors if all_selected else sectors[sectors["name"] == sector_name]

if all_selected:
    selected_points = points.copy()
else:
    selected_sector = selected_sectors.iloc[0]
    selected_points = points[
        (points["pk_km"] >= selected_sector["pk_start"])
        & (points["pk_km"] < selected_sector["pk_end"])
    ].copy()

if selected_points.empty:
    st.error("Aucun point trouvé pour la sélection.")
    st.stop()

lat_c = float(selected_points["latitude"].mean())
lon_c = float(selected_points["longitude"].mean())

# -----------------------------------------------------------------------------
# VUE RAPIDE
# -----------------------------------------------------------------------------
if module == "Vue rapide":
    st.subheader("Vue rapide des secteurs de 10 km")
    quick = sectors[[
        "name", "pk_start", "pk_end", "communes", "departments",
        "static_score", "static_level",
    ]].rename(columns={
        "name": "Secteur", "pk_start": "PK début", "pk_end": "PK fin",
        "communes": "Communes", "departments": "Départements",
        "static_score": "Indice structurel", "static_level": "Niveau structurel",
    })
    quick = quick.sort_values(
        ["Niveau structurel", "Indice structurel"],
        key=lambda col: col.map(LEVEL_RANK) if col.name == "Niveau structurel" else col,
        ascending=False,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Secteurs", len(quick))
    c2.metric("Rouges", int((quick["Niveau structurel"] == "ROUGE").sum()))
    c3.metric("Orange", int((quick["Niveau structurel"] == "ORANGE").sum()))
    c4.metric("Jaunes", int((quick["Niveau structurel"] == "JAUNE").sum()))
    st.dataframe(quick, use_container_width=True, hide_index=True, height=560)
    st.info(
        "Cette vue présente la sensibilité structurelle du terrain. Pour la météo, les orages et les alertes en cours, ouvre le module Alertes et prévisions."
    )

# -----------------------------------------------------------------------------
# ALERTES ET PREVISIONS
# -----------------------------------------------------------------------------
elif module == "Alertes et prévisions":
    st.subheader(f"Alertes et prévisions - {sector_name}")

    with st.spinner("Chargement des vigilances et des prévisions..."):
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_mf = pool.submit(load_mf_alerts)
            f_vc = pool.submit(load_vigicrues)
            f_fc = pool.submit(load_forecast, lat_c, lon_c)
            mf_alerts, mf_ok = f_mf.result()
            vc_alerts, vc_ok = f_vc.result()
            try:
                forecast = forecast_summary(f_fc.result())
                fc_ok = True
            except Exception:
                forecast, fc_ok = {}, False

    selected_deps = {clean_department(x) for x in selected_points["department_code"] if clean_department(x)}
    relevant_mf = [x for x in mf_alerts if not selected_deps or x["dep"] in selected_deps]
    mf_level = highest_level(relevant_mf)
    vc_level = highest_level(vc_alerts)

    a, b, c = st.columns(3)
    with a:
        st.markdown("### Vigilance par département")
        if not mf_ok:
            st.warning("Vigilance non vérifiée")
        elif not relevant_mf:
            st.success("Aucune vigilance non verte sur la sélection")
        else:
            alerts_df = pd.DataFrame(relevant_mf)
            alerts_df["rank"] = alerts_df["level"].map(LEVEL_RANK)
            alerts_df = alerts_df.sort_values(["rank", "dep"], ascending=[False, True])
            for dep, group in alerts_df.groupby("dep"):
                level = group.iloc[0]["level"]
                st.markdown(f"**Département {dep} : {level}**")
                for alert in group.to_dict("records"):
                    st.write(f"• {alert['phenomenon']} | {alert['day']}")

    with b:
        st.markdown("### Prévision du secteur")
        if not fc_ok:
            st.warning("Prévisions non vérifiées")
        else:
            st.metric("Pluie 24 h", f"{forecast['rain_24h']:.1f} mm")
            st.metric("Pluie 72 h", f"{forecast['rain_72h']:.1f} mm")
            st.metric("Pluie 7 jours", f"{forecast['rain_7d']:.1f} mm")
            if forecast["thunderstorm"]:
                st.warning("Orage possible dans les 7 prochains jours")

    with c:
        st.markdown("### Vigicrues")
        if not vc_ok:
            st.warning("Vigicrues non vérifié")
        else:
            non_green = [x for x in vc_alerts if x["level"] != "VERT"]
            if not non_green:
                st.success("Aucune vigilance non verte détectée")
            else:
                for item in sorted(non_green, key=lambda z: -LEVEL_RANK[z["level"]])[:15]:
                    st.write(f"**{item['level']}** | {item['name']}")

    st.divider()
    st.markdown("### ⛈️ Prévisions d’orage par commune")
    coords = selected_points.groupby("commune_name", as_index=False).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        pk_min=("pk_km", "min"),
        pk_max=("pk_km", "max"),
        department_code=("department_code", "first"),
    )

    with st.spinner(f"Analyse de {len(coords)} commune(s)..."):
        commune_forecasts = load_commune_forecasts(coords)

    if commune_forecasts.empty:
        st.warning("Prévisions communales indisponibles.")
    else:
        thunderstorms = commune_forecasts[commune_forecasts["Orage"]].copy()
        if thunderstorms.empty:
            st.success("Aucun signal d’orage détecté sur les communes sélectionnées pour les 7 prochains jours.")
        else:
            thunderstorms["Secteur 10 km"] = thunderstorms["PK min"].apply(
                lambda pk: f"PK {int(pk // 10) * 10:03d}-{int(pk // 10) * 10 + 10:03d}"
            )
            display_cols = [
                "Date", "Département", "Commune", "Secteur 10 km", "Phénomène",
                "Probabilité pluie (%)", "Pluie prévue (mm)", "Rafales max (km/h)",
            ]
            thunderstorms = thunderstorms.sort_values(
                ["Date", "Probabilité pluie (%)", "Pluie prévue (mm)"],
                ascending=[True, False, False],
            )
            st.warning(f"{len(thunderstorms)} signal(s) journalier(s) d’orage détecté(s).")
            st.dataframe(thunderstorms[display_cols], use_container_width=True, hide_index=True)
            st.caption("L’orage par commune est une prévision localisée, pas une vigilance communale officielle.")

# -----------------------------------------------------------------------------
# PLUIE HISTORIQUE
# -----------------------------------------------------------------------------
elif module == "Pluie historique":
    st.subheader(f"Pluviométrie journalière - {sector_name}")
    start_date = st.date_input("Début", date.today() - timedelta(days=365), max_value=date.today())
    end_date = st.date_input(
        "Fin", date.today() - timedelta(days=1), min_value=start_date, max_value=date.today()
    )
    coords = selected_points.groupby("commune_name")[["latitude", "longitude"]].mean().reset_index()
    names = sorted(coords["commune_name"].tolist())
    chosen_names = st.multiselect("Communes", names, default=names[:min(6, len(names))])
    coords = coords[coords["commune_name"].isin(chosen_names)]

    frames = []
    if not coords.empty:
        with st.spinner(f"Chargement de {len(coords)} commune(s)..."):
            with ThreadPoolExecutor(max_workers=min(8, len(coords))) as pool:
                jobs = {
                    pool.submit(load_daily_rain, r.latitude, r.longitude, start_date, end_date): r.commune_name
                    for r in coords.itertuples(index=False)
                }
                for future in as_completed(jobs):
                    try:
                        df = future.result()
                        df["commune"] = jobs[future]
                        frames.append(df)
                    except Exception:
                        pass

    rain = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if rain.empty:
        st.warning("Données indisponibles ou aucune commune sélectionnée.")
    else:
        fig = go.Figure()
        for name, group in rain.groupby("commune"):
            fig.add_scatter(x=group["date"], y=group["rain_mm"], mode="lines", name=name)
        fig.update_layout(height=430, yaxis_title="Pluie journalière (mm)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        maxima = rain.loc[rain.groupby("commune")["rain_mm"].idxmax(), ["commune", "date", "rain_mm"]]
        st.dataframe(maxima.rename(columns={
            "commune": "Commune", "date": "Date du maximum", "rain_mm": "Maximum (mm)",
        }), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# PIEZOMETRES ACTIFS
# -----------------------------------------------------------------------------
elif module == "Piézomètres actifs":
    st.subheader(f"Piézomètres qui émettent - {sector_name}")
    pk_min = selected_sectors["pk_start"].min() if not all_selected else None
    pk_max = selected_sectors["pk_end"].max() if not all_selected else None
    with st.spinner("Vérification des émissions..."):
        stations = active_piezometers(line, radius, station_limit, pk_min, pk_max)

    if stations.empty:
        st.warning("Aucun piézomètre actif ou en retard léger trouvé.")
    else:
        cols = [
            "code_bss", "nom_commune", "pk_km", "distance_km", "status",
            "last_date", "level", "delta_24h", "delta_7d",
        ]
        display = stations[[c for c in cols if c in stations.columns]].rename(columns={
            "code_bss": "Code BSS", "nom_commune": "Commune", "pk_km": "PK (km)",
            "distance_km": "Distance LGV (km)", "status": "Emission",
            "last_date": "Dernière mesure", "level": "Niveau NGF",
            "delta_24h": "Variation 24 h", "delta_7d": "Variation 7 j",
        })
        st.dataframe(display.sort_values("PK (km)"), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# RISQUE HISTORIQUE
# -----------------------------------------------------------------------------
elif module == "Risque historique":
    if all_selected:
        st.warning("Sélectionne un secteur précis de 10 km.")
        st.stop()

    st.subheader(f"Risque historique - {sector_name}")
    data_start = st.date_input("Début des données", date(2021, 1, 1), max_value=date.today())
    data_end = st.date_input(
        "Fin des données", date.today() - timedelta(days=1), min_value=data_start, max_value=date.today()
    )
    mode = st.radio("Consultation", ["Un jour donné", "Une période"], horizontal=True)

    if mode == "Un jour donné":
        target = st.date_input("Jour", data_end, min_value=data_start, max_value=data_end)
        period_start = period_end = target
    else:
        chosen_period = st.date_input(
            "Période",
            value=(max(data_start, data_end - timedelta(days=90)), data_end),
            min_value=data_start,
            max_value=data_end,
        )
        if not isinstance(chosen_period, (tuple, list)) or len(chosen_period) != 2:
            st.info("Choisis deux dates.")
            st.stop()
        period_start, period_end = chosen_period

    sector = selected_sectors.iloc[0]
    coords = selected_points.groupby("commune_name")[["latitude", "longitude"]].mean().reset_index()
    frames = []
    with st.spinner("Chargement des pluies du secteur..."):
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(coords)))) as pool:
            jobs = [
                pool.submit(load_daily_rain, r.latitude, r.longitude, data_start, data_end)
                for r in coords.itertuples(index=False)
            ]
            for future in as_completed(jobs):
                try:
                    frames.append(future.result())
                except Exception:
                    pass

    if not frames:
        st.error("Pluie indisponible.")
        st.stop()

    daily_max = pd.concat(frames).groupby("date", as_index=False)["rain_mm"].max()
    rain = add_rain_features(daily_max)
    piezo_history = pd.DataFrame()

    with st.spinner("Recherche d’un piézomètre actif proche..."):
        stations = active_piezometers(
            line, radius, min(20, station_limit), sector["pk_start"], sector["pk_end"]
        )

    if not stations.empty:
        station = stations.sort_values("distance_km").iloc[0]
        try:
            piezo_history = load_piezo_history(str(station["code_bss"]), data_start, data_end)
            st.caption(f"Piézomètre utilisé : {station['code_bss']} | PK {station['pk_km']:.1f}")
        except Exception:
            pass

    period = rain[
        (rain["date"].dt.date >= period_start)
        & (rain["date"].dt.date <= period_end)
    ]
    rows = []
    for _, rr in period.iterrows():
        context = historical_piezo_context(piezo_history, rr["date"]) if not piezo_history.empty else None
        result = calculate_risk(rr, sector, context, historical=True)
        rows.append({
            "Date": rr["date"], "Indice": result["score"], "Niveau": result["level"],
            "Confiance (%)": result["confidence"], "Pluie jour": rr["rain_mm"],
            "Pluie 3 j": rr["rain_3d"], "Pluie 7 j": rr["rain_7d"],
            "Pluie 30 j": rr["rain_30d"], "Facteurs": ", ".join(result["factors"]),
        })

    history = pd.DataFrame(rows)
    if history.empty:
        st.info("Aucune donnée sur la période.")
    else:
        fig = go.Figure()
        fig.add_scatter(x=history["Date"], y=history["Indice"], mode="lines+markers", name="Indice")
        for y, label, color in [
            (25, "Jaune", "#eab308"), (50, "Orange", "#ea580c"), (75, "Rouge", "#dc2626"),
        ]:
            fig.add_hline(y=y, line_dash="dash", line_color=color, annotation_text=label)
        fig.update_layout(height=430, yaxis=dict(title="Indice / 100", range=[0, 100]), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        worst = history.loc[history["Indice"].idxmax()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Niveau maximal", worst["Niveau"])
        c2.metric("Indice maximal", f"{worst['Indice']:.1f}/100")
        c3.metric("Date", worst["Date"].strftime("%d/%m/%Y"))
        c4.metric("Confiance", f"{worst['Confiance (%)']} %")
        st.dataframe(history.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export CSV",
            history.to_csv(index=False).encode("utf-8-sig"),
            f"risque_{sector['sector_id']}.csv",
            "text/csv",
        )
        st.caption(
            "Pour les dates passées, les anciennes prévisions et vigilances ne sont pas inventées. La confiance baisse lorsque des sources sont absentes."
        )

# -----------------------------------------------------------------------------
# CARTE SATELLITE
# -----------------------------------------------------------------------------
elif module == "Carte satellite":
    st.subheader(f"Carte satellite - {sector_name}")
    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=8 if all_selected else 11,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, GIS User Community",
        name="Satellite",
        max_zoom=19,
    ).add_to(m)

    for segment in snapshot.get("lgv_lines") or []:
        points_line = [
            [p["lat"], p["lon"]]
            for p in segment
            if isinstance(p, dict) and "lat" in p and "lon" in p
        ]
        if points_line:
            folium.PolyLine(points_line, color="#ef4444", weight=3, opacity=0.9).add_to(m)

    for _, sector in selected_sectors.iterrows():
        popup = (
            f"<b>{sector['name']}</b><br>"
            f"Niveau structurel : {sector['static_level']}<br>"
            f"Indice : {sector['static_score']:.0f}/100<br>"
            f"Communes : {sector['communes']}<br>"
            f"Action : {LEVEL_ACTION[sector['static_level']]}"
        )
        folium.CircleMarker(
            [sector["latitude"], sector["longitude"]],
            radius=8,
            color=LEVEL_COLOR[sector["static_level"]],
            fill=True,
            fill_opacity=0.85,
            tooltip=f"{sector['name']} | Structurel {sector['static_level']} | {sector['static_score']:.0f}/100",
            popup=folium.Popup(popup, max_width=420),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, height=600, returned_objects=[])
    st.caption("La couleur de la carte correspond à la sensibilité structurelle, pas à une alerte météo temps réel.")

st.caption(
    "Optimisations : secteurs fixes de 10 km, cache par source, chargement à la demande et requêtes parallèles limitées."
)
