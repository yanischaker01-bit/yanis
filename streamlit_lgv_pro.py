from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import altair as alt
import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium


SNAPSHOT_LATEST = Path("reports/streamlit_snapshot_latest.json")
SNAPSHOT_GLOB = "streamlit_snapshot_*.json"
REMOTE_SNAPSHOT_URLS = [
    "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json",
]
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_MODEL = "meteofrance_seamless"

RISK_ORDER = {"FAIBLE": 1, "MODERE": 2, "ELEVE": 3, "CRITIQUE": 4}
RISK_COLOR = {
    "FAIBLE": "#16a34a",
    "MODERE": "#ea580c",
    "ELEVE": "#dc2626",
    "CRITIQUE": "#7f1d1d",
    "INDETERMINE": "#6b7280",
}

RAIN_PERIODS = {
    "24h": ("rain_24h_mm", "weather_max_24h_mm"),
    "7 jours": ("rain_7d_mm", "weather_max_7d_mm"),
    "30 jours": ("rain_30d_mm", "weather_max_30d_mm"),
    "Mois courant": ("rain_month_mm", "weather_max_month_mm"),
}

RAIN_COMPONENT_THRESHOLDS = {
    "weather_max_24h_mm": (20.0, 50.0, 80.0, 120.0),
    "weather_max_7d_mm": (35.0, 70.0, 110.0, 160.0),
    "weather_max_30d_mm": (60.0, 110.0, 170.0, 240.0),
    "weather_max_month_mm": (60.0, 110.0, 170.0, 240.0),
}


def _risk_rank(level: str) -> int:
    return RISK_ORDER.get(str(level or "").upper(), 0)


def _risk_level_from_note(note_gc: float) -> str:
    if note_gc >= 80:
        return "CRITIQUE"
    if note_gc >= 60:
        return "ELEVE"
    if note_gc >= 40:
        return "MODERE"
    return "FAIBLE"


def _ai_level_from_probability(probability: float) -> str:
    if probability >= 0.85:
        return "CRITIQUE"
    if probability >= 0.65:
        return "ELEVE"
    if probability >= 0.40:
        return "MODERE"
    return "FAIBLE"


def _score_from_thresholds(value: float, thresholds: Tuple[float, float, float, float]) -> float:
    t1, t2, t3, t4 = thresholds
    if value >= t4:
        return 4.0
    if value >= t3:
        return 3.0
    if value >= t2:
        return 2.0
    if value >= t1:
        return 1.5
    return 1.0


def _score_from_presence_count(value: float, medium: float, high: float) -> float:
    if value >= high:
        return 4.0
    if value >= medium:
        return 3.0
    if value > 0.0:
        return 2.0
    return 1.0


def _find_snapshot() -> Path | None:
    if SNAPSHOT_LATEST.exists():
        return SNAPSHOT_LATEST
    reports = Path("reports")
    if not reports.exists():
        return None
    snapshots = sorted(reports.glob(SNAPSHOT_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0] if snapshots else None


@st.cache_data(show_spinner=False)
def _load_snapshot(path_str: str, mtime: float) -> Dict[str, object]:
    _ = mtime
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(show_spinner=False, ttl=300)
def _load_remote_snapshot(url: str) -> Dict[str, object]:
    try:
        response = requests.get(url, timeout=20)
        if response.status_code != 200 or not response.text.strip():
            return {}
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_snapshot_payload() -> Tuple[Dict[str, object], str]:
    local_path = _find_snapshot()
    if local_path is not None:
        return _load_snapshot(str(local_path), local_path.stat().st_mtime), f"local:{local_path}"

    for url in REMOTE_SNAPSHOT_URLS:
        payload = _load_remote_snapshot(url)
        if payload:
            return payload, url
    return {}, ""


def _safe_df(records: object) -> pd.DataFrame:
    if isinstance(records, list):
        try:
            return pd.DataFrame(records)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_row(df: pd.DataFrame, lat: float, lon: float) -> Dict[str, object]:
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return {}
    work = df.copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work = work.dropna(subset=["latitude", "longitude"])
    if work.empty:
        return {}
    work["_dist_km"] = work.apply(lambda r: _haversine_km(lat, lon, float(r["latitude"]), float(r["longitude"])), axis=1)
    row = work.sort_values("_dist_km").iloc[0].to_dict()
    row["_dist_km"] = round(float(row.get("_dist_km", 0.0)), 3)
    return row


@st.cache_data(show_spinner=False, ttl=21600)
def _load_sector_monthly_history(lat: float, lon: float, years_back: int) -> Dict[str, object]:
    now_utc = datetime.now(timezone.utc)
    start_year = max(2010, now_utc.year - max(int(years_back), 1) + 1)
    start_date = f"{start_year}-01-01"
    end_date = now_utc.strftime("%Y-%m-%d")
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "UTC",
        "models": OPEN_METEO_MODEL,
    }

    used_model = OPEN_METEO_MODEL
    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
        if response.status_code != 200:
            fallback_params = dict(params)
            fallback_params.pop("models", None)
            response = requests.get(OPEN_METEO_ARCHIVE_URL, params=fallback_params, timeout=30)
            used_model = "open_meteo_default"
        if response.status_code != 200 or not response.text.strip():
            return {"monthly": [], "climatology": [], "model": used_model, "error": f"HTTP {response.status_code}"}

        payload = response.json()
        entry = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(entry, dict):
            return {"monthly": [], "climatology": [], "model": used_model, "error": "payload invalide"}
        daily = entry.get("daily", {}) if isinstance(entry.get("daily"), dict) else {}
        times = daily.get("time", []) or []
        vals = daily.get("precipitation_sum", []) or []
        if not times or not vals:
            return {"monthly": [], "climatology": [], "model": used_model, "error": "serie vide"}

        df = pd.DataFrame({"date": pd.to_datetime(times, utc=True, errors="coerce"), "precip_mm": pd.to_numeric(vals, errors="coerce")})
        df = df.dropna(subset=["date", "precip_mm"])
        if df.empty:
            return {"monthly": [], "climatology": [], "model": used_model, "error": "serie vide"}

        monthly = (
            df.assign(
                year=lambda d: d["date"].dt.year,
                month=lambda d: d["date"].dt.month,
                ym=lambda d: d["date"].dt.to_period("M").astype(str),
            )
            .groupby(["ym", "year", "month"], as_index=False)["precip_mm"]
            .sum()
            .rename(columns={"precip_mm": "monthly_precip_mm"})
        )
        monthly["monthly_precip_mm"] = monthly["monthly_precip_mm"].round(1)

        clim = monthly.groupby("month", as_index=False)["monthly_precip_mm"].mean().rename(columns={"monthly_precip_mm": "climatology_mm"})
        clim["climatology_mm"] = clim["climatology_mm"].round(1)
        month_names = {
            1: "Jan", 2: "Fev", 3: "Mar", 4: "Avr", 5: "Mai", 6: "Juin",
            7: "Juil", 8: "Aou", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        clim["month_label"] = clim["month"].map(month_names)

        return {
            "monthly": monthly.sort_values("ym").to_dict(orient="records"),
            "climatology": clim.sort_values("month").to_dict(orient="records"),
            "model": used_model,
            "error": None,
        }
    except Exception as exc:
        return {"monthly": [], "climatology": [], "model": used_model, "error": str(exc)}


def _build_multi_commune_history(commune_rows: List[Dict[str, object]], years_back: int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames: List[pd.DataFrame] = []
    model_by_commune: Dict[str, str] = {}
    for com in commune_rows:
        cname = str(com.get("commune_name") or "Inconnue")
        ccode = str(com.get("commune_code") or "")
        commune_label = f"{cname} ({ccode})" if ccode else cname
        try:
            lat = float(com.get("latitude"))
            lon = float(com.get("longitude"))
        except (TypeError, ValueError):
            continue

        payload = _load_sector_monthly_history(lat, lon, years_back)
        model_by_commune[commune_label] = str(payload.get("model") or "")
        monthly = _safe_df(payload.get("monthly"))
        if monthly.empty:
            continue
        monthly["commune_name"] = cname
        monthly["commune_label"] = commune_label
        frames.append(monthly)
    if not frames:
        return pd.DataFrame(), model_by_commune
    out = pd.concat(frames, ignore_index=True)
    out["ym"] = out["ym"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(0).astype(int)
    out["month"] = pd.to_numeric(out["month"], errors="coerce").fillna(0).astype(int)
    out["monthly_precip_mm"] = pd.to_numeric(out["monthly_precip_mm"], errors="coerce")
    out = out.dropna(subset=["monthly_precip_mm"])
    return out.sort_values(["ym", "commune_label"]), model_by_commune


def _aggregate_communes(sectors_df: pd.DataFrame, commune_rain_col: str) -> pd.DataFrame:
    if sectors_df.empty:
        return pd.DataFrame()

    df = sectors_df.copy()
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["commune_name"] = df.get("commune_name", "Inconnue").fillna("Inconnue")
    df["sector_score"] = pd.to_numeric(df.get("score", 0.0), errors="coerce").fillna(0.0)
    df["rain_period_mm"] = pd.to_numeric(df.get(commune_rain_col, 0.0), errors="coerce").fillna(0.0)
    df["geotech_points"] = pd.to_numeric(df.get("geotech_points", 0.0), errors="coerce").fillna(0.0)
    df["piezometers"] = pd.to_numeric(df.get("piezometers", 0.0), errors="coerce").fillna(0.0)
    df["hydro_stations"] = pd.to_numeric(df.get("hydro_stations", 0.0), errors="coerce").fillna(0.0)
    df["ai_pred_probability"] = pd.to_numeric(df.get("ai_pred_probability", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    df["ai_pred_score"] = pd.to_numeric(df.get("ai_pred_score", 0.0), errors="coerce").fillna(0.0)
    df["ai_soil_fragility"] = pd.to_numeric(df.get("ai_soil_fragility", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    if "ai_pred_risk_level" not in df.columns:
        df["ai_pred_risk_level"] = df["ai_pred_probability"].map(_ai_level_from_probability)
    df["ai_pred_risk_level"] = df["ai_pred_risk_level"].fillna("INDETERMINE").astype(str)
    df["is_critical"] = (df.get("risk_level", "") == "CRITIQUE").astype(int)
    df["is_high"] = (df.get("risk_level", "") == "ELEVE").astype(int)
    df["is_moderate"] = (df.get("risk_level", "") == "MODERE").astype(int)
    df["is_ai_critical"] = (df.get("ai_pred_risk_level", "") == "CRITIQUE").astype(int)
    df["is_ai_high"] = (df.get("ai_pred_risk_level", "") == "ELEVE").astype(int)
    if "under_watch" in df.columns:
        df["is_watch"] = df["under_watch"].fillna(False).astype(bool).astype(int)
    else:
        df["is_watch"] = 0
    if "risk_level" in df.columns:
        df["risk_rank"] = df["risk_level"].map(lambda x: _risk_rank(str(x)))
    else:
        df["risk_rank"] = 0

    grouped = (
        df.groupby(["commune_name", "commune_code", "departement_code", "departement_name"], dropna=False)
        .agg(
            sector_count=("sector_id", "count"),
            avg_sector_score=("sector_score", "mean"),
            max_sector_score=("sector_score", "max"),
            avg_risk_rank=("risk_rank", "mean"),
            critical=("is_critical", "sum"),
            high=("is_high", "sum"),
            moderate=("is_moderate", "sum"),
            ai_critical=("is_ai_critical", "sum"),
            ai_high=("is_ai_high", "sum"),
            watch=("is_watch", "sum"),
            avg_rain_period_mm=("rain_period_mm", "mean"),
            max_rain_period_mm=("rain_period_mm", "max"),
            avg_geotech_points=("geotech_points", "mean"),
            max_geotech_points=("geotech_points", "max"),
            avg_piezometers=("piezometers", "mean"),
            max_piezometers=("piezometers", "max"),
            avg_hydro_stations=("hydro_stations", "mean"),
            max_hydro_stations=("hydro_stations", "max"),
            avg_ai_probability=("ai_pred_probability", "mean"),
            max_ai_probability=("ai_pred_probability", "max"),
            avg_ai_soil_fragility=("ai_soil_fragility", "mean"),
            max_ai_soil_fragility=("ai_soil_fragility", "max"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .reset_index()
    )

    grouped["note_gc_base"] = (
        (grouped["avg_sector_score"] / 4.0) * 68.0
        + ((grouped["critical"] * 12.0 + grouped["high"] * 6.0) / grouped["sector_count"].clip(lower=1))
        + grouped["avg_rain_period_mm"].clip(lower=0.0, upper=180.0) / 9.0
    ).clip(lower=0.0, upper=100.0)
    grouped["note_gc_base"] = grouped["note_gc_base"].round(1)
    grouped["latitude"] = pd.to_numeric(grouped.get("latitude"), errors="coerce").round(6)
    grouped["longitude"] = pd.to_numeric(grouped.get("longitude"), errors="coerce").round(6)

    rain_thresholds = RAIN_COMPONENT_THRESHOLDS.get(
        str(commune_rain_col),
        (20.0, 50.0, 80.0, 120.0),
    )
    grouped["weather_component_score"] = grouped["avg_rain_period_mm"].map(lambda v: _score_from_thresholds(float(v or 0.0), rain_thresholds))
    grouped["geotech_component_score"] = grouped["max_geotech_points"].map(
        lambda v: _score_from_presence_count(float(v or 0.0), medium=2.0, high=4.0)
    )
    grouped["piezo_component_score"] = grouped["max_piezometers"].map(
        lambda v: _score_from_presence_count(float(v or 0.0), medium=1.0, high=2.0)
    )
    grouped["hydro_component_score"] = grouped["max_hydro_stations"].map(
        lambda v: _score_from_presence_count(float(v or 0.0), medium=1.0, high=2.0)
    )
    grouped["ai_component_score"] = grouped["max_ai_probability"].map(
        lambda v: 4.0 if float(v or 0.0) >= 0.85 else (3.0 if float(v or 0.0) >= 0.65 else (2.0 if float(v or 0.0) >= 0.40 else 1.0))
    )

    grouped["weather_component_note"] = (grouped["weather_component_score"] / 4.0 * 100.0).round(1)
    grouped["geotech_component_note"] = (grouped["geotech_component_score"] / 4.0 * 100.0).round(1)
    grouped["piezo_component_note"] = (grouped["piezo_component_score"] / 4.0 * 100.0).round(1)
    grouped["hydro_component_note"] = (grouped["hydro_component_score"] / 4.0 * 100.0).round(1)
    grouped["ai_component_note"] = (grouped["ai_component_score"] / 4.0 * 100.0).round(1)

    grouped["note_gc"] = grouped["note_gc_base"]
    has_ai_signal = grouped["max_ai_probability"] > 0.0
    grouped.loc[has_ai_signal, "note_gc"] = (
        grouped.loc[has_ai_signal, "note_gc_base"] * 0.72
        + grouped.loc[has_ai_signal, "ai_component_note"] * 0.28
    ).clip(lower=0.0, upper=100.0)
    grouped["note_gc"] = grouped["note_gc"].round(1)
    grouped["global_gc_note"] = grouped["note_gc"]
    grouped["commune_risk_level"] = grouped["note_gc"].map(_risk_level_from_note)
    grouped["ai_commune_risk_level"] = grouped["max_ai_probability"].map(lambda v: _ai_level_from_probability(float(v or 0.0)))

    grouped["lgv_points_count"] = grouped["sector_count"]
    grouped["avg_point_score"] = grouped["avg_sector_score"]
    grouped["max_point_score"] = grouped["max_sector_score"]
    grouped = grouped.sort_values(["note_gc", "critical", "high"], ascending=[False, False, False]).reset_index(drop=True)
    return grouped


def _build_map(
    snapshot: Dict[str, object],
    weather_df: pd.DataFrame,
    commune_df: pd.DataFrame,
    sectors_df: pd.DataFrame,
    hydro_df: pd.DataFrame,
    piezo_df: pd.DataFrame,
    geotech_df: pd.DataFrame,
    lgv_communes_df: pd.DataFrame,
    fr_communes_geojson: Dict[str, object],
    rain_col_weather: str,
    min_risk: str,
    show_weather: bool,
    show_communes: bool,
    show_sectors: bool,
    show_hydro: bool,
    show_piezo: bool,
    show_geotech: bool,
    show_fr_layer: bool,
) -> folium.Map:
    m = folium.Map(location=[46.2, 0.2], zoom_start=7, tiles="CartoDB positron")

    for line in snapshot.get("lgv_lines", []) if isinstance(snapshot.get("lgv_lines"), list) else []:
        coords = []
        for pt in line if isinstance(line, list) else []:
            if isinstance(pt, dict) and "lat" in pt and "lon" in pt:
                coords.append((float(pt["lat"]), float(pt["lon"])))
        if len(coords) >= 2:
            folium.PolyLine(coords, color="#1d4ed8", weight=4, opacity=0.9, tooltip="Trace LGV SEA").add_to(m)

    if show_weather and not weather_df.empty:
        weather_layer = folium.FeatureGroup(name="Meteo", show=True)
        for _, row in weather_df.iterrows():
            lvl = str(row.get("risk_level", "INDETERMINE"))
            if _risk_rank(lvl) < _risk_rank(min_risk):
                continue
            rain = float(row.get(rain_col_weather, 0.0) or 0.0)
            popup = (
                f"<b>Station:</b> {row.get('station_id')}<br>"
                f"<b>Source:</b> {row.get('source')}<br>"
                f"<b>Commune station:</b> {row.get('station_commune_name', 'n/a')}<br>"
                f"<b>Risque:</b> {lvl}<br>"
                f"<b>Cumul filtre:</b> {rain:.1f} mm<br>"
                f"<b>Dist LGV:</b> {row.get('distance_to_lgv_km')} km"
            )
            folium.CircleMarker(
                [float(row["latitude"]), float(row["longitude"])],
                radius=5,
                color=RISK_COLOR.get(lvl, "#6b7280"),
                fill=True,
                fill_opacity=0.85,
                weight=1,
                popup=folium.Popup(popup, max_width=320),
            ).add_to(weather_layer)
        weather_layer.add_to(m)

    if show_communes and not commune_df.empty:
        commune_layer = folium.FeatureGroup(name="Communes", show=True)
        for _, row in commune_df.iterrows():
            lvl = str(row.get("commune_risk_level", "INDETERMINE"))
            if _risk_rank(lvl) < _risk_rank(min_risk):
                continue
            lat = pd.to_numeric(row.get("latitude"), errors="coerce")
            lon = pd.to_numeric(row.get("longitude"), errors="coerce")
            if pd.isna(lat) or pd.isna(lon):
                continue
            rain_avg = float(row.get("avg_rain_period_mm", 0.0) or 0.0)
            rain_max = float(row.get("max_rain_period_mm", 0.0) or 0.0)
            lgv_points = int(row.get("lgv_points_count", row.get("sector_count", 0)) or 0)
            radius = max(6, min(14, 6 + lgv_points))
            popup = (
                f"<b>Commune:</b> {row.get('commune_name')}<br>"
                f"<b>Code INSEE:</b> {row.get('commune_code', 'n/a')}<br>"
                f"<b>Risque:</b> {lvl}<br>"
                f"<b>Note GC:</b> {row.get('note_gc')} /100<br>"
                f"<b>Prediction IA commune:</b> {row.get('ai_commune_risk_level', 'n/a')}<br>"
                f"<b>Probabilite IA max:</b> {round(float(row.get('max_ai_probability', 0.0) or 0.0) * 100.0, 1)} %<br>"
                f"<b>Cumul moyen filtre:</b> {rain_avg:.1f} mm<br>"
                f"<b>Cumul max filtre:</b> {rain_max:.1f} mm<br>"
                f"<b>Points LGV dans commune:</b> {lgv_points}"
            )
            folium.CircleMarker(
                [float(lat), float(lon)],
                radius=radius,
                color=RISK_COLOR.get(lvl, "#6b7280"),
                fill=True,
                fill_opacity=0.30,
                weight=2,
                popup=folium.Popup(popup, max_width=360),
            ).add_to(commune_layer)
        commune_layer.add_to(m)

    if show_sectors and not sectors_df.empty:
        sectors_layer = folium.FeatureGroup(name="Secteurs IA", show=True)
        for _, row in sectors_df.iterrows():
            lvl = str(row.get("ai_pred_risk_level", row.get("risk_level", "INDETERMINE")))
            if _risk_rank(lvl) < _risk_rank(min_risk):
                continue
            lat = pd.to_numeric(row.get("latitude"), errors="coerce")
            lon = pd.to_numeric(row.get("longitude"), errors="coerce")
            if pd.isna(lat) or pd.isna(lon):
                continue
            ai_prob_raw = pd.to_numeric(row.get("ai_pred_probability"), errors="coerce")
            ai_prob = 0.0 if pd.isna(ai_prob_raw) else float(ai_prob_raw)
            soil_frag_raw = pd.to_numeric(row.get("ai_soil_fragility"), errors="coerce")
            soil_frag = 0.0 if pd.isna(soil_frag_raw) else float(soil_frag_raw)
            ai_conf_raw = pd.to_numeric(row.get("ai_confidence"), errors="coerce")
            ai_conf = 0.0 if pd.isna(ai_conf_raw) else float(ai_conf_raw)
            top_factors = row.get("ai_top_factors")
            if isinstance(top_factors, list):
                top_factors_txt = ", ".join([str(x) for x in top_factors if str(x).strip()]) or "n/a"
            else:
                top_factors_txt = str(top_factors or "n/a")
            popup = (
                f"<b>Secteur:</b> {row.get('sector_id')}<br>"
                f"<b>Commune:</b> {row.get('commune_name', 'n/a')}<br>"
                f"<b>Prediction IA:</b> {lvl}<br>"
                f"<b>Probabilite IA:</b> {ai_prob * 100.0:.1f}%<br>"
                f"<b>Score IA:</b> {row.get('ai_pred_score', 'n/a')}/4<br>"
                f"<b>Confiance IA:</b> {ai_conf * 100.0:.1f}%<br>"
                f"<b>Fragilite sol:</b> {soil_frag * 100.0:.1f}% ({row.get('ai_dominant_pedology', 'n/a')})<br>"
                f"<b>Type sol dominant:</b> {row.get('ai_dominant_soil_type', 'n/a')}<br>"
                f"<b>Pluie 24h/7j/30j:</b> {row.get('weather_max_24h_mm', 0)} / {row.get('weather_max_7d_mm', 0)} / {row.get('weather_max_30d_mm', 0)} mm<br>"
                f"<b>Facteurs IA:</b> {top_factors_txt}"
            )
            radius = max(5, min(14, 5 + ai_prob * 9.0))
            color = str(row.get("ai_pred_risk_color") or RISK_COLOR.get(lvl, "#6b7280"))
            folium.CircleMarker(
                [float(lat), float(lon)],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.45,
                weight=2,
                popup=folium.Popup(popup, max_width=420),
            ).add_to(sectors_layer)
        sectors_layer.add_to(m)

    if show_hydro and not hydro_df.empty:
        hydro_layer = folium.FeatureGroup(name="Hydro reseau", show=False)
        for _, row in hydro_df.iterrows():
            lvl = str(row.get("risk_level", "INDETERMINE"))
            if _risk_rank(lvl) < _risk_rank(min_risk):
                continue
            popup = (
                f"<b>Station:</b> {row.get('station_code')}<br>"
                f"<b>Riviere:</b> {row.get('river_name')}<br>"
                f"<b>Niveau:</b> {row.get('last_level_m')} m<br>"
                f"<b>Tendance:</b> {row.get('trend_mph')} m/h<br>"
                f"<b>Seuil urgence:</b> {row.get('emergency_threshold_m')} m<br>"
                f"<b>Ratio niveau/seuil:</b> {row.get('threshold_ratio')}<br>"
                f"<b>Depassement seuil:</b> {row.get('threshold_exceeded')}<br>"
                f"<b>Risque:</b> {lvl}"
            )
            folium.CircleMarker(
                [float(row["latitude"]), float(row["longitude"])],
                radius=6,
                color=RISK_COLOR.get(lvl, "#6b7280"),
                fill=True,
                fill_opacity=0.9,
                weight=1,
                popup=folium.Popup(popup, max_width=340),
            ).add_to(hydro_layer)
        hydro_layer.add_to(m)

    if show_piezo and not piezo_df.empty:
        piezo_layer = folium.FeatureGroup(name="Piezometres", show=False)
        for _, row in piezo_df.iterrows():
            lvl = str(row.get("risk_level", "INDETERMINE"))
            if _risk_rank(lvl) < _risk_rank(min_risk):
                continue
            popup = (
                f"<b>Piezometre:</b> {row.get('code_bss')}<br>"
                f"<b>Nom:</b> {row.get('name')}<br>"
                f"<b>Profondeur:</b> {row.get('depth_m')} m<br>"
                f"<b>Tendance:</b> {row.get('trend_depth_mpd')} m/j<br>"
                f"<b>Risque:</b> {lvl}"
            )
            folium.CircleMarker(
                [float(row["latitude"]), float(row["longitude"])],
                radius=6,
                color=RISK_COLOR.get(lvl, "#6b7280"),
                fill=True,
                fill_opacity=0.85,
                weight=1,
                popup=folium.Popup(popup, max_width=340),
            ).add_to(piezo_layer)
        piezo_layer.add_to(m)

    if show_geotech and not geotech_df.empty:
        geo_layer = folium.FeatureGroup(name="Geotech", show=False)
        for _, row in geotech_df.iterrows():
            lvl = str(row.get("risk_level", "INDETERMINE"))
            if _risk_rank(lvl) < _risk_rank(min_risk):
                continue
            popup = (
                f"<b>Point:</b> {row.get('point_id')}<br>"
                f"<b>Sol:</b> {row.get('soil_type')}<br>"
                f"<b>Pedologie:</b> {row.get('pedology_family')}<br>"
                f"<b>Lithologie:</b> {row.get('lithology_descr')} ({row.get('lithology_type')})<br>"
                f"<b>RGA:</b> {row.get('rga_label')}<br>"
                f"<b>MVT:</b> {row.get('mvt_count')}<br>"
                f"<b>Risque:</b> {lvl}"
            )
            folium.CircleMarker(
                [float(row["latitude"]), float(row["longitude"])],
                radius=5,
                color=RISK_COLOR.get(lvl, "#6b7280"),
                fill=True,
                fill_opacity=0.85,
                weight=1,
                popup=folium.Popup(popup, max_width=340),
            ).add_to(geo_layer)
        geo_layer.add_to(m)

    if show_fr_layer:
        fr_layer = folium.FeatureGroup(name="Couche geographique FR", show=False)
        has_geojson = isinstance(fr_communes_geojson, dict) and isinstance(fr_communes_geojson.get("features"), list) and bool(fr_communes_geojson.get("features"))
        if has_geojson:
            first_props = {}
            first_feature = fr_communes_geojson.get("features", [])[0]
            if isinstance(first_feature, dict) and isinstance(first_feature.get("properties"), dict):
                first_props = first_feature.get("properties") or {}
            alias_map = {
                "commune_name": "Commune",
                "commune_code": "Code INSEE",
                "departement_code": "Departement",
                "pk_start_km": "PK debut",
                "pk_end_km": "PK fin",
                "traversed_km": "Traverse (km)",
                "order_on_line": "Ordre sur ligne",
            }
            tooltip_fields = [f for f in ["commune_name", "commune_code", "departement_code", "pk_start_km", "pk_end_km", "traversed_km"] if f in first_props]
            tooltip_aliases = [alias_map.get(f, f) for f in tooltip_fields]
            tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, localize=True, sticky=False) if tooltip_fields else None
            folium.GeoJson(
                fr_communes_geojson,
                name="Communes traversees",
                style_function=lambda _: {"color": "#0f766e", "weight": 1.2, "fillColor": "#99f6e4", "fillOpacity": 0.08},
                highlight_function=lambda _: {"weight": 2.2, "fillOpacity": 0.20},
                tooltip=tooltip,
            ).add_to(fr_layer)
        elif not lgv_communes_df.empty:
            for _, row in lgv_communes_df.iterrows():
                lat = pd.to_numeric(row.get("centroid_latitude"), errors="coerce")
                lon = pd.to_numeric(row.get("centroid_longitude"), errors="coerce")
                if pd.isna(lat) or pd.isna(lon):
                    continue
                popup = (
                    f"<b>Commune LGV:</b> {row.get('commune_name')}<br>"
                    f"<b>Code INSEE:</b> {row.get('commune_code')}<br>"
                    f"<b>Departement:</b> {row.get('departement_code')}<br>"
                    f"<b>Traverse:</b> {row.get('traversed_km')} km"
                )
                folium.CircleMarker(
                    [float(lat), float(lon)],
                    radius=4,
                    color="#0f766e",
                    fill=True,
                    fill_opacity=0.60,
                    weight=1,
                    popup=folium.Popup(popup, max_width=320),
                ).add_to(fr_layer)
        fr_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


st.set_page_config(page_title="LGV SEA Pro Monitoring", page_icon=":chart_with_upwards_trend:", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
      .stMetric {border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; background: #ffffff;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LGV SEA - Rapport Streamlit Pro")
st.caption("Suivi hydrometeo et geotechnique avec classement par commune")

snapshot, snapshot_source = _load_snapshot_payload()
if not snapshot:
    st.error("Aucune donnee chargee. Le snapshot n'est pas disponible.")
    st.info("Verifie que GitHub Pages est actif puis recharge la page.")
    if st.button("Reessayer le chargement", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.stop()

weather_df = _safe_df(snapshot.get("weather"))
sectors_df = _safe_df((snapshot.get("sectors") or {}).get("sectors"))
hydro_df = _safe_df((snapshot.get("hydro_network") or {}).get("stations"))
piezo_df = _safe_df((snapshot.get("piezometers") or {}).get("stations"))
geotech_df = _safe_df((snapshot.get("geotech") or {}).get("points"))
alerts_df = _safe_df(snapshot.get("alerts"))
lgv_communes_obj = snapshot.get("lgv_communes") if isinstance(snapshot.get("lgv_communes"), dict) else {}
lgv_communes_df = _safe_df((lgv_communes_obj or {}).get("communes"))
fr_geo_obj = snapshot.get("fr_geography") if isinstance(snapshot.get("fr_geography"), dict) else {}
fr_communes_geojson: Dict[str, object] = {}
if isinstance(fr_geo_obj.get("communes_geojson"), dict):
    fr_communes_geojson = fr_geo_obj.get("communes_geojson") or {}
elif isinstance(lgv_communes_obj.get("communes_geojson"), dict):
    fr_communes_geojson = lgv_communes_obj.get("communes_geojson") or {}
metadata_obj = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), dict) else {}

if not weather_df.empty:
    for col in ["rain_24h_mm", "rain_7d_mm", "rain_30d_mm", "rain_month_mm", "distance_to_lgv_km"]:
        if col in weather_df.columns:
            weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")
    weather_df["risk_level"] = weather_df.get("rain_class", "INDETERMINE")
    if "station_commune_name" not in weather_df.columns:
        weather_df["station_commune_name"] = "Inconnue"
    weather_df["station_commune_name"] = weather_df["station_commune_name"].fillna("Inconnue")

if not sectors_df.empty:
    for col in [
        "score",
        "weather_max_24h_mm",
        "weather_max_7d_mm",
        "weather_max_30d_mm",
        "weather_max_month_mm",
        "ai_pred_probability",
        "ai_pred_score",
        "ai_confidence",
        "ai_soil_fragility",
    ]:
        if col in sectors_df.columns:
            sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")
    sectors_df["commune_name"] = sectors_df.get("commune_name", "Inconnue").fillna("Inconnue")
    if "ai_pred_probability" not in sectors_df.columns:
        sectors_df["ai_pred_probability"] = (
            pd.to_numeric(sectors_df.get("score", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0) / 4.0
        ).clip(lower=0.0, upper=1.0)
    else:
        sectors_df["ai_pred_probability"] = pd.to_numeric(sectors_df["ai_pred_probability"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    if "ai_pred_score" not in sectors_df.columns:
        sectors_df["ai_pred_score"] = (sectors_df["ai_pred_probability"] * 3.0 + 1.0).round(2)
    if "ai_pred_risk_level" not in sectors_df.columns:
        sectors_df["ai_pred_risk_level"] = sectors_df["ai_pred_probability"].map(lambda v: _ai_level_from_probability(float(v or 0.0)))
    sectors_df["ai_pred_risk_level"] = sectors_df["ai_pred_risk_level"].fillna("INDETERMINE").astype(str)
    if "ai_pred_risk_color" not in sectors_df.columns:
        sectors_df["ai_pred_risk_color"] = sectors_df["ai_pred_risk_level"].map(lambda lvl: RISK_COLOR.get(str(lvl), "#6b7280"))
    if "ai_soil_fragility" not in sectors_df.columns:
        sectors_df["ai_soil_fragility"] = 0.55
    sectors_df["ai_soil_fragility"] = pd.to_numeric(sectors_df["ai_soil_fragility"], errors="coerce").fillna(0.55).clip(lower=0.0, upper=1.0)
    if "ai_dominant_pedology" not in sectors_df.columns:
        sectors_df["ai_dominant_pedology"] = "Pedologie indeterminee"
    sectors_df["ai_dominant_pedology"] = sectors_df["ai_dominant_pedology"].fillna("Pedologie indeterminee").astype(str)
    if "ai_dominant_soil_type" not in sectors_df.columns:
        sectors_df["ai_dominant_soil_type"] = "Sols indetermines"
    sectors_df["ai_dominant_soil_type"] = sectors_df["ai_dominant_soil_type"].fillna("Sols indetermines").astype(str)
    if "ai_top_factors" not in sectors_df.columns:
        sectors_df["ai_top_factors"] = [[] for _ in range(len(sectors_df))]

if not hydro_df.empty:
    for col in ["last_level_m", "trend_mph", "emergency_threshold_m", "watch_threshold_m", "threshold_ratio", "distance_to_lgv_km"]:
        if col in hydro_df.columns:
            hydro_df[col] = pd.to_numeric(hydro_df[col], errors="coerce")
    if "threshold_exceeded" not in hydro_df.columns:
        hydro_df["threshold_exceeded"] = False
    hydro_df["threshold_exceeded"] = hydro_df["threshold_exceeded"].fillna(False).astype(bool)
    if "risk_level" not in hydro_df.columns:
        hydro_df["risk_level"] = "INDETERMINE"

if not geotech_df.empty:
    if "pedology_family" not in geotech_df.columns:
        geotech_df["pedology_family"] = "Pedologie indeterminee"
    geotech_df["pedology_family"] = geotech_df["pedology_family"].fillna("Pedologie indeterminee")

if not lgv_communes_df.empty:
    for col in ["pk_start_km", "pk_end_km", "traversed_km", "order_on_line", "centroid_latitude", "centroid_longitude"]:
        if col in lgv_communes_df.columns:
            lgv_communes_df[col] = pd.to_numeric(lgv_communes_df[col], errors="coerce")

if not alerts_df.empty and "level" in alerts_df.columns:
    alerts_df["rank"] = alerts_df["level"].map(lambda x: _risk_rank(str(x))).fillna(0)
    alerts_df = alerts_df.sort_values("rank", ascending=False)

with st.sidebar:
    st.subheader("Filtres")
    if st.button("Rafraichir snapshot", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    period_label = st.selectbox("Periode pluvio", list(RAIN_PERIODS.keys()), index=0)
    rain_col_weather, commune_rain_col = RAIN_PERIODS[period_label]
    min_risk = st.selectbox("Risque minimum", ["FAIBLE", "MODERE", "ELEVE", "CRITIQUE"], index=1)
    sector_risk_mode = st.selectbox("Filtre secteurs", ["IA predictive", "Operationnel"], index=0)

    sources = sorted(weather_df["source"].dropna().astype(str).unique().tolist()) if "source" in weather_df.columns else []
    selected_sources = st.multiselect("Sources meteo", sources, default=sources)

    communes = sorted(sectors_df["commune_name"].dropna().astype(str).unique().tolist()) if "commune_name" in sectors_df.columns else []
    selected_communes = st.multiselect("Communes", communes, default=communes)

    station_communes = (
        sorted(weather_df["station_commune_name"].dropna().astype(str).unique().tolist())
        if "station_commune_name" in weather_df.columns
        else []
    )
    selected_station_communes = st.multiselect("Communes des stations meteo", station_communes, default=station_communes)

    st.caption("Toutes les communes filtrees sont affichees (pas de limite a 25).")

    st.markdown("---")
    show_weather = st.checkbox("Layer meteo", value=True)
    show_communes = st.checkbox("Layer communes", value=True)
    show_sectors = st.checkbox("Layer secteurs IA", value=True)
    show_hydro = st.checkbox("Layer hydro", value=True)
    show_piezo = st.checkbox("Layer piezometres", value=False)
    show_geotech = st.checkbox("Layer geotech", value=False)
    show_fr_layer = st.checkbox("Layer geographie FR", value=False)

filtered_weather = weather_df.copy()
if not filtered_weather.empty and selected_sources:
    filtered_weather = filtered_weather[filtered_weather["source"].astype(str).isin(selected_sources)]
if not filtered_weather.empty and selected_station_communes:
    filtered_weather = filtered_weather[filtered_weather["station_commune_name"].astype(str).isin(selected_station_communes)]
if not filtered_weather.empty:
    filtered_weather = filtered_weather[filtered_weather["risk_level"].map(lambda x: _risk_rank(str(x))) >= _risk_rank(min_risk)]

filtered_sectors = sectors_df.copy()
if not filtered_sectors.empty and selected_communes:
    filtered_sectors = filtered_sectors[filtered_sectors["commune_name"].astype(str).isin(selected_communes)]
if not filtered_sectors.empty:
    sector_risk_col = "risk_level"
    if sector_risk_mode.startswith("IA") and "ai_pred_risk_level" in filtered_sectors.columns:
        sector_risk_col = "ai_pred_risk_level"
    filtered_sectors = filtered_sectors[filtered_sectors[sector_risk_col].map(lambda x: _risk_rank(str(x))) >= _risk_rank(min_risk)]

commune_df = _aggregate_communes(filtered_sectors, commune_rain_col)
if not commune_df.empty:
    commune_df["commune_code"] = commune_df.get("commune_code", "").fillna("").astype(str)
    commune_df["commune_label"] = commune_df.apply(
        lambda r: f"{str(r.get('commune_name') or 'Inconnue')} ({str(r.get('commune_code') or '')})"
        if str(r.get("commune_code") or "").strip()
        else str(r.get("commune_name") or "Inconnue"),
        axis=1,
    )

commune_pool = commune_df.copy()
selected_commune: Dict[str, object] = {}
history_years = 5
selected_compare_commune_labels: List[str] = []
compare_history_df = pd.DataFrame()
history_models: Dict[str, str] = {}
if not commune_pool.empty:
    with st.sidebar:
        st.markdown("---")
        st.subheader("Analyse commune")
        commune_labels = commune_pool["commune_label"].astype(str).tolist()
        chosen_commune_label = st.selectbox("Commune detail", commune_labels, index=0)
        selected_compare_commune_labels = st.multiselect(
            "Communes a comparer",
            commune_labels,
            default=commune_labels[: min(8, len(commune_labels))],
        )
        history_years = st.slider("Historique mensuel (ans)", min_value=2, max_value=5, value=5, step=1)
    selected_commune = commune_pool[commune_pool["commune_label"].astype(str) == chosen_commune_label].iloc[0].to_dict()

    compare_commune_rows: List[Dict[str, object]] = []
    for label in selected_compare_commune_labels:
        hit = commune_pool[commune_pool["commune_label"].astype(str) == str(label)]
        if not hit.empty:
            compare_commune_rows.append(hit.iloc[0].to_dict())
    compare_history_df, history_models = _build_multi_commune_history(compare_commune_rows, int(history_years))

    if not compare_history_df.empty:
        ym_options = sorted(compare_history_df["ym"].astype(str).unique().tolist())
        if ym_options:
            default_start = ym_options[max(0, len(ym_options) - 24)]
            default_end = ym_options[-1]
            with st.sidebar:
                ym_start, ym_end = st.select_slider(
                    "Periode historique comparee",
                    options=ym_options,
                    value=(default_start, default_end),
                )
            compare_history_df = compare_history_df[
                (compare_history_df["ym"].astype(str) >= str(ym_start)) & (compare_history_df["ym"].astype(str) <= str(ym_end))
            ]

history_payload = {"monthly": [], "climatology": [], "model": "", "error": "pas de commune"}
if selected_commune:
    history_payload = _load_sector_monthly_history(
        float(selected_commune["latitude"]), float(selected_commune["longitude"]), int(history_years)
    )
history_monthly_df = _safe_df(history_payload.get("monthly"))
history_clim_df = _safe_df(history_payload.get("climatology"))
risk_level = str(snapshot.get("risk_level", "INDETERMINE"))
score = float(snapshot.get("score", 0.0) or 0.0)

hydro_exceeded_count = int(hydro_df["threshold_exceeded"].sum()) if (not hydro_df.empty and "threshold_exceeded" in hydro_df.columns) else 0
total_lgv_communes = int(len(lgv_communes_df)) if not lgv_communes_df.empty else int(len(commune_df))
ai_critical_count = int((filtered_sectors.get("ai_pred_risk_level", pd.Series(dtype=str)) == "CRITIQUE").sum()) if not filtered_sectors.empty else 0
fragile_soil_count = (
    int((pd.to_numeric(filtered_sectors.get("ai_soil_fragility", 0.0), errors="coerce").fillna(0.0) >= 0.70).sum())
    if not filtered_sectors.empty
    else 0
)

col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
col1.metric("Risque global", risk_level)
col2.metric("Score global", f"{score:.2f}/4")
col3.metric("Stations meteo", int(len(filtered_weather)))
col4.metric("Points LGV filtres", int(len(filtered_sectors)))
col5.metric("Communes traversees LGV", total_lgv_communes)
col6.metric("Hydro seuil urgence", hydro_exceeded_count)
col7.metric("Secteurs IA critiques", ai_critical_count)
col8.metric("Secteurs sols fragiles", fragile_soil_count)

tabs = st.tabs(["Vue executive", "Carte dynamique", "Tables et alertes", "Metadata"])

with tabs[0]:
    left, right = st.columns([1.5, 1.0])

    with left:
        st.subheader("Classement complet des communes (note GC)")
        if commune_df.empty:
            st.info("Aucune commune pour les filtres courants.")
        else:
            ranked_communes = commune_df.sort_values("note_gc", ascending=False).copy()
            chart = (
                alt.Chart(ranked_communes)
                .mark_bar()
                .encode(
                    x=alt.X("note_gc:Q", title="Note GC /100"),
                    y=alt.Y("commune_label:N", sort=alt.SortField(field="note_gc", order="descending"), title="Commune"),
                    color=alt.Color(
                        "commune_risk_level:N",
                        scale=alt.Scale(
                            domain=["FAIBLE", "MODERE", "ELEVE", "CRITIQUE"],
                            range=[RISK_COLOR["FAIBLE"], RISK_COLOR["MODERE"], RISK_COLOR["ELEVE"], RISK_COLOR["CRITIQUE"]],
                        ),
                        legend=alt.Legend(title="Risque global"),
                    ),
                    tooltip=[
                        "commune_label",
                        "note_gc",
                        "weather_component_note",
                        "geotech_component_note",
                        "piezo_component_note",
                        "hydro_component_note",
                        "ai_component_note",
                        "max_ai_probability",
                        "lgv_points_count",
                    ],
                )
            )
            st.altair_chart(chart, use_container_width=True)

    with right:
        st.subheader("Distribution du risque communal")
        if commune_df.empty:
            st.info("Pas de commune filtree.")
        else:
            dist = (
                commune_df["commune_risk_level"]
                .value_counts()
                .rename_axis("risk_level")
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            chart_dist = (
                alt.Chart(dist)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("count:Q"),
                    color=alt.Color(
                        "risk_level:N",
                        scale=alt.Scale(
                            domain=["FAIBLE", "MODERE", "ELEVE", "CRITIQUE", "INDETERMINE"],
                            range=[
                                RISK_COLOR["FAIBLE"],
                                RISK_COLOR["MODERE"],
                                RISK_COLOR["ELEVE"],
                                RISK_COLOR["CRITIQUE"],
                                RISK_COLOR["INDETERMINE"],
                            ],
                        ),
                    ),
                    tooltip=["risk_level", "count"],
                )
            )
            st.altair_chart(chart_dist, use_container_width=True)

        st.subheader("Synthese pluie")
        if filtered_weather.empty or rain_col_weather not in filtered_weather.columns:
            st.info("Pas de donnees pluie pour ce filtre.")
        else:
            max_rain = float(filtered_weather[rain_col_weather].max())
            mean_rain = float(filtered_weather[rain_col_weather].mean())
            st.metric(f"Max {period_label}", f"{max_rain:.1f} mm")
            st.metric(f"Moyenne {period_label}", f"{mean_rain:.1f} mm")

    st.subheader("Composantes de risque par commune + note GC globale")
    if commune_df.empty:
        st.info("Pas de donnees composantes a afficher.")
    else:
        component_map = {
            "weather_component_note": "Risque pluie",
            "geotech_component_note": "Risque geotechnique",
            "piezo_component_note": "Risque nappes (piezo)",
            "hydro_component_note": "Risque hydro",
            "ai_component_note": "Risque IA pluie+sol",
            "note_gc": "Note GC globale",
        }
        comp_cols = ["commune_label"] + list(component_map.keys())
        comp_long = (
            commune_df[comp_cols]
            .melt(
                id_vars=["commune_label"],
                value_vars=list(component_map.keys()),
                var_name="component_key",
                value_name="component_note",
            )
            .assign(component_label=lambda d: d["component_key"].map(component_map))
        )
        comp_long = comp_long.merge(commune_df[["commune_label", "note_gc"]], on="commune_label", how="left")
        component_order = [
            "Risque pluie",
            "Risque geotechnique",
            "Risque nappes (piezo)",
            "Risque hydro",
            "Risque IA pluie+sol",
            "Note GC globale",
        ]
        heatmap = (
            alt.Chart(comp_long)
            .mark_rect()
            .encode(
                x=alt.X("component_label:N", sort=component_order, title="Composante"),
                y=alt.Y("commune_label:N", sort=alt.SortField(field="note_gc", order="descending"), title="Commune"),
                color=alt.Color(
                    "component_note:Q",
                    title="Note /100",
                    scale=alt.Scale(scheme="redyellowgreen", reverse=True),
                ),
                tooltip=[
                    "commune_label",
                    "component_label",
                    alt.Tooltip("component_note:Q", title="Note composante", format=".1f"),
                    alt.Tooltip("note_gc:Q", title="Note GC globale", format=".1f"),
                ],
            )
        )
        st.altair_chart(heatmap, use_container_width=True)

    st.subheader("Predictions IA par secteur (pluie + type de sol)")
    if filtered_sectors.empty:
        st.info("Pas de secteur filtre pour le modele IA.")
    else:
        ai_view = filtered_sectors.copy()
        ai_view["ai_pred_probability"] = pd.to_numeric(ai_view.get("ai_pred_probability", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        ai_view["ai_probability_pct"] = (ai_view["ai_pred_probability"] * 100.0).round(1)
        if "ai_pred_risk_level" not in ai_view.columns:
            ai_view["ai_pred_risk_level"] = ai_view["ai_pred_probability"].map(lambda v: _ai_level_from_probability(float(v or 0.0)))
        ai_chart_df = ai_view.sort_values("ai_pred_probability", ascending=False).head(40).copy()
        ai_chart_df["sector_label"] = ai_chart_df["sector_id"].astype(str) + " - " + ai_chart_df["commune_name"].astype(str)
        ai_chart = (
            alt.Chart(ai_chart_df)
            .mark_bar()
            .encode(
                x=alt.X("ai_probability_pct:Q", title="Probabilite IA (%)"),
                y=alt.Y("sector_label:N", sort="-x", title="Secteur"),
                color=alt.Color(
                    "ai_pred_risk_level:N",
                    scale=alt.Scale(
                        domain=["FAIBLE", "MODERE", "ELEVE", "CRITIQUE"],
                        range=[RISK_COLOR["FAIBLE"], RISK_COLOR["MODERE"], RISK_COLOR["ELEVE"], RISK_COLOR["CRITIQUE"]],
                    ),
                    legend=alt.Legend(title="Risque IA"),
                ),
                tooltip=[
                    "sector_id",
                    "commune_name",
                    "ai_pred_risk_level",
                    "ai_probability_pct",
                    "ai_soil_fragility",
                    "ai_dominant_pedology",
                    "ai_dominant_soil_type",
                    "weather_max_24h_mm",
                    "weather_max_7d_mm",
                    "weather_max_30d_mm",
                ],
            )
        )
        st.altair_chart(ai_chart, use_container_width=True)

    st.subheader("Profil pedologique LGV (points geotechniques)")
    if geotech_df.empty or "pedology_family" not in geotech_df.columns:
        st.info("Pedologie indisponible dans ce snapshot.")
    else:
        pedo = (
            geotech_df["pedology_family"]
            .fillna("Pedologie indeterminee")
            .astype(str)
            .value_counts()
            .rename_axis("pedology_family")
            .reset_index(name="count")
        )
        pedo_chart = (
            alt.Chart(pedo)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Nombre de points"),
                y=alt.Y("pedology_family:N", sort="-x", title="Famille pedologique"),
                tooltip=["pedology_family", "count"],
            )
        )
        st.altair_chart(pedo_chart, use_container_width=True)

    st.subheader("Top stations meteo")
    if filtered_weather.empty or rain_col_weather not in filtered_weather.columns:
        st.info("Pas de station meteo pour ce filtre.")
    else:
        top_stations = filtered_weather.sort_values(rain_col_weather, ascending=False).head(20).copy()
        top_stations["station_display"] = top_stations["station_id"].astype(str) + " (" + top_stations["source"].astype(str) + ")"
        chart_st = (
            alt.Chart(top_stations)
            .mark_bar()
            .encode(
                x=alt.X(f"{rain_col_weather}:Q", title=f"Cumul {period_label} (mm)"),
                y=alt.Y("station_display:N", sort="-x", title="Station"),
                color=alt.Color("risk_level:N", scale=alt.Scale(domain=list(RISK_COLOR.keys()), range=list(RISK_COLOR.values()))),
                tooltip=[
                    "station_id",
                    "source",
                    "station_commune_name",
                    rain_col_weather,
                    "distance_to_lgv_km",
                    "risk_level",
                    "date_obs_raw",
                ],
            )
        )
        st.altair_chart(chart_st, use_container_width=True)

    st.subheader("Comparaison pluvio entre communes (5 dernieres annees)")
    if compare_history_df.empty:
        st.info("Selectionne des communes avec historique disponible pour comparer les pluies mensuelles.")
    else:
        hist_line = (
            alt.Chart(compare_history_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("ym:N", sort=None, title="Mois"),
                y=alt.Y("monthly_precip_mm:Q", title="Pluie mensuelle (mm)"),
                color=alt.Color("commune_label:N", title="Commune"),
                tooltip=["ym", "commune_label", "monthly_precip_mm"],
            )
        )
        st.altair_chart(hist_line, use_container_width=True)

        month_compare = (
            compare_history_df.groupby(["commune_label", "month"], as_index=False)["monthly_precip_mm"]
            .mean()
            .rename(columns={"monthly_precip_mm": "mean_monthly_mm"})
        )
        month_labels = {
            1: "Jan",
            2: "Fev",
            3: "Mar",
            4: "Avr",
            5: "Mai",
            6: "Juin",
            7: "Juil",
            8: "Aou",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        month_compare["month_label"] = month_compare["month"].map(month_labels)
        month_order = ["Jan", "Fev", "Mar", "Avr", "Mai", "Juin", "Juil", "Aou", "Sep", "Oct", "Nov", "Dec"]
        climat_cmp = (
            alt.Chart(month_compare)
            .mark_bar()
            .encode(
                x=alt.X("month_label:N", sort=month_order, title="Mois"),
                y=alt.Y("mean_monthly_mm:Q", title="Moyenne mensuelle (mm)"),
                color=alt.Color("commune_label:N", title="Commune"),
                xOffset=alt.XOffset("commune_label:N"),
                tooltip=["commune_label", "month_label", "mean_monthly_mm"],
            )
        )
        st.altair_chart(climat_cmp, use_container_width=True)
        st.caption("Modele historique: " + ", ".join([f"{k}:{v}" for k, v in history_models.items() if v]))

        latest_rows: List[Dict[str, object]] = []
        if selected_compare_commune_labels:
            for label in selected_compare_commune_labels:
                hit = commune_pool[commune_pool["commune_label"].astype(str) == str(label)]
                if hit.empty:
                    continue
                com = hit.iloc[0].to_dict()
                nwx = _nearest_row(weather_df, float(com["latitude"]), float(com["longitude"]))
                latest_rows.append(
                    {
                        "commune": label,
                        "station_meteo": nwx.get("station_id"),
                        "commune_station": nwx.get("station_commune_name"),
                        "dist_station_km": nwx.get("_dist_km"),
                        "rain_24h_mm": nwx.get("rain_24h_mm"),
                        "rain_7d_mm": nwx.get("rain_7d_mm"),
                        "rain_30d_mm": nwx.get("rain_30d_mm"),
                        "date_obs": nwx.get("date_obs_raw"),
                    }
                )
        if latest_rows:
            st.markdown("**Dernieres mesures meteo par commune comparee**")
            st.dataframe(pd.DataFrame(latest_rows), use_container_width=True, hide_index=True)

    st.subheader("Analyse detaillee commune")
    if not selected_commune:
        st.info("Aucune commune disponible pour l'analyse detaillee.")
    else:
        commune_name = str(selected_commune.get("commune_name") or "Inconnue")
        commune_code = str(selected_commune.get("commune_code") or "N/A")

        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Commune", commune_name)
        s2.metric("Code INSEE", commune_code)
        s3.metric("Risque commune", str(selected_commune.get("commune_risk_level", "INDETERMINE")))
        s4.metric("Note GC globale", f"{float(selected_commune.get('note_gc', 0.0)):.1f}/100")
        s5.metric("Points LGV", int(float(selected_commune.get("lgv_points_count", 0) or 0)))
        s6.metric(
            "Risque IA commune",
            f"{selected_commune.get('ai_commune_risk_level', 'INDETERMINE')} ({float(selected_commune.get('max_ai_probability', 0.0) or 0.0) * 100.0:.0f}%)",
        )

        nearest_weather = _nearest_row(weather_df, float(selected_commune["latitude"]), float(selected_commune["longitude"]))
        nearest_hydro = _nearest_row(hydro_df, float(selected_commune["latitude"]), float(selected_commune["longitude"]))
        nearest_piezo = _nearest_row(piezo_df, float(selected_commune["latitude"]), float(selected_commune["longitude"]))

        cwx, chx, cpx = st.columns(3)
        with cwx:
            st.markdown("**Derniere mesure meteo proche**")
            if nearest_weather:
                st.write(
                    {
                        "station_id": nearest_weather.get("station_id"),
                        "source": nearest_weather.get("source"),
                        "commune_station": nearest_weather.get("station_commune_name"),
                        "distance_km": nearest_weather.get("_dist_km"),
                        "rain_24h_mm": nearest_weather.get("rain_24h_mm"),
                        "rain_7d_mm": nearest_weather.get("rain_7d_mm"),
                        "rain_30d_mm": nearest_weather.get("rain_30d_mm"),
                        "date_obs_raw": nearest_weather.get("date_obs_raw"),
                    }
                )
            else:
                st.info("Pas de mesure meteo proche.")
        with chx:
            st.markdown("**Derniere mesure hydro proche**")
            if nearest_hydro:
                st.write(
                    {
                        "station_code": nearest_hydro.get("station_code"),
                        "river_name": nearest_hydro.get("river_name"),
                        "distance_km": nearest_hydro.get("_dist_km"),
                        "last_level_m": nearest_hydro.get("last_level_m"),
                        "trend_mph": nearest_hydro.get("trend_mph"),
                        "last_obs_utc": nearest_hydro.get("last_obs_utc"),
                    }
                )
            else:
                st.info("Pas de mesure hydro proche.")
        with cpx:
            st.markdown("**Derniere mesure piezometre proche**")
            if nearest_piezo:
                st.write(
                    {
                        "code_bss": nearest_piezo.get("code_bss"),
                        "name": nearest_piezo.get("name"),
                        "distance_km": nearest_piezo.get("_dist_km"),
                        "depth_m": nearest_piezo.get("depth_m"),
                        "trend_depth_mpd": nearest_piezo.get("trend_depth_mpd"),
                        "last_date_utc": nearest_piezo.get("last_date_utc"),
                    }
                )
            else:
                st.info("Pas de piezometre proche.")

        st.markdown("**Historique meteo mensuel multi-annees (commune)**")
        if history_payload.get("error"):
            st.warning(f"Historique indisponible: {history_payload.get('error')}")
        elif history_monthly_df.empty:
            st.info("Pas d'historique mensuel disponible.")
        else:
            hist_chart = (
                alt.Chart(history_monthly_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("ym:N", sort=None, title="Mois"),
                    y=alt.Y("monthly_precip_mm:Q", title="Pluie mensuelle (mm)"),
                    color=alt.Color("year:N", title="Annee"),
                    tooltip=["ym", "year", "monthly_precip_mm"],
                )
            )
            st.altair_chart(hist_chart, use_container_width=True)

            if not history_clim_df.empty:
                clim_chart = (
                    alt.Chart(history_clim_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("month_label:N", title="Mois"),
                        y=alt.Y("climatology_mm:Q", title="Moyenne historique (mm/mois)"),
                        tooltip=["month_label", "climatology_mm"],
                    )
                )
                st.altair_chart(clim_chart, use_container_width=True)
            st.caption(f"Source historique: Open-Meteo archive ({history_payload.get('model')})")

with tabs[1]:
    st.subheader("Carte multi-couches")
    if commune_df.empty and filtered_weather.empty and filtered_sectors.empty and lgv_communes_df.empty:
        st.info("Pas de donnees cartographiques avec ces filtres.")
    else:
        m = _build_map(
            snapshot=snapshot,
            weather_df=filtered_weather,
            commune_df=commune_df,
            sectors_df=filtered_sectors,
            hydro_df=hydro_df,
            piezo_df=piezo_df,
            geotech_df=geotech_df,
            lgv_communes_df=lgv_communes_df,
            fr_communes_geojson=fr_communes_geojson,
            rain_col_weather=rain_col_weather,
            min_risk=min_risk,
            show_weather=show_weather,
            show_communes=show_communes,
            show_sectors=show_sectors,
            show_hydro=show_hydro,
            show_piezo=show_piezo,
            show_geotech=show_geotech,
            show_fr_layer=show_fr_layer,
        )
        st_folium(m, height=680, use_container_width=True)

with tabs[2]:
    st.subheader("Tableau communes")
    if commune_df.empty:
        st.info("Aucune commune disponible.")
    else:
        st.dataframe(
            commune_df[
                [
                    "commune_name",
                    "commune_code",
                    "departement_code",
                    "lgv_points_count",
                    "note_gc",
                    "commune_risk_level",
                    "ai_commune_risk_level",
                    "weather_component_note",
                    "geotech_component_note",
                    "piezo_component_note",
                    "hydro_component_note",
                    "ai_component_note",
                    "avg_ai_probability",
                    "max_ai_probability",
                    "avg_point_score",
                    "max_point_score",
                    "critical",
                    "high",
                    "moderate",
                    "ai_critical",
                    "ai_high",
                    "avg_rain_period_mm",
                    "max_rain_period_mm",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Points LGV filtres (detail)")
    if filtered_sectors.empty:
        st.info("Aucun point LGV filtre.")
    else:
        view_cols = [
            "sector_id",
            "commune_name",
            "risk_level",
            "score",
            commune_rain_col,
            "geotech_points",
            "piezometers",
            "hydro_stations",
            "ai_pred_risk_level",
            "ai_pred_probability",
            "ai_soil_fragility",
            "ai_dominant_pedology",
            "under_watch",
        ]
        present_cols = [c for c in view_cols if c in filtered_sectors.columns]
        sort_col = "ai_pred_probability" if "ai_pred_probability" in filtered_sectors.columns else "score"
        st.dataframe(filtered_sectors[present_cols].sort_values(sort_col, ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Stations meteo filtrees (commune/station)")
    if filtered_weather.empty:
        st.info("Aucune station meteo sur les filtres actifs.")
    else:
        wx_cols = [
            "station_id",
            "source",
            "station_commune_name",
            "distance_to_lgv_km",
            "rain_24h_mm",
            "rain_7d_mm",
            "rain_30d_mm",
            "rain_month_mm",
            "date_obs_raw",
        ]
        wx_cols = [c for c in wx_cols if c in filtered_weather.columns]
        st.dataframe(filtered_weather[wx_cols].sort_values("rain_24h_mm", ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Cours d'eau et ruisseaux - hauteurs et seuils d'urgence")
    if hydro_df.empty:
        st.info("Aucune station hydro reseau disponible.")
    else:
        hydro_view_cols = [
            "station_code",
            "station_name",
            "river_name",
            "distance_to_lgv_km",
            "last_level_m",
            "trend_mph",
            "watch_threshold_m",
            "emergency_threshold_m",
            "threshold_ratio",
            "threshold_exceeded",
            "risk_level",
            "last_obs_utc",
        ]
        hydro_view_cols = [c for c in hydro_view_cols if c in hydro_df.columns]
        hydro_view = hydro_df[hydro_view_cols].copy()
        if "threshold_ratio" in hydro_view.columns:
            hydro_view = hydro_view.sort_values("threshold_ratio", ascending=False, na_position="last")
        st.dataframe(hydro_view, use_container_width=True, hide_index=True)

    st.subheader("Communes traversees par la LGV SEA (liste exhaustive)")
    if lgv_communes_df.empty:
        st.info("Liste exhaustive des communes indisponible dans ce snapshot.")
    else:
        lgvc_cols = [
            "order_on_line",
            "commune_name",
            "commune_code",
            "departement_code",
            "departement_name",
            "pk_start_km",
            "pk_end_km",
            "traversed_km",
            "sample_count",
        ]
        lgvc_cols = [c for c in lgvc_cols if c in lgv_communes_df.columns]
        lgv_view = lgv_communes_df[lgvc_cols].copy()
        if "order_on_line" in lgv_view.columns:
            lgv_view = lgv_view.sort_values("order_on_line", na_position="last")
        st.dataframe(
            lgv_view,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Alertes actives")
    if alerts_df.empty:
        st.success("Aucune alerte active.")
    else:
        st.dataframe(alerts_df[["level", "type", "message"]], use_container_width=True, hide_index=True)

    recos = snapshot.get("recommendations", [])
    st.subheader("Recommandations")
    if isinstance(recos, list) and recos:
        for rec in recos:
            st.write(f"- {rec}")
    else:
        st.info("Pas de recommandation disponible.")

with tabs[3]:
    st.subheader("Resume technique")
    line_meta = metadata_obj.get("line_monitoring", {}) if isinstance(metadata_obj.get("line_monitoring"), dict) else {}
    st.write(
        {
            "line_name": line_meta.get("line_name", "LGV SEA"),
            "line_length_km": line_meta.get("line_length_km"),
            "communes_traversees": int(len(lgv_communes_df)) if not lgv_communes_df.empty else None,
            "risk_level_global": snapshot.get("risk_level"),
            "score_global": snapshot.get("score"),
        }
    )

    st.subheader("Methodes de calcul")
    methods_meta = metadata_obj.get("calculation_methods", {}) if isinstance(metadata_obj.get("calculation_methods"), dict) else {}
    if methods_meta:
        methods_df = pd.DataFrame(
            [{"bloc": str(k), "methode": str(v)} for k, v in methods_meta.items()]
        )
        st.dataframe(methods_df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune metadata de methode disponible.")

    st.subheader("Modele IA sectoriel")
    sectors_obj = snapshot.get("sectors") if isinstance(snapshot.get("sectors"), dict) else {}
    ai_model_meta = sectors_obj.get("ai_model") if isinstance(sectors_obj.get("ai_model"), dict) else {}
    if ai_model_meta:
        st.write(ai_model_meta)
    else:
        st.info("Metadata IA non disponible dans ce snapshot.")

    st.subheader("Sources de donnees")
    sources_meta = metadata_obj.get("sources", [])
    if isinstance(sources_meta, list) and sources_meta:
        src_df = pd.DataFrame(sources_meta)
        cols = [c for c in ["id", "label", "usage", "url"] if c in src_df.columns]
        st.dataframe(src_df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("Aucune liste de sources structuree disponible.")

    st.subheader("Frequence de mise a jour")
    update_meta = metadata_obj.get("update_frequency", {}) if isinstance(metadata_obj.get("update_frequency"), dict) else {}
    if update_meta:
        st.write(update_meta)
    else:
        st.info("Frequence de MAJ non renseignee dans le snapshot.")

    st.subheader("Couche geographique FR")
    fr_summary = {}
    if isinstance(fr_geo_obj.get("summary"), dict):
        fr_summary = fr_geo_obj.get("summary") or {}
    elif isinstance(lgv_communes_obj.get("summary"), dict):
        fr_summary = lgv_communes_obj.get("summary") or {}
    fr_feature_count = 0
    if isinstance(fr_communes_geojson, dict) and isinstance(fr_communes_geojson.get("features"), list):
        fr_feature_count = int(len(fr_communes_geojson.get("features", [])))
    st.write(
        {
            "geojson_communes_count": fr_feature_count,
            "departements_ref": lgv_communes_obj.get("departements_ref"),
            "summary": fr_summary,
        }
    )

    st.subheader("Sources pluviometres recommandees")
    pluviometer_candidates = metadata_obj.get("pluviometer_sources_candidates", [])
    if isinstance(pluviometer_candidates, list) and pluviometer_candidates:
        psrc_df = pd.DataFrame(pluviometer_candidates)
        cols = [c for c in ["label", "notes", "url"] if c in psrc_df.columns]
        st.dataframe(psrc_df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("Aucune recommandation pluviometre disponible.")

    st.subheader("Limites et hypotheses")
    st.markdown(
        """
        - Le chainage communal est **approximatif** (echantillonnage le long de la ligne).
        - La pedologie BRGM est issue d'une carte **1:1 000 000** (usage macro-territorial).
        - Les seuils hydro d'urgence dependent de la disponibilite des seuils publics par station.
        - Le modele IA pluie/sol est un outil d'aide a la priorisation, pas une decision autonome.
        - Les decisions travaux doivent etre confirmees par expertise terrain et inspection OA.
        """
    )

    st.subheader("Fraicheur des donnees")
    st.write(
        {
            "snapshot_timestamp_utc": snapshot.get("timestamp_utc"),
            "snapshot_source": snapshot_source,
            "weather_notice": snapshot.get("weather_notice"),
            "history_model": history_payload.get("model"),
            "history_models_compare": history_models,
            "ai_model": ai_model_meta,
            "fr_geojson_features": int(len(fr_communes_geojson.get("features", []))) if isinstance(fr_communes_geojson, dict) and isinstance(fr_communes_geojson.get("features"), list) else 0,
            "metadata_present": bool(metadata_obj),
        }
    )

ts = snapshot.get("timestamp_utc")
if ts:
    st.caption(f"Donnees snapshot: {ts}")
if snapshot_source:
    st.caption(f"Source snapshot: {snapshot_source}")
st.caption(f"Interface update: {datetime.now(timezone.utc).isoformat()}")
