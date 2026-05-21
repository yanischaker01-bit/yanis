from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import altair as alt
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium


SNAPSHOT_LATEST = Path("reports/streamlit_snapshot_latest.json")
SNAPSHOT_GLOB = "streamlit_snapshot_*.json"
REMOTE_SNAPSHOT_URLS = [
    "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json",
    "https://raw.githubusercontent.com/yanischaker01-bit/yanis/main/reports/streamlit_snapshot_latest.json",
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

WEATHER_SOURCE_RELIABILITY = {
    "SYNOP": 96.0,
    "METEOFRANCE": 93.0,
    "VIGICRUES": 88.0,
    "OPEN_METEO": 82.0,
}

WEATHER_ALERT_THRESHOLDS = {
    "FAIBLE": 25.0,
    "MODERE": 45.0,
    "ELEVE": 65.0,
    "CRITIQUE": 82.0,
}

WEATHER_OP_ACTIONS = {
    "FAIBLE": "Suivi normal (controle quotidien).",
    "MODERE": "Surveillance renforcee (controle terrain <24h).",
    "ELEVE": "Pre-alerte GC (controle terrain <12h).",
    "CRITIQUE": "Alerte urgence GC (controle immediat <2h).",
    "INDETERMINE": "Donnees insuffisantes (verification manuelle).",
}

SLIP_ALERT_THRESHOLDS = {
    "FAIBLE": 45.0,
    "MODERE": 60.0,
    "ELEVE": 75.0,
    "CRITIQUE": 88.0,
}

DEFAULT_MANUAL_PK_RANGES = [
    (98.244, 98.640),
    (119.590, 120.970),
    (102.700, 103.970),
    (104.700, 109.200),
    (114.890, 117.340),
    (2.500, 2.800),
    (54.770, 54.920),
    (2.500, 3.100),
    (20.090, 20.180),
    (80.370, 80.650),
    (216.700, 216.950),
]

MONTH_LABELS_SHORT = {
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
MONTH_ORDER_SHORT = [MONTH_LABELS_SHORT[m] for m in range(1, 13)]
MONTH_SHORT_TO_NUM = {v: k for k, v in MONTH_LABELS_SHORT.items()}

OBS_WINDOW_HOURS = {
    "Tout": None,
    "1 jour": 24.0,
    "7 jours": 24.0 * 7.0,
    "Mois courant": None,
}

WEATHER_SOURCE_FILTER_MODES = [
    "Toutes sources",
    "InfoClimat/SYNOP",
    "InfoClimat/SYNOP proches LGV SEA",
]

WEATHER_SOURCE_PROFILE_OPTIONS = [
    "Auto (recommande)",
    "Open-Meteo uniquement",
    "InfoClimat/SYNOP uniquement",
    "InfoClimat/SYNOP <= distance LGV",
]

DEFAULT_INFOCLIMAT_LGV_MAX_DISTANCE_KM = 15.0
DEFAULT_STRICT_RELIABLE_MODE = True

MAP_TILE_STYLES = {
    "Google Satellite": {
        "tiles": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attr": "Google",
    },
    "Google Hybrid": {
        "tiles": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        "attr": "Google",
    },
    "OpenStreetMap": {
        "tiles": "OpenStreetMap",
        "attr": "OpenStreetMap",
    },
    "CartoDB Positron": {
        "tiles": "CartoDB positron",
        "attr": "CartoDB",
    },
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


def _weather_level_from_index(index: float) -> str:
    val = float(index or 0.0)
    if val >= WEATHER_ALERT_THRESHOLDS["CRITIQUE"]:
        return "CRITIQUE"
    if val >= WEATHER_ALERT_THRESHOLDS["ELEVE"]:
        return "ELEVE"
    if val >= WEATHER_ALERT_THRESHOLDS["MODERE"]:
        return "MODERE"
    return "FAIBLE"


def _weather_freshness_level_from_age(hours: float) -> str:
    h = float(hours or 0.0)
    if h <= 6.0:
        return "TRES_RECENTE"
    if h <= 12.0:
        return "RECENTE"
    if h <= 24.0:
        return "VALIDE"
    if h <= 48.0:
        return "ANCIENNE"
    return "OBSOLETE"


def _weather_data_reliability_label(quality_note: float, obs_age_h: float) -> str:
    q = float(quality_note or 0.0)
    a = float(obs_age_h or 0.0)
    if q < 55.0 or a > 30.0:
        return "A_VERIFIER"
    if q < 70.0 or a > 18.0:
        return "SURVEILLER"
    return "OK"


def _weather_action_label(level: str, reliability: str) -> str:
    lvl = str(level or "INDETERMINE").upper()
    rel = str(reliability or "A_VERIFIER").upper()
    base = WEATHER_OP_ACTIONS.get(lvl, WEATHER_OP_ACTIONS["INDETERMINE"])
    if rel == "A_VERIFIER":
        return f"{base} Validation meteo + terrain obligatoire."
    if rel == "SURVEILLER":
        return f"{base} Controler la source meteo avant engagement."
    return base


def _source_reliability_note(source: str) -> float:
    src = str(source or "").strip().upper()
    if not src:
        return 70.0
    for key, note in WEATHER_SOURCE_RELIABILITY.items():
        if key in src:
            return float(note)
    return 75.0


def _to_non_negative_numeric(series: pd.Series | object, index: pd.Index) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if not isinstance(s, pd.Series):
        s = pd.Series(np.nan, index=index, dtype=float)
    else:
        s = s.reindex(index)
    return s.where(s >= 0.0)


def _to_float_or_none(value: object) -> float | None:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    return float(num)


def _format_mm(value: object, decimals: int = 1) -> str:
    num = _to_float_or_none(value)
    if num is None:
        return "N/A"
    d = max(0, int(decimals))
    return f"{num:.{d}f} mm"


def _normalize_weather_rain_columns(df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], pd.Series]:
    idx = df.index
    raw_24 = _to_non_negative_numeric(df["rain_24h_mm"] if "rain_24h_mm" in df.columns else pd.Series(np.nan, index=idx), idx)
    raw_7 = _to_non_negative_numeric(df["rain_7d_mm"] if "rain_7d_mm" in df.columns else pd.Series(np.nan, index=idx), idx)
    raw_30 = _to_non_negative_numeric(df["rain_30d_mm"] if "rain_30d_mm" in df.columns else pd.Series(np.nan, index=idx), idx)
    raw_month = _to_non_negative_numeric(df["rain_month_mm"] if "rain_month_mm" in df.columns else pd.Series(np.nan, index=idx), idx)
    raw_forecast = _to_non_negative_numeric(df["rain_forecast_mm"] if "rain_forecast_mm" in df.columns else pd.Series(np.nan, index=idx), idx)

    completeness_raw = pd.DataFrame(
        {
            "rain_24h_mm": raw_24,
            "rain_7d_mm": raw_7,
            "rain_30d_mm": raw_30,
            "rain_month_mm": raw_month,
            "rain_forecast_mm": raw_forecast,
        }
    )
    completeness_ratio = completeness_raw.notna().sum(axis=1) / float(len(completeness_raw.columns))

    rain_24 = raw_24.fillna(0.0)
    rain_7_base = raw_7.fillna(rain_24)
    rain_7 = pd.Series(np.maximum(rain_7_base.to_numpy(dtype=float), rain_24.to_numpy(dtype=float)), index=idx, dtype=float)
    rain_30_base = raw_30.fillna(rain_7)
    rain_30 = pd.Series(np.maximum(rain_30_base.to_numpy(dtype=float), rain_7.to_numpy(dtype=float)), index=idx, dtype=float)

    # Keep month as true calendar-month accumulation only; avoid silent 30d substitution.
    rain_month = raw_month.astype(float)
    rain_forecast = raw_forecast.fillna(rain_24)

    normalized = {
        "rain_24h_mm": rain_24.astype(float),
        "rain_7d_mm": rain_7.astype(float),
        "rain_30d_mm": rain_30.astype(float),
        "rain_month_mm": rain_month.astype(float),
        "rain_forecast_mm": rain_forecast.astype(float),
    }
    return normalized, completeness_ratio.astype(float)


def _snapshot_payload_timestamp(payload: Dict[str, object]) -> pd.Timestamp:
    if not isinstance(payload, dict):
        return pd.NaT
    return pd.to_datetime(payload.get("timestamp_utc"), utc=True, errors="coerce")


def _snapshot_age_hours(snapshot_ts: pd.Timestamp | None) -> float | None:
    if snapshot_ts is None or pd.isna(snapshot_ts):
        return None
    now_utc = pd.Timestamp.now(tz="UTC")
    age_h = (now_utc - snapshot_ts).total_seconds() / 3600.0
    return max(0.0, float(age_h))


def _http_get_with_retry(url: str, params: Dict[str, object], timeout: int = 30, max_attempts: int = 2) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            return requests.get(url, params=params, timeout=timeout)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(0.6)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("HTTP request failed")


def _build_weather_enhanced(weather_df: pd.DataFrame, snapshot_ts: pd.Timestamp | None) -> pd.DataFrame:
    if weather_df.empty:
        return weather_df.copy()

    df = weather_df.copy()
    raw_24_input = pd.to_numeric(df["rain_24h_mm"], errors="coerce") if "rain_24h_mm" in df.columns else pd.Series(np.nan, index=df.index)
    raw_7_input = pd.to_numeric(df["rain_7d_mm"], errors="coerce") if "rain_7d_mm" in df.columns else pd.Series(np.nan, index=df.index)
    raw_30_input = pd.to_numeric(df["rain_30d_mm"], errors="coerce") if "rain_30d_mm" in df.columns else pd.Series(np.nan, index=df.index)
    raw_month_input = pd.to_numeric(df["rain_month_mm"], errors="coerce") if "rain_month_mm" in df.columns else pd.Series(np.nan, index=df.index)
    rain_cols, completeness = _normalize_weather_rain_columns(df)
    for col, series in rain_cols.items():
        df[col] = series
    df["distance_to_lgv_km"] = pd.to_numeric(df.get("distance_to_lgv_km"), errors="coerce")

    ts_source = None
    if "date_obs_raw" in df.columns:
        ts_source = pd.to_datetime(df["date_obs_raw"], utc=True, errors="coerce")
    if ts_source is None or ts_source.isna().all():
        ts_source = pd.to_datetime(df.get("date"), utc=True, errors="coerce")
    df["obs_ts_utc"] = ts_source

    if snapshot_ts is None or pd.isna(snapshot_ts):
        snapshot_ts = pd.Timestamp.now(tz="UTC")
    delta_h = (snapshot_ts - df["obs_ts_utc"]).dt.total_seconds() / 3600.0
    df["obs_age_h"] = pd.to_numeric(delta_h, errors="coerce").fillna(48.0).clip(lower=0.0, upper=240.0)

    def _freshness_note(hours: float) -> float:
        h = float(hours or 0.0)
        if h <= 3.0:
            return 100.0
        if h <= 6.0:
            return 95.0
        if h <= 12.0:
            return 88.0
        if h <= 24.0:
            return 75.0
        if h <= 48.0:
            return 60.0
        return 45.0

    df["source_reliability_note"] = df.get("source", pd.Series("", index=df.index)).map(_source_reliability_note).astype(float)
    df["freshness_note"] = df["obs_age_h"].map(_freshness_note).astype(float)

    dist = pd.to_numeric(df.get("distance_to_lgv_km"), errors="coerce")
    df["proximity_note"] = ((1.0 - dist.fillna(2.5).clip(lower=0.0, upper=2.5) / 2.5) * 100.0).clip(lower=35.0, upper=100.0)

    df["completeness_note"] = (completeness * 100.0).clip(lower=20.0, upper=100.0)

    df["weather_quality_note"] = (
        df["source_reliability_note"] * 0.30
        + df["freshness_note"] * 0.35
        + df["completeness_note"] * 0.20
        + df["proximity_note"] * 0.15
    ).round(1)

    inconsistency_penalty = (
        (raw_7_input.notna() & raw_24_input.notna() & (raw_7_input + 0.10 < raw_24_input)).astype(float) * 8.0
        + (raw_30_input.notna() & raw_7_input.notna() & (raw_30_input + 0.10 < raw_7_input)).astype(float) * 8.0
        + (raw_month_input.notna() & raw_24_input.notna() & (raw_month_input + 0.10 < raw_24_input)).astype(float) * 6.0
    )
    extreme_penalty = (
        (raw_24_input.notna() & (raw_24_input > 350.0)).astype(float) * 18.0
        + (raw_7_input.notna() & (raw_7_input > 700.0)).astype(float) * 14.0
        + (raw_30_input.notna() & (raw_30_input > 1200.0)).astype(float) * 12.0
        + (raw_month_input.notna() & (raw_month_input > 1500.0)).astype(float) * 10.0
    )
    df["pluvio_quality_penalty"] = (inconsistency_penalty + extreme_penalty).clip(lower=0.0, upper=35.0)
    df["weather_quality_note"] = (df["weather_quality_note"] - df["pluvio_quality_penalty"]).clip(lower=0.0, upper=100.0).round(1)

    r24 = pd.to_numeric(df.get("rain_24h_mm"), errors="coerce").fillna(0.0).clip(lower=0.0)
    r7 = pd.to_numeric(df.get("rain_7d_mm"), errors="coerce").fillna(0.0).clip(lower=0.0)
    r30 = pd.to_numeric(df.get("rain_30d_mm"), errors="coerce").fillna(0.0).clip(lower=0.0)
    rf = pd.to_numeric(df.get("rain_forecast_mm"), errors="coerce").fillna(0.0).clip(lower=0.0)
    df["weather_alert_index"] = (
        (r24 / 90.0).clip(upper=1.6) * 40.0
        + (r7 / 140.0).clip(upper=1.5) * 30.0
        + (r30 / 260.0).clip(upper=1.5) * 20.0
        + (rf / 60.0).clip(upper=1.5) * 10.0
    ).clip(lower=0.0, upper=100.0).round(1)
    df["weather_alert_level"] = df["weather_alert_index"].map(_weather_level_from_index)

    base_level = df.get("risk_level", pd.Series("INDETERMINE", index=df.index)).fillna("INDETERMINE").astype(str)
    op_rank = np.maximum(
        base_level.map(lambda x: _risk_rank(str(x))).fillna(0).astype(int),
        df["weather_alert_level"].map(lambda x: _risk_rank(str(x))).fillna(0).astype(int),
    )
    rank_to_level = {0: "INDETERMINE", 1: "FAIBLE", 2: "MODERE", 3: "ELEVE", 4: "CRITIQUE"}
    df["meteo_operational_level"] = op_rank.map(lambda r: rank_to_level.get(int(r), "INDETERMINE"))

    quality_scale = (df["weather_quality_note"] / 100.0).clip(lower=0.30, upper=1.0)
    df["weather_watch_priority"] = (df["weather_alert_index"] * (0.55 + 0.45 * quality_scale)).round(1)

    def _quality_label(note: float) -> str:
        v = float(note or 0.0)
        if v >= 85.0:
            return "TRES_BONNE"
        if v >= 70.0:
            return "BONNE"
        if v >= 55.0:
            return "MOYENNE"
        return "A_VERIFIER"

    df["weather_quality_level"] = df["weather_quality_note"].map(_quality_label)
    df["obs_freshness_level"] = df["obs_age_h"].map(_weather_freshness_level_from_age)
    df["weather_data_reliability"] = [
        _weather_data_reliability_label(float(q), float(a))
        for q, a in zip(df["weather_quality_note"].tolist(), df["obs_age_h"].tolist())
    ]
    df["weather_action_label"] = [
        _weather_action_label(str(level), str(rel))
        for level, rel in zip(df["meteo_operational_level"].tolist(), df["weather_data_reliability"].tolist())
    ]
    return df


def _build_commune_weather_context(
    commune_rows: pd.DataFrame,
    weather_df: pd.DataFrame,
    radius_km: float = 12.0,
    min_points: int = 3,
) -> pd.DataFrame:
    if commune_rows.empty or weather_df.empty:
        return pd.DataFrame()

    weather = weather_df.copy()
    weather["latitude"] = pd.to_numeric(weather.get("latitude"), errors="coerce")
    weather["longitude"] = pd.to_numeric(weather.get("longitude"), errors="coerce")
    weather = weather.dropna(subset=["latitude", "longitude"])
    if weather.empty:
        return pd.DataFrame()

    w_lat = weather["latitude"].to_numpy(dtype=float)
    w_lon = weather["longitude"].to_numpy(dtype=float)
    w_r24 = pd.to_numeric(weather.get("rain_24h_mm"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w_r7 = pd.to_numeric(weather.get("rain_7d_mm"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w_r30 = pd.to_numeric(weather.get("rain_30d_mm"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w_rm = pd.to_numeric(weather.get("rain_month_mm"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w_rf = pd.to_numeric(weather.get("rain_forecast_mm"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w_quality = pd.to_numeric(weather.get("weather_quality_note"), errors="coerce").fillna(60.0).to_numpy(dtype=float)
    w_alert = pd.to_numeric(weather.get("weather_alert_index"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w_priority = pd.to_numeric(weather.get("weather_watch_priority"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w_age = pd.to_numeric(weather.get("obs_age_h"), errors="coerce").fillna(48.0).to_numpy(dtype=float)
    w_level = weather.get("meteo_operational_level", pd.Series("INDETERMINE", index=weather.index)).astype(str).to_numpy()

    out_rows: List[Dict[str, object]] = []
    for _, com in commune_rows.iterrows():
        label = str(com.get("commune_label") or "")
        lat = pd.to_numeric(com.get("latitude"), errors="coerce")
        lon = pd.to_numeric(com.get("longitude"), errors="coerce")
        if not label or pd.isna(lat) or pd.isna(lon):
            continue

        lat = float(lat)
        lon = float(lon)
        p1 = np.radians(lat)
        p2 = np.radians(w_lat)
        dlat = np.radians(w_lat - lat)
        dlon = np.radians(w_lon - lon)
        a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
        dist_km = 2.0 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        if dist_km.size == 0:
            continue

        idx = np.where(dist_km <= float(radius_km))[0]
        if idx.size < int(min_points):
            idx = np.argsort(dist_km)[: max(int(min_points), 1)]
        if idx.size == 0:
            continue

        sel_dist = dist_km[idx]
        weights = 1.0 / np.maximum(sel_dist, 0.5)
        weights = weights / np.maximum(weights.sum(), 1e-9)

        def _wavg(arr: np.ndarray) -> float:
            return float(np.sum(arr[idx] * weights))

        lvl_rank = [_risk_rank(str(x)) for x in w_level[idx]]
        max_rank = max(lvl_rank) if lvl_rank else 0
        lvl_map = {0: "INDETERMINE", 1: "FAIBLE", 2: "MODERE", 3: "ELEVE", 4: "CRITIQUE"}
        weather_level = lvl_map.get(int(max_rank), "INDETERMINE")

        quality_note = _wavg(w_quality)
        age_h = _wavg(w_age)
        reliability_flag = _weather_data_reliability_label(quality_note, age_h)
        obs_freshness = _weather_freshness_level_from_age(age_h)
        action_label = _weather_action_label(weather_level, reliability_flag)

        out_rows.append(
            {
                "commune_label": label,
                "weather_points_used": int(idx.size),
                "weather_mean_dist_km": round(float(np.mean(sel_dist)), 3),
                "weather_quality_note_commune": round(quality_note, 1),
                "weather_obs_age_h_commune": round(age_h, 1),
                "weather_alert_index_commune": round(_wavg(w_alert), 1),
                "weather_watch_priority_commune": round(_wavg(w_priority), 1),
                "weather_24h_commune_mm": round(_wavg(w_r24), 1),
                "weather_7d_commune_mm": round(_wavg(w_r7), 1),
                "weather_30d_commune_mm": round(_wavg(w_r30), 1),
                "weather_month_commune_mm": round(_wavg(w_rm), 1),
                "weather_forecast_commune_mm": round(_wavg(w_rf), 1),
                "weather_alert_level_commune": weather_level,
                "weather_reliability_flag": reliability_flag,
                "weather_obs_freshness_commune": obs_freshness,
                "weather_action_commune": action_label,
            }
        )

    return pd.DataFrame(out_rows)


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
        response = _http_get_with_retry(url, params={}, timeout=12, max_attempts=2)
        if response.status_code != 200 or not response.text.strip():
            return {}
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_snapshot_payload() -> Tuple[Dict[str, object], str]:
    candidates: List[Tuple[str, Dict[str, object]]] = []
    local_path = _find_snapshot()
    if local_path is not None:
        try:
            local_payload = _load_snapshot(str(local_path), local_path.stat().st_mtime)
            if local_payload:
                candidates.append((f"local:{local_path}", local_payload))
        except Exception:
            pass

    for url in REMOTE_SNAPSHOT_URLS:
        payload = _load_remote_snapshot(url)
        if payload:
            candidates.append((url, payload))

    if not candidates:
        return {}, ""

    def _candidate_rank(item: Tuple[str, Dict[str, object]]) -> Tuple[int, int, int]:
        source, payload = item
        ts = _snapshot_payload_timestamp(payload)
        has_ts = 0 if pd.isna(ts) else 1
        ts_val = int(ts.value) if has_ts else -1
        local_bonus = 1 if str(source).startswith("local:") else 0
        return has_ts, ts_val, local_bonus

    best_source, best_payload = max(candidates, key=_candidate_rank)
    return best_payload, best_source


def _safe_df(records: object) -> pd.DataFrame:
    if isinstance(records, list):
        try:
            return pd.DataFrame(records)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _prepare_weather_df(weather_df: pd.DataFrame, snapshot_ts: pd.Timestamp | None) -> pd.DataFrame:
    if weather_df.empty:
        return weather_df.copy()

    out = weather_df.copy()
    for col in ["rain_24h_mm", "rain_7d_mm", "rain_30d_mm", "rain_month_mm", "distance_to_lgv_km", "latitude", "longitude"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["risk_level"] = out.get("rain_class", "INDETERMINE")
    if "station_commune_name" not in out.columns:
        out["station_commune_name"] = "Inconnue"
    out["station_commune_name"] = out["station_commune_name"].fillna("Inconnue")
    out = _build_weather_enhanced(out, snapshot_ts)
    out = _deduplicate_weather_rows(out)
    out = _stabilize_weather_station_ids(out)
    return out


def _unique_text_values(values: List[object]) -> List[str]:
    out: List[str] = []
    seen = set()
    for val in values:
        txt = str(val).strip()
        if not txt or txt.lower() == "nan":
            continue
        if txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


def _multiselect_with_all(label: str, options: List[str], key: str) -> List[str]:
    clean_options = _unique_text_values(options)
    if not clean_options:
        st.multiselect(label, ["Tout"], default=["Tout"], key=key, disabled=True)
        return []
    ui_options = ["Tout"] + clean_options
    selected = st.multiselect(label, ui_options, default=["Tout"], key=key)
    if not selected or "Tout" in selected:
        return clean_options
    return [opt for opt in clean_options if opt in selected]


def _ensure_month_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "month" in out.columns:
        out["month"] = pd.to_numeric(out["month"], errors="coerce")
    elif "ym" in out.columns:
        out["month"] = pd.to_numeric(out["ym"].astype(str).str.slice(5, 7), errors="coerce")
    else:
        out["month"] = np.nan
    out["month"] = out["month"].fillna(0).astype(int)
    return out


def _filter_history_months(df: pd.DataFrame, selected_month_numbers: List[int]) -> pd.DataFrame:
    if df.empty or not selected_month_numbers:
        return df.copy()
    work = _ensure_month_column(df)
    month_set = set(int(m) for m in selected_month_numbers if int(m) in MONTH_LABELS_SHORT)
    if not month_set:
        return work
    return work[work["month"].isin(month_set)].copy()


def _build_monthly_climatology(monthly_df: pd.DataFrame) -> pd.DataFrame:
    if monthly_df.empty:
        return pd.DataFrame()
    work = _ensure_month_column(monthly_df)
    work["monthly_precip_mm"] = pd.to_numeric(work.get("monthly_precip_mm"), errors="coerce")
    work = work.dropna(subset=["month", "monthly_precip_mm"])
    if work.empty:
        return pd.DataFrame()
    out = (
        work.groupby("month", as_index=False)["monthly_precip_mm"]
        .mean()
        .rename(columns={"monthly_precip_mm": "climatology_mm"})
    )
    out["climatology_mm"] = pd.to_numeric(out["climatology_mm"], errors="coerce").round(1)
    out["month_label"] = out["month"].map(MONTH_LABELS_SHORT)
    return out.sort_values("month")


def _commune_short_label(label: object) -> str:
    txt = str(label or "").strip()
    if not txt:
        return "Inconnue"
    if " (" in txt:
        txt = txt.split(" (", 1)[0].strip()
    if len(txt) <= 22:
        return txt
    return txt[:19].rstrip() + "..."


def _normalize_text_for_match(value: object) -> str:
    txt = str(value or "").strip().lower()
    if not txt:
        return ""
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = re.sub(r"[^a-z0-9]+", " ", txt).strip()
    return txt


def _location_match_strength(commune_name: object, station_commune_name: object) -> float:
    a = _normalize_text_for_match(commune_name)
    b = _normalize_text_for_match(station_commune_name)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    tokens_a = {tok for tok in a.split(" ") if tok}
    tokens_b = {tok for tok in b.split(" ") if tok}
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a.intersection(tokens_b))
    if overlap <= 0:
        return 0.0
    return float(overlap) / float(max(len(tokens_a), len(tokens_b)))


def _prepare_monthly_history_chart_df(df: pd.DataFrame, default_commune_label: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "ym" not in out.columns:
        if "year" in out.columns and "month" in out.columns:
            yy = pd.to_numeric(out.get("year"), errors="coerce").fillna(0).astype(int).astype(str)
            mm = pd.to_numeric(out.get("month"), errors="coerce").fillna(0).astype(int).map(lambda v: f"{int(v):02d}")
            out["ym"] = yy + "-" + mm
        else:
            return pd.DataFrame()
    out["ym"] = out["ym"].astype(str)
    out["ym_date"] = pd.to_datetime(out["ym"] + "-01", utc=True, errors="coerce")
    out["monthly_precip_mm"] = pd.to_numeric(out.get("monthly_precip_mm"), errors="coerce")
    if "year" not in out.columns:
        out["year"] = pd.to_numeric(out["ym"].str.slice(0, 4), errors="coerce")
    else:
        out["year"] = pd.to_numeric(out.get("year"), errors="coerce")
    if "month" not in out.columns:
        out["month"] = pd.to_numeric(out["ym"].str.slice(5, 7), errors="coerce")
    else:
        out["month"] = pd.to_numeric(out.get("month"), errors="coerce")
    if "commune_label" not in out.columns:
        out["commune_label"] = str(default_commune_label or "Inconnue")
    out["commune_label"] = out["commune_label"].fillna(str(default_commune_label or "Inconnue")).astype(str)
    out = out.dropna(subset=["ym_date", "monthly_precip_mm", "year", "month"])
    if out.empty:
        return pd.DataFrame()
    out["monthly_precip_mm"] = out["monthly_precip_mm"].clip(lower=0.0)
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out["month_label"] = out["month"].map(MONTH_LABELS_SHORT)
    out["commune_short"] = out["commune_label"].map(_commune_short_label)
    return out.sort_values(["commune_label", "ym_date"]).reset_index(drop=True)


def _build_history_month_lookup(history_df: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    if history_df.empty:
        return {}, {}
    work = history_df.copy()
    if "commune_label" not in work.columns:
        return {}, {}
    work["commune_label"] = work["commune_label"].fillna("").astype(str)
    if "ym" not in work.columns:
        if "year" not in work.columns or "month" not in work.columns:
            return {}, {}
        yy = pd.to_numeric(work.get("year"), errors="coerce").fillna(0).astype(int).astype(str)
        mm = pd.to_numeric(work.get("month"), errors="coerce").fillna(0).astype(int).map(lambda v: f"{int(v):02d}")
        work["ym"] = yy + "-" + mm
    work["ym"] = work["ym"].astype(str)
    work["monthly_precip_mm"] = pd.to_numeric(work.get("monthly_precip_mm"), errors="coerce")
    work = work.dropna(subset=["monthly_precip_mm"])
    if work.empty:
        return {}, {}

    exact_map: Dict[Tuple[str, str], float] = {}
    for row in work[["commune_label", "ym", "monthly_precip_mm"]].itertuples(index=False):
        exact_map[(str(row.commune_label), str(row.ym))] = float(row.monthly_precip_mm)

    latest_map: Dict[str, float] = {}
    latest_rows = work.sort_values("ym").groupby("commune_label", as_index=False).tail(1)
    for row in latest_rows[["commune_label", "monthly_precip_mm"]].itertuples(index=False):
        latest_map[str(row.commune_label)] = float(row.monthly_precip_mm)

    return exact_map, latest_map


def _align_history_month_with_reference(history_df: pd.DataFrame, reference: Dict[str, object]) -> Tuple[pd.DataFrame, bool]:
    if history_df.empty or not isinstance(reference, dict):
        return history_df.copy(), False

    source_mode = str(reference.get("source_mode") or "").upper()
    source = reference.get("source")
    reliability = str(reference.get("weather_data_reliability") or "").upper()
    dist_km = pd.to_numeric(reference.get("distance_km"), errors="coerce")
    is_infoclimat_source = _is_infoclimat_station_source(source)
    if source_mode not in {"INFOCLIMAT_STATION", "INFOCLIMAT_NEAREST_FALLBACK"}:
        return history_df.copy(), False
    if not is_infoclimat_source:
        return history_df.copy(), False
    if reliability == "A_VERIFIER":
        return history_df.copy(), False
    if source_mode == "INFOCLIMAT_NEAREST_FALLBACK":
        if not pd.isna(dist_km) and float(dist_km) > float(DEFAULT_INFOCLIMAT_LGV_MAX_DISTANCE_KM):
            return history_df.copy(), False

    obs_date_raw = str(reference.get("obs_date") or "").strip()
    ref_month = pd.to_numeric(reference.get("rain_month_mm"), errors="coerce")
    if not obs_date_raw or pd.isna(ref_month):
        return history_df.copy(), False

    obs_ts = pd.to_datetime(obs_date_raw, utc=True, errors="coerce")
    if pd.isna(obs_ts):
        return history_df.copy(), False
    ym = f"{int(obs_ts.year):04d}-{int(obs_ts.month):02d}"

    out = history_df.copy()
    if "ym" not in out.columns:
        if "year" in out.columns and "month" in out.columns:
            yy = pd.to_numeric(out.get("year"), errors="coerce").fillna(0).astype(int).astype(str)
            mm = pd.to_numeric(out.get("month"), errors="coerce").fillna(0).astype(int).map(lambda v: f"{int(v):02d}")
            out["ym"] = yy + "-" + mm
        else:
            return out, False

    out["ym"] = out["ym"].astype(str)
    out["monthly_precip_mm"] = pd.to_numeric(out.get("monthly_precip_mm"), errors="coerce")
    month_mask = out["ym"] == ym
    if month_mask.any():
        hist_vals = pd.to_numeric(out.loc[month_mask, "monthly_precip_mm"], errors="coerce").dropna()
        if not hist_vals.empty and source_mode == "INFOCLIMAT_NEAREST_FALLBACK":
            hist_val = float(hist_vals.iloc[-1])
            if abs(float(ref_month) - hist_val) > 25.0:
                return out, False
        out.loc[month_mask, "monthly_precip_mm"] = float(ref_month)
        return out, True

    new_row: Dict[str, object] = {
        "ym": ym,
        "year": int(obs_ts.year),
        "month": int(obs_ts.month),
        "monthly_precip_mm": float(ref_month),
    }
    for col in ["commune_label", "commune_name", "history_imputed"]:
        if col in out.columns:
            if col == "history_imputed":
                new_row[col] = False
            else:
                vals = out[col].dropna().astype(str).tolist()
                new_row[col] = vals[0] if vals else ""
    out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)
    return out, True


def _is_infoclimat_station_source(source: object) -> bool:
    src = str(source or "").strip().lower()
    if not src:
        return False
    return ("infoclimat" in src) or ("synop" in src)


def _is_open_meteo_station_source(source: object) -> bool:
    src = str(source or "").strip().lower()
    if not src:
        return False
    return ("open" in src) and ("meteo" in src)


def _apply_weather_source_filter(
    weather_df: pd.DataFrame,
    source_mode: str,
    infoclimat_max_distance_km: float,
) -> pd.DataFrame:
    if weather_df.empty:
        return weather_df.copy()

    mode = str(source_mode or WEATHER_SOURCE_FILTER_MODES[0])
    if mode == "Toutes sources":
        return weather_df.copy()

    out = weather_df.copy()
    src_series = out.get("source", pd.Series("", index=out.index)).fillna("").astype(str)
    is_infoclimat = src_series.map(_is_infoclimat_station_source).fillna(False)

    if mode == "InfoClimat/SYNOP":
        return out[is_infoclimat].copy()

    if mode == "InfoClimat/SYNOP proches LGV SEA":
        dist = pd.to_numeric(out.get("distance_to_lgv_km"), errors="coerce")
        max_km = max(0.5, float(infoclimat_max_distance_km or DEFAULT_INFOCLIMAT_LGV_MAX_DISTANCE_KM))
        return out[is_infoclimat & dist.notna() & (dist <= max_km)].copy()

    return out


def _apply_weather_source_profile(
    weather_df: pd.DataFrame,
    profile: str,
    infoclimat_max_distance_km: float,
) -> pd.DataFrame:
    if weather_df.empty:
        return weather_df.copy()

    mode = str(profile or WEATHER_SOURCE_PROFILE_OPTIONS[0])
    out = weather_df.copy()
    src_series = out.get("source", pd.Series("", index=out.index)).fillna("").astype(str)
    src_lc = src_series.str.lower()
    is_infoclimat = src_series.map(_is_infoclimat_station_source).fillna(False)
    is_open_meteo = src_lc.str.contains("open") & src_lc.str.contains("meteo")

    if mode == "Open-Meteo uniquement":
        return out[is_open_meteo.fillna(False)].copy()
    if mode == "InfoClimat/SYNOP uniquement":
        return out[is_infoclimat].copy()
    if mode == "InfoClimat/SYNOP <= distance LGV":
        dist = pd.to_numeric(out.get("distance_to_lgv_km"), errors="coerce")
        max_km = max(0.5, float(infoclimat_max_distance_km or DEFAULT_INFOCLIMAT_LGV_MAX_DISTANCE_KM))
        return out[is_infoclimat & dist.notna() & (dist <= max_km)].copy()
    return out


def _apply_pluvio_quality_gate(weather_df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    if weather_df.empty or not enabled:
        return weather_df.copy()

    out = weather_df.copy()
    quality = pd.to_numeric(out.get("weather_quality_note"), errors="coerce").fillna(0.0)
    age_h = pd.to_numeric(out.get("obs_age_h"), errors="coerce").fillna(9999.0)
    rain_24h = pd.to_numeric(out.get("rain_24h_mm"), errors="coerce")
    rain_7d = pd.to_numeric(out.get("rain_7d_mm"), errors="coerce")
    rain_30d = pd.to_numeric(out.get("rain_30d_mm"), errors="coerce")
    rain_month = pd.to_numeric(out.get("rain_month_mm"), errors="coerce")

    coherence_ok = (
        (rain_7d.isna() | rain_24h.isna() | (rain_7d + 0.10 >= rain_24h))
        & (rain_30d.isna() | rain_7d.isna() | (rain_30d + 0.10 >= rain_7d))
        & (rain_month.isna() | rain_24h.isna() | (rain_month + 0.10 >= rain_24h))
    )
    extreme_ok = (
        (rain_24h.fillna(0.0) >= 0.0)
        & (rain_24h.fillna(0.0) <= 350.0)
        & (rain_7d.fillna(0.0) >= 0.0)
        & (rain_7d.fillna(0.0) <= 700.0)
        & (rain_30d.fillna(0.0) >= 0.0)
        & (rain_30d.fillna(0.0) <= 1200.0)
        & (rain_month.fillna(0.0) >= 0.0)
        & (rain_month.fillna(0.0) <= 1500.0)
    )

    strict_mask = (quality >= 50.0) & (age_h <= 96.0) & coherence_ok.fillna(False) & extreme_ok.fillna(False)
    strict_out = out[strict_mask].copy()
    if not strict_out.empty:
        return strict_out

    fallback_mask = (quality >= 40.0) & (age_h <= 144.0) & extreme_ok.fillna(False)
    fallback_out = out[fallback_mask].copy()
    return fallback_out if not fallback_out.empty else out


def _select_weather_pool_for_ui(
    filtered_weather: pd.DataFrame,
    weather_for_context: pd.DataFrame,
    weather_df: pd.DataFrame,
    strict_reliable_mode: bool,
) -> pd.DataFrame:
    if strict_reliable_mode:
        if not filtered_weather.empty:
            return filtered_weather.copy()
        return weather_for_context.copy()

    if not filtered_weather.empty:
        return filtered_weather.copy()
    if not weather_for_context.empty:
        return weather_for_context.copy()
    return weather_df.copy()


def _deduplicate_weather_rows(weather_df: pd.DataFrame) -> pd.DataFrame:
    if weather_df.empty:
        return weather_df.copy()

    out = weather_df.copy()
    source_norm = out.get("source", pd.Series("", index=out.index)).fillna("").astype(str).str.strip().str.lower()
    lat_round = pd.to_numeric(out.get("latitude"), errors="coerce").round(5)
    lon_round = pd.to_numeric(out.get("longitude"), errors="coerce").round(5)

    obs_ts = pd.to_datetime(out.get("obs_ts_utc"), utc=True, errors="coerce")
    if obs_ts.isna().all():
        obs_ts = pd.to_datetime(out.get("date_obs_raw"), utc=True, errors="coerce")
    obs_key = obs_ts.dt.strftime("%Y-%m-%dT%H:%M")
    station_id = out.get("station_id", pd.Series("", index=out.index)).fillna("").astype(str)

    lat_key = lat_round.map(lambda v: f"{v:.5f}" if pd.notna(v) else "nan")
    lon_key = lon_round.map(lambda v: f"{v:.5f}" if pd.notna(v) else "nan")
    geo_key = source_norm + "|" + lat_key + "|" + lon_key + "|" + obs_key.fillna("")
    geo_key = geo_key.where(~(lat_round.isna() | lon_round.isna() | obs_key.isna()), source_norm + "|sid|" + station_id)

    out["_geo_dedup_key"] = geo_key
    out["_sort_quality"] = pd.to_numeric(out.get("weather_quality_note"), errors="coerce").fillna(0.0)
    out["_sort_alert"] = pd.to_numeric(out.get("weather_alert_index"), errors="coerce").fillna(0.0)
    out["_sort_age"] = pd.to_numeric(out.get("obs_age_h"), errors="coerce").fillna(9999.0)
    out["_sort_dist"] = pd.to_numeric(out.get("distance_to_lgv_km"), errors="coerce").fillna(9999.0)

    out = out.sort_values(
        ["_sort_quality", "_sort_alert", "_sort_age", "_sort_dist"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    out = out.drop_duplicates(subset=["_geo_dedup_key"], keep="first").copy()
    out = out.drop(columns=["_geo_dedup_key", "_sort_quality", "_sort_alert", "_sort_age", "_sort_dist"], errors="ignore")
    return out


def _stabilize_weather_station_ids(weather_df: pd.DataFrame) -> pd.DataFrame:
    if weather_df.empty:
        return weather_df.copy()

    out = weather_df.copy()
    if "station_id" not in out.columns:
        out["station_id"] = "station_inconnue"
    if "station_id_raw" not in out.columns:
        out["station_id_raw"] = out["station_id"]

    station_id = out["station_id"].fillna("").astype(str).str.strip()
    station_id = station_id.replace("", "station_inconnue")
    dup_mask = station_id.duplicated(keep=False)
    if dup_mask.any():
        lat = pd.to_numeric(out.get("latitude"), errors="coerce").round(4)
        lon = pd.to_numeric(out.get("longitude"), errors="coerce").round(4)
        suffix = (
            lat.map(lambda v: f"{v:.4f}" if pd.notna(v) else "na")
            + ","
            + lon.map(lambda v: f"{v:.4f}" if pd.notna(v) else "na")
        )
        station_id = station_id.where(~dup_mask, station_id + "@" + suffix)
    out["station_id"] = station_id
    return out


def _apply_weather_obs_window(
    weather_df: pd.DataFrame,
    window_label: str,
    snapshot_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    if weather_df.empty:
        return weather_df.copy()
    label = str(window_label or "Tout")
    if label == "Tout":
        return weather_df.copy()

    ref_ts = snapshot_ts if snapshot_ts is not None and not pd.isna(snapshot_ts) else pd.Timestamp.now(tz="UTC")
    obs_ts = pd.to_datetime(weather_df.get("obs_ts_utc"), utc=True, errors="coerce")
    if label == "Mois courant":
        mask = (obs_ts.dt.year == int(ref_ts.year)) & (obs_ts.dt.month == int(ref_ts.month))
    else:
        max_hours = OBS_WINDOW_HOURS.get(label)
        if max_hours is None:
            return weather_df.copy()
        age_h = (ref_ts - obs_ts).dt.total_seconds() / 3600.0
        mask = pd.to_numeric(age_h, errors="coerce").fillna(9999.0) <= float(max_hours)
    return weather_df[mask.fillna(False)].copy()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _choose_weather_signal_column(df: pd.DataFrame, preferred: str, fallbacks: List[str]) -> str:
    if df.empty:
        return preferred

    ordered: List[str] = []
    for col in [preferred] + fallbacks:
        if col not in ordered:
            ordered.append(col)

    def _signal_score(col: str) -> Tuple[int, float]:
        if col not in df.columns:
            return (-1, -1.0)
        s = pd.to_numeric(df[col], errors="coerce")
        if not s.notna().any():
            return (0, -1.0)
        non_zero = int((s.fillna(0.0) > 0.0).sum())
        max_val = float(s.max(skipna=True))
        return (non_zero, max_val)

    pref_non_zero, pref_max = _signal_score(preferred)
    if pref_non_zero > 0 and pref_max > 0.0:
        return preferred

    best_col = preferred
    best_score = (pref_non_zero, pref_max)
    for col in ordered[1:]:
        score = _signal_score(col)
        if score > best_score:
            best_col = col
            best_score = score
    return best_col


def _slip_level_from_index(index: float) -> str:
    val = float(index or 0.0)
    if val >= SLIP_ALERT_THRESHOLDS["CRITIQUE"]:
        return "CRITIQUE"
    if val >= SLIP_ALERT_THRESHOLDS["ELEVE"]:
        return "ELEVE"
    if val >= SLIP_ALERT_THRESHOLDS["MODERE"]:
        return "MODERE"
    return "FAIBLE"


def _parse_manual_pk_ranges(raw_text: str) -> List[Tuple[float, float]]:
    if not str(raw_text or "").strip():
        return []

    ranges: List[Tuple[float, float]] = []
    seen = set()
    chunks = re.split(r"[;\n]+", str(raw_text))
    for chunk in chunks:
        numbers = re.findall(r"-?\d+(?:[.,]\d+)?", chunk)
        if len(numbers) < 2:
            continue
        try:
            start = float(numbers[0].replace(",", "."))
            end = float(numbers[1].replace(",", "."))
        except ValueError:
            continue
        a, b = (start, end) if start <= end else (end, start)
        key = (round(a, 3), round(b, 3))
        if key in seen:
            continue
        seen.add(key)
        ranges.append((float(key[0]), float(key[1])))
    return ranges


def _build_slip_assessment(
    sectors_df: pd.DataFrame,
    manual_pk_ranges: List[Tuple[float, float]],
) -> pd.DataFrame:
    if sectors_df.empty:
        return sectors_df.copy()

    work = sectors_df.copy()
    for col, default in [
        ("pk_km", np.nan),
        ("score", 0.0),
        ("ai_pred_probability", 0.0),
        ("ai_soil_fragility", 0.0),
        ("weather_max_24h_mm", 0.0),
        ("weather_max_7d_mm", 0.0),
        ("weather_max_30d_mm", 0.0),
        ("hydro_stations", 0.0),
        ("geotech_points", 0.0),
        ("piezometers", 0.0),
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default)
        else:
            work[col] = default

    ai_note = (work["ai_pred_probability"].clip(lower=0.0, upper=1.0) * 100.0).clip(lower=0.0, upper=100.0)
    soil_note = (work["ai_soil_fragility"].clip(lower=0.0, upper=1.0) * 100.0).clip(lower=0.0, upper=100.0)
    weather_note = (
        (work["weather_max_24h_mm"] / 80.0).clip(upper=1.5) * 30.0
        + (work["weather_max_7d_mm"] / 120.0).clip(upper=1.5) * 35.0
        + (work["weather_max_30d_mm"] / 240.0).clip(upper=1.5) * 35.0
    ).clip(lower=0.0, upper=100.0)
    hydro_note = (work["hydro_stations"].clip(lower=0.0, upper=5.0) / 5.0 * 100.0).clip(lower=0.0, upper=100.0)
    geotech_note = (work["geotech_points"].clip(lower=0.0, upper=4.0) / 4.0 * 100.0).clip(lower=0.0, upper=100.0)
    piezo_note = (work["piezometers"].clip(lower=0.0, upper=2.0) / 2.0 * 100.0).clip(lower=0.0, upper=100.0)
    score_note = (work["score"].clip(lower=0.0, upper=4.0) / 4.0 * 100.0).clip(lower=0.0, upper=100.0)

    work["slip_ai_note"] = ai_note.round(1)
    work["slip_soil_note"] = soil_note.round(1)
    work["slip_weather_note"] = weather_note.round(1)
    work["slip_hydro_note"] = hydro_note.round(1)
    work["slip_geotech_note"] = geotech_note.round(1)
    work["slip_piezo_note"] = piezo_note.round(1)
    work["slip_score_note"] = score_note.round(1)

    slip_index = (
        ai_note * 0.30
        + soil_note * 0.20
        + weather_note * 0.22
        + hydro_note * 0.12
        + geotech_note * 0.10
        + piezo_note * 0.04
        + score_note * 0.02
    ).clip(lower=0.0, upper=100.0)

    manual_watch = pd.Series(False, index=work.index)
    if manual_pk_ranges:
        pk_vals = pd.to_numeric(work["pk_km"], errors="coerce")
        for start, end in manual_pk_ranges:
            manual_watch = manual_watch | ((pk_vals >= float(start)) & (pk_vals <= float(end)))
    work["manual_watch_pk"] = manual_watch.fillna(False).astype(bool)

    work["slip_index"] = np.where(
        work["manual_watch_pk"],
        np.minimum(100.0, slip_index + 6.0),
        slip_index,
    ).round(1)
    work["slip_level"] = work["slip_index"].map(_slip_level_from_index)

    def _top_drivers(row: pd.Series) -> str:
        drivers = [
            ("IA", float(row.get("slip_ai_note", 0.0))),
            ("Sol", float(row.get("slip_soil_note", 0.0))),
            ("Pluie", float(row.get("slip_weather_note", 0.0))),
            ("Hydro", float(row.get("slip_hydro_note", 0.0))),
            ("Geotech", float(row.get("slip_geotech_note", 0.0))),
            ("Piezo", float(row.get("slip_piezo_note", 0.0))),
        ]
        drivers = sorted(drivers, key=lambda x: x[1], reverse=True)
        return " | ".join([f"{label}:{value:.0f}" for label, value in drivers[:3]])

    work["slip_drivers"] = work.apply(_top_drivers, axis=1)
    return work


def _build_slip_corridors(
    sectors_df: pd.DataFrame,
    alert_threshold: float,
) -> pd.DataFrame:
    if sectors_df.empty or "pk_km" not in sectors_df.columns or "slip_index" not in sectors_df.columns:
        return pd.DataFrame()

    work = sectors_df.copy()
    if "sector_id" not in work.columns:
        work["sector_id"] = [f"S{i+1}" for i in range(len(work))]
    if "commune_name" not in work.columns:
        work["commune_name"] = "Inconnue"
    for col, default in [("ai_pred_probability", 0.0), ("weather_max_30d_mm", 0.0), ("ai_soil_fragility", 0.0)]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default)
        else:
            work[col] = default
    work["pk_km"] = pd.to_numeric(work.get("pk_km"), errors="coerce")
    work["slip_index"] = pd.to_numeric(work.get("slip_index"), errors="coerce")
    work = work.dropna(subset=["pk_km", "slip_index"]).sort_values("pk_km").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    focus = work[
        (work["slip_index"] >= float(alert_threshold))
        | (work.get("manual_watch_pk", pd.Series(False, index=work.index)).fillna(False).astype(bool))
    ].copy()
    if focus.empty:
        return pd.DataFrame()

    step = pd.to_numeric(work["pk_km"].diff(), errors="coerce").dropna()
    median_step = float(step[step > 0].median()) if not step.empty and (step > 0).any() else 5.0
    max_gap = max(2.0, median_step * 1.35)

    focus = focus.sort_values("pk_km").reset_index(drop=True)
    corridor_ids: List[int] = []
    cid = 1
    prev_pk = None
    for _, row in focus.iterrows():
        pk = float(row["pk_km"])
        if prev_pk is not None and (pk - prev_pk) > max_gap:
            cid += 1
        corridor_ids.append(cid)
        prev_pk = pk
    focus["slip_corridor_raw_id"] = corridor_ids
    focus["slip_corridor_id"] = focus["slip_corridor_raw_id"].map(lambda x: f"GLI-{int(x):03d}")

    grouped = (
        focus.groupby(["slip_corridor_raw_id", "slip_corridor_id"], as_index=False)
        .agg(
            pk_start_km=("pk_km", "min"),
            pk_end_km=("pk_km", "max"),
            sector_count=("sector_id", "count"),
            slip_index_max=("slip_index", "max"),
            slip_index_mean=("slip_index", "mean"),
            manual_watch_count=("manual_watch_pk", "sum"),
            critical_count=("slip_level", lambda s: int((s.astype(str) == "CRITIQUE").sum())),
            commune_dominante=("commune_name", lambda s: str(s.mode().iloc[0]) if not s.mode().empty else str(s.iloc[0])),
            max_ai_probability=("ai_pred_probability", "max"),
            max_weather_30d_mm=("weather_max_30d_mm", "max"),
            max_soil_fragility=("ai_soil_fragility", "max"),
        )
    )
    grouped["corridor_length_km"] = (grouped["pk_end_km"] - grouped["pk_start_km"]).clip(lower=0.0).round(2)
    grouped["slip_index_max"] = pd.to_numeric(grouped["slip_index_max"], errors="coerce").round(1)
    grouped["slip_index_mean"] = pd.to_numeric(grouped["slip_index_mean"], errors="coerce").round(1)
    grouped["slip_level"] = grouped["slip_index_max"].map(_slip_level_from_index)
    grouped["max_ai_probability"] = pd.to_numeric(grouped["max_ai_probability"], errors="coerce").round(3)
    grouped["max_weather_30d_mm"] = pd.to_numeric(grouped["max_weather_30d_mm"], errors="coerce").round(1)
    grouped["max_soil_fragility"] = pd.to_numeric(grouped["max_soil_fragility"], errors="coerce").round(3)
    return grouped.sort_values(["slip_index_max", "corridor_length_km"], ascending=[False, False]).reset_index(drop=True)


def _select_best_weather_row(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    commune_name: object = "",
) -> Dict[str, object]:
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return {}
    work = df.copy()
    lat_series = pd.to_numeric(work["latitude"], errors="coerce")
    lon_series = pd.to_numeric(work["longitude"], errors="coerce")
    valid_mask = lat_series.notna() & lon_series.notna()
    if not valid_mask.any():
        return {}
    work = work.loc[valid_mask].reset_index(drop=True)
    lat_vals = lat_series.loc[valid_mask].astype(float).to_numpy()
    lon_vals = lon_series.loc[valid_mask].astype(float).to_numpy()

    r = 6371.0
    p1 = np.radians(float(lat))
    p2 = np.radians(lat_vals)
    dlat = np.radians(lat_vals - float(lat))
    dlon = np.radians(lon_vals - float(lon))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    dist_km = 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    if dist_km.size == 0:
        return {}

    work["_dist_km"] = pd.Series(dist_km, index=work.index, dtype=float)
    quality = pd.to_numeric(work.get("weather_quality_note"), errors="coerce").fillna(55.0)
    obs_age = pd.to_numeric(work.get("obs_age_h"), errors="coerce").fillna(120.0)
    src = work.get("source", pd.Series("", index=work.index)).fillna("").astype(str)
    infoclimat_flag = src.map(_is_infoclimat_station_source).astype(int)
    infoclimat_bonus = np.where(
        infoclimat_flag == 1,
        np.where(work["_dist_km"] <= 10.0, 180.0, np.where(work["_dist_km"] <= 20.0, 135.0, 70.0)),
        0.0,
    )
    reliability = work.get("weather_data_reliability", pd.Series("A_VERIFIER", index=work.index)).fillna("A_VERIFIER").astype(str)
    reliability_bonus = reliability.map({"OK": 20.0, "SURVEILLER": 8.0, "A_VERIFIER": -12.0}).fillna(0.0)
    station_commune = work.get("station_commune_name", pd.Series("", index=work.index)).fillna("").astype(str)
    loc_match = station_commune.map(lambda s: _location_match_strength(commune_name, s))

    work["_selection_score"] = (
        infoclimat_bonus
        + quality * 1.2
        + reliability_bonus
        + loc_match * 45.0
        - work["_dist_km"] * 3.2
        - obs_age * 0.22
    )
    work = work.sort_values(["_selection_score", "_dist_km"], ascending=[False, True], kind="mergesort")
    row = work.iloc[0].to_dict()
    row["_dist_km"] = round(float(row.get("_dist_km", np.nan)), 3) if pd.notna(pd.to_numeric(row.get("_dist_km"), errors="coerce")) else np.nan
    return row


def _nearest_row(df: pd.DataFrame, lat: float, lon: float) -> Dict[str, object]:
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return {}
    work = df.copy()
    lat_series = pd.to_numeric(work["latitude"], errors="coerce")
    lon_series = pd.to_numeric(work["longitude"], errors="coerce")
    valid_mask = lat_series.notna() & lon_series.notna()
    if not valid_mask.any():
        return {}
    work = work.loc[valid_mask].reset_index(drop=True)
    lat_vals = lat_series.loc[valid_mask].astype(float).to_numpy()
    lon_vals = lon_series.loc[valid_mask].astype(float).to_numpy()

    r = 6371.0
    p1 = np.radians(float(lat))
    p2 = np.radians(lat_vals)
    dlat = np.radians(lat_vals - float(lat))
    dlon = np.radians(lon_vals - float(lon))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    dist_km = 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    if dist_km.size == 0:
        return {}
    best_idx = int(np.nanargmin(dist_km))
    row = work.iloc[best_idx].to_dict()
    row["_dist_km"] = round(float(dist_km[best_idx]), 3)
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
        response = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30, max_attempts=2)
        if response.status_code != 200:
            fallback_params = dict(params)
            fallback_params.pop("models", None)
            response = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=fallback_params, timeout=30, max_attempts=2)
            used_model = "open_meteo_default"
        if response.status_code != 200 or not response.text.strip():
            return {
                "monthly": [],
                "climatology": [],
                "model": used_model,
                "latest_obs_date": None,
                "recent_rain_30d_mm": None,
                "recent_rain_month_mm": None,
                "error": f"HTTP {response.status_code}",
            }

        payload = response.json()
        entry = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(entry, dict):
            return {
                "monthly": [],
                "climatology": [],
                "model": used_model,
                "latest_obs_date": None,
                "recent_rain_30d_mm": None,
                "recent_rain_month_mm": None,
                "error": "payload invalide",
            }
        daily = entry.get("daily", {}) if isinstance(entry.get("daily"), dict) else {}
        times = daily.get("time", []) or []
        vals = daily.get("precipitation_sum", []) or []
        if not times or not vals:
            return {
                "monthly": [],
                "climatology": [],
                "model": used_model,
                "latest_obs_date": None,
                "recent_rain_30d_mm": None,
                "recent_rain_month_mm": None,
                "error": "serie vide",
            }

        df = pd.DataFrame({"date": pd.to_datetime(times, utc=True, errors="coerce"), "precip_mm": pd.to_numeric(vals, errors="coerce")})
        df = df.dropna(subset=["date", "precip_mm"])
        if df.empty:
            return {
                "monthly": [],
                "climatology": [],
                "model": used_model,
                "latest_obs_date": None,
                "recent_rain_30d_mm": None,
                "recent_rain_month_mm": None,
                "error": "serie vide",
            }
        df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"]).sort_values("date")
        if df.empty:
            return {
                "monthly": [],
                "climatology": [],
                "model": used_model,
                "latest_obs_date": None,
                "recent_rain_30d_mm": None,
                "recent_rain_month_mm": None,
                "error": "serie vide",
            }

        last_date = pd.Timestamp(df["date"].max()).tz_convert("UTC")
        lower_30d = last_date - pd.Timedelta(days=29)
        month_start = pd.Timestamp(year=int(last_date.year), month=int(last_date.month), day=1, tz="UTC")
        recent_rain_30d = float(df.loc[df["date"] >= lower_30d, "precip_mm"].sum())
        recent_rain_month = float(df.loc[df["date"] >= month_start, "precip_mm"].sum())

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

        clim = _build_monthly_climatology(monthly)

        return {
            "monthly": monthly.sort_values("ym").to_dict(orient="records"),
            "climatology": clim.sort_values("month").to_dict(orient="records"),
            "model": used_model,
            "latest_obs_date": last_date.strftime("%Y-%m-%d"),
            "recent_rain_30d_mm": round(max(0.0, recent_rain_30d), 2),
            "recent_rain_month_mm": round(max(0.0, recent_rain_month), 2),
            "error": None,
        }
    except Exception as exc:
        return {
            "monthly": [],
            "climatology": [],
            "model": used_model,
            "latest_obs_date": None,
            "recent_rain_30d_mm": None,
            "recent_rain_month_mm": None,
            "error": str(exc),
        }


@st.cache_data(show_spinner=False, ttl=21600)
def _load_sector_daily_precip_on_date(lat: float, lon: float, target_day_iso: str) -> Dict[str, object]:
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "start_date": str(target_day_iso),
        "end_date": str(target_day_iso),
        "daily": "precipitation_sum",
        "timezone": "UTC",
        "models": OPEN_METEO_MODEL,
    }
    used_model = OPEN_METEO_MODEL
    try:
        response = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30, max_attempts=2)
        if response.status_code != 200:
            fallback_params = dict(params)
            fallback_params.pop("models", None)
            response = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=fallback_params, timeout=30, max_attempts=2)
            used_model = "open_meteo_default"
        if response.status_code != 200 or not response.text.strip():
            return {"date": str(target_day_iso), "precip_mm": None, "model": used_model, "error": f"HTTP {response.status_code}"}

        payload = response.json()
        entry = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(entry, dict):
            return {"date": str(target_day_iso), "precip_mm": None, "model": used_model, "error": "payload invalide"}
        daily = entry.get("daily", {}) if isinstance(entry.get("daily"), dict) else {}
        times = daily.get("time", []) or []
        vals = daily.get("precipitation_sum", []) or []
        if not times or not vals:
            return {"date": str(target_day_iso), "precip_mm": None, "model": used_model, "error": "serie vide"}

        day_df = pd.DataFrame(
            {
                "date": pd.to_datetime(times, utc=True, errors="coerce"),
                "precip_mm": pd.to_numeric(vals, errors="coerce"),
            }
        ).dropna(subset=["date", "precip_mm"])
        if day_df.empty:
            return {"date": str(target_day_iso), "precip_mm": None, "model": used_model, "error": "serie vide"}

        precip_val = float(day_df["precip_mm"].iloc[-1])
        return {"date": str(target_day_iso), "precip_mm": round(max(0.0, precip_val), 2), "model": used_model, "error": None}
    except Exception as exc:
        return {"date": str(target_day_iso), "precip_mm": None, "model": used_model, "error": str(exc)}


@st.cache_data(show_spinner=False, ttl=21600)
def _load_sector_recent_weather_archive(lat: float, lon: float, lookback_days: int = 45) -> Dict[str, object]:
    now_utc = datetime.now(timezone.utc)
    lookback = max(32, min(int(lookback_days), 120))
    start_date = (now_utc - timedelta(days=lookback)).strftime("%Y-%m-%d")
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
        response = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30, max_attempts=2)
        if response.status_code != 200:
            fallback_params = dict(params)
            fallback_params.pop("models", None)
            response = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=fallback_params, timeout=30, max_attempts=2)
            used_model = "open_meteo_default"
        if response.status_code != 200 or not response.text.strip():
            return {"error": f"HTTP {response.status_code}", "model": used_model}

        payload = response.json()
        entry = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(entry, dict):
            return {"error": "payload invalide", "model": used_model}
        daily = entry.get("daily", {}) if isinstance(entry.get("daily"), dict) else {}
        times = daily.get("time", []) or []
        vals = daily.get("precipitation_sum", []) or []
        if not times or not vals:
            return {"error": "serie vide", "model": used_model}

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(times, utc=True, errors="coerce"),
                "precip_mm": pd.to_numeric(vals, errors="coerce"),
            }
        )
        df = df.dropna(subset=["date"])
        if df.empty:
            return {"error": "serie vide", "model": used_model}
        df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"]).sort_values("date")
        if df.empty:
            return {"error": "serie vide", "model": used_model}

        last_date = pd.Timestamp(df["date"].max()).tz_convert("UTC")
        lower_7d = last_date - pd.Timedelta(days=6)
        lower_30d = last_date - pd.Timedelta(days=29)
        month_start = pd.Timestamp(year=int(last_date.year), month=int(last_date.month), day=1, tz="UTC")

        rain_24h = float(df.loc[df["date"] == last_date, "precip_mm"].sum())
        rain_7d = float(df.loc[df["date"] >= lower_7d, "precip_mm"].sum())
        rain_30d = float(df.loc[df["date"] >= lower_30d, "precip_mm"].sum())
        rain_month = float(df.loc[df["date"] >= month_start, "precip_mm"].sum())
        rain_24h = max(0.0, rain_24h)
        rain_7d = max(rain_24h, rain_7d)
        rain_30d = max(rain_7d, rain_30d)
        rain_month = max(rain_24h, rain_month)

        weather_alert_index = (
            min(rain_24h / 90.0, 1.6) * 40.0
            + min(rain_7d / 140.0, 1.5) * 30.0
            + min(rain_30d / 260.0, 1.5) * 20.0
        )
        weather_alert_index = max(0.0, min(weather_alert_index, 100.0))

        return {
            "station_id": "commune_archive_ref",
            "source": "open_meteo_archive_commune",
            "commune_station": "centre_commune",
            "distance_km": 0.0,
            "date_obs_raw": last_date.strftime("%Y-%m-%d"),
            "rain_24h_mm": round(rain_24h, 2),
            "rain_7d_mm": round(rain_7d, 2),
            "rain_30d_mm": round(rain_30d, 2),
            "rain_month_mm": round(rain_month, 2),
            "precipitation_mm": round(rain_24h, 2),
            "weather_alert_index": round(weather_alert_index, 1),
            "meteo_operational_level": _weather_level_from_index(weather_alert_index),
            "meteo_model": used_model,
            "error": None,
        }
    except Exception as exc:
        return {"error": str(exc), "model": used_model}


@st.cache_data(show_spinner=False, ttl=1200)
def _load_live_synop_weather_pack() -> Tuple[pd.DataFrame, str]:
    try:
        from meteo_test import LGVSeaMonitor

        monitor = LGVSeaMonitor()
        pack = monitor.fetch_pluviometry_synop()
        df = pack.get("all") if isinstance(pack.get("all"), pd.DataFrame) else pd.DataFrame()
        notice = str(pack.get("notice") or "").strip()
        src_url = str(pack.get("source_url") or "").strip()
        if src_url:
            notice = f"{notice} | source={src_url}" if notice else f"source={src_url}"
        return df.copy(), notice
    except Exception as exc:
        return pd.DataFrame(), f"SYNOP live indisponible: {exc}"


@st.cache_data(show_spinner=False, ttl=1200)
def _load_live_open_meteo_weather_pack() -> Tuple[pd.DataFrame, str]:
    try:
        from meteo_test import LGVSeaMonitor

        monitor = LGVSeaMonitor()
        pack = monitor.fetch_pluviometry_open_meteo()
        df = pack.get("all") if isinstance(pack.get("all"), pd.DataFrame) else pd.DataFrame()
        notice = str(pack.get("notice") or "").strip()
        src_url = str(pack.get("source_url") or "").strip()
        if src_url:
            notice = f"{notice} | source={src_url}" if notice else f"source={src_url}"
        return df.copy(), notice
    except Exception as exc:
        return pd.DataFrame(), f"Open-Meteo live indisponible: {exc}"


def _target_march_year(reference_ts: pd.Timestamp | None) -> int:
    ts = reference_ts
    if ts is None or pd.isna(ts):
        ts = pd.Timestamp.now(tz="UTC")
    if int(ts.month) < 3:
        return int(ts.year) - 1
    return int(ts.year)


@st.cache_data(show_spinner=False, ttl=43200)
def _synop_month_totals_for_stations(year: int, month: int, station_ids: Tuple[str, ...]) -> pd.DataFrame:
    wanted_ids = sorted({str(s).strip() for s in station_ids if str(s).strip()})
    if not wanted_ids:
        return pd.DataFrame(columns=["station_id", "synop_month_mm", "synop_obs_days"])

    try:
        from meteo_test import LGVSeaMonitor

        monitor = LGVSeaMonitor()
        gz_path, _ = monitor._download_synop_archive_gz(int(year))
        if not gz_path:
            return pd.DataFrame(columns=["station_id", "synop_month_mm", "synop_obs_days"])
    except Exception:
        return pd.DataFrame(columns=["station_id", "synop_month_mm", "synop_obs_days"])

    try:
        preview = pd.read_csv(gz_path, sep=";", nrows=1, dtype=str, compression="gzip")
        available = set(preview.columns.tolist())
        wanted_cols = [
            "numer_sta",
            "geo_id_wmo",
            "station",
            "id",
            "ID",
            "date",
            "Date",
            "validity_time",
            "reference_time",
            "rr1",
            "rr3",
            "rr6",
            "rr12",
            "rr24",
            "rr",
            "precipitation",
        ]
        usecols = [c for c in wanted_cols if c in available]
        if not usecols:
            return pd.DataFrame(columns=["station_id", "synop_month_mm", "synop_obs_days"])

        raw = pd.read_csv(
            gz_path,
            sep=";",
            dtype=str,
            usecols=usecols,
            compression="gzip",
            low_memory=False,
        )
        if raw.empty:
            return pd.DataFrame(columns=["station_id", "synop_month_mm", "synop_obs_days"])

        def _coalesce_text(frame: pd.DataFrame, cols: List[str]) -> pd.Series:
            out = pd.Series("", index=frame.index, dtype="string")
            for c in cols:
                if c not in frame.columns:
                    continue
                cur = frame[c].astype("string").fillna("").str.strip()
                out = out.where(out.str.len() > 0, cur)
            return out

        sid = _coalesce_text(raw, ["numer_sta", "geo_id_wmo", "station", "id", "ID"]).astype(str).str.strip()
        obs_raw = _coalesce_text(raw, ["date", "Date", "validity_time", "reference_time"])
        obs_dt = pd.to_datetime(obs_raw, format="%Y%m%d%H%M%S", errors="coerce", utc=True)
        obs_fallback = pd.to_datetime(obs_raw, errors="coerce", utc=True)
        obs_dt = obs_dt.where(obs_dt.notna(), obs_fallback)

        mask = sid.isin(wanted_ids) & obs_dt.notna()
        if not mask.any():
            return pd.DataFrame(columns=["station_id", "synop_month_mm", "synop_obs_days"])
        mask = mask & (obs_dt.dt.year == int(year)) & (obs_dt.dt.month == int(month))
        if not mask.any():
            return pd.DataFrame(columns=["station_id", "synop_month_mm", "synop_obs_days"])

        rr_cols = [c for c in ["rr24", "rr12", "rr6", "rr3", "rr1", "rr", "precipitation"] if c in raw.columns]
        if not rr_cols:
            return pd.DataFrame(columns=["station_id", "synop_month_mm", "synop_obs_days"])
        rr = raw[rr_cols].apply(pd.to_numeric, errors="coerce")
        rr = rr.where((rr >= 0.0) & (rr < 900.0))
        rain_day = rr.max(axis=1, skipna=True)

        subset = pd.DataFrame(
            {
                "station_id": sid,
                "day": obs_dt.dt.strftime("%Y-%m-%d"),
                "rain_day_mm": pd.to_numeric(rain_day, errors="coerce"),
            }
        )
        subset = subset[mask].dropna(subset=["station_id", "day", "rain_day_mm"])
        if subset.empty:
            return pd.DataFrame(columns=["station_id", "synop_month_mm", "synop_obs_days"])
        subset["rain_day_mm"] = subset["rain_day_mm"].clip(lower=0.0)
        subset = subset.groupby(["station_id", "day"], as_index=False)["rain_day_mm"].max()

        out = (
            subset.groupby("station_id", as_index=False)
            .agg(synop_month_mm=("rain_day_mm", "sum"), synop_obs_days=("day", "nunique"))
            .sort_values("station_id")
        )
        out["synop_month_mm"] = pd.to_numeric(out["synop_month_mm"], errors="coerce").round(2)
        out["synop_obs_days"] = pd.to_numeric(out["synop_obs_days"], errors="coerce").fillna(0).astype(int)
        return out
    except Exception:
        return pd.DataFrame(columns=["station_id", "synop_month_mm", "synop_obs_days"])


@st.cache_data(show_spinner=False, ttl=43200)
def _open_meteo_month_total_at_point(lat: float, lon: float, year: int, month: int) -> Dict[str, object]:
    start = pd.Timestamp(year=int(year), month=int(month), day=1, tz="UTC")
    end = (start + pd.offsets.MonthEnd(1)).tz_convert("UTC")
    params = {
        "latitude": f"{float(lat):.6f}",
        "longitude": f"{float(lon):.6f}",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": "precipitation_sum",
        "timezone": "UTC",
        "models": OPEN_METEO_MODEL,
    }
    used_model = OPEN_METEO_MODEL
    try:
        response = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30, max_attempts=2)
        if response.status_code != 200:
            fallback_params = dict(params)
            fallback_params.pop("models", None)
            response = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=fallback_params, timeout=30, max_attempts=2)
            used_model = "open_meteo_default"
        if response.status_code != 200 or not response.text.strip():
            return {"month_mm": np.nan, "model": used_model, "error": f"HTTP {response.status_code}"}

        payload = response.json()
        entry = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(entry, dict):
            return {"month_mm": np.nan, "model": used_model, "error": "payload invalide"}
        daily = entry.get("daily", {}) if isinstance(entry.get("daily"), dict) else {}
        vals = daily.get("precipitation_sum", []) or []
        if not vals:
            return {"month_mm": np.nan, "model": used_model, "error": "serie vide"}
        series = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
        if series.empty:
            return {"month_mm": np.nan, "model": used_model, "error": "serie vide"}
        return {"month_mm": round(float(series.clip(lower=0.0).sum()), 2), "model": used_model, "error": None}
    except Exception as exc:
        return {"month_mm": np.nan, "model": used_model, "error": str(exc)}


def _build_march_source_comparison(
    infoclimat_df: pd.DataFrame,
    open_meteo_df: pd.DataFrame,
    march_year: int,
    top_n: int = 3,
) -> pd.DataFrame:
    info = infoclimat_df.copy() if isinstance(infoclimat_df, pd.DataFrame) else pd.DataFrame()
    open_df = open_meteo_df.copy() if isinstance(open_meteo_df, pd.DataFrame) else pd.DataFrame()

    if info.empty and open_df.empty:
        return pd.DataFrame()

    if not info.empty:
        info["distance_to_lgv_km"] = pd.to_numeric(info.get("distance_to_lgv_km"), errors="coerce")
        info = info.dropna(subset=["distance_to_lgv_km"]).sort_values("distance_to_lgv_km").head(max(1, int(top_n))).copy()

    if not open_df.empty:
        open_df["distance_to_lgv_km"] = pd.to_numeric(open_df.get("distance_to_lgv_km"), errors="coerce")
        open_df = open_df.dropna(subset=["distance_to_lgv_km"]).sort_values("distance_to_lgv_km").head(max(1, int(top_n))).copy()

    info_ids = tuple(sorted({str(s).strip() for s in info.get("station_id", pd.Series(dtype=str)).astype(str).tolist() if str(s).strip()}))
    synop_month = _synop_month_totals_for_stations(int(march_year), 3, info_ids) if info_ids else pd.DataFrame()
    synop_map: Dict[str, float] = {}
    synop_days_map: Dict[str, int] = {}
    if not synop_month.empty:
        for row in synop_month.to_dict(orient="records"):
            sid = str(row.get("station_id") or "").strip()
            synop_map[sid] = float(pd.to_numeric(row.get("synop_month_mm"), errors="coerce"))
            synop_days_map[sid] = int(pd.to_numeric(row.get("synop_obs_days"), errors="coerce"))

    max_rows = max(len(info), len(open_df), 1)
    rows: List[Dict[str, object]] = []
    for i in range(max_rows):
        info_row = info.iloc[i].to_dict() if i < len(info) else {}
        open_row = open_df.iloc[i].to_dict() if i < len(open_df) else {}

        info_station = str(info_row.get("station_id") or "") if info_row else ""
        info_dist = pd.to_numeric(info_row.get("distance_to_lgv_km"), errors="coerce") if info_row else np.nan
        info_lat = pd.to_numeric(info_row.get("latitude"), errors="coerce") if info_row else np.nan
        info_lon = pd.to_numeric(info_row.get("longitude"), errors="coerce") if info_row else np.nan

        open_station = str(open_row.get("station_id") or "") if open_row else ""
        open_dist = pd.to_numeric(open_row.get("distance_to_lgv_km"), errors="coerce") if open_row else np.nan
        open_lat = pd.to_numeric(open_row.get("latitude"), errors="coerce") if open_row else np.nan
        open_lon = pd.to_numeric(open_row.get("longitude"), errors="coerce") if open_row else np.nan

        if (not open_row) and info_row:
            open_station = f"openmeteo@{info_station}" if info_station else "openmeteo@coord_infoclimat"
            open_dist = info_dist
            open_lat = info_lat
            open_lon = info_lon

        synop_mm = synop_map.get(info_station, np.nan) if info_station else np.nan
        synop_days = synop_days_map.get(info_station, 0) if info_station else 0

        open_mm = np.nan
        open_model = ""
        if pd.notna(open_lat) and pd.notna(open_lon):
            om = _open_meteo_month_total_at_point(float(open_lat), float(open_lon), int(march_year), 3)
            open_mm = pd.to_numeric(om.get("month_mm"), errors="coerce")
            open_model = str(om.get("model") or "")

        delta = np.nan
        if pd.notna(synop_mm) and pd.notna(open_mm):
            delta = float(synop_mm) - float(open_mm)

        rows.append(
            {
                "rang": int(i + 1),
                "station_infoclimat": info_station or None,
                "distance_infoclimat_km": None if pd.isna(info_dist) else round(float(info_dist), 3),
                f"mars_{int(march_year)}_infoclimat_mm": None if pd.isna(synop_mm) else round(float(synop_mm), 2),
                "jours_synop_mars": int(synop_days),
                "station_openmeteo": open_station or None,
                "distance_openmeteo_km": None if pd.isna(open_dist) else round(float(open_dist), 3),
                f"mars_{int(march_year)}_openmeteo_mm": None if pd.isna(open_mm) else round(float(open_mm), 2),
                "modele_openmeteo": open_model or None,
                "ecart_infoclimat_moins_openmeteo_mm": None if pd.isna(delta) else round(float(delta), 2),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out


def _build_commune_weather_reference(
    commune_row: Dict[str, object],
    weather_df: pd.DataFrame,
    snapshot_ts: pd.Timestamp | None,
    max_station_km: float = 15.0,
    max_obs_age_h: float = 72.0,
    max_infoclimat_fallback_km: float = 80.0,
    max_infoclimat_fallback_obs_h: float = 168.0,
    allow_open_meteo_fallback: bool = True,
) -> Dict[str, object]:
    try:
        lat = float(commune_row.get("latitude"))
        lon = float(commune_row.get("longitude"))
    except Exception:
        return {
            "source_mode": "AUCUNE_DONNEE",
            "source_label": "Coordonnees commune invalides",
            "source": None,
            "ref_latitude": None,
            "ref_longitude": None,
            "station_id": None,
            "commune_station": None,
            "distance_km": None,
            "obs_date": None,
            "rain_24h_mm": None,
            "rain_7d_mm": None,
            "rain_30d_mm": None,
            "rain_month_mm": None,
            "weather_alert_index": None,
            "weather_quality_note": None,
            "weather_data_reliability": "A_VERIFIER",
        }

    commune_name = str(commune_row.get("commune_name") or commune_row.get("commune_label") or "")
    nearest = (
        _select_best_weather_row(weather_df, lat, lon, commune_name=commune_name)
        if isinstance(weather_df, pd.DataFrame) and not weather_df.empty
        else {}
    )
    dist_km = pd.to_numeric(nearest.get("_dist_km"), errors="coerce") if nearest else pd.NA
    obs_age_h = pd.to_numeric(nearest.get("obs_age_h"), errors="coerce") if nearest else pd.NA
    if (pd.isna(obs_age_h) or obs_age_h is None) and nearest:
        obs_raw = pd.to_datetime(nearest.get("date_obs_raw"), utc=True, errors="coerce")
        ref_ts = snapshot_ts if snapshot_ts is not None and not pd.isna(snapshot_ts) else pd.Timestamp.now(tz="UTC")
        if not pd.isna(obs_raw):
            obs_age_h = (ref_ts - obs_raw).total_seconds() / 3600.0

    nearest_is_infoclimat = bool(nearest) and _is_infoclimat_station_source(nearest.get("source"))
    nearest_ok = (
        bool(nearest)
        and nearest_is_infoclimat
        and (pd.isna(dist_km) or float(dist_km) <= float(max_station_km))
        and (pd.isna(obs_age_h) or float(obs_age_h) <= float(max_obs_age_h))
    )
    if nearest_ok:
        return {
            "source_mode": "INFOCLIMAT_STATION",
            "source_label": "InfoClimat/SYNOP station commune",
            "source": nearest.get("source"),
            "ref_latitude": pd.to_numeric(nearest.get("latitude"), errors="coerce"),
            "ref_longitude": pd.to_numeric(nearest.get("longitude"), errors="coerce"),
            "station_id": nearest.get("station_id"),
            "commune_station": nearest.get("station_commune_name"),
            "distance_km": None if pd.isna(dist_km) else round(float(dist_km), 3),
            "obs_date": nearest.get("date_obs_raw"),
            "rain_24h_mm": pd.to_numeric(nearest.get("rain_24h_mm"), errors="coerce"),
            "rain_7d_mm": pd.to_numeric(nearest.get("rain_7d_mm"), errors="coerce"),
            "rain_30d_mm": pd.to_numeric(nearest.get("rain_30d_mm"), errors="coerce"),
            "rain_month_mm": pd.to_numeric(nearest.get("rain_month_mm"), errors="coerce"),
            "weather_alert_index": pd.to_numeric(nearest.get("weather_alert_index"), errors="coerce"),
            "weather_quality_note": pd.to_numeric(nearest.get("weather_quality_note"), errors="coerce"),
            "weather_data_reliability": nearest.get("weather_data_reliability"),
        }

    nearest_info_fallback_ok = (
        bool(nearest)
        and nearest_is_infoclimat
        and (pd.isna(dist_km) or float(dist_km) <= float(max_infoclimat_fallback_km))
        and (pd.isna(obs_age_h) or float(obs_age_h) <= float(max_infoclimat_fallback_obs_h))
    )
    if nearest_info_fallback_ok:
        return {
            "source_mode": "INFOCLIMAT_NEAREST_FALLBACK",
            "source_label": "InfoClimat/SYNOP station la plus proche",
            "source": nearest.get("source"),
            "ref_latitude": pd.to_numeric(nearest.get("latitude"), errors="coerce"),
            "ref_longitude": pd.to_numeric(nearest.get("longitude"), errors="coerce"),
            "station_id": nearest.get("station_id"),
            "commune_station": nearest.get("station_commune_name"),
            "distance_km": None if pd.isna(dist_km) else round(float(dist_km), 3),
            "obs_date": nearest.get("date_obs_raw"),
            "rain_24h_mm": pd.to_numeric(nearest.get("rain_24h_mm"), errors="coerce"),
            "rain_7d_mm": pd.to_numeric(nearest.get("rain_7d_mm"), errors="coerce"),
            "rain_30d_mm": pd.to_numeric(nearest.get("rain_30d_mm"), errors="coerce"),
            "rain_month_mm": pd.to_numeric(nearest.get("rain_month_mm"), errors="coerce"),
            "weather_alert_index": pd.to_numeric(nearest.get("weather_alert_index"), errors="coerce"),
            "weather_quality_note": pd.to_numeric(nearest.get("weather_quality_note"), errors="coerce"),
            "weather_data_reliability": nearest.get("weather_data_reliability"),
        }

    if allow_open_meteo_fallback:
        archive = _load_sector_recent_weather_archive(lat, lon, lookback_days=45)
        if not archive.get("error"):
            return {
                "source_mode": "OPEN_METEO_ARCHIVE_FALLBACK",
                "source_label": "Fallback fiable (Open-Meteo archive)",
                "source": archive.get("source"),
                "ref_latitude": pd.to_numeric(lat, errors="coerce"),
                "ref_longitude": pd.to_numeric(lon, errors="coerce"),
                "station_id": archive.get("station_id"),
                "commune_station": archive.get("commune_station"),
                "distance_km": archive.get("distance_km"),
                "obs_date": archive.get("date_obs_raw"),
                "rain_24h_mm": pd.to_numeric(archive.get("rain_24h_mm"), errors="coerce"),
                "rain_7d_mm": pd.to_numeric(archive.get("rain_7d_mm"), errors="coerce"),
                "rain_30d_mm": pd.to_numeric(archive.get("rain_30d_mm"), errors="coerce"),
                "rain_month_mm": pd.to_numeric(archive.get("rain_month_mm"), errors="coerce"),
                "weather_alert_index": pd.to_numeric(archive.get("weather_alert_index"), errors="coerce"),
                "weather_quality_note": np.nan,
                "weather_data_reliability": "OK",
            }

    if nearest and allow_open_meteo_fallback:
        return {
            "source_mode": "NEAREST_STATION_FALLBACK",
            "source_label": "Fallback station proche (hors InfoClimat)",
            "source": nearest.get("source"),
            "ref_latitude": pd.to_numeric(nearest.get("latitude"), errors="coerce"),
            "ref_longitude": pd.to_numeric(nearest.get("longitude"), errors="coerce"),
            "station_id": nearest.get("station_id"),
            "commune_station": nearest.get("station_commune_name"),
            "distance_km": None if pd.isna(dist_km) else round(float(dist_km), 3),
            "obs_date": nearest.get("date_obs_raw"),
            "rain_24h_mm": pd.to_numeric(nearest.get("rain_24h_mm"), errors="coerce"),
            "rain_7d_mm": pd.to_numeric(nearest.get("rain_7d_mm"), errors="coerce"),
            "rain_30d_mm": pd.to_numeric(nearest.get("rain_30d_mm"), errors="coerce"),
            "rain_month_mm": pd.to_numeric(nearest.get("rain_month_mm"), errors="coerce"),
            "weather_alert_index": pd.to_numeric(nearest.get("weather_alert_index"), errors="coerce"),
            "weather_quality_note": pd.to_numeric(nearest.get("weather_quality_note"), errors="coerce"),
            "weather_data_reliability": nearest.get("weather_data_reliability"),
        }

    return {
        "source_mode": "AUCUNE_DONNEE",
        "source_label": "Aucune source meteo exploitable",
        "source": None,
        "ref_latitude": None,
        "ref_longitude": None,
        "station_id": None,
        "commune_station": None,
        "distance_km": None,
        "obs_date": None,
        "rain_24h_mm": None,
        "rain_7d_mm": None,
        "rain_30d_mm": None,
        "rain_month_mm": None,
        "weather_alert_index": None,
        "weather_quality_note": None,
        "weather_data_reliability": "A_VERIFIER",
    }


@st.cache_data(show_spinner=False, ttl=21600)
def _build_multi_commune_history(commune_rows: List[Dict[str, object]], years_back: int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames: List[pd.DataFrame] = []
    missing_labels: List[Tuple[str, str]] = []
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
            missing_labels.append((commune_label, cname))
            continue
        monthly["commune_name"] = cname
        monthly["commune_label"] = commune_label
        monthly["history_imputed"] = False
        frames.append(monthly)

    now_utc = datetime.now(timezone.utc)
    start_year = max(2010, now_utc.year - max(int(years_back), 1) + 1)
    periods = pd.period_range(start=f"{start_year}-01", end=now_utc.strftime("%Y-%m"), freq="M")
    timeline = pd.DataFrame({"ym": periods.astype(str)})
    timeline["year"] = pd.to_numeric(timeline["ym"].str.slice(0, 4), errors="coerce").fillna(start_year).astype(int)
    timeline["month"] = pd.to_numeric(timeline["ym"].str.slice(5, 7), errors="coerce").fillna(1).astype(int)

    month_pattern: Dict[int, float] = {}
    default_monthly = 70.0
    if frames:
        hist_all = pd.concat(frames, ignore_index=True)
        hist_all["month"] = pd.to_numeric(hist_all.get("month"), errors="coerce")
        hist_all["monthly_precip_mm"] = pd.to_numeric(hist_all.get("monthly_precip_mm"), errors="coerce")
        pattern = (
            hist_all.dropna(subset=["month", "monthly_precip_mm"])
            .groupby("month", as_index=False)["monthly_precip_mm"]
            .median()
        )
        for _, row in pattern.iterrows():
            m = int(row["month"])
            month_pattern[m] = round(float(row["monthly_precip_mm"]), 1)
        if month_pattern:
            default_monthly = float(pd.Series(list(month_pattern.values())).median())

    for commune_label, cname in missing_labels:
        fallback = timeline.copy()
        fallback["monthly_precip_mm"] = fallback["month"].map(lambda m: month_pattern.get(int(m), default_monthly))
        fallback["commune_name"] = cname
        fallback["commune_label"] = commune_label
        fallback["history_imputed"] = True
        frames.append(fallback)

    if not frames:
        return pd.DataFrame(), model_by_commune

    out = pd.concat(frames, ignore_index=True)
    out["ym"] = out["ym"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(0).astype(int)
    out["month"] = pd.to_numeric(out["month"], errors="coerce").fillna(0).astype(int)
    out["monthly_precip_mm"] = pd.to_numeric(out["monthly_precip_mm"], errors="coerce").fillna(default_monthly)
    if "history_imputed" not in out.columns:
        out["history_imputed"] = False
    out["history_imputed"] = out["history_imputed"].fillna(False).astype(bool)
    return out.sort_values(["ym", "commune_label"]), model_by_commune


def _season_totals_from_monthly(monthly_df: pd.DataFrame) -> Tuple[float | None, float | None, int | None]:
    if monthly_df.empty:
        return None, None, None
    work = monthly_df.copy()
    work["year"] = pd.to_numeric(work.get("year"), errors="coerce")
    work["month"] = pd.to_numeric(work.get("month"), errors="coerce")
    work["monthly_precip_mm"] = pd.to_numeric(work.get("monthly_precip_mm"), errors="coerce")
    work = work.dropna(subset=["year", "month", "monthly_precip_mm"])
    if work.empty:
        return None, None, None
    work["year"] = work["year"].astype(int)
    work["month"] = work["month"].astype(int)
    latest_year = int(work["year"].max())

    winter_mask = ((work["year"] == latest_year - 1) & (work["month"] == 12)) | (
        (work["year"] == latest_year) & (work["month"].isin([1, 2]))
    )
    spring_mask = (work["year"] == latest_year) & (work["month"].isin([3, 4, 5]))

    winter_val = pd.to_numeric(work.loc[winter_mask, "monthly_precip_mm"], errors="coerce").sum(min_count=1)
    spring_val = pd.to_numeric(work.loc[spring_mask, "monthly_precip_mm"], errors="coerce").sum(min_count=1)
    winter = None if pd.isna(winter_val) else round(float(winter_val), 1)
    spring = None if pd.isna(spring_val) else round(float(spring_val), 1)
    return winter, spring, latest_year


def _build_sector_segmentation_compare(
    sectors_df: pd.DataFrame,
    segment_km: int,
    base_km: float,
) -> pd.DataFrame:
    if sectors_df.empty:
        return pd.DataFrame()
    work = sectors_df.copy().reset_index(drop=True)
    work["score"] = pd.to_numeric(work.get("score"), errors="coerce")
    work["ai_pred_probability"] = pd.to_numeric(work.get("ai_pred_probability"), errors="coerce")
    work["weather_max_24h_mm"] = pd.to_numeric(work.get("weather_max_24h_mm"), errors="coerce")
    work["weather_max_7d_mm"] = pd.to_numeric(work.get("weather_max_7d_mm"), errors="coerce")
    work["weather_max_30d_mm"] = pd.to_numeric(work.get("weather_max_30d_mm"), errors="coerce")
    work["hydro_stations"] = pd.to_numeric(work.get("hydro_stations"), errors="coerce")
    work["pk_km"] = pd.to_numeric(work.get("pk_km"), errors="coerce")

    sector_series = work["sector_id"].astype(str) if "sector_id" in work.columns else pd.Series([""] * len(work), index=work.index)
    sector_num = pd.to_numeric(sector_series.str.extract(r"(\d+)")[0], errors="coerce")
    work["_order"] = sector_num.fillna(pd.Series(range(1, len(work) + 1), index=work.index)).astype(int)
    work = work.sort_values("_order").reset_index(drop=True)

    try:
        group_size = max(1, int(round(float(segment_km) / max(float(base_km), 0.1))))
    except Exception:
        group_size = 1
    work["segment_index"] = (work.index // group_size) + 1

    rank_map = {"FAIBLE": 1, "MODERE": 2, "ELEVE": 3, "CRITIQUE": 4}
    inv_rank = {v: k for k, v in rank_map.items()}
    risk_series = work["risk_level"] if "risk_level" in work.columns else pd.Series(["INDETERMINE"] * len(work), index=work.index)
    ai_risk_series = work["ai_pred_risk_level"] if "ai_pred_risk_level" in work.columns else risk_series
    work["_risk_rank"] = risk_series.map(lambda x: rank_map.get(str(x), 0))
    work["_ai_rank"] = ai_risk_series.map(lambda x: rank_map.get(str(x), 0))

    grouped = (
        work.groupby("segment_index", as_index=False)
        .agg(
            sector_count=("sector_id", "count"),
            pk_start_km=("pk_km", "min"),
            pk_end_km=("pk_km", "max"),
            avg_score=("score", "mean"),
            max_score=("score", "max"),
            ai_max_probability=("ai_pred_probability", "max"),
            rain_24h_max=("weather_max_24h_mm", "max"),
            rain_7d_max=("weather_max_7d_mm", "max"),
            rain_30d_max=("weather_max_30d_mm", "max"),
            hydro_stations_max=("hydro_stations", "max"),
            risk_rank_max=("_risk_rank", "max"),
            ai_rank_max=("_ai_rank", "max"),
        )
    )
    grouped["segment_id"] = grouped["segment_index"].map(lambda i: f"SEG-{int(i):03d}")
    grouped["segment_km"] = int(segment_km)
    grouped["risk_level"] = grouped["risk_rank_max"].map(lambda x: inv_rank.get(int(x), "INDETERMINE"))
    grouped["ai_pred_risk_level"] = grouped["ai_rank_max"].map(lambda x: inv_rank.get(int(x), "INDETERMINE"))
    grouped["avg_score"] = pd.to_numeric(grouped["avg_score"], errors="coerce").round(2)
    grouped["max_score"] = pd.to_numeric(grouped["max_score"], errors="coerce").round(2)
    grouped["ai_max_probability"] = pd.to_numeric(grouped["ai_max_probability"], errors="coerce").round(4)
    grouped["pk_start_km"] = pd.to_numeric(grouped["pk_start_km"], errors="coerce").round(2)
    grouped["pk_end_km"] = pd.to_numeric(grouped["pk_end_km"], errors="coerce").round(2)
    return grouped.sort_values("segment_index")


def _build_pluvio_ranking(
    commune_rows: List[Dict[str, object]],
    weather_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    if not commune_rows:
        return pd.DataFrame()

    hist = history_df.copy() if isinstance(history_df, pd.DataFrame) else pd.DataFrame()
    if not hist.empty:
        hist["commune_label"] = hist.get("commune_label", "").astype(str)

    rows: List[Dict[str, object]] = []
    for com in commune_rows:
        cname = str(com.get("commune_name") or "Inconnue")
        ccode = str(com.get("commune_code") or "")
        label = f"{cname} ({ccode})" if ccode else cname
        try:
            lat = float(com.get("latitude"))
            lon = float(com.get("longitude"))
        except (TypeError, ValueError):
            continue

        wx = (
            _select_best_weather_row(weather_df, lat, lon, commune_name=cname)
            if isinstance(weather_df, pd.DataFrame)
            else {}
        )
        r1j = pd.to_numeric(wx.get("rain_24h_mm"), errors="coerce")
        r7j = pd.to_numeric(wx.get("rain_7d_mm"), errors="coerce")
        r30j = pd.to_numeric(wx.get("rain_30d_mm"), errors="coerce")
        r1m_raw = pd.to_numeric(wx.get("rain_month_mm"), errors="coerce")
        r1m = r1m_raw

        winter = None
        spring = None
        season_year = None
        max_monthly = None
        history_mode = "ESTIME_RECENT"
        if not hist.empty:
            h = hist[hist["commune_label"] == label]
            if (pd.isna(r1m) or float(r1m) < 0.0) and not h.empty:
                try:
                    now_utc = datetime.now(timezone.utc)
                    cur_ym = f"{int(now_utc.year):04d}-{int(now_utc.month):02d}"
                    h_ym = h.copy()
                    if "ym" not in h_ym.columns and "year" in h_ym.columns and "month" in h_ym.columns:
                        h_ym["ym"] = (
                            pd.to_numeric(h_ym.get("year"), errors="coerce").fillna(0).astype(int).astype(str)
                            + "-"
                            + pd.to_numeric(h_ym.get("month"), errors="coerce").fillna(0).astype(int).map(lambda v: f"{int(v):02d}")
                        )
                    if "ym" in h_ym.columns:
                        hm = h_ym[h_ym["ym"].astype(str) == cur_ym]
                        if not hm.empty:
                            hm_val = pd.to_numeric(hm.get("monthly_precip_mm"), errors="coerce").dropna()
                            if not hm_val.empty:
                                r1m = float(hm_val.iloc[-1])
                except Exception:
                    pass
            winter, spring, season_year = _season_totals_from_monthly(h)
            if not h.empty:
                max_monthly_val = pd.to_numeric(h.get("monthly_precip_mm"), errors="coerce").max()
                max_monthly = None if pd.isna(max_monthly_val) else round(float(max_monthly_val), 1)
                history_mode = "ARCHIVE"

        if max_monthly is None:
            monthly_candidates = [x for x in [r1m, r30j, r7j] if not pd.isna(x)]
            if monthly_candidates:
                max_monthly = round(float(max(monthly_candidates)), 1)
        if winter is None:
            if not pd.isna(r30j) and float(r30j) > 0.0:
                winter = round(float(r30j) * 3.0, 1)
            elif not pd.isna(r7j) and float(r7j) > 0.0:
                winter = round(float(r7j) * 12.0, 1)
        if spring is None:
            if not pd.isna(r30j) and float(r30j) > 0.0:
                spring = round(float(r30j) * 3.0, 1)
            elif not pd.isna(r7j) and float(r7j) > 0.0:
                spring = round(float(r7j) * 12.0, 1)
        if season_year is None:
            season_year = int(datetime.now(timezone.utc).year)

        rows.append(
            {
                "commune_label": label,
                "commune_name": cname,
                "commune_code": ccode,
                "cum_1j_mm": None if pd.isna(r1j) else round(float(r1j), 1),
                "cum_1_semaine_mm": None if pd.isna(r7j) else round(float(r7j), 1),
                "cum_1_mois_mm": None if pd.isna(r1m) else round(float(r1m), 1),
                "cum_hiver_mm": winter,
                "cum_printemps_mm": spring,
                "max_mensuel_mm": max_monthly,
                "annee_saison_ref": season_year,
                "histo_mode": history_mode,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    metric_cols = ["cum_1j_mm", "cum_1_semaine_mm", "cum_1_mois_mm", "cum_hiver_mm", "cum_printemps_mm", "max_mensuel_mm"]
    realtime_cols = {"cum_1j_mm", "cum_1_semaine_mm", "cum_1_mois_mm"}
    for col in metric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if col in realtime_cols:
            if out[col].notna().any():
                fallback = float(out[col].median(skipna=True))
            else:
                fallback = 0.0
            out[col] = out[col].fillna(fallback)
        rank_col = f"rang_{col.replace('cum_', '').replace('_mm', '')}"
        if out[col].notna().any():
            out[rank_col] = out[col].rank(method="min", ascending=False).astype("Int64")
        else:
            out[rank_col] = pd.Series([pd.NA] * len(out), dtype="Int64")
    return out.sort_values(["cum_1_mois_mm", "cum_1_semaine_mm", "cum_1j_mm"], ascending=False, na_position="last")


def _aggregate_communes(sectors_df: pd.DataFrame, commune_rain_col: str) -> pd.DataFrame:
    if sectors_df.empty:
        return pd.DataFrame()

    df = sectors_df.copy()
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["commune_name"] = df.get("commune_name", "Inconnue").fillna("Inconnue")
    df["sector_score"] = pd.to_numeric(df.get("score", 0.0), errors="coerce").fillna(0.0)
    df["rain_period_mm"] = pd.to_numeric(df.get(commune_rain_col, 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
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

    grouped["risk_score_base"] = (
        (grouped["avg_sector_score"] / 4.0) * 68.0
        + ((grouped["critical"] * 12.0 + grouped["high"] * 6.0) / grouped["sector_count"].clip(lower=1))
        + grouped["avg_rain_period_mm"].clip(lower=0.0, upper=180.0) / 9.0
    ).clip(lower=0.0, upper=100.0)
    grouped["risk_score_base"] = grouped["risk_score_base"].round(1)
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

    grouped["risk_score"] = grouped["risk_score_base"]
    has_ai_signal = grouped["max_ai_probability"] > 0.0
    grouped.loc[has_ai_signal, "risk_score"] = (
        grouped.loc[has_ai_signal, "risk_score_base"] * 0.72
        + grouped.loc[has_ai_signal, "ai_component_note"] * 0.28
    ).clip(lower=0.0, upper=100.0)
    grouped["risk_score"] = grouped["risk_score"].round(1)
    grouped["commune_risk_level"] = grouped["risk_score"].map(_risk_level_from_note)
    grouped["ai_commune_risk_level"] = grouped["max_ai_probability"].map(lambda v: _ai_level_from_probability(float(v or 0.0)))
    grouped["note_gc"] = grouped["risk_score"]

    grouped["lgv_points_count"] = grouped["sector_count"]
    grouped["avg_point_score"] = grouped["avg_sector_score"]
    grouped["max_point_score"] = grouped["max_sector_score"]
    grouped = grouped.sort_values(["risk_score", "critical", "high"], ascending=[False, False, False]).reset_index(drop=True)
    return grouped


def _extract_lgv_lines_from_snapshot(snapshot: Dict[str, object]) -> List[List[Tuple[float, float]]]:
    lines: List[List[Tuple[float, float]]] = []
    raw_lines = snapshot.get("lgv_lines", []) if isinstance(snapshot, dict) else []
    for line in raw_lines if isinstance(raw_lines, list) else []:
        coords: List[Tuple[float, float]] = []
        for pt in line if isinstance(line, list) else []:
            if isinstance(pt, dict) and "lat" in pt and "lon" in pt:
                try:
                    coords.append((float(pt["lat"]), float(pt["lon"])))
                except Exception:
                    continue
        if len(coords) >= 2:
            lines.append(coords)
    return lines


def _create_base_map(location: List[float], zoom_start: int, map_style: str) -> folium.Map:
    style = MAP_TILE_STYLES.get(str(map_style), MAP_TILE_STYLES["Google Satellite"])
    tile_value = str(style.get("tiles") or "")
    if tile_value.lower().startswith("http"):
        m = folium.Map(location=location, zoom_start=zoom_start, tiles=None)
        folium.TileLayer(
            tiles=tile_value,
            attr=str(style.get("attr") or "Map"),
            name=str(map_style),
            overlay=False,
            control=False,
        ).add_to(m)
        return m
    return folium.Map(location=location, zoom_start=zoom_start, tiles=tile_value)


def _build_map(
    snapshot: Dict[str, object],
    weather_df: pd.DataFrame,
    commune_df: pd.DataFrame,
    sectors_df: pd.DataFrame,
    slip_corridors_df: pd.DataFrame,
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
    show_slip: bool,
    slip_alert_threshold: float,
    show_fr_layer: bool,
    map_style: str = "Google Satellite",
) -> folium.Map:
    m = _create_base_map(location=[46.2, 0.2], zoom_start=7, map_style=map_style)

    for coords in _extract_lgv_lines_from_snapshot(snapshot):
        folium.PolyLine(coords, color="#1d4ed8", weight=4, opacity=0.9, tooltip="Trace LGV SEA").add_to(m)

    if show_weather and not weather_df.empty:
        weather_layer = folium.FeatureGroup(name="Meteo", show=True)
        for _, row in weather_df.iterrows():
            lvl = str(row.get("meteo_operational_level", row.get("risk_level", "INDETERMINE")))
            if _risk_rank(lvl) < _risk_rank(min_risk):
                continue
            rain = float(row.get(rain_col_weather, 0.0) or 0.0)
            quality_note = pd.to_numeric(row.get("weather_quality_note"), errors="coerce")
            alert_idx = pd.to_numeric(row.get("weather_alert_index"), errors="coerce")
            obs_age = pd.to_numeric(row.get("obs_age_h"), errors="coerce")
            data_reliability = str(
                row.get(
                    "weather_data_reliability",
                    _weather_data_reliability_label(
                        0.0 if pd.isna(quality_note) else float(quality_note),
                        0.0 if pd.isna(obs_age) else float(obs_age),
                    ),
                )
            )
            freshness_label = str(row.get("obs_freshness_level", _weather_freshness_level_from_age(0.0 if pd.isna(obs_age) else float(obs_age))))
            action_label = str(row.get("weather_action_label", _weather_action_label(lvl, data_reliability)))
            popup = (
                f"<b>Station:</b> {row.get('station_id')}<br>"
                f"<b>Source:</b> {row.get('source')}<br>"
                f"<b>Commune station:</b> {row.get('station_commune_name', 'n/a')}<br>"
                f"<b>Localisation:</b> {float(row.get('latitude')):.6f}, {float(row.get('longitude')):.6f}<br>"
                f"<b>Risque meteo operationnel:</b> {lvl}<br>"
                f"<b>Cumul filtre:</b> {rain:.1f} mm<br>"
                f"<b>Indice alerte meteo:</b> {0.0 if pd.isna(alert_idx) else float(alert_idx):.1f}/100<br>"
                f"<b>Qualite mesure:</b> {0.0 if pd.isna(quality_note) else float(quality_note):.1f}/100 ({row.get('weather_quality_level', 'n/a')})<br>"
                f"<b>Anciennete obs:</b> {0.0 if pd.isna(obs_age) else float(obs_age):.1f} h ({freshness_label})<br>"
                f"<b>Fiabilite donnee:</b> {data_reliability}<br>"
                f"<b>Action recommandee:</b> {action_label}<br>"
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
                f"<b>Score risque global:</b> {row.get('risk_score', row.get('note_gc'))} /100<br>"
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

    if show_slip and not sectors_df.empty and "slip_index" in sectors_df.columns:
        slip_layer = folium.FeatureGroup(name="Zones glissement", show=True)
        slip_work = sectors_df.copy()
        slip_work["pk_km"] = pd.to_numeric(slip_work.get("pk_km"), errors="coerce")
        slip_work["slip_index"] = pd.to_numeric(slip_work.get("slip_index"), errors="coerce").fillna(0.0)
        slip_work["slip_level"] = slip_work.get("slip_level", pd.Series("FAIBLE", index=slip_work.index)).astype(str)
        slip_work["manual_watch_pk"] = slip_work.get("manual_watch_pk", pd.Series(False, index=slip_work.index)).fillna(False).astype(bool)

        slip_focus = slip_work[
            (slip_work["slip_index"] >= float(slip_alert_threshold))
            | (slip_work["manual_watch_pk"])
        ].copy()
        if str(min_risk).upper() != "TOUT":
            slip_focus = slip_focus[
                slip_focus["slip_level"].map(lambda x: _risk_rank(str(x))) >= _risk_rank(str(min_risk).upper())
            ]

        for _, row in slip_focus.iterrows():
            lat = pd.to_numeric(row.get("latitude"), errors="coerce")
            lon = pd.to_numeric(row.get("longitude"), errors="coerce")
            if pd.isna(lat) or pd.isna(lon):
                continue
            lvl = str(row.get("slip_level", "FAIBLE"))
            slip_index = float(pd.to_numeric(row.get("slip_index"), errors="coerce") or 0.0)
            radius = max(6, min(15, 5 + slip_index / 12.0))
            popup = (
                f"<b>Zone glissement:</b> {row.get('sector_id')}<br>"
                f"<b>PK:</b> {row.get('pk_km')}<br>"
                f"<b>Niveau glissement:</b> {lvl}<br>"
                f"<b>Indice glissement:</b> {slip_index:.1f}/100<br>"
                f"<b>Commune:</b> {row.get('commune_name', 'n/a')}<br>"
                f"<b>Drivers:</b> {row.get('slip_drivers', 'n/a')}<br>"
                f"<b>Zone PK manuelle:</b> {row.get('manual_watch_pk', False)}"
            )
            folium.CircleMarker(
                [float(lat), float(lon)],
                radius=radius,
                color=RISK_COLOR.get(lvl, "#6b7280"),
                fill=True,
                fill_opacity=0.55,
                weight=2,
                popup=folium.Popup(popup, max_width=420),
            ).add_to(slip_layer)

        if not slip_corridors_df.empty:
            for _, corr in slip_corridors_df.iterrows():
                cid = str(corr.get("slip_corridor_id", ""))
                cmax = float(pd.to_numeric(corr.get("slip_index_max"), errors="coerce") or 0.0)
                lvl = str(corr.get("slip_level", _slip_level_from_index(cmax)))
                pk_start = float(pd.to_numeric(corr.get("pk_start_km"), errors="coerce") or 0.0)
                pk_end = float(pd.to_numeric(corr.get("pk_end_km"), errors="coerce") or 0.0)
                cpoints = slip_work[
                    (pd.to_numeric(slip_work.get("pk_km"), errors="coerce") >= pk_start)
                    & (pd.to_numeric(slip_work.get("pk_km"), errors="coerce") <= pk_end)
                ].copy()
                cpoints = cpoints.sort_values("pk_km")
                coords: List[Tuple[float, float]] = []
                for _, crow in cpoints.iterrows():
                    clat = pd.to_numeric(crow.get("latitude"), errors="coerce")
                    clon = pd.to_numeric(crow.get("longitude"), errors="coerce")
                    if pd.isna(clat) or pd.isna(clon):
                        continue
                    coords.append((float(clat), float(clon)))
                if len(coords) >= 2:
                    tooltip = (
                        f"{cid} | PK {float(corr.get('pk_start_km', 0.0)):.2f}-{float(corr.get('pk_end_km', 0.0)):.2f} | "
                        f"indice max={cmax:.1f}"
                    )
                    folium.PolyLine(
                        coords,
                        color=RISK_COLOR.get(lvl, "#7f1d1d"),
                        weight=max(3, min(8, int(round(cmax / 18.0)))),
                        opacity=0.85,
                        tooltip=tooltip,
                    ).add_to(slip_layer)
        slip_layer.add_to(m)

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


def _build_weather_focus_map(
    snapshot: Dict[str, object],
    weather_df: pd.DataFrame,
    rain_col_weather: str,
    min_risk: str,
    max_labels: int = 80,
    map_style: str = "Google Satellite",
) -> folium.Map:
    lgv_lines = _extract_lgv_lines_from_snapshot(snapshot)
    center = [46.2, 0.2]
    if lgv_lines:
        flat_pts = [pt for line in lgv_lines for pt in line]
        if flat_pts:
            center = [float(np.mean([p[0] for p in flat_pts])), float(np.mean([p[1] for p in flat_pts]))]

    m = _create_base_map(location=center, zoom_start=7, map_style=map_style)
    for coords in lgv_lines:
        folium.PolyLine(coords, color="#0b4f6c", weight=4, opacity=0.95, tooltip="Trace LGV SEA").add_to(m)

    if weather_df.empty:
        return m

    wdf = weather_df.copy()
    wdf["latitude"] = pd.to_numeric(wdf.get("latitude"), errors="coerce")
    wdf["longitude"] = pd.to_numeric(wdf.get("longitude"), errors="coerce")
    wdf[rain_col_weather] = pd.to_numeric(wdf.get(rain_col_weather), errors="coerce")
    wdf = wdf.dropna(subset=["latitude", "longitude", rain_col_weather])
    if wdf.empty:
        return m

    lvl_col = wdf.get("meteo_operational_level", wdf.get("risk_level", "INDETERMINE")).astype(str)
    wdf = wdf.assign(_lvl=lvl_col)
    if str(min_risk).upper() != "TOUT":
        wdf = wdf[wdf["_lvl"].map(lambda x: _risk_rank(str(x))) >= _risk_rank(str(min_risk))]
    if wdf.empty:
        return m

    wdf["distance_to_lgv_km"] = pd.to_numeric(wdf.get("distance_to_lgv_km"), errors="coerce")
    wdf = wdf.sort_values(
        by=["distance_to_lgv_km", rain_col_weather],
        ascending=[True, False],
        na_position="last",
    ).head(max(1, int(max_labels)))

    for i, (_, row) in enumerate(wdf.iterrows()):
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        lvl = str(row.get("_lvl", "INDETERMINE"))
        rain_val = float(row.get(rain_col_weather, 0.0) or 0.0)
        station_id = str(row.get("station_id") or "station")
        commune_station = str(row.get("station_commune_name") or "")

        folium.CircleMarker(
            [lat, lon],
            radius=6,
            color=RISK_COLOR.get(lvl, "#6b7280"),
            fill=True,
            fill_opacity=0.9,
            weight=1,
            popup=folium.Popup(
                f"<b>{station_id}</b><br>"
                f"Commune: {commune_station}<br>"
                f"Localisation: {lat:.6f}, {lon:.6f}<br>"
                f"Pluie ({rain_col_weather}): {rain_val:.1f} mm<br>"
                f"Risque meteo: {lvl}<br>"
                f"Distance LGV: {row.get('distance_to_lgv_km')} km<br>"
                f"Date obs: {row.get('date_obs_raw')}",
                max_width=340,
            ),
        ).add_to(m)

        lat_off = 0.006 if (i % 2 == 0) else -0.006
        lon_off = 0.008 if (i % 3 == 0) else 0.012
        label_txt = f"{station_id} | {rain_val:.1f} mm"
        safe_txt = label_txt.replace("<", "").replace(">", "")
        label_html = (
            "<div style=\""
            "font-size:11px;font-weight:700;color:#111827;"
            "background-color:rgba(255,255,255,0.92);"
            "border:1px solid #d1d5db;border-radius:4px;"
            "padding:1px 4px;white-space:nowrap;\">"
            f"{safe_txt}</div>"
        )
        folium.Marker(
            [lat + lat_off, lon + lon_off],
            icon=folium.features.DivIcon(icon_size=(160, 20), icon_anchor=(0, 0), html=label_html),
        ).add_to(m)

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
# Snapshot vide ({}) = collecte en cours ou fallback workflow → continuer avec données vides
if not isinstance(snapshot, dict):
    st.error("Snapshot invalide. Verifie que GitHub Pages est actif puis recharge la page.")
    if st.button("Reessayer le chargement", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.stop()
if not snapshot:
    snapshot = {
        "timestamp_utc": None,
        "risk_level": "INDETERMINE",
        "score": 0,
        "weather_notice": "Donnees en cours de collecte (prochain cycle automatique dans ~1h)",
        "alerts": [],
        "weather": [],
        "sectors": {},
        "hydro_network": {},
        "piezometers": {},
        "geotech": {},
        "lgv_communes": {},
        "fr_geography": {},
        "metadata": {"line_monitoring": {"sector_length_km": 5.0}},
        "commune_ranking": [],
    }
    st.warning("⏳ Snapshot en cours de génération. Les données seront disponibles au prochain cycle (env. 1h).")

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
line_meta = metadata_obj.get("line_monitoring", {}) if isinstance(metadata_obj.get("line_monitoring"), dict) else {}
sector_base_km_raw = pd.to_numeric(line_meta.get("sector_length_km"), errors="coerce")
sector_base_km = float(sector_base_km_raw) if not pd.isna(sector_base_km_raw) else 5.0
snapshot_ts = pd.to_datetime(snapshot.get("timestamp_utc"), utc=True, errors="coerce")
snapshot_age_h = _snapshot_age_hours(snapshot_ts)

weather_df = _prepare_weather_df(weather_df, snapshot_ts)

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
    for rain_col in ["weather_max_24h_mm", "weather_max_7d_mm", "weather_max_30d_mm", "weather_max_month_mm"]:
        if rain_col in sectors_df.columns:
            sectors_df[rain_col] = pd.to_numeric(sectors_df[rain_col], errors="coerce").fillna(0.0).clip(lower=0.0)
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

weather_obs_window = "Tout"
weather_exact_day_enabled = False
weather_exact_day = datetime.now(timezone.utc).date()
weather_source_profile = WEATHER_SOURCE_PROFILE_OPTIONS[0]
weather_source_mode = WEATHER_SOURCE_FILTER_MODES[2]
infoclimat_max_distance_km = DEFAULT_INFOCLIMAT_LGV_MAX_DISTANCE_KM
enforce_pluvio_quality = True
strict_reliable_mode = DEFAULT_STRICT_RELIABLE_MODE
enable_live_synop_assist = True
march_compare_top_n = 3
auto_live_open_meteo_compare = True
live_synop_notice = ""
live_synop_added_count = 0
live_open_meteo_notice = ""
live_open_meteo_added_count = 0
map_basemap_style = "Google Satellite"
selected_history_month_labels = MONTH_ORDER_SHORT.copy()
selected_history_month_numbers = list(MONTH_LABELS_SHORT.keys())
with st.sidebar:
    st.subheader("Filtres")
    if st.button("Rafraichir snapshot", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    period_options = list(RAIN_PERIODS.keys())
    period_default_idx = period_options.index("Mois courant") if "Mois courant" in period_options else 0
    period_label = st.selectbox("Periode pluvio", period_options, index=period_default_idx)
    rain_col_weather, commune_rain_col = RAIN_PERIODS[period_label]
    weather_obs_window = st.selectbox(
        "Recherche pluie (fenetre obs)",
        list(OBS_WINDOW_HOURS.keys()),
        index=1,
        help="Filtre les stations meteo selon la date d'observation (1 jour, 7 jours ou mois courant).",
    )
    weather_exact_day_enabled = st.checkbox("Filtrer stations sur jour exact", value=False)
    if weather_exact_day_enabled:
        weather_exact_day = st.date_input(
            "Jour exact observations meteo",
            value=datetime.now(timezone.utc).date(),
            min_value=datetime(2010, 1, 1, tzinfo=timezone.utc).date(),
            max_value=datetime.now(timezone.utc).date(),
        )
    selected_history_month_labels = _multiselect_with_all(
        "Mois historiques (Jan-Dec)",
        MONTH_ORDER_SHORT,
        key="flt_hist_months",
    )
    selected_history_month_numbers = [
        int(MONTH_SHORT_TO_NUM[m])
        for m in selected_history_month_labels
        if str(m) in MONTH_SHORT_TO_NUM
    ]
    if not selected_history_month_numbers:
        selected_history_month_numbers = list(MONTH_LABELS_SHORT.keys())

    min_risk = st.selectbox("Risque minimum", ["Tout", "FAIBLE", "MODERE", "ELEVE", "CRITIQUE"], index=2)
    weather_risk_mode = st.selectbox(
        "Filtre meteo",
        ["Operationnel renforce (recommande)", "Pluie brute (niveau source)"],
        index=0,
    )
    sector_risk_mode = st.selectbox("Filtre secteurs", ["IA predictive", "Operationnel"], index=0)
    ai_probability_min_pct = st.slider(
        "Seuil IA predictive min (%)",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        help="Filtre les secteurs selon la probabilite IA predictive.",
    )
    enable_live_synop_assist = st.checkbox(
        "Secours SYNOP live si snapshot incomplet",
        value=True,
        help="Ajoute des stations SYNOP live (source officielle Meteo-France) si le snapshot est incomplet ou stale.",
    )
    march_compare_top_n = st.slider(
        "Comparatif Mars: nb stations proches",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )
    auto_live_open_meteo_compare = st.checkbox(
        "Comparatif Mars: completer Open-Meteo live si absent",
        value=True,
    )

    if enable_live_synop_assist:
        src_series_for_count = weather_df.get("source", pd.Series("", index=weather_df.index)).fillna("").astype(str)
        snapshot_infoclimat_count = int(src_series_for_count.map(_is_infoclimat_station_source).fillna(False).sum())
        if snapshot_infoclimat_count <= 0:
            live_synop_df, live_synop_notice = _load_live_synop_weather_pack()
            if not live_synop_df.empty:
                live_ref_ts = snapshot_ts if snapshot_ts is not None and not pd.isna(snapshot_ts) else pd.Timestamp.now(tz="UTC")
                live_synop_df = _prepare_weather_df(live_synop_df, live_ref_ts)
                before_rows = int(len(weather_df))
                weather_df = pd.concat([weather_df, live_synop_df], ignore_index=True)
                weather_df = _prepare_weather_df(weather_df, live_ref_ts)
                live_synop_added_count = max(0, int(len(weather_df)) - before_rows)

    sources = sorted(weather_df["source"].dropna().astype(str).unique().tolist()) if "source" in weather_df.columns else []
    selected_sources = _multiselect_with_all("Sources meteo", sources, key="flt_sources")
    weather_source_profile = st.selectbox(
        "Source de data meteo",
        WEATHER_SOURCE_PROFILE_OPTIONS,
        index=0,
        help="Choix rapide de la source: Open-Meteo, InfoClimat/SYNOP, ou mode auto.",
    )
    weather_source_mode = st.selectbox(
        "Mode source meteo",
        WEATHER_SOURCE_FILTER_MODES,
        index=2,
        help="Permet de cibler les stations InfoClimat/SYNOP, avec option proximite LGV SEA.",
        disabled=(weather_source_profile != "Auto (recommande)"),
    )
    infoclimat_max_distance_km = st.slider(
        "Distance max InfoClimat a la LGV (km)",
        min_value=1.0,
        max_value=50.0,
        value=float(DEFAULT_INFOCLIMAT_LGV_MAX_DISTANCE_KM),
        step=0.5,
        disabled=not (
            weather_source_profile == "InfoClimat/SYNOP <= distance LGV"
            or (weather_source_profile == "Auto (recommande)" and weather_source_mode == "InfoClimat/SYNOP proches LGV SEA")
        ),
    )
    enforce_pluvio_quality = st.checkbox(
        "Qualite pluvio renforcee (recommande)",
        value=True,
        help="Exclut automatiquement les observations pluvio les moins fiables (age, incoherences, valeurs extremes).",
    )
    strict_reliable_mode = st.checkbox(
        "Mode donnees fiables strict",
        value=bool(DEFAULT_STRICT_RELIABLE_MODE),
        help="Si active: pas de fallback Open-Meteo quand aucune station InfoClimat/SYNOP fiable n'est disponible.",
    )

    commune_values: List[str] = []
    if "commune_name" in sectors_df.columns:
        commune_values.extend(sectors_df["commune_name"].dropna().astype(str).unique().tolist())
    if "commune_name" in lgv_communes_df.columns:
        commune_values.extend(lgv_communes_df["commune_name"].dropna().astype(str).unique().tolist())
    communes = sorted(_unique_text_values(commune_values))
    selected_communes = _multiselect_with_all("Communes", communes, key="flt_communes")

    station_communes = (
        sorted(weather_df["station_commune_name"].dropna().astype(str).unique().tolist())
        if "station_commune_name" in weather_df.columns
        else []
    )
    selected_station_communes = _multiselect_with_all(
        "Communes des stations meteo",
        station_communes,
        key="flt_station_communes",
    )
    weather_min_quality = st.slider("Qualite meteo min (/100, 0=desactive)", min_value=0, max_value=100, value=0, step=5)
    weather_max_age_h = st.slider("Age max observations meteo (h, 0=desactive)", min_value=0, max_value=120, value=0, step=6)
    st.caption("Lecture meteo: qualite<55 ou age>30h = A_VERIFIER. Alerte critique >= 82/100.")

    hydro_sources = sorted(hydro_df["source"].dropna().astype(str).unique().tolist()) if "source" in hydro_df.columns else []
    selected_hydro_sources = _multiselect_with_all("Sources hydro", hydro_sources, key="flt_hydro_sources")

    hydro_rivers = sorted(hydro_df["river_name"].dropna().astype(str).unique().tolist()) if "river_name" in hydro_df.columns else []
    selected_hydro_rivers = _multiselect_with_all("Cours d'eau / ruisseaux", hydro_rivers, key="flt_hydro_rivers")
    hydro_risk_filter = st.selectbox("Risque hydro", ["Tout", "FAIBLE", "MODERE", "ELEVE", "CRITIQUE"], index=0)
    hydro_only_exceeded = st.checkbox("Hydro: uniquement seuil urgence depasse", value=False)
    slip_alert_threshold = st.slider("Seuil alerte glissement (/100)", min_value=40, max_value=95, value=68, step=2)
    enable_manual_pk_watch = st.checkbox("Activer zones PK sous surveillance (capture)", value=True)
    default_manual_pk_text = "; ".join([f"{a:.3f}-{b:.3f}" for a, b in DEFAULT_MANUAL_PK_RANGES])
    manual_pk_watch_text = st.text_area(
        "PK sous surveillance (format: 98.244-98.640; 119.590-120.970; ...)",
        value=default_manual_pk_text,
        height=90,
    )

    st.caption("Toutes les communes filtrees sont affichees (pas de limite a 25).")

    st.markdown("---")
    map_basemap_style = st.selectbox(
        "Fond de carte",
        list(MAP_TILE_STYLES.keys()),
        index=0,
    )
    show_weather = st.checkbox("Layer meteo", value=True)
    show_communes = st.checkbox("Layer communes", value=True)
    show_sectors = st.checkbox("Layer secteurs IA", value=True)
    show_hydro = st.checkbox("Layer hydro", value=True)
    show_piezo = st.checkbox("Layer piezometres", value=False)
    show_geotech = st.checkbox("Layer geotech", value=False)
    show_slip = st.checkbox("Layer zones glissement", value=True)
    show_fr_layer = st.checkbox("Layer geographie FR", value=False)

weather_for_context = weather_df.copy()
if not weather_for_context.empty:
    weather_for_context = _apply_weather_source_profile(
        weather_for_context,
        weather_source_profile,
        float(infoclimat_max_distance_km),
    )
if not weather_for_context.empty and selected_sources:
    weather_for_context = weather_for_context[weather_for_context["source"].astype(str).isin(selected_sources)]
if not weather_for_context.empty and weather_source_profile == "Auto (recommande)":
    weather_for_context = _apply_weather_source_filter(
        weather_for_context,
        weather_source_mode,
        float(infoclimat_max_distance_km),
    )
if not weather_for_context.empty:
    weather_for_context = _apply_pluvio_quality_gate(weather_for_context, bool(enforce_pluvio_quality))
if not weather_for_context.empty and selected_station_communes:
    weather_for_context = weather_for_context[weather_for_context["station_commune_name"].astype(str).isin(selected_station_communes)]
if not weather_for_context.empty and int(weather_min_quality) > 0 and "weather_quality_note" in weather_for_context.columns:
    weather_for_context = weather_for_context[
        pd.to_numeric(weather_for_context["weather_quality_note"], errors="coerce").fillna(0.0) >= float(weather_min_quality)
    ]
if not weather_for_context.empty and int(weather_max_age_h) > 0 and "obs_age_h" in weather_for_context.columns:
    weather_for_context = weather_for_context[
        pd.to_numeric(weather_for_context["obs_age_h"], errors="coerce").fillna(999.0) <= float(weather_max_age_h)
    ]
if not weather_for_context.empty:
    weather_for_context = _apply_weather_obs_window(weather_for_context, weather_obs_window, snapshot_ts)
if not weather_for_context.empty and weather_exact_day_enabled:
    obs_day = pd.to_datetime(weather_for_context.get("obs_ts_utc"), utc=True, errors="coerce").dt.date
    weather_for_context = weather_for_context[obs_day == weather_exact_day]

filtered_weather = weather_for_context.copy()
if not filtered_weather.empty and str(min_risk).upper() != "TOUT":
    weather_risk_col = "meteo_operational_level"
    if weather_risk_mode.startswith("Pluie") or weather_risk_col not in filtered_weather.columns:
        weather_risk_col = "risk_level"
    filtered_weather = filtered_weather[filtered_weather[weather_risk_col].map(lambda x: _risk_rank(str(x))) >= _risk_rank(min_risk)]

if int(live_synop_added_count) > 0:
    st.info(
        f"Secours SYNOP live actif: +{int(live_synop_added_count)} station(s) ajoutee(s) "
        "depuis la source officielle Meteo-France."
    )
elif enable_live_synop_assist and live_synop_notice:
    st.caption(f"Notice SYNOP live: {live_synop_notice}")

scope_label = "snapshot + secours live" if enable_live_synop_assist else "snapshot"
if weather_for_context.empty and not weather_df.empty:
    if weather_source_profile == "Open-Meteo uniquement":
        st.warning(f"Aucune donnee Open-Meteo disponible dans ce {scope_label}.")
    elif weather_source_profile in {"InfoClimat/SYNOP uniquement", "InfoClimat/SYNOP <= distance LGV"}:
        if weather_source_profile == "InfoClimat/SYNOP <= distance LGV":
            st.warning(
                f"Aucune station InfoClimat/SYNOP <= {float(infoclimat_max_distance_km):.1f} km de la LGV SEA dans ce {scope_label}."
            )
        else:
            st.warning(f"Aucune station InfoClimat/SYNOP disponible dans ce {scope_label}.")
    elif weather_source_mode != "Toutes sources":
        if weather_source_mode == "InfoClimat/SYNOP proches LGV SEA":
            st.warning(
                f"Aucune station InfoClimat/SYNOP <= {float(infoclimat_max_distance_km):.1f} km de la LGV SEA dans ce {scope_label}."
            )
        else:
            st.warning(f"Aucune station InfoClimat/SYNOP disponible dans ce {scope_label}.")
    if strict_reliable_mode:
        st.info("Mode strict actif: les estimations Open-Meteo ne sont pas utilisees pour remplacer ces donnees.")

weather_signal_df = _select_weather_pool_for_ui(
    filtered_weather,
    weather_for_context,
    weather_df,
    strict_reliable_mode=bool(strict_reliable_mode),
)
effective_rain_col_weather = _choose_weather_signal_column(
    weather_signal_df,
    rain_col_weather,
    ["rain_30d_mm", "rain_7d_mm", "rain_24h_mm", "rain_month_mm"],
)
RAIN_COL_LABELS = {
    "rain_24h_mm": "24h",
    "rain_7d_mm": "7 jours",
    "rain_30d_mm": "30 jours",
    "rain_month_mm": "Mois courant",
}
effective_period_label = RAIN_COL_LABELS.get(effective_rain_col_weather, period_label)
weather_signal_fallback = effective_rain_col_weather != rain_col_weather

march_year = _target_march_year(snapshot_ts)
march_period_start = pd.Timestamp(year=int(march_year), month=3, day=1).strftime("%Y-%m-%d")
march_period_end = (pd.Timestamp(year=int(march_year), month=3, day=1) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
infoclimat_candidates = weather_df[
    weather_df.get("source", pd.Series("", index=weather_df.index)).fillna("").astype(str).map(_is_infoclimat_station_source).fillna(False)
].copy() if not weather_df.empty else pd.DataFrame()
open_meteo_candidates = weather_df[
    weather_df.get("source", pd.Series("", index=weather_df.index)).fillna("").astype(str).map(_is_open_meteo_station_source).fillna(False)
].copy() if not weather_df.empty else pd.DataFrame()

if open_meteo_candidates.empty and auto_live_open_meteo_compare:
    live_open_meteo_df, live_open_meteo_notice = _load_live_open_meteo_weather_pack()
    if not live_open_meteo_df.empty:
        live_ref_ts = snapshot_ts if snapshot_ts is not None and not pd.isna(snapshot_ts) else pd.Timestamp.now(tz="UTC")
        live_open_meteo_df = _prepare_weather_df(live_open_meteo_df, live_ref_ts)
        open_meteo_candidates = live_open_meteo_df.copy()
        live_open_meteo_added_count = int(len(live_open_meteo_df))

march_comparison_df = _build_march_source_comparison(
    infoclimat_df=infoclimat_candidates,
    open_meteo_df=open_meteo_candidates,
    march_year=int(march_year),
    top_n=int(march_compare_top_n),
)

st.subheader("Comparatif Mars - InfoClimat vs Open-Meteo (stations proches)")
st.caption(
    f"Periode comparee: {march_period_start} -> {march_period_end} | "
    f"Top {int(march_compare_top_n)} station(s) les plus proches par source."
)
if int(live_open_meteo_added_count) > 0:
    st.caption(
        f"Open-Meteo live ajoute pour comparatif: {int(live_open_meteo_added_count)} point(s)."
    )
elif auto_live_open_meteo_compare and live_open_meteo_notice:
    st.caption(f"Notice Open-Meteo live: {live_open_meteo_notice}")

if march_comparison_df.empty:
    st.warning(
        "Comparatif Mars indisponible: source InfoClimat/SYNOP ou Open-Meteo manquante."
    )
else:
    st.dataframe(march_comparison_df, use_container_width=True, hide_index=True)

manual_pk_ranges = _parse_manual_pk_ranges(manual_pk_watch_text) if enable_manual_pk_watch else []
if not sectors_df.empty:
    sectors_df = _build_slip_assessment(sectors_df, manual_pk_ranges)

filtered_sectors = sectors_df.copy()
if not filtered_sectors.empty and selected_communes:
    filtered_sectors = filtered_sectors[filtered_sectors["commune_name"].astype(str).isin(selected_communes)]
if not filtered_sectors.empty and str(min_risk).upper() != "TOUT":
    sector_risk_col = "risk_level"
    if sector_risk_mode.startswith("IA") and "ai_pred_risk_level" in filtered_sectors.columns:
        sector_risk_col = "ai_pred_risk_level"
    filtered_sectors = filtered_sectors[filtered_sectors[sector_risk_col].map(lambda x: _risk_rank(str(x))) >= _risk_rank(min_risk)]
if not filtered_sectors.empty and "ai_pred_probability" in filtered_sectors.columns and int(ai_probability_min_pct) > 0:
    min_prob = float(ai_probability_min_pct) / 100.0
    filtered_sectors = filtered_sectors[
        pd.to_numeric(filtered_sectors["ai_pred_probability"], errors="coerce").fillna(0.0) >= min_prob
    ]

slip_source_df = filtered_sectors if not filtered_sectors.empty else sectors_df
slip_corridors_df = _build_slip_corridors(slip_source_df, float(slip_alert_threshold))
if not slip_source_df.empty and "slip_index" in slip_source_df.columns:
    slip_focus_df = slip_source_df[
        (pd.to_numeric(slip_source_df.get("slip_index"), errors="coerce").fillna(0.0) >= float(slip_alert_threshold))
        | (slip_source_df.get("manual_watch_pk", pd.Series(False, index=slip_source_df.index)).fillna(False).astype(bool))
    ].copy()
else:
    slip_focus_df = pd.DataFrame()

filtered_hydro = hydro_df.copy()
if not filtered_hydro.empty and selected_hydro_sources and "source" in filtered_hydro.columns:
    filtered_hydro = filtered_hydro[filtered_hydro["source"].astype(str).isin(selected_hydro_sources)]
if not filtered_hydro.empty and selected_hydro_rivers and "river_name" in filtered_hydro.columns:
    filtered_hydro = filtered_hydro[filtered_hydro["river_name"].astype(str).isin(selected_hydro_rivers)]
if not filtered_hydro.empty and str(hydro_risk_filter).upper() != "TOUT" and "risk_level" in filtered_hydro.columns:
    filtered_hydro = filtered_hydro[
        filtered_hydro["risk_level"].map(lambda x: _risk_rank(str(x))) >= _risk_rank(str(hydro_risk_filter).upper())
    ]
if not filtered_hydro.empty and hydro_only_exceeded and "threshold_exceeded" in filtered_hydro.columns:
    filtered_hydro = filtered_hydro[filtered_hydro["threshold_exceeded"].fillna(False).astype(bool)]
if not filtered_hydro.empty and str(min_risk).upper() != "TOUT" and "risk_level" in filtered_hydro.columns:
    filtered_hydro = filtered_hydro[filtered_hydro["risk_level"].map(lambda x: _risk_rank(str(x))) >= _risk_rank(min_risk)]

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
if not lgv_communes_df.empty:
    lgv_pool = lgv_communes_df.copy()
    lgv_pool["commune_name"] = lgv_pool.get("commune_name", "Inconnue").fillna("Inconnue").astype(str)
    lgv_pool["commune_code"] = lgv_pool.get("commune_code", "").fillna("").astype(str)
    if "departement_code" not in lgv_pool.columns:
        lgv_pool["departement_code"] = ""
    if "departement_name" not in lgv_pool.columns:
        lgv_pool["departement_name"] = ""
    lgv_pool["latitude"] = pd.to_numeric(lgv_pool.get("centroid_latitude"), errors="coerce")
    lgv_pool["longitude"] = pd.to_numeric(lgv_pool.get("centroid_longitude"), errors="coerce")
    lgv_pool = lgv_pool[
        ["commune_name", "commune_code", "departement_code", "departement_name", "latitude", "longitude"]
    ].drop_duplicates(subset=["commune_code", "commune_name"], keep="first")

    if commune_pool.empty:
        commune_pool = lgv_pool.copy()
    else:
        existing_codes = set(commune_pool.get("commune_code", pd.Series(dtype=str)).fillna("").astype(str).tolist())
        existing_names = set(commune_pool.get("commune_name", pd.Series(dtype=str)).fillna("").astype(str).tolist())
        missing_rows = lgv_pool[
            ~lgv_pool.apply(
                lambda r: (str(r.get("commune_code") or "") in existing_codes and str(r.get("commune_code") or "").strip() != "")
                or (str(r.get("commune_name") or "") in existing_names),
                axis=1,
            )
        ]
        if not missing_rows.empty:
            commune_pool = pd.concat([commune_pool, missing_rows], ignore_index=True, sort=False)

if not commune_pool.empty:
    commune_pool["commune_code"] = commune_pool.get("commune_code", "").fillna("").astype(str)
    commune_pool["commune_label"] = commune_pool.apply(
        lambda r: f"{str(r.get('commune_name') or 'Inconnue')} ({str(r.get('commune_code') or '')})"
        if str(r.get("commune_code") or "").strip()
        else str(r.get("commune_name") or "Inconnue"),
        axis=1,
    )

    weather_ctx_source = _select_weather_pool_for_ui(
        filtered_weather,
        weather_for_context,
        weather_df,
        strict_reliable_mode=bool(strict_reliable_mode),
    )
    weather_context_df = _build_commune_weather_context(
        commune_pool[["commune_label", "latitude", "longitude"]],
        weather_ctx_source,
        radius_km=12.0,
        min_points=3,
    )
    if not weather_context_df.empty:
        ctx_cols = [c for c in weather_context_df.columns if c != "commune_label"]
        commune_pool = commune_pool.drop(columns=[c for c in ctx_cols if c in commune_pool.columns], errors="ignore")
        commune_pool = commune_pool.merge(weather_context_df, on="commune_label", how="left")
        if not commune_df.empty and "commune_label" in commune_df.columns:
            commune_df = commune_df.drop(columns=[c for c in ctx_cols if c in commune_df.columns], errors="ignore")
            commune_df = commune_df.merge(weather_context_df, on="commune_label", how="left")

    if not commune_df.empty and "weather_alert_index_commune" in commune_df.columns:
        base_weather_note = pd.to_numeric(commune_df.get("weather_component_note"), errors="coerce").fillna(0.0)
        ctx_alert = pd.to_numeric(commune_df.get("weather_alert_index_commune"), errors="coerce").fillna(base_weather_note)
        ctx_quality = pd.to_numeric(commune_df.get("weather_quality_note_commune"), errors="coerce").fillna(60.0).clip(lower=0.0, upper=100.0)
        ctx_priority = pd.to_numeric(commune_df.get("weather_watch_priority_commune"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)

        commune_df["weather_component_note_raw"] = base_weather_note.round(1)
        commune_df["weather_component_note"] = (
            base_weather_note * 0.65 + ctx_alert * 0.35
        ).clip(lower=0.0, upper=100.0).round(1)

        quality_factor = (ctx_quality / 100.0).clip(lower=0.30, upper=1.0)
        weather_effective_note = commune_df["weather_component_note"] * (0.55 + 0.45 * quality_factor)
        commune_df["risk_score"] = (
            pd.to_numeric(commune_df.get("risk_score"), errors="coerce").fillna(0.0) * 0.84
            + weather_effective_note * 0.16
        ).clip(lower=0.0, upper=100.0)
        high_watch_mask = ctx_priority >= 85.0
        commune_df.loc[high_watch_mask, "risk_score"] = (
            commune_df.loc[high_watch_mask, "risk_score"] + 3.0
        ).clip(upper=100.0)
        commune_df["risk_score"] = pd.to_numeric(commune_df["risk_score"], errors="coerce").fillna(0.0).round(1)
        commune_df["commune_risk_level"] = commune_df["risk_score"].map(_risk_level_from_note)
        commune_df["note_gc"] = commune_df["risk_score"]

        sync_cols = [
            "commune_risk_level",
            "risk_score",
            "note_gc",
            "weather_component_note",
            "weather_component_note_raw",
            "weather_points_used",
            "weather_mean_dist_km",
            "weather_quality_note_commune",
            "weather_obs_age_h_commune",
            "weather_alert_index_commune",
            "weather_watch_priority_commune",
            "weather_24h_commune_mm",
            "weather_7d_commune_mm",
            "weather_30d_commune_mm",
            "weather_month_commune_mm",
            "weather_forecast_commune_mm",
            "weather_alert_level_commune",
            "weather_reliability_flag",
            "weather_obs_freshness_commune",
            "weather_action_commune",
        ]
        sync_cols = [c for c in sync_cols if c in commune_df.columns]
        if sync_cols:
            sync_df = commune_df[["commune_label"] + sync_cols].drop_duplicates(subset=["commune_label"], keep="first")
            commune_pool = commune_pool.drop(columns=[c for c in sync_cols if c in commune_pool.columns], errors="ignore")
            commune_pool = commune_pool.merge(sync_df, on="commune_label", how="left")

    defaults = {
        "commune_risk_level": "INDETERMINE",
        "risk_score": 0.0,
        "lgv_points_count": 0,
        "ai_commune_risk_level": "INDETERMINE",
        "max_ai_probability": 0.0,
        "weather_quality_note_commune": 0.0,
        "weather_alert_index_commune": 0.0,
        "weather_watch_priority_commune": 0.0,
        "weather_reliability_flag": "A_VERIFIER",
        "weather_obs_freshness_commune": "OBSOLETE",
        "weather_action_commune": WEATHER_OP_ACTIONS["INDETERMINE"],
    }
    for col, default in defaults.items():
        if col not in commune_pool.columns:
            commune_pool[col] = default
        else:
            commune_pool[col] = commune_pool[col].fillna(default)

selected_commune: Dict[str, object] = {}
history_years = 5
history_fetch_limit = 0
history_compare_enabled = False
selected_compare_commune_labels: List[str] = []
compare_commune_rows: List[Dict[str, object]] = []
compare_history_df = pd.DataFrame()
compare_history_full_df = pd.DataFrame()
history_models: Dict[str, str] = {}
selected_daily_day = datetime.now(timezone.utc).date()
selected_daily_day_precip: Dict[str, object] = {"date": None, "precip_mm": None, "model": "", "error": "pas de commune"}
selected_commune_weather_reference: Dict[str, object] = {"source_mode": "AUCUNE_DONNEE", "source_label": "pas de commune"}
if not commune_pool.empty:
    with st.sidebar:
        st.markdown("---")
        st.subheader("Analyse commune")
        commune_labels = commune_pool["commune_label"].astype(str).tolist()
        chosen_commune_label = st.selectbox("Commune detail", commune_labels, index=0)
        history_compare_enabled = st.checkbox("Activer comparaison historique (plus lent)", value=False)
        compare_all_communes = st.checkbox("Comparer toutes les communes", value=False, key="analysis_compare_all")
        if compare_all_communes:
            selected_compare_commune_labels = list(commune_labels)
        elif not history_compare_enabled:
            selected_compare_commune_labels = [chosen_commune_label]
        else:
            selected_compare_commune_labels = st.multiselect(
                "Communes a comparer",
                commune_labels,
                default=commune_labels[: min(12, len(commune_labels))],
                key="analysis_compare_communes",
            )
        history_years = st.slider("Historique mensuel (ans)", min_value=2, max_value=10, value=5, step=1)
        selected_daily_day = st.date_input(
            "Recherche pluvio (jour precis)",
            value=datetime.now(timezone.utc).date(),
            min_value=datetime(2010, 1, 1, tzinfo=timezone.utc).date(),
            max_value=datetime.now(timezone.utc).date(),
            help="Charge la pluie journaliere archivee Open-Meteo pour la commune selectionnee.",
        )
        history_fetch_limit = st.slider(
            "Max communes historique (0 = toutes)",
            min_value=0,
            max_value=max(0, len(commune_labels)),
            value=min(40, len(commune_labels)),
            step=1 if len(commune_labels) <= 40 else 5,
        )
    selected_commune = commune_pool[commune_pool["commune_label"].astype(str) == chosen_commune_label].iloc[0].to_dict()
    if chosen_commune_label not in selected_compare_commune_labels:
        selected_compare_commune_labels = [chosen_commune_label] + list(selected_compare_commune_labels)

    for label in selected_compare_commune_labels:
        hit = commune_pool[commune_pool["commune_label"].astype(str) == str(label)]
        if not hit.empty:
            compare_commune_rows.append(hit.iloc[0].to_dict())
    if history_compare_enabled and compare_commune_rows:
        if int(history_fetch_limit) > 0:
            compare_history_targets = compare_commune_rows[: int(history_fetch_limit)]
        else:
            compare_history_targets = compare_commune_rows
        compare_history_df, history_models = _build_multi_commune_history(compare_history_targets, int(history_years))
        compare_history_full_df = compare_history_df.copy()

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

    if selected_history_month_numbers:
        compare_history_df = _filter_history_months(compare_history_df, selected_history_month_numbers)
        compare_history_full_df = _filter_history_months(compare_history_full_df, selected_history_month_numbers)

    if selected_commune:
        try:
            selected_daily_day_precip = _load_sector_daily_precip_on_date(
                float(selected_commune["latitude"]),
                float(selected_commune["longitude"]),
                selected_daily_day.isoformat(),
            )
        except Exception as exc:
            selected_daily_day_precip = {
                "date": selected_daily_day.isoformat(),
                "precip_mm": None,
                "model": "",
                "error": str(exc),
            }
        try:
            weather_reference_pool = _select_weather_pool_for_ui(
                filtered_weather,
                weather_for_context,
                weather_df,
                strict_reliable_mode=bool(strict_reliable_mode),
            )
            selected_commune_weather_reference = _build_commune_weather_reference(
                selected_commune,
                weather_reference_pool,
                snapshot_ts,
                allow_open_meteo_fallback=not bool(strict_reliable_mode),
            )
        except Exception as exc:
            selected_commune_weather_reference = {
                "source_mode": "AUCUNE_DONNEE",
                "source_label": f"erreur reference commune: {exc}",
            }

history_payload = {"monthly": [], "climatology": [], "model": "", "error": "pas de commune"}
history_month_aligned = False
if selected_commune:
    hist_lat = pd.to_numeric(selected_commune_weather_reference.get("ref_latitude"), errors="coerce")
    hist_lon = pd.to_numeric(selected_commune_weather_reference.get("ref_longitude"), errors="coerce")
    if pd.isna(hist_lat) or pd.isna(hist_lon):
        hist_lat = pd.to_numeric(selected_commune.get("latitude"), errors="coerce")
        hist_lon = pd.to_numeric(selected_commune.get("longitude"), errors="coerce")
    history_payload = _load_sector_monthly_history(
        float(hist_lat), float(hist_lon), int(history_years)
    )
history_monthly_df = _safe_df(history_payload.get("monthly"))
history_clim_df = _safe_df(history_payload.get("climatology"))
if selected_commune and history_monthly_df.empty:
    selected_label = str(
        f"{selected_commune.get('commune_name', 'Inconnue')} ({selected_commune.get('commune_code', '')})"
        if str(selected_commune.get("commune_code") or "").strip()
        else str(selected_commune.get("commune_name", "Inconnue"))
    )
    fallback_monthly = pd.DataFrame()
    if not compare_history_full_df.empty:
        hit = compare_history_full_df[compare_history_full_df["commune_label"].astype(str) == selected_label]
        if not hit.empty:
            fallback_monthly = hit.copy()
        else:
            month_pool = compare_history_full_df.copy()
            month_pool["ym"] = month_pool["ym"].astype(str)
            month_pool["monthly_precip_mm"] = pd.to_numeric(month_pool["monthly_precip_mm"], errors="coerce")
            fallback_monthly = (
                month_pool.groupby("ym", as_index=False)["monthly_precip_mm"]
                .median()
                .dropna(subset=["monthly_precip_mm"])
            )
            if not fallback_monthly.empty:
                fallback_monthly["year"] = pd.to_numeric(fallback_monthly["ym"].str.slice(0, 4), errors="coerce").fillna(0).astype(int)
                fallback_monthly["month"] = pd.to_numeric(fallback_monthly["ym"].str.slice(5, 7), errors="coerce").fillna(0).astype(int)
                fallback_monthly["commune_label"] = selected_label
                fallback_monthly["commune_name"] = str(selected_commune.get("commune_name") or "Inconnue")

    if not fallback_monthly.empty:
        history_monthly_df = fallback_monthly.copy()
        history_payload["error"] = None
        history_payload["model"] = str(history_payload.get("model") or "fallback_reference_compare")
        clim_calc = (
            history_monthly_df.groupby("month", as_index=False)["monthly_precip_mm"]
            .mean()
            .rename(columns={"monthly_precip_mm": "climatology_mm"})
        )
        clim_calc["climatology_mm"] = pd.to_numeric(clim_calc["climatology_mm"], errors="coerce").round(1)
        clim_calc["month_label"] = clim_calc["month"].map(MONTH_LABELS_SHORT)
        history_clim_df = clim_calc

if selected_commune and not history_monthly_df.empty:
    history_monthly_df, history_month_aligned = _align_history_month_with_reference(
        history_monthly_df,
        selected_commune_weather_reference,
    )

history_monthly_df = _filter_history_months(history_monthly_df, selected_history_month_numbers)
history_clim_df = _filter_history_months(history_clim_df, selected_history_month_numbers)
if history_clim_df.empty and not history_monthly_df.empty:
    history_clim_df = _build_monthly_climatology(history_monthly_df)

pluvio_ranking_df = _build_pluvio_ranking(
    commune_pool.to_dict(orient="records") if not commune_pool.empty else compare_commune_rows,
    _select_weather_pool_for_ui(
        filtered_weather,
        weather_for_context,
        weather_df,
        strict_reliable_mode=bool(strict_reliable_mode),
    ),
    compare_history_full_df,
)
risk_level = str(snapshot.get("risk_level", "INDETERMINE"))
score = float(snapshot.get("score", 0.0) or 0.0)

hydro_exceeded_count = (
    int(filtered_hydro["threshold_exceeded"].sum())
    if (not filtered_hydro.empty and "threshold_exceeded" in filtered_hydro.columns)
    else 0
)
total_lgv_communes = int(len(lgv_communes_df)) if not lgv_communes_df.empty else int(len(commune_df))
ai_critical_count = int((filtered_sectors.get("ai_pred_risk_level", pd.Series(dtype=str)) == "CRITIQUE").sum()) if not filtered_sectors.empty else 0
fragile_soil_count = (
    int((pd.to_numeric(filtered_sectors.get("ai_soil_fragility", 0.0), errors="coerce").fillna(0.0) >= 0.70).sum())
    if not filtered_sectors.empty
    else 0
)
slip_high_count = (
    int((pd.to_numeric(slip_source_df.get("slip_index"), errors="coerce").fillna(0.0) >= float(slip_alert_threshold)).sum())
    if not slip_source_df.empty
    else 0
)
slip_critical_count = (
    int((slip_source_df.get("slip_level", pd.Series(dtype=str)).astype(str) == "CRITIQUE").sum())
    if not slip_source_df.empty
    else 0
)
manual_watch_count = (
    int(slip_source_df.get("manual_watch_pk", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    if not slip_source_df.empty
    else 0
)

col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)
col1.metric("Risque global", risk_level)
col2.metric("Score global", f"{score:.2f}/4")
col3.metric("Stations meteo", int(len(filtered_weather)))
col4.metric("Points LGV filtres", int(len(filtered_sectors)))
col5.metric("Communes traversees LGV", total_lgv_communes)
col6.metric("Hydro seuil urgence", hydro_exceeded_count)
col7.metric("Secteurs IA critiques", ai_critical_count)
col8.metric("Secteurs sols fragiles", fragile_soil_count)
col9.metric("Zones glissement >= seuil", slip_high_count)
col10.metric("PK surveillance manuelle", manual_watch_count)
if snapshot_age_h is None:
    col11.metric("Age snapshot", "N/A")
else:
    col11.metric("Age snapshot", f"{snapshot_age_h:.1f} h")

if snapshot_ts is None or pd.isna(snapshot_ts):
    st.caption(f"Snapshot source: {snapshot_source} | timestamp: inconnu")
else:
    st.caption(
        "Snapshot source: "
        + f"{snapshot_source} | timestamp_utc={snapshot_ts.isoformat()} | age={0.0 if snapshot_age_h is None else snapshot_age_h:.1f} h"
    )

tabs = st.tabs(["Vue executive", "Carte dynamique", "Tables et alertes", "Metadata"])

with tabs[0]:
    left, right = st.columns([1.5, 1.0])
    weather_summary_df = pd.DataFrame()

    with left:
        st.subheader("Classement complet des communes (score risque)")
        if commune_df.empty:
            st.info("Aucune commune pour les filtres courants.")
        else:
            ranked_communes = commune_df.sort_values("risk_score", ascending=False).copy()
            chart = (
                alt.Chart(ranked_communes)
                .mark_bar()
                .encode(
                    x=alt.X("risk_score:Q", title="Score risque /100"),
                    y=alt.Y("commune_label:N", sort=alt.SortField(field="risk_score", order="descending"), title="Commune"),
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
                        "risk_score",
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
        if filtered_weather.empty or effective_rain_col_weather not in filtered_weather.columns:
            st.info("Pas de donnees pluie pour ce filtre.")
        else:
            weather_summary_df = filtered_weather.copy()
            max_rain = float(pd.to_numeric(weather_summary_df[effective_rain_col_weather], errors="coerce").fillna(0.0).max())
            mean_rain = float(pd.to_numeric(weather_summary_df[effective_rain_col_weather], errors="coerce").fillna(0.0).mean())
            st.metric(f"Max {effective_period_label}", f"{max_rain:.1f} mm")
            st.metric(f"Moyenne {effective_period_label}", f"{mean_rain:.1f} mm")
            if weather_signal_fallback:
                st.caption(
                    f"Periode demandee '{period_label}' sans signal exploitable sur ce snapshot. "
                    f"Affichage bascule sur '{effective_period_label}'."
                )
            if "weather_quality_note" in weather_summary_df.columns:
                q_mean = pd.to_numeric(weather_summary_df["weather_quality_note"], errors="coerce").mean()
                st.metric("Qualite meteo moyenne", f"{0.0 if pd.isna(q_mean) else float(q_mean):.1f}/100")
            if "obs_age_h" in weather_summary_df.columns:
                stale_count = int((pd.to_numeric(weather_summary_df["obs_age_h"], errors="coerce").fillna(999.0) > 24.0).sum())
                st.metric("Stations > 24h", stale_count)
            if "weather_alert_index" in weather_summary_df.columns:
                alert_mean = pd.to_numeric(weather_summary_df["weather_alert_index"], errors="coerce").fillna(0.0).mean()
                critical_alert = int((pd.to_numeric(weather_summary_df["weather_alert_index"], errors="coerce").fillna(0.0) >= WEATHER_ALERT_THRESHOLDS["CRITIQUE"]).sum())
                st.metric("Indice alerte moyen", f"{float(alert_mean):.1f}/100")
                st.metric("Stations alerte critique", critical_alert)
            if "weather_data_reliability" in weather_summary_df.columns:
                to_verify_count = int((weather_summary_df["weather_data_reliability"].astype(str) == "A_VERIFIER").sum())
                st.metric("Stations A_VERIFIER", to_verify_count)

    if not weather_summary_df.empty:
        st.markdown("**Lecture meteo sans ambiguite (regles de decision)**")
        st.caption(
            "Les stations sont classees par indice d'alerte, qualite et anciennete. "
            "Le niveau operationnel guide l'action GC."
        )
        weather_legend_df = pd.DataFrame(
            [
                {
                    "Niveau operationnel": "FAIBLE",
                    "Indice alerte meteo": "< 45",
                    "Action GC": WEATHER_OP_ACTIONS["FAIBLE"],
                },
                {
                    "Niveau operationnel": "MODERE",
                    "Indice alerte meteo": "45 a 64.9",
                    "Action GC": WEATHER_OP_ACTIONS["MODERE"],
                },
                {
                    "Niveau operationnel": "ELEVE",
                    "Indice alerte meteo": "65 a 81.9",
                    "Action GC": WEATHER_OP_ACTIONS["ELEVE"],
                },
                {
                    "Niveau operationnel": "CRITIQUE",
                    "Indice alerte meteo": ">= 82",
                    "Action GC": WEATHER_OP_ACTIONS["CRITIQUE"],
                },
            ]
        )
        st.dataframe(weather_legend_df, use_container_width=True, hide_index=True)

        if "meteo_operational_level" in weather_summary_df.columns:
            level_dist_df = (
                weather_summary_df["meteo_operational_level"]
                .fillna("INDETERMINE")
                .astype(str)
                .value_counts()
                .rename_axis("meteo_operational_level")
                .reset_index(name="count")
            )
            level_chart = (
                alt.Chart(level_dist_df)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Nombre de stations"),
                    y=alt.Y("meteo_operational_level:N", sort="-x", title="Niveau meteo"),
                    color=alt.Color(
                        "meteo_operational_level:N",
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
                    tooltip=["meteo_operational_level", "count"],
                )
            )
            st.altair_chart(level_chart, use_container_width=True)

    st.subheader("Composantes de risque par commune + score global")
    if commune_df.empty:
        st.info("Pas de donnees composantes a afficher.")
    else:
        component_map = {
            "weather_component_note": "Risque pluie",
            "geotech_component_note": "Risque geotechnique",
            "piezo_component_note": "Risque nappes (piezo)",
            "hydro_component_note": "Risque hydro",
            "ai_component_note": "Risque IA pluie+sol",
            "risk_score": "Score risque global",
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
        comp_long = comp_long.merge(commune_df[["commune_label", "risk_score"]], on="commune_label", how="left")
        component_order = [
            "Risque pluie",
            "Risque geotechnique",
            "Risque nappes (piezo)",
            "Risque hydro",
            "Risque IA pluie+sol",
            "Score risque global",
        ]
        heatmap = (
            alt.Chart(comp_long)
            .mark_rect()
            .encode(
                x=alt.X("component_label:N", sort=component_order, title="Composante"),
                y=alt.Y("commune_label:N", sort=alt.SortField(field="risk_score", order="descending"), title="Commune"),
                color=alt.Color(
                    "component_note:Q",
                    title="Note /100",
                    scale=alt.Scale(scheme="redyellowgreen", reverse=True),
                ),
                tooltip=[
                    "commune_label",
                    "component_label",
                    alt.Tooltip("component_note:Q", title="Note composante", format=".1f"),
                    alt.Tooltip("risk_score:Q", title="Score risque global", format=".1f"),
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

    st.subheader("Comparatif decoupage secteurs (5 / 10 / 20 km)")
    segment_km = st.selectbox("Decoupage de comparaison", [5, 10, 20], index=1, key="segment_compare_km")
    if sector_base_km > float(segment_km):
        st.warning(
            f"Snapshot actuel base sur des secteurs ~{sector_base_km:.1f} km: "
            f"la comparaison {segment_km} km est une approximation. Regenerer en base 5 km pour precision max."
        )
    segment_source_df = filtered_sectors if not filtered_sectors.empty else sectors_df
    segment_df = _build_sector_segmentation_compare(segment_source_df, int(segment_km), float(sector_base_km))
    if segment_df.empty:
        st.info("Pas de donnees secteurs pour le comparatif de decoupage.")
    else:
        seg_cols = [
            "segment_id",
            "segment_km",
            "pk_start_km",
            "pk_end_km",
            "sector_count",
            "risk_level",
            "ai_pred_risk_level",
            "avg_score",
            "max_score",
            "ai_max_probability",
            "rain_24h_max",
            "rain_7d_max",
            "rain_30d_max",
            "hydro_stations_max",
        ]
        seg_cols = [c for c in seg_cols if c in segment_df.columns]
        st.dataframe(segment_df[seg_cols], use_container_width=True, hide_index=True)
        seg_chart = (
            alt.Chart(segment_df)
            .mark_bar()
            .encode(
                x=alt.X("ai_max_probability:Q", title="Probabilite IA max"),
                y=alt.Y("segment_id:N", sort="-x", title="Segment"),
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
                tooltip=seg_cols,
            )
        )
        st.altair_chart(seg_chart, use_container_width=True)

    st.subheader("Surveillance glissement (profil PK sur 300 km)")
    if slip_source_df.empty or "slip_index" not in slip_source_df.columns:
        st.info("Aucun indicateur glissement disponible sur les secteurs.")
    else:
        gl_df = slip_source_df.copy()
        gl_df["pk_km"] = pd.to_numeric(gl_df.get("pk_km"), errors="coerce")
        gl_df["slip_index"] = pd.to_numeric(gl_df.get("slip_index"), errors="coerce")
        gl_df["slip_level"] = gl_df.get("slip_level", pd.Series("FAIBLE", index=gl_df.index)).astype(str)
        gl_df["manual_watch_pk"] = gl_df.get("manual_watch_pk", pd.Series(False, index=gl_df.index)).fillna(False).astype(bool)
        gl_df = gl_df.dropna(subset=["pk_km", "slip_index"]).sort_values("pk_km")

        if gl_df.empty:
            st.info("Profil glissement indisponible sur ce filtre.")
        else:
            st.caption(
                f"Seuil alerte glissement actif: {float(slip_alert_threshold):.0f}/100 | "
                f"Secteurs critiques: {slip_critical_count} | Zones manuelles activees: {manual_watch_count}"
            )
            if enable_manual_pk_watch and manual_pk_ranges:
                st.caption(
                    "PK surveilles (manuel): "
                    + ", ".join([f"{a:.3f}-{b:.3f}" for a, b in manual_pk_ranges[:12]])
                )

            profile_base = alt.Chart(gl_df)
            profile_line = profile_base.mark_line(color="#0f172a").encode(
                x=alt.X("pk_km:Q", title="PK (km)"),
                y=alt.Y("slip_index:Q", title="Indice glissement (/100)"),
                tooltip=[
                    "sector_id",
                    "commune_name",
                    "pk_km",
                    "slip_index",
                    "slip_level",
                    "manual_watch_pk",
                    "slip_drivers",
                ],
            )
            profile_points = profile_base.mark_circle(size=55, opacity=0.85).encode(
                x="pk_km:Q",
                y="slip_index:Q",
                color=alt.Color(
                    "slip_level:N",
                    scale=alt.Scale(
                        domain=["FAIBLE", "MODERE", "ELEVE", "CRITIQUE"],
                        range=[RISK_COLOR["FAIBLE"], RISK_COLOR["MODERE"], RISK_COLOR["ELEVE"], RISK_COLOR["CRITIQUE"]],
                    ),
                    legend=alt.Legend(title="Niveau glissement"),
                ),
                shape=alt.Shape(
                    "manual_watch_pk:N",
                    scale=alt.Scale(domain=[False, True], range=["circle", "diamond"]),
                    legend=alt.Legend(title="PK manuel"),
                ),
            )
            threshold_rule = alt.Chart(pd.DataFrame({"y": [float(slip_alert_threshold)]})).mark_rule(
                color="#7f1d1d",
                strokeDash=[6, 4],
            ).encode(y="y:Q")
            st.altair_chart((profile_line + profile_points + threshold_rule).interactive(), use_container_width=True)

            pk_bin_df = gl_df.copy()
            pk_bin_df["pk_bin_start"] = (np.floor(pk_bin_df["pk_km"] / 10.0) * 10.0).astype(int)
            pk_bin_df["pk_bin_label"] = pk_bin_df["pk_bin_start"].map(lambda v: f"PK {int(v)}-{int(v)+10}")
            pk_bin_df = (
                pk_bin_df.groupby(["pk_bin_start", "pk_bin_label"], as_index=False)
                .agg(
                    slip_index_max=("slip_index", "max"),
                    slip_index_mean=("slip_index", "mean"),
                    sector_count=("sector_id", "count"),
                )
                .sort_values("pk_bin_start")
            )
            pk_bin_chart = (
                alt.Chart(pk_bin_df)
                .mark_bar()
                .encode(
                    x=alt.X("pk_bin_label:N", title="Troncon 10 km"),
                    y=alt.Y("slip_index_max:Q", title="Indice glissement max (/100)"),
                    color=alt.Color("slip_index_max:Q", scale=alt.Scale(scheme="orangered"), title="Intensite"),
                    tooltip=["pk_bin_label", "slip_index_max", "slip_index_mean", "sector_count"],
                )
            )
            st.altair_chart(pk_bin_chart, use_container_width=True)

            if slip_corridors_df.empty:
                st.info("Aucun corridor glissement au-dessus du seuil sur ce filtre.")
            else:
                corr_chart_df = slip_corridors_df.head(25).copy()
                corr_chart = (
                    alt.Chart(corr_chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("slip_index_max:Q", title="Indice glissement max (/100)"),
                        y=alt.Y("slip_corridor_id:N", sort="-x", title="Corridor"),
                        color=alt.Color(
                            "slip_level:N",
                            scale=alt.Scale(
                                domain=["FAIBLE", "MODERE", "ELEVE", "CRITIQUE"],
                                range=[RISK_COLOR["FAIBLE"], RISK_COLOR["MODERE"], RISK_COLOR["ELEVE"], RISK_COLOR["CRITIQUE"]],
                            ),
                        ),
                        tooltip=[
                            "slip_corridor_id",
                            "pk_start_km",
                            "pk_end_km",
                            "corridor_length_km",
                            "slip_index_max",
                            "slip_index_mean",
                            "sector_count",
                            "manual_watch_count",
                            "commune_dominante",
                        ],
                    )
                )
                st.altair_chart(corr_chart, use_container_width=True)

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
    if filtered_weather.empty or effective_rain_col_weather not in filtered_weather.columns:
        st.info("Pas de station meteo pour ce filtre.")
    else:
        station_rank_mode = st.selectbox(
            "Classement stations meteo",
            ["Priorite de surveillance", f"Cumul {effective_period_label}"],
            index=0,
            key="station_rank_mode",
        )
        station_rank_col = "weather_watch_priority" if "weather_watch_priority" in filtered_weather.columns else effective_rain_col_weather
        station_rank_title = "Priorite de surveillance (/100)"
        if station_rank_mode.startswith("Cumul"):
            station_rank_col = effective_rain_col_weather
            station_rank_title = f"Cumul {effective_period_label} (mm)"

        top_stations = filtered_weather.sort_values(station_rank_col, ascending=False).head(25).copy()
        top_stations["station_commune_name"] = top_stations.get("station_commune_name", "Inconnue").fillna("Inconnue").astype(str)
        top_stations["station_display"] = (
            top_stations["station_id"].astype(str)
            + " | "
            + top_stations["station_commune_name"]
            + " | "
            + top_stations["source"].astype(str)
        )
        station_color_col = "meteo_operational_level" if "meteo_operational_level" in top_stations.columns else "risk_level"
        station_tooltip = [
            c
            for c in [
                "station_id",
                "source",
                "station_commune_name",
                effective_rain_col_weather,
                "weather_watch_priority",
                "distance_to_lgv_km",
                "risk_level",
                "meteo_operational_level",
                "weather_alert_index",
                "weather_quality_note",
                "weather_quality_level",
                "weather_data_reliability",
                "weather_action_label",
                "obs_age_h",
                "obs_freshness_level",
                "date_obs_raw",
            ]
            if c in top_stations.columns
        ]
        chart_st = (
            alt.Chart(top_stations)
            .mark_bar()
            .encode(
                x=alt.X(f"{station_rank_col}:Q", title=station_rank_title),
                y=alt.Y("station_display:N", sort="-x", title="Station"),
                color=alt.Color(f"{station_color_col}:N", scale=alt.Scale(domain=list(RISK_COLOR.keys()), range=list(RISK_COLOR.values()))),
                tooltip=station_tooltip,
            )
        )
        st.altair_chart(chart_st, use_container_width=True)
        st.caption("Tri recommande: priorite de surveillance. Si la fiabilite meteo est faible, verification terrain obligatoire.")

    st.subheader("Qualite et priorite meteo des stations")
    if filtered_weather.empty:
        st.info("Pas de station meteo pour l'analyse qualite/alerte.")
    elif "weather_quality_note" not in filtered_weather.columns or "weather_watch_priority" not in filtered_weather.columns:
        st.info("Indicateurs qualite meteo indisponibles dans ce snapshot.")
    else:
        qdf = filtered_weather.copy()
        qdf["weather_quality_note"] = pd.to_numeric(qdf.get("weather_quality_note"), errors="coerce").fillna(0.0)
        qdf["weather_watch_priority"] = pd.to_numeric(qdf.get("weather_watch_priority"), errors="coerce").fillna(0.0)
        qdf["rain_24h_mm"] = pd.to_numeric(qdf.get("rain_24h_mm"), errors="coerce").fillna(0.0)
        qdf["meteo_operational_level"] = qdf.get("meteo_operational_level", qdf.get("risk_level", "INDETERMINE")).astype(str)
        scatter = (
            alt.Chart(qdf)
            .mark_circle(opacity=0.85)
            .encode(
                x=alt.X("weather_quality_note:Q", title="Qualite meteo (/100)"),
                y=alt.Y("weather_watch_priority:Q", title="Priorite de surveillance (/100)"),
                size=alt.Size("rain_24h_mm:Q", title="Pluie 24h (mm)"),
                color=alt.Color(
                    "meteo_operational_level:N",
                    scale=alt.Scale(domain=["FAIBLE", "MODERE", "ELEVE", "CRITIQUE", "INDETERMINE"], range=[RISK_COLOR["FAIBLE"], RISK_COLOR["MODERE"], RISK_COLOR["ELEVE"], RISK_COLOR["CRITIQUE"], RISK_COLOR["INDETERMINE"]]),
                    legend=alt.Legend(title="Risque meteo"),
                ),
                tooltip=[
                    "station_id",
                    "source",
                    "station_commune_name",
                    "weather_quality_note",
                    "weather_quality_level",
                    "weather_alert_index",
                    "weather_watch_priority",
                    "obs_age_h",
                    "rain_24h_mm",
                    "rain_7d_mm",
                    "rain_30d_mm",
                ],
            )
        )
        st.altair_chart(scatter, use_container_width=True)
        st.caption(
            "Lecture: haut-droite = stations a traiter en priorite. "
            "Qualite<55 ou age>30h => indicateur a confirmer avant decision travaux."
        )

    st.subheader("Comparaison pluvio entre communes (multi-annees)")
    if not history_compare_enabled:
        st.info("Active 'comparaison historique' dans la barre laterale pour charger le comparatif multi-communes.")
    elif int(history_fetch_limit) > 0 and len(selected_compare_commune_labels) > int(history_fetch_limit):
        st.info(
            f"Historique calcule sur {history_fetch_limit} communes max pour performance "
            f"(selection actuelle: {len(selected_compare_commune_labels)})."
        )
    if history_compare_enabled and compare_history_df.empty:
        st.info("Selectionne des communes avec historique disponible pour comparer les pluies mensuelles.")
    elif history_compare_enabled:
        compare_plot_df = _prepare_monthly_history_chart_df(compare_history_df)
        if compare_plot_df.empty:
            st.info("Historique compare inutilisable apres nettoyage des dates/valeurs.")
        else:
            commune_count_initial = int(compare_plot_df["commune_label"].nunique())
            commune_count = commune_count_initial
            max_plot_communes = commune_count
            if commune_count > 8:
                max_plot_communes = st.slider(
                    "Communes affichees sur le graphe multi-annees",
                    min_value=3,
                    max_value=commune_count,
                    value=min(12, commune_count),
                    step=1 if commune_count <= 30 else 2,
                    key="compare_history_plot_communes",
                )
            if commune_count > max_plot_communes:
                latest_by_commune = (
                    compare_plot_df.sort_values("ym_date")
                    .groupby("commune_label", as_index=False)
                    .tail(1)[["commune_label", "monthly_precip_mm"]]
                    .sort_values("monthly_precip_mm", ascending=False)
                )
                keep_labels = latest_by_commune["commune_label"].head(int(max_plot_communes)).astype(str).tolist()
                compare_plot_df = compare_plot_df[compare_plot_df["commune_label"].astype(str).isin(keep_labels)].copy()
                commune_count = int(compare_plot_df["commune_label"].nunique())

            point_enabled = len(compare_plot_df) <= 900 and commune_count <= 12
            hist_line_base = (
                alt.Chart(compare_plot_df)
                .mark_line(point=point_enabled)
                .encode(
                    x=alt.X("ym_date:T", title="Mois", axis=alt.Axis(format="%Y-%m", labelAngle=-55)),
                    y=alt.Y("monthly_precip_mm:Q", title="Pluie mensuelle (mm)"),
                    color=alt.Color("commune_label:N", title="Commune"),
                    tooltip=["commune_label", "ym", "monthly_precip_mm"],
                )
            )
            chart_obj = hist_line_base
            if commune_count <= 18:
                last_points = compare_plot_df.sort_values("ym_date").groupby("commune_label", as_index=False).tail(1).copy()
                labels = (
                    alt.Chart(last_points)
                    .mark_text(align="left", baseline="middle", dx=6, fontSize=11)
                    .encode(
                        x=alt.X("ym_date:T"),
                        y=alt.Y("monthly_precip_mm:Q"),
                        text=alt.Text("commune_short:N"),
                        color=alt.Color("commune_label:N", legend=None),
                        tooltip=["commune_label", "ym", "monthly_precip_mm"],
                    )
                )
                chart_obj = hist_line_base + labels
            st.altair_chart(chart_obj.interactive(), use_container_width=True)
            if not point_enabled:
                st.caption("Points des courbes masques pour ameliorer lisibilite/performance sur gros volumes.")
            if commune_count > 18:
                st.caption("Etiquettes fin de courbe desactivees (>18 communes) pour eviter la surcharge visuelle.")
            if commune_count_initial > commune_count:
                st.caption(
                    f"Affichage limite a {commune_count} commune(s) sur {commune_count_initial} pour garder un rendu lisible."
                )

            month_compare = (
                compare_plot_df.groupby(["commune_label", "commune_short", "month", "month_label"], as_index=False)["monthly_precip_mm"]
                .mean()
                .rename(columns={"monthly_precip_mm": "mean_monthly_mm"})
            )
            if commune_count <= 8:
                climat_cmp = (
                    alt.Chart(month_compare)
                    .mark_bar()
                    .encode(
                        x=alt.X("month_label:N", sort=MONTH_ORDER_SHORT, title="Mois"),
                        y=alt.Y("mean_monthly_mm:Q", title="Moyenne mensuelle (mm)"),
                        color=alt.Color("commune_label:N", title="Commune"),
                        xOffset=alt.XOffset("commune_label:N"),
                        tooltip=["commune_label", "month_label", "mean_monthly_mm"],
                    )
                )
            else:
                climat_cmp = (
                    alt.Chart(month_compare)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("month_label:N", sort=MONTH_ORDER_SHORT, title="Mois"),
                        y=alt.Y("mean_monthly_mm:Q", title="Moyenne mensuelle (mm)"),
                        color=alt.Color("commune_label:N", title="Commune"),
                        tooltip=["commune_label", "month_label", "mean_monthly_mm"],
                    )
                )
            st.altair_chart(climat_cmp, use_container_width=True)
            st.caption(
                f"Comparaison sur {commune_count} commune(s). Modele historique: "
                + ", ".join([f"{k}:{v}" for k, v in history_models.items() if v])
            )
            st.caption(
                "Historique mensuel multi-annees: Open-Meteo archive (homogene sur toutes les communes). "
                "Mesures recentes par commune: InfoClimat/SYNOP prioritaire, puis station InfoClimat proche, "
                "puis fallback archive si necessaire."
            )

        history_month_exact, history_month_latest = _build_history_month_lookup(compare_history_full_df)
        latest_rows: List[Dict[str, object]] = []
        if selected_compare_commune_labels:
            for label in selected_compare_commune_labels:
                hit = commune_pool[commune_pool["commune_label"].astype(str) == str(label)]
                if hit.empty:
                    continue
                com = hit.iloc[0].to_dict()
                wx_source = _select_weather_pool_for_ui(
                    filtered_weather,
                    weather_for_context,
                    weather_df,
                    strict_reliable_mode=bool(strict_reliable_mode),
                )
                ref = _build_commune_weather_reference(
                    com,
                    wx_source,
                    snapshot_ts,
                    allow_open_meteo_fallback=not bool(strict_reliable_mode),
                )
                ref_obs = pd.to_datetime(ref.get("obs_date"), utc=True, errors="coerce")
                ref_ym = (
                    f"{int(ref_obs.year):04d}-{int(ref_obs.month):02d}"
                    if not pd.isna(ref_obs)
                    else ""
                )
                hist_month_mm = history_month_exact.get((str(label), ref_ym), np.nan)
                if pd.isna(hist_month_mm):
                    hist_month_mm = history_month_latest.get(str(label), np.nan)
                ref_month_mm = pd.to_numeric(ref.get("rain_month_mm"), errors="coerce")
                delta_hist_ref = (
                    abs(float(ref_month_mm) - float(hist_month_mm))
                    if (not pd.isna(ref_month_mm) and not pd.isna(hist_month_mm))
                    else np.nan
                )
                latest_rows.append(
                    {
                        "commune": label,
                        "mode_source": ref.get("source_mode"),
                        "source_label": ref.get("source_label"),
                        "source_brute": ref.get("source"),
                        "station_meteo": ref.get("station_id"),
                        "commune_station": ref.get("commune_station"),
                        "latitude_ref": ref.get("ref_latitude"),
                        "longitude_ref": ref.get("ref_longitude"),
                        "dist_station_km": ref.get("distance_km"),
                        "rain_24h_mm": ref.get("rain_24h_mm"),
                        "rain_7d_mm": ref.get("rain_7d_mm"),
                        "rain_30d_mm": ref.get("rain_30d_mm"),
                        "rain_month_mm": ref.get("rain_month_mm"),
                        "hist_month_mm": None if pd.isna(hist_month_mm) else round(float(hist_month_mm), 2),
                        "delta_hist_vs_ref_month_mm": None if pd.isna(delta_hist_ref) else round(float(delta_hist_ref), 2),
                        "date_obs": ref.get("obs_date"),
                        "ym_ref": ref_ym,
                        "reliability": ref.get("weather_data_reliability"),
                    }
                )
        if latest_rows:
            st.markdown("**Dernieres mesures meteo par commune comparee**")
            latest_df = pd.DataFrame(latest_rows)
            preferred_cols = [
                "commune",
                "mode_source",
                "source_label",
                "station_meteo",
                "commune_station",
                "dist_station_km",
                "rain_24h_mm",
                "rain_7d_mm",
                "rain_30d_mm",
                "rain_month_mm",
                "hist_month_mm",
                "delta_hist_vs_ref_month_mm",
                "date_obs",
                "ym_ref",
                "reliability",
                "source_brute",
                "latitude_ref",
                "longitude_ref",
            ]
            latest_df = latest_df[[c for c in preferred_cols if c in latest_df.columns]]
            st.dataframe(latest_df, use_container_width=True, hide_index=True)
            mode_counts = latest_df.get("mode_source", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
            if mode_counts:
                st.caption(
                    "Comparatif sources communes: "
                    + ", ".join([f"{k}={v}" for k, v in mode_counts.items()])
                )
            if "delta_hist_vs_ref_month_mm" in latest_df.columns:
                delta_series = pd.to_numeric(latest_df["delta_hist_vs_ref_month_mm"], errors="coerce")
                if delta_series.notna().any():
                    st.caption(
                        "Ecart historique vs reference (mois courant): "
                        + f"moyenne={float(delta_series.mean()):.2f} mm | max={float(delta_series.max()):.2f} mm"
                    )
            st.caption(
                "Regle appliquee: InfoClimat/SYNOP prioritaire (station commune), "
                "puis station InfoClimat la plus proche; Open-Meteo archive uniquement en dernier recours."
            )
            st.caption(
                "Lecture periodes: 30j = cumul glissant des 30 derniers jours; "
                "mois = cumul du mois calendaire courant."
            )

    st.subheader("Classement cumuls pluvio (1J / 1 semaine / mois calendaire / hiver / printemps / max mensuel / annees)")
    if pluvio_ranking_df.empty:
        st.info("Classement indisponible: selectionne des communes avec mesures meteo et historique.")
    else:
        period_options = ["1J", "1 semaine", "Mois courant (calendaire)", "Hiver", "Printemps", "Max mensuel", "Annees"]
        selected_periods = st.multiselect(
            "Filtres periodes classement",
            period_options,
            default=period_options,
            key="pluvio_period_filter",
        )
        ranking_sort_options = {
            "1J": "cum_1j_mm",
            "1 semaine": "cum_1_semaine_mm",
            "Mois courant (calendaire)": "cum_1_mois_mm",
            "Hiver": "cum_hiver_mm",
            "Printemps": "cum_printemps_mm",
            "Max mensuel": "max_mensuel_mm",
            "Annees": "cum_annees_mm",
        }
        ranking_df = pluvio_ranking_df.copy()
        selected_years: List[int] = []
        if "Annees" in selected_periods and not compare_history_full_df.empty:
            year_vals = sorted(pd.to_numeric(compare_history_full_df.get("year"), errors="coerce").dropna().astype(int).unique().tolist())
            if year_vals:
                selected_years = st.multiselect(
                    "Filtre annees (historique)",
                    year_vals,
                    default=year_vals[-min(5, len(year_vals)):],
                    key="pluvio_year_filter",
                )
                year_work = compare_history_full_df.copy()
                year_work["year"] = pd.to_numeric(year_work.get("year"), errors="coerce")
                if selected_years:
                    year_work = year_work[year_work["year"].isin(selected_years)]
                annual = (
                    year_work.groupby("commune_label", as_index=False)["monthly_precip_mm"]
                    .sum()
                    .rename(columns={"monthly_precip_mm": "cum_annees_mm"})
                )
                annual["cum_annees_mm"] = pd.to_numeric(annual["cum_annees_mm"], errors="coerce").round(1)
                ranking_df = ranking_df.merge(annual, on="commune_label", how="left")
                if ranking_df["cum_annees_mm"].notna().any():
                    annual_fallback = float(pd.to_numeric(ranking_df["cum_annees_mm"], errors="coerce").median(skipna=True))
                else:
                    annual_fallback = 0.0
                ranking_df["cum_annees_mm"] = pd.to_numeric(ranking_df["cum_annees_mm"], errors="coerce").fillna(annual_fallback)
                ranking_df["rang_annees"] = ranking_df["cum_annees_mm"].rank(method="min", ascending=False).astype("Int64")
            else:
                ranking_df["cum_annees_mm"] = np.nan
                ranking_df["rang_annees"] = pd.Series([pd.NA] * len(ranking_df), dtype="Int64")
        else:
            ranking_df["cum_annees_mm"] = np.nan
            ranking_df["rang_annees"] = pd.Series([pd.NA] * len(ranking_df), dtype="Int64")

        active_sort_options = {k: v for k, v in ranking_sort_options.items() if k in selected_periods} or {"Mois courant (calendaire)": "cum_1_mois_mm"}
        ranking_sort_label = st.selectbox(
            "Classement principal",
            list(active_sort_options.keys()),
            index=min(2, max(0, len(active_sort_options) - 1)),
            key="pluvio_ranking_sort",
        )
        ranking_sort_col = active_sort_options[ranking_sort_label]
        ranking_df = ranking_df.sort_values(ranking_sort_col, ascending=False, na_position="last").copy()

        col_map = {
            "1J": ["cum_1j_mm", "rang_1j"],
            "1 semaine": ["cum_1_semaine_mm", "rang_1_semaine"],
            "Mois courant (calendaire)": ["cum_1_mois_mm", "rang_1_mois"],
            "Hiver": ["cum_hiver_mm", "rang_hiver"],
            "Printemps": ["cum_printemps_mm", "rang_printemps"],
            "Max mensuel": ["max_mensuel_mm", "rang_max_mensuel"],
            "Annees": ["cum_annees_mm", "rang_annees"],
        }
        ranking_cols = ["commune_label"]
        for p in selected_periods:
            ranking_cols.extend(col_map.get(p, []))
        if any(p in selected_periods for p in ["Hiver", "Printemps"]):
            ranking_cols.append("annee_saison_ref")
        if "histo_mode" in ranking_df.columns:
            ranking_cols.append("histo_mode")
        ranking_cols = [c for c in ranking_cols if c in ranking_df.columns]
        st.dataframe(ranking_df[ranking_cols], use_container_width=True, hide_index=True)
        if "histo_mode" in ranking_df.columns:
            mode_count = ranking_df["histo_mode"].astype(str).value_counts().to_dict()
            st.caption(
                "Mode historique classement: "
                + ", ".join([f"{k}={v}" for k, v in mode_count.items()])
            )
        if ranking_sort_col in ranking_df.columns:
            ranking_sort_series = pd.to_numeric(ranking_df[ranking_sort_col], errors="coerce")
            if not ranking_sort_series.notna().any():
                st.info(
                    f"Aucune valeur disponible pour '{ranking_sort_label}'. "
                    "Active la comparaison historique pour les indicateurs saisonniers/annuels."
                )
            elif float(ranking_sort_series.max()) <= 0.0:
                st.warning(
                    f"Toutes les valeurs de '{ranking_sort_label}' sont nulles ou proches de 0 sur ce snapshot."
                )

        rank_count = len(ranking_df)
        top_rank_count = rank_count
        if rank_count > 0:
            top_rank_count = st.slider(
                "Communes affichees dans le graphe classement",
                min_value=1,
                max_value=rank_count,
                value=min(40, rank_count),
                step=1 if rank_count <= 40 else 5,
                key="pluvio_ranking_topn",
            )
        chart_df = ranking_df.copy()
        if ranking_sort_col in chart_df.columns:
            chart_df[ranking_sort_col] = pd.to_numeric(chart_df[ranking_sort_col], errors="coerce")
            chart_df = chart_df[chart_df[ranking_sort_col].notna()].copy()
        chart_df = chart_df.head(int(top_rank_count)).copy()
        if chart_df.empty:
            st.info("Graphe indisponible: aucune valeur exploitable pour ce classement.")
        else:
            bar_rank = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X(f"{ranking_sort_col}:Q", title=f"Cumul {ranking_sort_label} (mm)"),
                    y=alt.Y("commune_label:N", sort="-x", title="Commune"),
                    tooltip=[c for c in ranking_cols if c in chart_df.columns],
                )
            )
            st.altair_chart(bar_rank, use_container_width=True)

        if "max_mensuel_mm" in ranking_df.columns:
            max_month_df = ranking_df[["commune_label", "max_mensuel_mm"]].copy()
            max_month_df["max_mensuel_mm"] = pd.to_numeric(max_month_df["max_mensuel_mm"], errors="coerce")
            max_month_df = max_month_df[max_month_df["max_mensuel_mm"].notna()].sort_values("max_mensuel_mm", ascending=False)
            st.markdown("**Comparatif des maxima mensuels (historique)**")
            if max_month_df.empty:
                st.info("Maxima mensuels indisponibles: active la comparaison historique multi-annees.")
            else:
                st.dataframe(max_month_df, use_container_width=True, hide_index=True)

    st.subheader("Analyse detaillee commune")
    if not selected_commune:
        st.info("Aucune commune disponible pour l'analyse detaillee.")
    else:
        commune_name = str(selected_commune.get("commune_name") or "Inconnue")
        commune_code = str(selected_commune.get("commune_code") or "N/A")
        weather_quality_val = pd.to_numeric(selected_commune.get("weather_quality_note_commune"), errors="coerce")
        weather_quality_val = 0.0 if pd.isna(weather_quality_val) else float(weather_quality_val)
        weather_priority_val = pd.to_numeric(selected_commune.get("weather_watch_priority_commune"), errors="coerce")
        weather_priority_val = 0.0 if pd.isna(weather_priority_val) else float(weather_priority_val)
        weather_points_val = pd.to_numeric(selected_commune.get("weather_points_used"), errors="coerce")
        weather_points_val = 0 if pd.isna(weather_points_val) else int(float(weather_points_val))

        s1, s2, s3, s4, s5, s6, s7, s8 = st.columns(8)
        s1.metric("Commune", commune_name)
        s2.metric("Code INSEE", commune_code)
        s3.metric("Risque commune", str(selected_commune.get("commune_risk_level", "INDETERMINE")))
        s4.metric("Score risque global", f"{float(selected_commune.get('risk_score', selected_commune.get('note_gc', 0.0))):.1f}/100")
        s5.metric("Points LGV", int(float(selected_commune.get("lgv_points_count", 0) or 0)))
        s6.metric(
            "Risque IA commune",
            f"{selected_commune.get('ai_commune_risk_level', 'INDETERMINE')} ({float(selected_commune.get('max_ai_probability', 0.0) or 0.0) * 100.0:.0f}%)",
        )
        s7.metric(
            "Qualite meteo",
            f"{weather_quality_val:.1f}/100",
        )
        s8.metric(
            "Priorite meteo",
            f"{weather_priority_val:.1f}/100",
        )
        st.caption(
            "Meteo commune: "
            + f"alerte={selected_commune.get('weather_alert_level_commune', 'INDETERMINE')} | "
            + f"fiabilite={selected_commune.get('weather_reliability_flag', 'A_VERIFIER')} | "
            + f"fraicheur={selected_commune.get('weather_obs_freshness_commune', 'OBSOLETE')} | "
            + f"action={selected_commune.get('weather_action_commune', WEATHER_OP_ACTIONS['INDETERMINE'])} | "
            + f"stations utilisees={weather_points_val}"
        )
        daily_precip_val = selected_daily_day_precip.get("precip_mm")
        daily_precip_err = selected_daily_day_precip.get("error")
        if daily_precip_err:
            st.warning(
                "Recherche pluvio jour precis indisponible "
                + f"({selected_daily_day_precip.get('date')}): {daily_precip_err}"
            )
        else:
            st.metric(
                f"Pluie du {selected_daily_day_precip.get('date')}",
                f"{0.0 if daily_precip_val is None else float(daily_precip_val):.2f} mm",
            )
            st.caption(f"Source jour precis: Open-Meteo archive ({selected_daily_day_precip.get('model')})")

        detail_weather_pool = _select_weather_pool_for_ui(
            filtered_weather,
            weather_for_context,
            weather_df,
            strict_reliable_mode=bool(strict_reliable_mode),
        )
        nearest_weather_snapshot = _select_best_weather_row(
            detail_weather_pool,
            float(selected_commune["latitude"]),
            float(selected_commune["longitude"]),
            commune_name=selected_commune.get("commune_name"),
        )
        nearest_hydro = _nearest_row(filtered_hydro, float(selected_commune["latitude"]), float(selected_commune["longitude"]))
        nearest_piezo = _nearest_row(piezo_df, float(selected_commune["latitude"]), float(selected_commune["longitude"]))

        cwx, chx, cpx = st.columns(3)
        with cwx:
            st.markdown("**Derniere mesure meteo proche (coherente historique commune)**")
            ref_mode = str(selected_commune_weather_reference.get("source_mode") or "AUCUNE_DONNEE")
            if ref_mode != "AUCUNE_DONNEE":
                ref_24h = _to_float_or_none(selected_commune_weather_reference.get("rain_24h_mm"))
                ref_7d = _to_float_or_none(selected_commune_weather_reference.get("rain_7d_mm"))
                ref_30d = _to_float_or_none(selected_commune_weather_reference.get("rain_30d_mm"))
                ref_month = _to_float_or_none(selected_commune_weather_reference.get("rain_month_mm"))
                m24, m7, m30, mm = st.columns(4)
                m24.metric("24h", _format_mm(ref_24h))
                m7.metric("7 jours", _format_mm(ref_7d))
                m30.metric("30 jours", _format_mm(ref_30d))
                mm.metric("Mois courant", _format_mm(ref_month))

                ref_table = pd.DataFrame(
                    [
                        {
                            "mode_source": ref_mode,
                            "source_label": selected_commune_weather_reference.get("source_label"),
                            "source": selected_commune_weather_reference.get("source"),
                            "station_id": selected_commune_weather_reference.get("station_id"),
                            "commune_station": selected_commune_weather_reference.get("commune_station"),
                            "distance_km": selected_commune_weather_reference.get("distance_km"),
                            "latitude_ref": selected_commune_weather_reference.get("ref_latitude"),
                            "longitude_ref": selected_commune_weather_reference.get("ref_longitude"),
                            "rain_24h_mm": ref_24h,
                            "rain_7d_mm": ref_7d,
                            "rain_30d_mm": ref_30d,
                            "rain_month_mm": ref_month,
                            "weather_alert_index": selected_commune_weather_reference.get("weather_alert_index"),
                            "weather_quality_note": selected_commune_weather_reference.get("weather_quality_note"),
                            "weather_data_reliability": selected_commune_weather_reference.get("weather_data_reliability"),
                            "date_obs_raw": selected_commune_weather_reference.get("obs_date"),
                        }
                    ]
                )
                st.dataframe(ref_table, use_container_width=True, hide_index=True)
                st.caption(
                    "Definition periodes: 30j = cumul glissant 30 jours; mois = cumul du mois calendaire en cours."
                )
                hist_ref_date = str(history_payload.get("latest_obs_date") or "")
                hist_ref_30d = pd.to_numeric(history_payload.get("recent_rain_30d_mm"), errors="coerce")
                hist_ref_month = pd.to_numeric(history_payload.get("recent_rain_month_mm"), errors="coerce")
                ref_30d_num = pd.to_numeric(ref_30d, errors="coerce")
                ref_month_num = pd.to_numeric(ref_month, errors="coerce")
                if ref_mode == "OPEN_METEO_ARCHIVE_FALLBACK":
                    if not pd.isna(hist_ref_30d) and not pd.isna(ref_30d_num):
                        delta_30d = abs(float(hist_ref_30d) - float(ref_30d_num))
                        st.caption(
                            f"Controle coherence 30j ({hist_ref_date}): "
                            f"historique={float(hist_ref_30d):.1f} mm | mesure ref={float(ref_30d_num):.1f} mm | ecart={delta_30d:.1f} mm"
                        )
                    if not pd.isna(hist_ref_month) and not pd.isna(ref_month_num):
                        delta_month = abs(float(hist_ref_month) - float(ref_month_num))
                        st.caption(
                            f"Controle coherence mois courant ({hist_ref_date}): "
                            f"historique={float(hist_ref_month):.1f} mm | mesure ref={float(ref_month_num):.1f} mm | ecart={delta_month:.1f} mm"
                        )
                elif not pd.isna(hist_ref_30d):
                    st.caption(
                        f"Reference archive commune ({hist_ref_date}): 30j={float(hist_ref_30d):.1f} mm | "
                        f"mois={0.0 if pd.isna(hist_ref_month) else float(hist_ref_month):.1f} mm "
                        "(source secondaire comparee a la station)."
                    )
                if nearest_weather_snapshot:
                    with st.expander("Comparer avec station snapshot la plus proche"):
                        near_month = pd.to_numeric(nearest_weather_snapshot.get("rain_month_mm"), errors="coerce")
                        near_30d = pd.to_numeric(nearest_weather_snapshot.get("rain_30d_mm"), errors="coerce")
                        near_table = pd.DataFrame(
                            [
                                {
                                    "station_id": nearest_weather_snapshot.get("station_id"),
                                    "source": nearest_weather_snapshot.get("source"),
                                    "commune_station": nearest_weather_snapshot.get("station_commune_name"),
                                    "distance_km": nearest_weather_snapshot.get("_dist_km"),
                                    "rain_24h_mm": nearest_weather_snapshot.get("rain_24h_mm"),
                                    "rain_7d_mm": nearest_weather_snapshot.get("rain_7d_mm"),
                                    "rain_30d_mm": near_30d,
                                    "rain_month_mm": near_month,
                                    "date_obs_raw": nearest_weather_snapshot.get("date_obs_raw"),
                                }
                            ]
                        )
                        st.dataframe(near_table, use_container_width=True, hide_index=True)
                        if not pd.isna(ref_month_num) and not pd.isna(near_month):
                            delta_month_snap = abs(float(ref_month_num) - float(near_month))
                            st.caption(
                                f"Ecart reference vs station snapshot (mois courant): {delta_month_snap:.2f} mm"
                            )
                        if not pd.isna(ref_30d_num) and not pd.isna(near_30d):
                            delta_30d_snap = abs(float(ref_30d_num) - float(near_30d))
                            st.caption(
                                f"Ecart reference vs station snapshot (30j): {delta_30d_snap:.2f} mm"
                            )
            elif nearest_weather_snapshot:
                fallback_table = pd.DataFrame(
                    [
                        {
                            "mode_source": "NEAREST_STATION_FALLBACK",
                            "source_label": "Fallback station snapshot",
                            "station_id": nearest_weather_snapshot.get("station_id"),
                            "source": nearest_weather_snapshot.get("source"),
                            "commune_station": nearest_weather_snapshot.get("station_commune_name"),
                            "distance_km": nearest_weather_snapshot.get("_dist_km"),
                            "rain_24h_mm": nearest_weather_snapshot.get("rain_24h_mm"),
                            "rain_7d_mm": nearest_weather_snapshot.get("rain_7d_mm"),
                            "rain_30d_mm": nearest_weather_snapshot.get("rain_30d_mm"),
                            "rain_month_mm": nearest_weather_snapshot.get("rain_month_mm"),
                            "weather_alert_index": nearest_weather_snapshot.get("weather_alert_index"),
                            "weather_watch_priority": nearest_weather_snapshot.get("weather_watch_priority"),
                            "weather_quality_note": nearest_weather_snapshot.get("weather_quality_note"),
                            "weather_data_reliability": nearest_weather_snapshot.get("weather_data_reliability"),
                            "obs_age_h": nearest_weather_snapshot.get("obs_age_h"),
                            "obs_freshness_level": nearest_weather_snapshot.get("obs_freshness_level"),
                            "weather_action_label": nearest_weather_snapshot.get("weather_action_label"),
                            "date_obs_raw": nearest_weather_snapshot.get("date_obs_raw"),
                        }
                    ]
                )
                st.dataframe(fallback_table, use_container_width=True, hide_index=True)
                st.caption("Reference commune indisponible: fallback station snapshot.")
            else:
                st.info("Pas de mesure meteo proche.")
        with chx:
            st.markdown("**Derniere mesure hydro proche**")
            if nearest_hydro:
                hydro_table = pd.DataFrame(
                    [
                        {
                            "station_code": nearest_hydro.get("station_code"),
                            "river_name": nearest_hydro.get("river_name"),
                            "distance_km": nearest_hydro.get("_dist_km"),
                            "last_level_m": nearest_hydro.get("last_level_m"),
                            "trend_mph": nearest_hydro.get("trend_mph"),
                            "last_obs_utc": nearest_hydro.get("last_obs_utc"),
                        }
                    ]
                )
                st.dataframe(hydro_table, use_container_width=True, hide_index=True)
            else:
                st.info("Pas de mesure hydro proche.")
        with cpx:
            st.markdown("**Derniere mesure piezometre proche**")
            if nearest_piezo:
                piezo_table = pd.DataFrame(
                    [
                        {
                            "code_bss": nearest_piezo.get("code_bss"),
                            "name": nearest_piezo.get("name"),
                            "distance_km": nearest_piezo.get("_dist_km"),
                            "depth_m": nearest_piezo.get("depth_m"),
                            "trend_depth_mpd": nearest_piezo.get("trend_depth_mpd"),
                            "last_date_utc": nearest_piezo.get("last_date_utc"),
                        }
                    ]
                )
                st.dataframe(piezo_table, use_container_width=True, hide_index=True)
            else:
                st.info("Pas de piezometre proche.")

        st.markdown("**Historique meteo mensuel multi-annees (commune)**")
        if history_payload.get("error"):
            st.warning(f"Historique indisponible: {history_payload.get('error')}")
        elif history_monthly_df.empty:
            st.info("Pas d'historique mensuel disponible.")
        else:
            selected_commune_label = (
                f"{commune_name} ({commune_code})" if str(commune_code).strip() and str(commune_code) != "N/A" else commune_name
            )
            hist_plot_df = _prepare_monthly_history_chart_df(history_monthly_df, default_commune_label=selected_commune_label)
            if hist_plot_df.empty:
                st.info("Historique mensuel non exploitable apres nettoyage des dates/valeurs.")
            else:
                point_enabled = len(hist_plot_df) <= 96
                hist_base = (
                    alt.Chart(hist_plot_df)
                    .mark_line(point=point_enabled)
                    .encode(
                        x=alt.X("ym_date:T", title="Mois", axis=alt.Axis(format="%Y-%m", labelAngle=-55)),
                        y=alt.Y("monthly_precip_mm:Q", title="Pluie mensuelle (mm)"),
                        color=alt.Color("year:N", title="Annee"),
                        tooltip=["commune_label", "ym", "year", "monthly_precip_mm"],
                    )
                )
                year_count = int(hist_plot_df["year"].nunique())
                chart_obj = hist_base
                if year_count <= 8:
                    last_year_points = hist_plot_df.sort_values("ym_date").groupby("year", as_index=False).tail(1)
                    year_labels = (
                        alt.Chart(last_year_points)
                        .mark_text(align="left", baseline="middle", dx=6, fontSize=11)
                        .encode(
                            x=alt.X("ym_date:T"),
                            y=alt.Y("monthly_precip_mm:Q"),
                            text=alt.Text("year:N"),
                            color=alt.Color("year:N", legend=None),
                        )
                    )
                    chart_obj = hist_base + year_labels
                st.altair_chart(chart_obj.interactive(), use_container_width=True)
                st.caption(f"Commune: {selected_commune_label}")
                if not point_enabled:
                    st.caption("Points des courbes masques pour garder un rendu fluide.")

            clim_plot_df = history_clim_df.copy()
            if not clim_plot_df.empty:
                clim_plot_df = _ensure_month_column(clim_plot_df)
                clim_plot_df["climatology_mm"] = pd.to_numeric(clim_plot_df.get("climatology_mm"), errors="coerce")
                clim_plot_df["month_label"] = clim_plot_df["month"].map(MONTH_LABELS_SHORT)
                clim_plot_df = clim_plot_df.dropna(subset=["month_label", "climatology_mm"]).sort_values("month")
            if not clim_plot_df.empty:
                clim_base = (
                    alt.Chart(clim_plot_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("month_label:N", sort=MONTH_ORDER_SHORT, title="Mois"),
                        y=alt.Y("climatology_mm:Q", title="Moyenne historique (mm/mois)"),
                        tooltip=["month_label", "climatology_mm"],
                    )
                )
                clim_labels = clim_base.mark_text(dy=-8, fontSize=10).encode(text=alt.Text("climatology_mm:Q", format=".1f"))
                st.altair_chart(clim_base + clim_labels, use_container_width=True)
            st.caption(
                "Source historique: Open-Meteo archive "
                + f"({history_payload.get('model')}) | point_ref="
                + f"{selected_commune_weather_reference.get('source_mode')} "
                + f"({selected_commune_weather_reference.get('source_label')})"
            )
            if history_month_aligned:
                st.caption(
                    "Mois courant aligne sur la mesure de reference commune pour coherence de lecture "
                    "(derniere mesure vs historique mensuel)."
                )

with tabs[1]:
    st.subheader("Carte multi-couches")
    if commune_df.empty and filtered_weather.empty and filtered_sectors.empty and filtered_hydro.empty and lgv_communes_df.empty:
        st.info("Pas de donnees cartographiques avec ces filtres.")
    else:
        m = _build_map(
            snapshot=snapshot,
            weather_df=filtered_weather,
            commune_df=commune_df,
            sectors_df=filtered_sectors,
            slip_corridors_df=slip_corridors_df,
            hydro_df=filtered_hydro,
            piezo_df=piezo_df,
            geotech_df=geotech_df,
            lgv_communes_df=lgv_communes_df,
            fr_communes_geojson=fr_communes_geojson,
            rain_col_weather=effective_rain_col_weather,
            min_risk=min_risk,
            show_weather=show_weather,
            show_communes=show_communes,
            show_sectors=show_sectors,
            show_hydro=show_hydro,
            show_piezo=show_piezo,
            show_geotech=show_geotech,
            show_slip=show_slip,
            slip_alert_threshold=float(slip_alert_threshold),
            show_fr_layer=show_fr_layer,
            map_style=map_basemap_style,
        )
        st_folium(m, height=680, use_container_width=True)
        st.caption(f"Fond de carte actif: {map_basemap_style}.")

    st.markdown("**Carte meteo dediee (etiquettes pluie visibles)**")
    if filtered_weather.empty or effective_rain_col_weather not in filtered_weather.columns:
        st.info("Pas de stations meteo disponibles pour la carte dediee.")
    else:
        max_label_default = min(80, int(len(filtered_weather)))
        max_label_default = max(1, max_label_default)
        max_label_value = st.slider(
            "Nombre max d'etiquettes meteo visibles",
            min_value=1,
            max_value=max(1, int(len(filtered_weather))),
            value=max_label_default,
            step=1 if len(filtered_weather) <= 120 else 5,
            key="weather_focus_label_limit",
        )
        m_weather = _build_weather_focus_map(
            snapshot=snapshot,
            weather_df=filtered_weather,
            rain_col_weather=effective_rain_col_weather,
            min_risk=min_risk,
            max_labels=int(max_label_value),
            map_style=map_basemap_style,
        )
        st_folium(m_weather, height=640, use_container_width=True, key="weather_focus_map")
        st.caption(
            f"Etiquettes affichees: pluie {effective_period_label} en mm, station et valeur. "
            "Mode de selection: stations les plus proches de la LGV puis cumul le plus eleve."
        )

with tabs[2]:
    st.subheader("Tableau communes")
    if commune_df.empty:
        st.info("Aucune commune disponible.")
    else:
        commune_cols = [
            "commune_name",
            "commune_code",
            "departement_code",
            "lgv_points_count",
            "risk_score",
            "commune_risk_level",
            "ai_commune_risk_level",
            "weather_component_note",
            "weather_component_note_raw",
            "geotech_component_note",
            "piezo_component_note",
            "hydro_component_note",
            "ai_component_note",
            "weather_points_used",
            "weather_mean_dist_km",
            "weather_quality_note_commune",
            "weather_obs_age_h_commune",
            "weather_alert_index_commune",
            "weather_watch_priority_commune",
            "weather_alert_level_commune",
            "weather_reliability_flag",
            "weather_obs_freshness_commune",
            "weather_action_commune",
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
        commune_cols = [c for c in commune_cols if c in commune_df.columns]
        commune_view = commune_df[commune_cols].copy()
        if "risk_score" in commune_view.columns:
            commune_view = commune_view.rename(columns={"risk_score": "score_risque_global"})
        st.dataframe(commune_view, use_container_width=True, hide_index=True)

    st.subheader("Points LGV filtres (detail)")
    if filtered_sectors.empty:
        st.info("Aucun point LGV filtre.")
    else:
        view_cols = [
            "sector_id",
            "pk_km",
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
            "slip_index",
            "slip_level",
            "manual_watch_pk",
            "slip_drivers",
            "under_watch",
        ]
        present_cols = [c for c in view_cols if c in filtered_sectors.columns]
        sort_col = "slip_index" if "slip_index" in filtered_sectors.columns else ("ai_pred_probability" if "ai_pred_probability" in filtered_sectors.columns else "score")
        st.dataframe(filtered_sectors[present_cols].sort_values(sort_col, ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Zones glissement detectees (corridors PK)")
    if slip_corridors_df.empty:
        st.info("Aucun corridor glissement detecte au-dessus du seuil actif.")
    else:
        corr_cols = [
            "slip_corridor_id",
            "pk_start_km",
            "pk_end_km",
            "corridor_length_km",
            "slip_level",
            "slip_index_max",
            "slip_index_mean",
            "sector_count",
            "critical_count",
            "manual_watch_count",
            "commune_dominante",
            "max_ai_probability",
            "max_weather_30d_mm",
            "max_soil_fragility",
        ]
        corr_cols = [c for c in corr_cols if c in slip_corridors_df.columns]
        st.dataframe(slip_corridors_df[corr_cols], use_container_width=True, hide_index=True)

    st.subheader("PK sous surveillance manuelle (expert GC)")
    if not enable_manual_pk_watch or not manual_pk_ranges:
        st.info("Aucune zone PK manuelle active.")
    else:
        manual_df = pd.DataFrame(
            [{"pk_start_km": round(a, 3), "pk_end_km": round(b, 3), "length_km": round(b - a, 3)} for a, b in manual_pk_ranges]
        ).sort_values("pk_start_km")
        st.dataframe(manual_df, use_container_width=True, hide_index=True)

    st.subheader("Stations meteo filtrees (commune/station)")
    if filtered_weather.empty:
        st.info("Aucune station meteo sur les filtres actifs.")
    else:
        wx_view = filtered_weather.copy()
        for col, default in [
            ("weather_alert_index", 0.0),
            ("weather_watch_priority", 0.0),
            ("weather_quality_note", 0.0),
            ("obs_age_h", 240.0),
            ("latitude", np.nan),
            ("longitude", np.nan),
            ("distance_to_lgv_km", np.nan),
            ("rain_24h_mm", 0.0),
            ("rain_7d_mm", 0.0),
            ("rain_30d_mm", 0.0),
            ("rain_month_mm", 0.0),
        ]:
            if col in wx_view.columns:
                wx_view[col] = pd.to_numeric(wx_view[col], errors="coerce").fillna(default)
            else:
                wx_view[col] = default
        wx_view["meteo_operational_level"] = wx_view.get("meteo_operational_level", wx_view.get("risk_level", "INDETERMINE")).astype(str)
        if "weather_data_reliability" not in wx_view.columns:
            wx_view["weather_data_reliability"] = [
                _weather_data_reliability_label(float(q), float(a))
                for q, a in zip(wx_view["weather_quality_note"].tolist(), wx_view["obs_age_h"].tolist())
            ]
        if "obs_freshness_level" not in wx_view.columns:
            wx_view["obs_freshness_level"] = wx_view["obs_age_h"].map(_weather_freshness_level_from_age)
        if "weather_action_label" not in wx_view.columns:
            wx_view["weather_action_label"] = [
                _weather_action_label(str(lvl), str(rel))
                for lvl, rel in zip(wx_view["meteo_operational_level"].tolist(), wx_view["weather_data_reliability"].tolist())
            ]

        wx_view = wx_view.sort_values(
            ["weather_watch_priority", "weather_alert_index", "rain_24h_mm"],
            ascending=[False, False, False],
            na_position="last",
        )
        wx_view = wx_view.rename(
            columns={
                "station_id": "station",
                "source": "source_meteo",
                "station_commune_name": "commune_station",
                "distance_to_lgv_km": "distance_lgv_km",
                "meteo_operational_level": "niveau_operationnel",
                "weather_alert_index": "indice_alerte_meteo_100",
                "weather_watch_priority": "priorite_surveillance_100",
                "weather_quality_note": "qualite_meteo_100",
                "weather_data_reliability": "fiabilite_donnee",
                "weather_action_label": "action_recommandee_gc",
                "obs_age_h": "age_observation_h",
                "obs_freshness_level": "fraicheur_observation",
                "date_obs_raw": "date_observation_utc",
            }
        )
        st.dataframe(
            wx_view[
                [
                    c
                    for c in [
                        "station",
                        "source_meteo",
                        "commune_station",
                        "latitude",
                        "longitude",
                        "distance_lgv_km",
                        "rain_24h_mm",
                        "rain_7d_mm",
                        "rain_30d_mm",
                        "rain_month_mm",
                        "niveau_operationnel",
                        "indice_alerte_meteo_100",
                        "priorite_surveillance_100",
                        "qualite_meteo_100",
                        "fiabilite_donnee",
                        "action_recommandee_gc",
                        "age_observation_h",
                        "fraicheur_observation",
                        "date_observation_utc",
                    ]
                    if c in wx_view.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Tri applique: priorite_surveillance_100, puis indice_alerte_meteo_100, puis pluie_24h.")

    st.subheader("Cours d'eau et ruisseaux - hauteurs et seuils d'urgence")
    if filtered_hydro.empty:
        st.info("Aucune station hydro reseau disponible.")
    else:
        river_count = int(filtered_hydro["river_name"].dropna().astype(str).nunique()) if "river_name" in filtered_hydro.columns else 0
        source_count = int(filtered_hydro["source"].dropna().astype(str).nunique()) if "source" in filtered_hydro.columns else 0
        st.caption(
            f"Stations affichees: {len(filtered_hydro)} | Cours d'eau: {river_count} | Sources: {source_count} | "
            f"Seuil urgence depasse: {hydro_exceeded_count}"
        )
        hydro_view_cols = [
            "station_code",
            "station_name",
            "river_name",
            "source",
            "distance_to_lgv_km",
            "last_level_m",
            "trend_mph",
            "watch_threshold_m",
            "emergency_threshold_m",
            "threshold_ratio",
            "threshold_exceeded",
            "risk_level",
            "risk_reason",
            "last_obs_utc",
        ]
        hydro_view_cols = [c for c in hydro_view_cols if c in filtered_hydro.columns]
        hydro_view = filtered_hydro[hydro_view_cols].copy()
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

    st.subheader("Logique meteo renforcee (Streamlit)")
    st.markdown(
        """
        - Chaque station est evaluee par une **qualite meteo** (/100):
          source + fraicheur + completude + proximite LGV.
        - Un **indice d'alerte meteo** (/100) combine pluie 24h, 7j, 30j et pluie previsionnelle.
        - Le **risque meteo operationnel** prend le max entre classe pluie brute et alerte renforcee.
        - Au niveau commune, les stations proches sont agregees par distance pour calculer:
          qualite, alerte, priorite de surveillance, fiabilite.
        - Le score communal final reintegre ce signal meteo avec un poids adapte a la fiabilite des observations.
        """
    )

    st.subheader("Metadata meteo detaillee (sans ambiguite)")
    st.markdown(
        """
        - **Objectif**: transformer une pluie brute en decision GC actionnable.
        - **Unite**: tous les scores meteo sont normalises sur 100.
        - **Principe**: une valeur elevee = priorite de surveillance plus forte.
        """
    )
    weather_formula_df = pd.DataFrame(
        [
            {
                "Champ metadata": "weather_quality_note",
                "Formule / logique": "0.30*source_reliability + 0.35*freshness + 0.20*completude + 0.15*proximite",
                "Interpretation": "Qualite intrinsique de la donnee station (0-100).",
            },
            {
                "Champ metadata": "weather_alert_index",
                "Formule / logique": "40% pluie24h + 30% pluie7j + 20% pluie30j + 10% pluie_prevision",
                "Interpretation": "Intensite hydrometeo combinee (0-100).",
            },
            {
                "Champ metadata": "meteo_operational_level",
                "Formule / logique": "max(risk_level_source, niveau(weather_alert_index))",
                "Interpretation": "Niveau final pour pilotage exploitation.",
            },
            {
                "Champ metadata": "weather_watch_priority",
                "Formule / logique": "weather_alert_index * (0.55 + 0.45 * quality_scale)",
                "Interpretation": "Priorite de traitement tenant compte de la fiabilite.",
            },
            {
                "Champ metadata": "weather_data_reliability",
                "Formule / logique": "OK / SURVEILLER / A_VERIFIER selon qualite et age obs",
                "Interpretation": "Statut de confiance minimum avant engagement travaux.",
            },
        ]
    )
    st.dataframe(weather_formula_df, use_container_width=True, hide_index=True)

    weather_threshold_df = pd.DataFrame(
        [
            {
                "Niveau": "FAIBLE",
                "Seuil indice alerte": "< 45",
                "Action exploitation GC": WEATHER_OP_ACTIONS["FAIBLE"],
            },
            {
                "Niveau": "MODERE",
                "Seuil indice alerte": "45 a 64.9",
                "Action exploitation GC": WEATHER_OP_ACTIONS["MODERE"],
            },
            {
                "Niveau": "ELEVE",
                "Seuil indice alerte": "65 a 81.9",
                "Action exploitation GC": WEATHER_OP_ACTIONS["ELEVE"],
            },
            {
                "Niveau": "CRITIQUE",
                "Seuil indice alerte": ">= 82",
                "Action exploitation GC": WEATHER_OP_ACTIONS["CRITIQUE"],
            },
        ]
    )
    st.markdown("**Seuils de niveau meteo operationnel**")
    st.dataframe(weather_threshold_df, use_container_width=True, hide_index=True)

    reliability_df = pd.DataFrame(
        [
            {
                "Statut fiabilite": "OK",
                "Condition": "qualite>=70 ET age_obs<=18h",
                "Decision": "donnee exploitable directement",
            },
            {
                "Statut fiabilite": "SURVEILLER",
                "Condition": "55<=qualite<70 OU 18h<age_obs<=30h",
                "Decision": "confirmer via 2e source meteo/hydro",
            },
            {
                "Statut fiabilite": "A_VERIFIER",
                "Condition": "qualite<55 OU age_obs>30h",
                "Decision": "verification terrain/telemesure obligatoire",
            },
        ]
    )
    st.markdown("**Regles de fiabilite des donnees meteo**")
    st.dataframe(reliability_df, use_container_width=True, hide_index=True)

    st.subheader("Logique glissement de terrain (surveillance GC)")
    st.markdown(
        """
        - L'indice glissement est calcule par secteur PK (0-100) via: IA + fragilite sol + pluie + hydro + geotech + piezo.
        - Les corridors glissement sont des regroupements de secteurs contigus au-dessus du seuil d'alerte.
        - Les zones PK manuelles (capture expert) peuvent surclasser localement la priorite de surveillance.
        - Le rendu carte montre les secteurs critiques + polylignes de corridors pour prioriser les rondes terrain.
        """
    )
    slip_method_df = pd.DataFrame(
        [
            {"Composante": "IA prediction", "Poids": "30%", "Indicateur": "ai_pred_probability"},
            {"Composante": "Fragilite pedologique", "Poids": "20%", "Indicateur": "ai_soil_fragility"},
            {"Composante": "Pression pluie", "Poids": "22%", "Indicateur": "weather_max_24h/7d/30d"},
            {"Composante": "Hydro reseau", "Poids": "12%", "Indicateur": "hydro_stations"},
            {"Composante": "Geotech", "Poids": "10%", "Indicateur": "geotech_points"},
            {"Composante": "Piezo", "Poids": "4%", "Indicateur": "piezometers"},
            {"Composante": "Score secteur", "Poids": "2%", "Indicateur": "score"},
        ]
    )
    st.dataframe(slip_method_df, use_container_width=True, hide_index=True)

    slip_levels_df = pd.DataFrame(
        [
            {"Niveau glissement": "FAIBLE", "Seuil indice": "< 60", "Action GC": "Surveillance normale"},
            {"Niveau glissement": "MODERE", "Seuil indice": "60-74.9", "Action GC": "Controle renforce"},
            {"Niveau glissement": "ELEVE", "Seuil indice": "75-87.9", "Action GC": "Inspection terrain prioritaire"},
            {"Niveau glissement": "CRITIQUE", "Seuil indice": ">= 88", "Action GC": "Intervention urgente"},
        ]
    )
    st.dataframe(slip_levels_df, use_container_width=True, hide_index=True)

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
        update_df = pd.DataFrame(
            [{"Bloc": str(k), "Frequence declaree": str(v)} for k, v in update_meta.items()]
        )
        st.dataframe(update_df, use_container_width=True, hide_index=True)
    else:
        st.info("Frequence de MAJ non renseignee dans le snapshot.")
    st.markdown(
        """
        **Frequences cibles recommandees (pilotage GC):**
        - Meteo station (pluie/obs): 5 a 60 min selon source.
        - Hydro (hauteurs cours d'eau): 5 a 30 min sur stations telemetrees.
        - Consolidation snapshot decisionnel: horaire (minimum quotidien).
        - Historique climatologique: recalcul mensuel ou a chaque ingestion majeure.
        """
    )

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
