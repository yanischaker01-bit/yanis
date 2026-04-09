from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

try:
    import altair as alt

    ALT_AVAILABLE = True
    ALT_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - defensive import fallback for cloud env mismatches
    alt = None
    ALT_AVAILABLE = False
    ALT_IMPORT_ERROR = str(exc)


SNAPSHOT_LATEST = Path("reports/streamlit_snapshot_latest.json")
REMOTE_SNAPSHOT_URLS = [
    "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json",
    "https://raw.githubusercontent.com/yanischaker01-bit/yanis/main/reports/streamlit_snapshot_latest.json",
]

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_MODEL_METEOFRANCE = "meteofrance_seamless"
NASA_POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

HISTORY_MIN_DATE = date(2026, 1, 1)
RAIN_METRICS = {
    "24h": "rain_24h_mm",
    "7 jours": "rain_7d_mm",
    "30 jours": "rain_30d_mm",
    "Mois courant": "rain_month_mm",
}
HISTORY_SOURCES = [
    "Open-Meteo MeteoFrance",
    "Open-Meteo Standard",
    "NASA POWER",
]
HISTORY_SOURCE_COLORS = {
    "Open-Meteo MeteoFrance": "#1d4ed8",
    "Open-Meteo Standard": "#0f766e",
    "NASA POWER": "#b45309",
}
SOURCE_RELIABILITY_HINTS = {
    "SYNOP": 96.0,
    "INFOCLIMAT": 94.0,
    "METEOFRANCE": 93.0,
    "OPEN_METEO": 82.0,
    "NASA": 78.0,
}


def _http_get_with_retry(url: str, params: Dict[str, object], timeout: int = 30, max_attempts: int = 2) -> requests.Response:
    last_exc: Exception | None = None
    session = requests.Session()
    session.trust_env = False
    for _ in range(max(1, int(max_attempts))):
        try:
            return session.get(url, params=params, timeout=int(timeout))
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("HTTP request failed")


def _extract_lgv_lines(payload: Dict[str, object]) -> List[List[Tuple[float, float]]]:
    lines: List[List[Tuple[float, float]]] = []
    raw_lines = payload.get("lgv_lines", []) if isinstance(payload, dict) else []
    for line in raw_lines if isinstance(raw_lines, list) else []:
        coords: List[Tuple[float, float]] = []
        for pt in line if isinstance(line, list) else []:
            if not isinstance(pt, dict):
                continue
            lat = pd.to_numeric(pt.get("lat"), errors="coerce")
            lon = pd.to_numeric(pt.get("lon"), errors="coerce")
            if pd.isna(lat) or pd.isna(lon):
                continue
            coords.append((float(lat), float(lon)))
        if len(coords) >= 2:
            lines.append(coords)
    return lines


def _safe_weather_df(payload: Dict[str, object]) -> pd.DataFrame:
    df = pd.DataFrame(payload.get("weather") or [])
    if df.empty:
        return df
    for col in [
        "distance_to_lgv_km",
        "latitude",
        "longitude",
        "rain_24h_mm",
        "rain_7d_mm",
        "rain_30d_mm",
        "rain_month_mm",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, default in [
        ("station_id", "station_inconnue"),
        ("source", "source_inconnue"),
        ("station_commune_name", "Inconnue"),
        ("date_obs_raw", ""),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default).astype(str)
    if "obs_ts_utc" in df.columns:
        obs_ts = pd.to_datetime(df["obs_ts_utc"], utc=True, errors="coerce")
    else:
        obs_ts = pd.to_datetime(df["date_obs_raw"], utc=True, errors="coerce")
    df["_obs_ts"] = obs_ts
    df = df.sort_values("_obs_ts", ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["station_id", "source"], keep="first")
    return df.reset_index(drop=True)


def _multiselect_all(label: str, options: List[str], key: str) -> List[str]:
    clean = sorted({str(v).strip() for v in options if str(v).strip()})
    if not clean:
        st.multiselect(label, ["Tout"], default=["Tout"], key=key, disabled=True)
        return []
    ui_opts = ["Tout"] + clean
    picked = st.multiselect(label, ui_opts, default=["Tout"], key=key)
    if not picked or "Tout" in picked:
        return clean
    return [v for v in clean if v in picked]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = np.radians(float(lat1))
    p2 = np.radians(float(lat2))
    dlat = np.radians(float(lat2) - float(lat1))
    dlon = np.radians(float(lon2) - float(lon1))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def _source_reliability_note(source: object) -> float:
    txt = str(source or "").strip().upper()
    if not txt:
        return 68.0
    if "SYNOP" in txt:
        return SOURCE_RELIABILITY_HINTS["SYNOP"]
    if "INFOCLIMAT" in txt:
        return SOURCE_RELIABILITY_HINTS["INFOCLIMAT"]
    if "METEOFRANCE" in txt:
        return SOURCE_RELIABILITY_HINTS["METEOFRANCE"]
    if ("OPEN" in txt) and ("METEO" in txt):
        return SOURCE_RELIABILITY_HINTS["OPEN_METEO"]
    if "NASA" in txt:
        return SOURCE_RELIABILITY_HINTS["NASA"]
    return 70.0


def _freshness_note(hours: float) -> float:
    h = float(hours)
    if h <= 6.0:
        return 100.0
    if h <= 12.0:
        return 92.0
    if h <= 24.0:
        return 84.0
    if h <= 48.0:
        return 60.0
    return 35.0


def _reliability_class(note: float) -> str:
    val = float(note)
    if val >= 85.0:
        return "FIABLE"
    if val >= 65.0:
        return "SURVEILLER"
    return "A_VERIFIER"


def _build_proximity_quality(
    stations_df: pd.DataFrame,
    snapshot_ts: pd.Timestamp | None,
    metric_col: str,
    compare_radius_km: float,
    min_neighbors: int,
) -> pd.DataFrame:
    if stations_df.empty:
        return stations_df.copy()
    work = stations_df.copy()
    for col in ["latitude", "longitude", metric_col, "rain_24h_mm", "rain_7d_mm", "rain_30d_mm", "rain_month_mm"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "_obs_ts" not in work.columns:
        work["_obs_ts"] = pd.to_datetime(work.get("date_obs_raw"), utc=True, errors="coerce")
    obs_ts = pd.to_datetime(work["_obs_ts"], utc=True, errors="coerce")
    if snapshot_ts is not None and not pd.isna(snapshot_ts):
        obs_age_h = (snapshot_ts - obs_ts).dt.total_seconds() / 3600.0
    else:
        obs_age_h = pd.Series([np.nan] * len(work), index=work.index, dtype=float)
    work["obs_age_h"] = pd.to_numeric(obs_age_h, errors="coerce").fillna(999.0).clip(lower=0.0)
    work["source_note"] = work.get("source", pd.Series("", index=work.index)).map(_source_reliability_note)
    work["freshness_note"] = work["obs_age_h"].map(_freshness_note)

    metrics_for_consistency = [c for c in ["rain_24h_mm", "rain_7d_mm", "rain_30d_mm", "rain_month_mm"] if c in work.columns]
    near_count_list: List[int] = []
    near_med_metric_list: List[float] = []
    near_delta_metric_mm_list: List[float] = []
    near_delta_metric_pct_list: List[float] = []
    coherence_note_list: List[float] = []

    for idx, row in work.iterrows():
        lat = pd.to_numeric(row.get("latitude"), errors="coerce")
        lon = pd.to_numeric(row.get("longitude"), errors="coerce")
        station_id = str(row.get("station_id") or "")
        if pd.isna(lat) or pd.isna(lon):
            near_count_list.append(0)
            near_med_metric_list.append(np.nan)
            near_delta_metric_mm_list.append(np.nan)
            near_delta_metric_pct_list.append(np.nan)
            coherence_note_list.append(45.0)
            continue

        neighbors = work.copy()
        neighbors = neighbors[neighbors["station_id"].astype(str) != station_id].copy()
        if neighbors.empty:
            near_count_list.append(0)
            near_med_metric_list.append(np.nan)
            near_delta_metric_mm_list.append(np.nan)
            near_delta_metric_pct_list.append(np.nan)
            coherence_note_list.append(45.0)
            continue

        neighbors["_dist_station_km"] = neighbors.apply(
            lambda r: _haversine_km(
                float(lat),
                float(lon),
                float(pd.to_numeric(r.get("latitude"), errors="coerce")),
                float(pd.to_numeric(r.get("longitude"), errors="coerce")),
            )
            if (pd.notna(pd.to_numeric(r.get("latitude"), errors="coerce")) and pd.notna(pd.to_numeric(r.get("longitude"), errors="coerce")))
            else np.nan,
            axis=1,
        )
        neighbors = neighbors[
            pd.to_numeric(neighbors["_dist_station_km"], errors="coerce").fillna(9999.0) <= float(compare_radius_km)
        ].copy()
        neighbors = neighbors.dropna(subset=["_dist_station_km"])
        near_count = int(len(neighbors))
        near_count_list.append(near_count)
        if near_count <= 0:
            near_med_metric_list.append(np.nan)
            near_delta_metric_mm_list.append(np.nan)
            near_delta_metric_pct_list.append(np.nan)
            coherence_note_list.append(45.0)
            continue

        metric_val = pd.to_numeric(row.get(metric_col), errors="coerce")
        near_med_metric = pd.to_numeric(neighbors.get(metric_col), errors="coerce").median(skipna=True)
        near_med_metric_list.append(float(near_med_metric) if pd.notna(near_med_metric) else np.nan)
        if pd.isna(metric_val) or pd.isna(near_med_metric):
            near_delta_metric_mm_list.append(np.nan)
            near_delta_metric_pct_list.append(np.nan)
        else:
            delta_mm = abs(float(metric_val) - float(near_med_metric))
            delta_pct = 100.0 * delta_mm / max(5.0, float(near_med_metric))
            near_delta_metric_mm_list.append(delta_mm)
            near_delta_metric_pct_list.append(delta_pct)

        consistency_notes: List[float] = []
        for mcol in metrics_for_consistency:
            val = pd.to_numeric(row.get(mcol), errors="coerce")
            med = pd.to_numeric(neighbors.get(mcol), errors="coerce").median(skipna=True)
            if pd.isna(val) or pd.isna(med):
                continue
            rel = abs(float(val) - float(med)) / max(5.0, float(med))
            metric_note = max(0.0, 100.0 - rel * 120.0)
            consistency_notes.append(metric_note)

        if near_count < int(min_neighbors):
            base_consistency = float(np.mean(consistency_notes)) if consistency_notes else 58.0
            coherence_note_list.append(min(72.0, base_consistency))
        else:
            coherence_note_list.append(float(np.mean(consistency_notes)) if consistency_notes else 65.0)

    work["near_station_count"] = pd.Series(near_count_list, index=work.index, dtype=int)
    work["near_median_metric_mm"] = pd.Series(near_med_metric_list, index=work.index, dtype=float)
    work["near_delta_metric_mm"] = pd.Series(near_delta_metric_mm_list, index=work.index, dtype=float)
    work["near_delta_metric_pct"] = pd.Series(near_delta_metric_pct_list, index=work.index, dtype=float)
    work["coherence_note"] = pd.Series(coherence_note_list, index=work.index, dtype=float).clip(lower=0.0, upper=100.0)

    work["reliability_score"] = (
        0.42 * pd.to_numeric(work["source_note"], errors="coerce").fillna(70.0)
        + 0.23 * pd.to_numeric(work["freshness_note"], errors="coerce").fillna(40.0)
        + 0.35 * pd.to_numeric(work["coherence_note"], errors="coerce").fillna(55.0)
    ).clip(lower=0.0, upper=100.0)
    work["reliability_class"] = work["reliability_score"].map(_reliability_class)
    work["reliability_reason"] = np.where(
        work["near_station_count"] < int(min_neighbors),
        "Peu de stations voisines pour confirmer la coherence",
        np.where(
            work["near_delta_metric_pct"].fillna(0.0) >= 60.0,
            "Ecart eleve vs mediane des stations proches",
            "Coherence locale satisfaisante",
        ),
    )
    return work


def _nearest_neighbors_for_station(
    stations_df: pd.DataFrame,
    station_id: str,
    metric_col: str,
    compare_radius_km: float,
) -> pd.DataFrame:
    if stations_df.empty:
        return pd.DataFrame()
    base = stations_df[stations_df["station_id"].astype(str) == str(station_id)].copy()
    if base.empty:
        return pd.DataFrame()
    ref = base.iloc[0]
    ref_lat = pd.to_numeric(ref.get("latitude"), errors="coerce")
    ref_lon = pd.to_numeric(ref.get("longitude"), errors="coerce")
    if pd.isna(ref_lat) or pd.isna(ref_lon):
        return pd.DataFrame()

    work = stations_df.copy()
    work = work[work["station_id"].astype(str) != str(station_id)].copy()
    if work.empty:
        return pd.DataFrame()
    work["distance_station_ref_km"] = work.apply(
        lambda r: _haversine_km(
            float(ref_lat),
            float(ref_lon),
            float(pd.to_numeric(r.get("latitude"), errors="coerce")),
            float(pd.to_numeric(r.get("longitude"), errors="coerce")),
        )
        if (pd.notna(pd.to_numeric(r.get("latitude"), errors="coerce")) and pd.notna(pd.to_numeric(r.get("longitude"), errors="coerce")))
        else np.nan,
        axis=1,
    )
    work = work.dropna(subset=["distance_station_ref_km"])
    work = work[
        pd.to_numeric(work["distance_station_ref_km"], errors="coerce").fillna(9999.0) <= float(compare_radius_km)
    ].copy()
    if work.empty:
        return work
    work[metric_col] = pd.to_numeric(work.get(metric_col), errors="coerce").fillna(0.0)
    ref_metric = pd.to_numeric(ref.get(metric_col), errors="coerce")
    if pd.isna(ref_metric):
        work["delta_metric_mm"] = np.nan
    else:
        work["delta_metric_mm"] = (work[metric_col] - float(ref_metric)).abs()
    work = work.sort_values(["distance_station_ref_km", "delta_metric_mm"], ascending=[True, True], na_position="last")
    return work


@st.cache_data(show_spinner=False, ttl=300)
def _load_snapshot_payload() -> Tuple[Dict[str, object], str]:
    errors: List[str] = []
    if SNAPSHOT_LATEST.exists():
        try:
            payload = pd.read_json(SNAPSHOT_LATEST, typ="series").to_dict()
            if isinstance(payload, dict):
                return payload, f"local:{SNAPSHOT_LATEST.as_posix()}"
        except Exception as exc:
            errors.append(f"local:{exc}")
    for url in REMOTE_SNAPSHOT_URLS:
        try:
            resp = _http_get_with_retry(url, params={}, timeout=25, max_attempts=2)
            if resp.status_code != 200:
                errors.append(f"{url}:HTTP{resp.status_code}")
                continue
            payload = resp.json()
            if isinstance(payload, dict):
                return payload, f"remote:{url}"
            errors.append(f"{url}:payload_non_dict")
        except Exception as exc:
            errors.append(f"{url}:{exc}")
    raise RuntimeError("Snapshot indisponible: " + " | ".join(errors))


@st.cache_data(show_spinner=False, ttl=21600)
def _fetch_open_meteo_history(
    lat: float,
    lon: float,
    start_day: str,
    end_day: str,
    source_label: str,
    model: str | None = None,
) -> Tuple[pd.DataFrame, str]:
    params: Dict[str, object] = {
        "latitude": f"{float(lat):.6f}",
        "longitude": f"{float(lon):.6f}",
        "start_date": str(start_day),
        "end_date": str(end_day),
        "daily": "precipitation_sum",
        "timezone": "UTC",
    }
    if model:
        params["models"] = str(model)
    try:
        resp = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30, max_attempts=2)
        if resp.status_code != 200:
            return pd.DataFrame(), f"{source_label}: HTTP {resp.status_code}"
        payload = resp.json()
        daily = payload.get("daily") if isinstance(payload, dict) else {}
        times = daily.get("time") if isinstance(daily, dict) else []
        values = daily.get("precipitation_sum") if isinstance(daily, dict) else []
        out = pd.DataFrame({"date": times, "precip_mm": values})
        if out.empty:
            return out, f"{source_label}: vide"
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out["precip_mm"] = pd.to_numeric(out["precip_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
        out = out.dropna(subset=["date"]).copy()
        out["source"] = source_label
        return out[["date", "precip_mm", "source"]], ""
    except Exception as exc:
        return pd.DataFrame(), f"{source_label}: {exc}"


@st.cache_data(show_spinner=False, ttl=21600)
def _fetch_nasa_power_history(lat: float, lon: float, start_day: str, end_day: str) -> Tuple[pd.DataFrame, str]:
    start_fmt = str(start_day).replace("-", "")
    end_fmt = str(end_day).replace("-", "")
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": f"{float(lon):.6f}",
        "latitude": f"{float(lat):.6f}",
        "start": start_fmt,
        "end": end_fmt,
        "format": "JSON",
    }
    try:
        resp = _http_get_with_retry(NASA_POWER_DAILY_URL, params=params, timeout=35, max_attempts=2)
        if resp.status_code != 200:
            return pd.DataFrame(), f"NASA POWER: HTTP {resp.status_code}"
        payload = resp.json()
        data_obj = (
            (((payload or {}).get("properties") or {}).get("parameter") or {}).get("PRECTOTCORR", {})
            if isinstance(payload, dict)
            else {}
        )
        if not isinstance(data_obj, dict) or not data_obj:
            return pd.DataFrame(), "NASA POWER: vide"
        rows: List[Dict[str, object]] = []
        for ymd, val in data_obj.items():
            dt = pd.to_datetime(str(ymd), format="%Y%m%d", utc=True, errors="coerce")
            mm = pd.to_numeric(val, errors="coerce")
            if pd.isna(dt) or pd.isna(mm):
                continue
            if float(mm) < 0.0:
                continue
            rows.append({"date": dt, "precip_mm": float(mm), "source": "NASA POWER"})
        out = pd.DataFrame(rows)
        if out.empty:
            return out, "NASA POWER: aucune valeur exploitable"
        out["precip_mm"] = pd.to_numeric(out["precip_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
        return out[["date", "precip_mm", "source"]], ""
    except Exception as exc:
        return pd.DataFrame(), f"NASA POWER: {exc}"


def _load_history_multi_source(
    lat: float,
    lon: float,
    start_day: date,
    end_day: date,
    selected_sources: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    start_iso = str(start_day.isoformat())
    end_iso = str(end_day.isoformat())
    blocks: List[pd.DataFrame] = []
    notices: List[str] = []

    if "Open-Meteo MeteoFrance" in selected_sources:
        df, note = _fetch_open_meteo_history(
            lat=lat,
            lon=lon,
            start_day=start_iso,
            end_day=end_iso,
            source_label="Open-Meteo MeteoFrance",
            model=OPEN_METEO_MODEL_METEOFRANCE,
        )
        if not df.empty:
            blocks.append(df)
        if note:
            notices.append(note)

    if "Open-Meteo Standard" in selected_sources:
        df, note = _fetch_open_meteo_history(
            lat=lat,
            lon=lon,
            start_day=start_iso,
            end_day=end_iso,
            source_label="Open-Meteo Standard",
            model=None,
        )
        if not df.empty:
            blocks.append(df)
        if note:
            notices.append(note)

    if "NASA POWER" in selected_sources:
        df, note = _fetch_nasa_power_history(lat=lat, lon=lon, start_day=start_iso, end_day=end_iso)
        if not df.empty:
            blocks.append(df)
        if note:
            notices.append(note)

    if not blocks:
        return pd.DataFrame(), notices
    history = pd.concat(blocks, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"], utc=True, errors="coerce")
    history["precip_mm"] = pd.to_numeric(history["precip_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
    history = history.dropna(subset=["date"])
    history = history.sort_values(["source", "date"]).reset_index(drop=True)
    return history, notices


def _build_map(lgv_lines: List[List[Tuple[float, float]]], stations_df: pd.DataFrame, rain_col: str) -> folium.Map:
    center = [46.2, 0.2]
    all_pts = [pt for line in lgv_lines for pt in line]
    if all_pts:
        center = [float(np.mean([p[0] for p in all_pts])), float(np.mean([p[1] for p in all_pts]))]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    for line in lgv_lines:
        folium.PolyLine(line, color="#1d4ed8", weight=4, opacity=0.85, tooltip="Trace LGV SEA").add_to(m)

    if stations_df.empty:
        return m

    metric_vals = pd.to_numeric(stations_df.get(rain_col), errors="coerce")
    vmax = float(metric_vals.max()) if metric_vals.notna().any() else 1.0
    vmax = max(vmax, 1.0)
    for _, row in stations_df.iterrows():
        lat = pd.to_numeric(row.get("latitude"), errors="coerce")
        lon = pd.to_numeric(row.get("longitude"), errors="coerce")
        val = pd.to_numeric(row.get(rain_col), errors="coerce")
        if pd.isna(lat) or pd.isna(lon) or pd.isna(val):
            continue
        ratio = max(0.0, min(1.0, float(val) / vmax))
        rclass = str(row.get("reliability_class") or "")
        if rclass == "A_VERIFIER":
            color = "#7f1d1d"
        elif rclass == "SURVEILLER":
            color = "#b45309"
        else:
            color = "#16a34a" if ratio < 0.33 else ("#ea580c" if ratio < 0.66 else "#dc2626")
        rel_score = pd.to_numeric(row.get("reliability_score"), errors="coerce")
        near_delta_pct = pd.to_numeric(row.get("near_delta_metric_pct"), errors="coerce")
        popup = (
            f"<b>Station:</b> {row.get('station_id')}<br>"
            f"<b>Commune:</b> {row.get('station_commune_name')}<br>"
            f"<b>Source:</b> {row.get('source')}<br>"
            f"<b>Distance LGV:</b> {row.get('distance_to_lgv_km')} km<br>"
            f"<b>{rain_col}:</b> {float(val):.1f} mm<br>"
            f"<b>Score fiabilite:</b> {0.0 if pd.isna(rel_score) else float(rel_score):.1f}/100 ({row.get('reliability_class', 'N/A')})<br>"
            f"<b>Ecart vs stations proches:</b> {0.0 if pd.isna(near_delta_pct) else float(near_delta_pct):.1f}%<br>"
            f"<b>Obs:</b> {row.get('date_obs_raw')}"
        )
        folium.CircleMarker(
            [float(lat), float(lon)],
            radius=5 + 6 * ratio,
            color=color,
            fill=True,
            fill_opacity=0.85,
            weight=1,
            popup=folium.Popup(popup, max_width=380),
            tooltip=f"{row.get('station_id')} | {float(val):.1f} mm",
        ).add_to(m)

    return m


st.set_page_config(page_title="LGV SEA Pluvio Stations Pro", page_icon=":umbrella:", layout="wide")
st.title("LGV SEA - Pluviometrie Stations Pro")
st.caption(
    "Version pro: fiabilisation des mesures, comparaison entre stations proches, historique multi-sources (depuis 2026) et carte operative."
)
if not ALT_AVAILABLE:
    st.warning(
        "Altair indisponible sur cet environnement Cloud: bascule automatique en mode graphique Streamlit natif. "
        + f"Detail import: {ALT_IMPORT_ERROR}"
    )

try:
    snapshot, snapshot_source = _load_snapshot_payload()
except Exception as exc:
    st.error(str(exc))
    st.stop()

weather_df = _safe_weather_df(snapshot)
lgv_lines = _extract_lgv_lines(snapshot)
snapshot_ts = pd.to_datetime(snapshot.get("timestamp_utc"), utc=True, errors="coerce")

with st.sidebar:
    st.subheader("Filtres stations")
    if st.button("Rafraichir", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    metric_label = st.selectbox("Indicateur pluvio", list(RAIN_METRICS.keys()), index=1)
    metric_col = RAIN_METRICS[metric_label]

    max_distance_km = st.slider("Distance max a la LGV (km)", min_value=1.0, max_value=80.0, value=25.0, step=0.5)
    compare_radius_km = st.slider(
        "Rayon comparaison stations proches (km)",
        min_value=2.0,
        max_value=40.0,
        value=15.0,
        step=0.5,
    )
    min_neighbors = st.slider(
        "Voisins min pour fiabiliser",
        min_value=1,
        max_value=8,
        value=2,
        step=1,
    )
    incoherence_alert_pct = st.slider(
        "Seuil alerte incoherence (%)",
        min_value=10,
        max_value=150,
        value=45,
        step=5,
        help="Ecart relatif (vs mediane des stations proches) au-dela duquel la station est marquee incoherente.",
    )

    src_options = sorted(weather_df.get("source", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    selected_sources = _multiselect_all("Sources snapshot", src_options, key="plv_sources")

    commune_options = sorted(weather_df.get("station_commune_name", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    selected_communes = _multiselect_all("Communes stations", commune_options, key="plv_communes")

filtered_stations = weather_df.copy()
if not filtered_stations.empty and "distance_to_lgv_km" in filtered_stations.columns:
    filtered_stations = filtered_stations[
        pd.to_numeric(filtered_stations["distance_to_lgv_km"], errors="coerce").fillna(9999.0) <= float(max_distance_km)
    ]
if not filtered_stations.empty and selected_sources:
    filtered_stations = filtered_stations[filtered_stations["source"].astype(str).isin(selected_sources)]
if not filtered_stations.empty and selected_communes:
    filtered_stations = filtered_stations[filtered_stations["station_commune_name"].astype(str).isin(selected_communes)]

station_options = filtered_stations.get("station_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
station_options = sorted(station_options)
with st.sidebar:
    selected_stations = _multiselect_all("Stations", station_options, key="plv_station_ids")
if not filtered_stations.empty and selected_stations:
    filtered_stations = filtered_stations[filtered_stations["station_id"].astype(str).isin(selected_stations)]

if filtered_stations.empty:
    st.warning("Aucune station sur ce filtre.")
    st.stop()

if metric_col not in filtered_stations.columns:
    filtered_stations[metric_col] = 0.0
filtered_stations[metric_col] = pd.to_numeric(filtered_stations.get(metric_col), errors="coerce").fillna(0.0).clip(lower=0.0)
filtered_stations = _build_proximity_quality(
    stations_df=filtered_stations,
    snapshot_ts=snapshot_ts,
    metric_col=metric_col,
    compare_radius_km=float(compare_radius_km),
    min_neighbors=int(min_neighbors),
)
filtered_stations["incoherence_flag"] = (
    pd.to_numeric(filtered_stations.get("near_delta_metric_pct"), errors="coerce").fillna(0.0) >= float(incoherence_alert_pct)
)
filtered_stations = filtered_stations.sort_values([metric_col, "distance_to_lgv_km"], ascending=[False, True], na_position="last")

top_n_max = max(1, int(len(filtered_stations)))
with st.sidebar:
    top_n = st.slider("Top stations (graphe)", min_value=1, max_value=top_n_max, value=min(25, top_n_max), step=1)
    st.markdown("---")
    st.subheader("Historique (depuis 2026)")
    history_station_default = str(filtered_stations.iloc[0]["station_id"])
    history_station_id = st.selectbox(
        "Station historique",
        options=sorted(filtered_stations["station_id"].astype(str).unique().tolist()),
        index=sorted(filtered_stations["station_id"].astype(str).unique().tolist()).index(history_station_default),
    )
    history_sources = st.multiselect(
        "Sources historiques",
        options=HISTORY_SOURCES,
        default=HISTORY_SOURCES,
    )
    today_utc = datetime.now(timezone.utc).date()
    history_start = st.date_input(
        "Date debut",
        value=HISTORY_MIN_DATE,
        min_value=HISTORY_MIN_DATE,
        max_value=today_utc,
    )
    history_end = st.date_input(
        "Date fin",
        value=today_utc,
        min_value=HISTORY_MIN_DATE,
        max_value=today_utc,
    )

if history_end < history_start:
    history_start, history_end = history_end, history_start

selected_station_df = filtered_stations[filtered_stations["station_id"].astype(str) == str(history_station_id)].copy()
selected_station = selected_station_df.iloc[0].to_dict() if not selected_station_df.empty else filtered_stations.iloc[0].to_dict()

reliability_series = pd.to_numeric(filtered_stations.get("reliability_score"), errors="coerce").fillna(0.0)
reliability_class = filtered_stations.get("reliability_class", pd.Series("A_VERIFIER", index=filtered_stations.index)).astype(str)
incoherence_count = int(filtered_stations.get("incoherence_flag", pd.Series(False, index=filtered_stations.index)).fillna(False).sum())

k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
k1.metric("Stations filtrees", int(len(filtered_stations)))
k2.metric("Communes", int(filtered_stations["station_commune_name"].astype(str).nunique()))
k3.metric(f"Max {metric_label}", f"{float(filtered_stations[metric_col].max()):.1f} mm")
k4.metric(f"Moyenne {metric_label}", f"{float(filtered_stations[metric_col].mean()):.1f} mm")
k5.metric("Distance max filtre", f"{float(max_distance_km):.1f} km")
k6.metric("Fiabilite moyenne", f"{float(reliability_series.mean()):.1f}/100")
k7.metric("Stations FIABLE", int((reliability_class == "FIABLE").sum()))
k8.metric("Incoherences proximite", incoherence_count)

if snapshot_ts is not None and not pd.isna(snapshot_ts):
    st.caption(f"Snapshot: {snapshot_source} | timestamp_utc={snapshot_ts.isoformat()}")
else:
    st.caption(f"Snapshot: {snapshot_source} | timestamp inconnu")

st.subheader("Carte stations pluvio autour de la LGV SEA")
map_obj = _build_map(lgv_lines, filtered_stations, metric_col)
st_folium(map_obj, height=640, use_container_width=True)

st.subheader(f"Top {int(top_n)} stations - {metric_label}")
top_df = filtered_stations.head(int(top_n)).copy()
top_df["station_label"] = top_df["station_id"].astype(str) + " | " + top_df["station_commune_name"].astype(str)
if ALT_AVAILABLE:
    bar_chart = (
        alt.Chart(top_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{metric_col}:Q", title=f"Pluie {metric_label} (mm)"),
            y=alt.Y("station_label:N", sort="-x", title="Station"),
            color=alt.Color("source:N", title="Source snapshot"),
            tooltip=["station_id", "station_commune_name", "source", "distance_to_lgv_km", metric_col, "date_obs_raw"],
        )
    )
    st.altair_chart(bar_chart, use_container_width=True)
else:
    st.caption("Mode fallback sans Altair: classement tabulaire.")
    fallback_top = top_df[["station_label", metric_col, "source", "distance_to_lgv_km"]].copy()
    st.dataframe(
        fallback_top.sort_values(metric_col, ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Comparaison inter-stations de proximite (mode pro)")
pro_view = filtered_stations.copy()
pro_view["metric_station_mm"] = pd.to_numeric(pro_view.get(metric_col), errors="coerce")
pro_view["metric_mediane_voisins_mm"] = pd.to_numeric(pro_view.get("near_median_metric_mm"), errors="coerce")
pro_view["ecart_voisins_mm"] = pd.to_numeric(pro_view.get("near_delta_metric_mm"), errors="coerce")
pro_view["ecart_voisins_pct"] = pd.to_numeric(pro_view.get("near_delta_metric_pct"), errors="coerce")
pro_view["fiabilite_100"] = pd.to_numeric(pro_view.get("reliability_score"), errors="coerce").fillna(0.0)
pro_view["nb_voisins"] = pd.to_numeric(pro_view.get("near_station_count"), errors="coerce").fillna(0).astype(int)
pro_view["incoherent"] = pro_view.get("incoherence_flag", pd.Series(False, index=pro_view.index)).fillna(False).astype(bool)

scatter_df = pro_view.dropna(subset=["metric_station_mm", "metric_mediane_voisins_mm"]).copy()
if scatter_df.empty:
    st.info("Comparaison proximite indisponible sur ce filtre.")
else:
    if ALT_AVAILABLE:
        scatter = (
            alt.Chart(scatter_df)
            .mark_circle(size=85, opacity=0.85)
            .encode(
                x=alt.X("metric_mediane_voisins_mm:Q", title=f"Mediane voisins proches - {metric_label} (mm)"),
                y=alt.Y("metric_station_mm:Q", title=f"Station - {metric_label} (mm)"),
                color=alt.Color("reliability_class:N", title="Fiabilite"),
                shape=alt.Shape("incoherent:N", title="Incoherence"),
                tooltip=[
                    "station_id",
                    "station_commune_name",
                    "source",
                    "distance_to_lgv_km",
                    "metric_station_mm",
                    "metric_mediane_voisins_mm",
                    "ecart_voisins_mm",
                    "ecart_voisins_pct",
                    "nb_voisins",
                    "fiabilite_100",
                    "reliability_class",
                    "reliability_reason",
                ],
            )
        )
        st.altair_chart(scatter.interactive(), use_container_width=True)
    else:
        st.caption("Mode fallback: tableau comparatif (Altair indisponible).")
        st.dataframe(
            scatter_df[
                [
                    "station_id",
                    "station_commune_name",
                    "metric_station_mm",
                    "metric_mediane_voisins_mm",
                    "ecart_voisins_mm",
                    "ecart_voisins_pct",
                    "fiabilite_100",
                    "reliability_class",
                    "reliability_reason",
                ]
            ].sort_values("ecart_voisins_pct", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

worst_df = pro_view.sort_values(["incoherent", "ecart_voisins_pct", "fiabilite_100"], ascending=[False, False, True], na_position="last").head(30).copy()
if not worst_df.empty:
    worst_df["station_label"] = worst_df["station_id"].astype(str) + " | " + worst_df["station_commune_name"].astype(str)
    if ALT_AVAILABLE:
        worst_chart = (
            alt.Chart(worst_df)
            .mark_bar()
            .encode(
                x=alt.X("ecart_voisins_pct:Q", title=f"Ecart relatif vs mediane voisins ({metric_label}, %)"),
                y=alt.Y("station_label:N", sort="-x", title="Station"),
                color=alt.Color("reliability_class:N", title="Fiabilite"),
                tooltip=[
                    "station_id",
                    "station_commune_name",
                    "source",
                    "nb_voisins",
                    "ecart_voisins_mm",
                    "ecart_voisins_pct",
                    "fiabilite_100",
                    "reliability_class",
                    "reliability_reason",
                ],
            )
        )
        st.altair_chart(worst_chart, use_container_width=True)
    else:
        st.caption("Mode fallback sans Altair: ecarts majeurs en tableau.")
        fallback_worst = worst_df[
            ["station_label", "ecart_voisins_pct", "ecart_voisins_mm", "reliability_score", "reliability_class"]
        ].copy()
        st.dataframe(
            fallback_worst.sort_values("ecart_voisins_pct", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

neighbor_df = _nearest_neighbors_for_station(
    stations_df=filtered_stations,
    station_id=str(history_station_id),
    metric_col=metric_col,
    compare_radius_km=float(compare_radius_km),
)
st.markdown(f"**Voisins de comparaison pour la station {history_station_id}**")
if neighbor_df.empty:
    st.info("Aucune station voisine dans le rayon de comparaison.")
else:
    ncols = [
        "station_id",
        "station_commune_name",
        "source",
        "distance_station_ref_km",
        "distance_to_lgv_km",
        metric_col,
        "delta_metric_mm",
        "reliability_score",
        "reliability_class",
        "near_delta_metric_pct",
    ]
    ncols = [c for c in ncols if c in neighbor_df.columns]
    st.dataframe(neighbor_df[ncols].head(40), use_container_width=True, hide_index=True)

st.subheader("Table stations filtrees")
station_cols = [
    "station_id",
    "source",
    "station_commune_name",
    "distance_to_lgv_km",
    "latitude",
    "longitude",
    "rain_24h_mm",
    "rain_7d_mm",
    "rain_30d_mm",
    "rain_month_mm",
    "near_station_count",
    "near_median_metric_mm",
    "near_delta_metric_mm",
    "near_delta_metric_pct",
    "reliability_score",
    "reliability_class",
    "reliability_reason",
    "date_obs_raw",
]
station_cols = [c for c in station_cols if c in filtered_stations.columns]
st.dataframe(filtered_stations[station_cols], use_container_width=True, hide_index=True)

st.subheader(f"Historique station: {selected_station.get('station_id')} ({selected_station.get('station_commune_name')})")
if not history_sources:
    st.info("Selectionne au moins une source historique.")
else:
    station_lat = pd.to_numeric(selected_station.get("latitude"), errors="coerce")
    station_lon = pd.to_numeric(selected_station.get("longitude"), errors="coerce")
    if pd.isna(station_lat) or pd.isna(station_lon):
        st.warning("Coordonnees station invalides pour charger l'historique.")
    else:
        hist_df, hist_notices = _load_history_multi_source(
            lat=float(station_lat),
            lon=float(station_lon),
            start_day=history_start,
            end_day=history_end,
            selected_sources=history_sources,
        )
        if hist_notices:
            st.caption(" | ".join(hist_notices[:4]))
        if hist_df.empty:
            st.warning("Historique indisponible sur cette periode/source.")
        else:
            hist_df = hist_df.copy()
            hist_df["date"] = pd.to_datetime(hist_df["date"], utc=True, errors="coerce")
            hist_df["precip_mm"] = pd.to_numeric(hist_df["precip_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
            hist_df = hist_df.dropna(subset=["date"]).sort_values(["source", "date"])

            roll_df = hist_df.copy()
            roll_df["rolling_7d_mm"] = roll_df.groupby("source")["precip_mm"].transform(
                lambda s: s.rolling(window=7, min_periods=1).sum()
            )

            monthly = hist_df.copy()
            monthly["ym"] = monthly["date"].dt.strftime("%Y-%m")
            monthly = (
                monthly.groupby(["source", "ym"], as_index=False)["precip_mm"]
                .sum()
                .rename(columns={"precip_mm": "monthly_mm"})
            )

            if ALT_AVAILABLE:
                daily_chart = (
                    alt.Chart(hist_df)
                    .mark_line(point=False)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("precip_mm:Q", title="Pluie journaliere (mm)"),
                        color=alt.Color(
                            "source:N",
                            scale=alt.Scale(
                                domain=list(HISTORY_SOURCE_COLORS.keys()),
                                range=list(HISTORY_SOURCE_COLORS.values()),
                            ),
                        ),
                        tooltip=["source", "date", "precip_mm"],
                    )
                )
                st.altair_chart(daily_chart.interactive(), use_container_width=True)

                roll_chart = (
                    alt.Chart(roll_df)
                    .mark_line(point=False, strokeDash=[8, 3])
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("rolling_7d_mm:Q", title="Cumul glissant 7 jours (mm)"),
                        color=alt.Color("source:N", title="Source historique"),
                        tooltip=["source", "date", "rolling_7d_mm"],
                    )
                )
                st.altair_chart(roll_chart.interactive(), use_container_width=True)

                monthly_chart = (
                    alt.Chart(monthly)
                    .mark_bar()
                    .encode(
                        x=alt.X("ym:N", title="Mois"),
                        y=alt.Y("monthly_mm:Q", title="Cumul mensuel (mm)"),
                        color=alt.Color("source:N", title="Source historique"),
                        xOffset=alt.XOffset("source:N"),
                        tooltip=["source", "ym", "monthly_mm"],
                    )
                )
                st.altair_chart(monthly_chart, use_container_width=True)
            else:
                st.caption("Mode fallback: graphiques historiques en rendu Streamlit natif.")
                daily_fallback = (
                    hist_df.pivot_table(index="date", columns="source", values="precip_mm", aggfunc="mean")
                    .sort_index()
                    .fillna(0.0)
                )
                st.markdown("**Historique journalier (tableau pivot)**")
                st.dataframe(daily_fallback.reset_index(), use_container_width=True, hide_index=True)

                roll_fallback = (
                    roll_df.pivot_table(index="date", columns="source", values="rolling_7d_mm", aggfunc="mean")
                    .sort_index()
                    .fillna(0.0)
                )
                st.markdown("**Cumul glissant 7 jours (tableau pivot)**")
                st.dataframe(roll_fallback.reset_index(), use_container_width=True, hide_index=True)

                monthly_fallback = (
                    monthly.pivot_table(index="ym", columns="source", values="monthly_mm", aggfunc="mean")
                    .sort_index()
                    .fillna(0.0)
                )
                st.markdown("**Cumuls mensuels (tableau pivot)**")
                st.dataframe(monthly_fallback.reset_index(), use_container_width=True, hide_index=True)

            summary = (
                hist_df.groupby("source", as_index=False)
                .agg(
                    jours=("date", "count"),
                    total_mm=("precip_mm", "sum"),
                    moyenne_mm_j=("precip_mm", "mean"),
                    max_journalier_mm=("precip_mm", "max"),
                )
                .sort_values("total_mm", ascending=False)
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
