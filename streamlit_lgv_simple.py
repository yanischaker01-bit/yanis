from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

from meteo_test import LGVSeaMonitor


SNAPSHOT_LATEST = Path("reports/streamlit_snapshot_latest.json")
REMOTE_SNAPSHOT_URLS = [
    "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json",
    "https://raw.githubusercontent.com/yanischaker01-bit/yanis/main/reports/streamlit_snapshot_latest.json",
]
DEFAULT_INFOCLIMAT_MAX_DISTANCE_KM = 15.0


def _is_infoclimat_source(source: object) -> bool:
    src = str(source or "").strip().lower()
    return ("infoclimat" in src) or ("synop" in src)


def _is_open_meteo_source(source: object) -> bool:
    src = str(source or "").strip().lower()
    return ("open" in src) and ("meteo" in src)


@st.cache_data(show_spinner=False, ttl=600)
def _load_snapshot() -> Tuple[Dict[str, object], str]:
    errors: List[str] = []
    if SNAPSHOT_LATEST.exists():
        try:
            payload = pd.read_json(SNAPSHOT_LATEST, typ="series").to_dict()
            return payload, f"local:{SNAPSHOT_LATEST.as_posix()}"
        except Exception as exc:
            errors.append(f"local:{exc}")

    session = requests.Session()
    session.trust_env = False
    for url in REMOTE_SNAPSHOT_URLS:
        try:
            resp = session.get(url, timeout=25)
            if resp.status_code != 200:
                errors.append(f"{url}:HTTP{resp.status_code}")
                continue
            payload = resp.json()
            if isinstance(payload, dict):
                return payload, f"remote:{url}"
        except Exception as exc:
            errors.append(f"{url}:{exc}")
    raise RuntimeError("Snapshot indisponible: " + " | ".join(errors))


def _normalize_weather_df(payload: Dict[str, object]) -> pd.DataFrame:
    weather_df = pd.DataFrame(payload.get("weather") or [])
    if weather_df.empty:
        return weather_df
    for col in ["distance_to_lgv_km", "rain_24h_mm", "rain_7d_mm", "rain_30d_mm", "rain_month_mm", "latitude", "longitude"]:
        if col in weather_df.columns:
            weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")
    weather_df["source"] = weather_df.get("source", pd.Series("", index=weather_df.index)).fillna("").astype(str)
    weather_df["station_id"] = weather_df.get("station_id", pd.Series("", index=weather_df.index)).fillna("").astype(str)
    weather_df["date_obs_raw"] = weather_df.get("date_obs_raw", pd.Series("", index=weather_df.index)).fillna("").astype(str)
    return weather_df


@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_live_synop_df() -> Tuple[pd.DataFrame, str, str]:
    try:
        monitor = LGVSeaMonitor()
        pack = monitor.fetch_pluviometry_synop()
        all_df = pack.get("all") if isinstance(pack.get("all"), pd.DataFrame) else pd.DataFrame()
        notice = str(pack.get("notice") or "").strip()
        source_url = str(pack.get("source_url") or "").strip()
        if all_df.empty:
            return all_df, notice or "SYNOP live vide", source_url
        out = all_df.copy()
        for col in ["distance_to_lgv_km", "rain_24h_mm", "rain_7d_mm", "rain_30d_mm", "rain_month_mm", "latitude", "longitude"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out["source"] = out.get("source", pd.Series("", index=out.index)).fillna("").astype(str)
        out["station_id"] = out.get("station_id", pd.Series("", index=out.index)).fillna("").astype(str)
        out["date_obs_raw"] = out.get("date_obs_raw", pd.Series("", index=out.index)).fillna("").astype(str)
        return out, notice, source_url
    except Exception as exc:
        return pd.DataFrame(), f"SYNOP live indisponible: {exc}", ""


def _extract_lgv_lines(payload: Dict[str, object]) -> List[List[Tuple[float, float]]]:
    lines = []
    raw = payload.get("lgv_lines") or []
    for line in raw:
        coords: List[Tuple[float, float]] = []
        if not isinstance(line, list):
            continue
        for pt in line:
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


def _build_station_pool(
    weather_df: pd.DataFrame,
    live_synop_df: pd.DataFrame,
    max_distance_km: float,
    allow_open_meteo_fallback: bool,
) -> Tuple[pd.DataFrame, str]:
    if weather_df.empty:
        if not live_synop_df.empty:
            live_dist = pd.to_numeric(live_synop_df.get("distance_to_lgv_km"), errors="coerce")
            live_info = live_synop_df[live_dist.notna() & (live_dist <= float(max_distance_km))].copy()
            if not live_info.empty:
                live_info = live_info.sort_values(["distance_to_lgv_km", "rain_24h_mm"], ascending=[True, False])
                return live_info, "live_infoclimat"
        return weather_df.copy(), "empty"

    dist = pd.to_numeric(weather_df.get("distance_to_lgv_km"), errors="coerce")
    src = weather_df.get("source", pd.Series("", index=weather_df.index)).fillna("").astype(str)
    is_info = src.map(_is_infoclimat_source)

    info_pool = weather_df[is_info & dist.notna() & (dist <= float(max_distance_km))].copy()
    if not info_pool.empty:
        info_pool = info_pool.sort_values(["distance_to_lgv_km", "rain_24h_mm"], ascending=[True, False])
        return info_pool, "infoclimat"

    # Snapshot can be stale: if no InfoClimat there, try live SYNOP fetch.
    if not live_synop_df.empty:
        live_dist = pd.to_numeric(live_synop_df.get("distance_to_lgv_km"), errors="coerce")
        live_info = live_synop_df[live_dist.notna() & (live_dist <= float(max_distance_km))].copy()
        if not live_info.empty:
            live_info = live_info.sort_values(["distance_to_lgv_km", "rain_24h_mm"], ascending=[True, False])
            return live_info, "live_infoclimat"

    if allow_open_meteo_fallback:
        is_open = src.map(_is_open_meteo_source)
        fallback = weather_df[is_open].copy()
        if not fallback.empty:
            fallback = fallback.sort_values(["distance_to_lgv_km", "rain_24h_mm"], ascending=[True, False])
            return fallback, "open_meteo_fallback"

    return pd.DataFrame(columns=list(weather_df.columns)), "none"


def _make_map(lines: List[List[Tuple[float, float]]], stations_df: pd.DataFrame) -> folium.Map:
    center = [46.2, 0.2]
    if lines and lines[0]:
        center = [lines[0][0][0], lines[0][0][1]]

    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")
    for line in lines:
        folium.PolyLine(line, color="#1d4ed8", weight=4, opacity=0.9, tooltip="Trace LGV SEA").add_to(m)

    if not stations_df.empty:
        for row in stations_df.to_dict(orient="records"):
            lat = pd.to_numeric(row.get("latitude"), errors="coerce")
            lon = pd.to_numeric(row.get("longitude"), errors="coerce")
            if pd.isna(lat) or pd.isna(lon):
                continue
            src = str(row.get("source") or "")
            color = "#0f766e" if _is_infoclimat_source(src) else "#1d4ed8"
            popup = (
                f"<b>Station:</b> {row.get('station_id')}<br>"
                f"<b>Source:</b> {src}<br>"
                f"<b>Dist LGV:</b> {row.get('distance_to_lgv_km')} km<br>"
                f"<b>Pluie 24h:</b> {row.get('rain_24h_mm')} mm<br>"
                f"<b>Pluie 7j:</b> {row.get('rain_7d_mm')} mm<br>"
                f"<b>Obs:</b> {row.get('date_obs_raw')}"
            )
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=7,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                tooltip=f"{row.get('station_id')} ({src})",
                popup=folium.Popup(popup, max_width=420),
            ).add_to(m)
    return m


st.set_page_config(page_title="LGV SEA Simple Fiable", page_icon=":umbrella:", layout="wide")
st.title("LGV SEA - Version simple (sources fiables)")
st.caption("Priorite: InfoClimat/SYNOP (Meteo-France). Fallback Open-Meteo uniquement si vous l'activez.")

try:
    payload, loaded_from = _load_snapshot()
except Exception as exc:
    st.error(str(exc))
    st.stop()

weather_df = _normalize_weather_df(payload)
lgv_lines = _extract_lgv_lines(payload)
snapshot_ts = pd.to_datetime(payload.get("timestamp_utc"), utc=True, errors="coerce")

with st.sidebar:
    st.header("Parametres")
    max_distance_km = st.slider(
        "Distance max InfoClimat/SYNOP a la LGV (km)",
        min_value=1.0,
        max_value=50.0,
        value=float(DEFAULT_INFOCLIMAT_MAX_DISTANCE_KM),
        step=0.5,
    )
    allow_open_meteo_fallback = st.checkbox(
        "Autoriser fallback Open-Meteo si aucune station InfoClimat",
        value=False,
    )
    enable_live_synop = st.checkbox(
        "Essayer SYNOP live si snapshot incomplet",
        value=True,
    )
    show_raw_table = st.checkbox("Afficher table brute des stations", value=False)

live_synop_df = pd.DataFrame()
live_synop_notice = ""
live_synop_source = ""
if enable_live_synop:
    live_synop_df, live_synop_notice, live_synop_source = _fetch_live_synop_df()

stations_df, pool_mode = _build_station_pool(
    weather_df=weather_df,
    live_synop_df=live_synop_df,
    max_distance_km=float(max_distance_km),
    allow_open_meteo_fallback=bool(allow_open_meteo_fallback),
)

col1, col2, col3 = st.columns(3)
col1.metric("Stations retenues", int(len(stations_df)))
if not stations_df.empty:
    col2.metric("Distance min LGV (km)", round(float(pd.to_numeric(stations_df["distance_to_lgv_km"], errors="coerce").min()), 3))
    col3.metric("Pluie max 24h (mm)", round(float(pd.to_numeric(stations_df["rain_24h_mm"], errors="coerce").max()), 1))
else:
    col2.metric("Distance min LGV (km)", "-")
    col3.metric("Pluie max 24h (mm)", "-")

if pool_mode == "infoclimat":
    st.success(f"Mode fiable actif: {len(stations_df)} station(s) InfoClimat/SYNOP <= {max_distance_km:.1f} km.")
elif pool_mode == "live_infoclimat":
    st.success(f"Mode fiable actif (SYNOP live): {len(stations_df)} station(s) <= {max_distance_km:.1f} km.")
elif pool_mode == "open_meteo_fallback":
    st.warning("Aucune station InfoClimat/SYNOP dans le filtre: fallback Open-Meteo active.")
else:
    st.error("Aucune station fiable trouvee avec ce filtre. Augmentez la distance (ex: 15-25 km).")

meta_notice = str(payload.get("weather_notice") or "").strip()
if meta_notice:
    st.info(f"Notice pipeline: {meta_notice}")
if enable_live_synop and live_synop_notice:
    st.caption(f"Notice SYNOP live: {live_synop_notice}")
if enable_live_synop and live_synop_source:
    st.caption(f"Source SYNOP live: {live_synop_source}")

source_counts = (
    weather_df["source"].value_counts(dropna=False).to_dict()
    if not weather_df.empty and "source" in weather_df.columns
    else {}
)
st.caption(
    f"Snapshot: {snapshot_ts if not pd.isna(snapshot_ts) else 'inconnu'} | "
    f"charge depuis {loaded_from} | sources snapshot: {source_counts}"
)

m = _make_map(lgv_lines, stations_df)
st_folium(m, height=620, use_container_width=True, key="map_simple_fiable")

if stations_df.empty:
    st.info("Pas de table detaillee: aucune station retenue.")
else:
    cols = [c for c in [
        "station_id",
        "source",
        "distance_to_lgv_km",
        "rain_24h_mm",
        "rain_7d_mm",
        "rain_30d_mm",
        "rain_month_mm",
        "date_obs_raw",
        "station_commune_name",
    ] if c in stations_df.columns]
    st.subheader("Stations retenues")
    st.dataframe(
        stations_df[cols].sort_values("distance_to_lgv_km"),
        use_container_width=True,
        hide_index=True,
    )

if show_raw_table:
    st.subheader("Table brute snapshot (diagnostic)")
    st.dataframe(weather_df, use_container_width=True, hide_index=True)

st.caption(f"Dernier rafraichissement UI: {datetime.utcnow().isoformat()}Z")
