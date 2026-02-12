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


def _build_multi_sector_history(sector_rows: List[Dict[str, object]], years_back: int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames: List[pd.DataFrame] = []
    model_by_sector: Dict[str, str] = {}
    for sec in sector_rows:
        sid = str(sec.get("sector_id"))
        lat = float(sec.get("latitude"))
        lon = float(sec.get("longitude"))
        payload = _load_sector_monthly_history(lat, lon, years_back)
        model_by_sector[sid] = str(payload.get("model") or "")
        monthly = _safe_df(payload.get("monthly"))
        if monthly.empty:
            continue
        monthly["sector_id"] = sid
        monthly["commune_name"] = str(sec.get("commune_name") or "Inconnue")
        monthly["sector_label"] = monthly["sector_id"] + " - " + monthly["commune_name"]
        frames.append(monthly)
    if not frames:
        return pd.DataFrame(), model_by_sector
    out = pd.concat(frames, ignore_index=True)
    out["ym"] = out["ym"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(0).astype(int)
    out["month"] = pd.to_numeric(out["month"], errors="coerce").fillna(0).astype(int)
    out["monthly_precip_mm"] = pd.to_numeric(out["monthly_precip_mm"], errors="coerce")
    out = out.dropna(subset=["monthly_precip_mm"])
    return out.sort_values(["ym", "sector_id"]), model_by_sector


def _aggregate_communes(sectors_df: pd.DataFrame, sector_rain_col: str) -> pd.DataFrame:
    if sectors_df.empty:
        return pd.DataFrame()

    df = sectors_df.copy()
    df["commune_name"] = df.get("commune_name", "Inconnue").fillna("Inconnue")
    df["sector_score"] = pd.to_numeric(df.get("score", 0.0), errors="coerce").fillna(0.0)
    df["rain_period_mm"] = pd.to_numeric(df.get(sector_rain_col, 0.0), errors="coerce").fillna(0.0)
    df["is_critical"] = (df.get("risk_level", "") == "CRITIQUE").astype(int)
    df["is_high"] = (df.get("risk_level", "") == "ELEVE").astype(int)
    df["is_moderate"] = (df.get("risk_level", "") == "MODERE").astype(int)
    df["is_watch"] = df.get("under_watch", False).astype(bool).astype(int)
    df["risk_rank"] = df.get("risk_level", "").map(lambda x: _risk_rank(str(x)))

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
            watch=("is_watch", "sum"),
            avg_rain_period_mm=("rain_period_mm", "mean"),
            max_rain_period_mm=("rain_period_mm", "max"),
        )
        .reset_index()
    )

    grouped["note_gc"] = (
        (grouped["avg_sector_score"] / 4.0) * 68.0
        + ((grouped["critical"] * 12.0 + grouped["high"] * 6.0) / grouped["sector_count"].clip(lower=1))
        + grouped["avg_rain_period_mm"].clip(lower=0.0, upper=180.0) / 9.0
    ).clip(lower=0.0, upper=100.0)
    grouped["note_gc"] = grouped["note_gc"].round(1)
    grouped["commune_risk_level"] = grouped["note_gc"].map(_risk_level_from_note)
    grouped = grouped.sort_values(["note_gc", "critical", "high"], ascending=[False, False, False]).reset_index(drop=True)
    return grouped


def _build_map(
    snapshot: Dict[str, object],
    weather_df: pd.DataFrame,
    sectors_df: pd.DataFrame,
    hydro_df: pd.DataFrame,
    piezo_df: pd.DataFrame,
    geotech_df: pd.DataFrame,
    rain_col_weather: str,
    sector_rain_col: str,
    min_risk: str,
    show_weather: bool,
    show_sectors: bool,
    show_hydro: bool,
    show_piezo: bool,
    show_geotech: bool,
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

    if show_sectors and not sectors_df.empty:
        sector_layer = folium.FeatureGroup(name="Secteurs", show=True)
        for _, row in sectors_df.iterrows():
            lvl = str(row.get("risk_level", "INDETERMINE"))
            if _risk_rank(lvl) < _risk_rank(min_risk):
                continue
            rain = float(row.get(sector_rain_col, 0.0) or 0.0)
            popup = (
                f"<b>{row.get('sector_id')}</b><br>"
                f"<b>Commune:</b> {row.get('commune_name')}<br>"
                f"<b>Risque:</b> {lvl}<br>"
                f"<b>Score:</b> {row.get('score')}<br>"
                f"<b>Cumul filtre:</b> {rain:.1f} mm<br>"
                f"<b>Geo/Piezo/Hydro:</b> {row.get('geotech_points')}/{row.get('piezometers')}/{row.get('hydro_stations')}"
            )
            folium.CircleMarker(
                [float(row["latitude"]), float(row["longitude"])],
                radius=8,
                color=RISK_COLOR.get(lvl, "#6b7280"),
                fill=True,
                fill_opacity=0.35,
                weight=2,
                popup=folium.Popup(popup, max_width=360),
            ).add_to(sector_layer)
        sector_layer.add_to(m)

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

if not weather_df.empty:
    for col in ["rain_24h_mm", "rain_7d_mm", "rain_30d_mm", "rain_month_mm", "distance_to_lgv_km"]:
        if col in weather_df.columns:
            weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")
    weather_df["risk_level"] = weather_df.get("rain_class", "INDETERMINE")
    if "station_commune_name" not in weather_df.columns:
        weather_df["station_commune_name"] = "Inconnue"
    weather_df["station_commune_name"] = weather_df["station_commune_name"].fillna("Inconnue")

if not sectors_df.empty:
    for col in ["score", "weather_max_24h_mm", "weather_max_7d_mm", "weather_max_30d_mm", "weather_max_month_mm"]:
        if col in sectors_df.columns:
            sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")
    sectors_df["commune_name"] = sectors_df.get("commune_name", "Inconnue").fillna("Inconnue")

if not alerts_df.empty and "level" in alerts_df.columns:
    alerts_df["rank"] = alerts_df["level"].map(lambda x: _risk_rank(str(x))).fillna(0)
    alerts_df = alerts_df.sort_values("rank", ascending=False)

with st.sidebar:
    st.subheader("Filtres")
    if st.button("Rafraichir snapshot", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    period_label = st.selectbox("Periode pluvio", list(RAIN_PERIODS.keys()), index=0)
    rain_col_weather, sector_rain_col = RAIN_PERIODS[period_label]
    min_risk = st.selectbox("Risque minimum", ["FAIBLE", "MODERE", "ELEVE", "CRITIQUE"], index=1)

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

    top_n = st.slider("Top communes", min_value=5, max_value=25, value=12, step=1)

    st.markdown("---")
    show_weather = st.checkbox("Layer meteo", value=True)
    show_sectors = st.checkbox("Layer secteurs", value=True)
    show_hydro = st.checkbox("Layer hydro", value=True)
    show_piezo = st.checkbox("Layer piezometres", value=False)
    show_geotech = st.checkbox("Layer geotech", value=False)

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
    filtered_sectors = filtered_sectors[filtered_sectors["risk_level"].map(lambda x: _risk_rank(str(x))) >= _risk_rank(min_risk)]

commune_df = _aggregate_communes(filtered_sectors, sector_rain_col)
sector_pool = filtered_sectors if not filtered_sectors.empty else sectors_df
selected_sector: Dict[str, object] = {}
history_years = 5
selected_compare_sector_ids: List[str] = []
compare_history_df = pd.DataFrame()
history_models: Dict[str, str] = {}
if not sector_pool.empty:
    with st.sidebar:
        st.markdown("---")
        st.subheader("Analyse secteur")
        sector_ids = sector_pool["sector_id"].astype(str).tolist()
        chosen_sector_id = st.selectbox("Secteur detail", sector_ids, index=0)
        selected_compare_sector_ids = st.multiselect(
            "Secteurs a comparer",
            sector_ids,
            default=sector_ids[: min(4, len(sector_ids))],
        )
        history_years = st.slider("Historique mensuel (ans)", min_value=2, max_value=5, value=5, step=1)
    selected_sector = sector_pool[sector_pool["sector_id"].astype(str) == chosen_sector_id].iloc[0].to_dict()

    compare_sector_rows: List[Dict[str, object]] = []
    for sid in selected_compare_sector_ids:
        hit = sector_pool[sector_pool["sector_id"].astype(str) == str(sid)]
        if not hit.empty:
            compare_sector_rows.append(hit.iloc[0].to_dict())
    compare_history_df, history_models = _build_multi_sector_history(compare_sector_rows, int(history_years))

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

history_payload = {"monthly": [], "climatology": [], "model": "", "error": "pas de secteur"}
if selected_sector:
    history_payload = _load_sector_monthly_history(float(selected_sector["latitude"]), float(selected_sector["longitude"]), int(history_years))
history_monthly_df = _safe_df(history_payload.get("monthly"))
history_clim_df = _safe_df(history_payload.get("climatology"))
risk_level = str(snapshot.get("risk_level", "INDETERMINE"))
score = float(snapshot.get("score", 0.0) or 0.0)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Risque global", risk_level)
col2.metric("Score global", f"{score:.2f}/4")
col3.metric("Stations meteo", int(len(filtered_weather)))
col4.metric("Secteurs filtres", int(len(filtered_sectors)))
col5.metric("Communes suivies", int(len(commune_df)))

tabs = st.tabs(["Vue executive", "Carte dynamique", "Tables et alertes", "Metadata"])

with tabs[0]:
    left, right = st.columns([1.5, 1.0])

    with left:
        st.subheader("Classement des communes (note GC)")
        if commune_df.empty:
            st.info("Aucune commune pour les filtres courants.")
        else:
            top_communes = commune_df.head(top_n).copy()
            top_communes["commune_label"] = top_communes["commune_name"] + " (" + top_communes["sector_count"].astype(str) + " sec.)"
            chart = (
                alt.Chart(top_communes)
                .mark_bar()
                .encode(
                    x=alt.X("note_gc:Q", title="Note GC /100"),
                    y=alt.Y("commune_label:N", sort="-x", title="Commune"),
                    color=alt.Color(
                        "commune_risk_level:N",
                        scale=alt.Scale(
                            domain=["FAIBLE", "MODERE", "ELEVE", "CRITIQUE"],
                            range=[RISK_COLOR["FAIBLE"], RISK_COLOR["MODERE"], RISK_COLOR["ELEVE"], RISK_COLOR["CRITIQUE"]],
                        ),
                        legend=alt.Legend(title="Risque"),
                    ),
                    tooltip=[
                        "commune_name",
                        "note_gc",
                        "sector_count",
                        "avg_sector_score",
                        "critical",
                        "high",
                        "avg_rain_period_mm",
                    ],
                )
            )
            st.altair_chart(chart, use_container_width=True)

    with right:
        st.subheader("Distribution des secteurs")
        if filtered_sectors.empty:
            st.info("Pas de secteur filtre.")
        else:
            dist = (
                filtered_sectors["risk_level"]
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

    st.subheader("Comparaison pluvio entre secteurs (5 dernieres annees)")
    if compare_history_df.empty:
        st.info("Selectionne des secteurs avec historique disponible pour comparer les pluies mensuelles.")
    else:
        hist_line = (
            alt.Chart(compare_history_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("ym:N", sort=None, title="Mois"),
                y=alt.Y("monthly_precip_mm:Q", title="Pluie mensuelle (mm)"),
                color=alt.Color("sector_label:N", title="Secteur"),
                tooltip=["ym", "sector_label", "monthly_precip_mm"],
            )
        )
        st.altair_chart(hist_line, use_container_width=True)

        month_compare = (
            compare_history_df.groupby(["sector_label", "month"], as_index=False)["monthly_precip_mm"]
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
                color=alt.Color("sector_label:N", title="Secteur"),
                xOffset=alt.XOffset("sector_label:N"),
                tooltip=["sector_label", "month_label", "mean_monthly_mm"],
            )
        )
        st.altair_chart(climat_cmp, use_container_width=True)
        st.caption("Modele historique: " + ", ".join([f"{k}:{v}" for k, v in history_models.items() if v]))

        latest_rows: List[Dict[str, object]] = []
        if selected_compare_sector_ids:
            for sid in selected_compare_sector_ids:
                hit = sector_pool[sector_pool["sector_id"].astype(str) == str(sid)]
                if hit.empty:
                    continue
                sec = hit.iloc[0].to_dict()
                nwx = _nearest_row(weather_df, float(sec["latitude"]), float(sec["longitude"]))
                latest_rows.append(
                    {
                        "sector_id": sid,
                        "commune_secteur": sec.get("commune_name"),
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
            st.markdown("**Dernieres mesures meteo par secteur compare**")
            st.dataframe(pd.DataFrame(latest_rows), use_container_width=True, hide_index=True)

    st.subheader("Analyse detaillee secteur")
    if not selected_sector:
        st.info("Aucun secteur disponible pour l'analyse detaillee.")
    else:
        sec_commune = str(selected_sector.get("commune_name") or "Inconnue")
        commune_note = None
        if not commune_df.empty:
            match = commune_df[commune_df["commune_name"].astype(str) == sec_commune]
            if not match.empty:
                commune_note = float(match.iloc[0].get("note_gc", 0.0))

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Secteur", str(selected_sector.get("sector_id")))
        s2.metric("Commune", sec_commune)
        s3.metric("Risque secteur", str(selected_sector.get("risk_level")))
        s4.metric("Score secteur", str(selected_sector.get("score")))
        s5.metric("Note GC commune", f"{commune_note:.1f}/100" if commune_note is not None else "N/A")

        nearest_weather = _nearest_row(weather_df, float(selected_sector["latitude"]), float(selected_sector["longitude"]))
        nearest_hydro = _nearest_row(hydro_df, float(selected_sector["latitude"]), float(selected_sector["longitude"]))
        nearest_piezo = _nearest_row(piezo_df, float(selected_sector["latitude"]), float(selected_sector["longitude"]))

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

        st.markdown("**Historique meteo mensuel multi-annees (secteur)**")
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
    if filtered_sectors.empty and filtered_weather.empty:
        st.info("Pas de donnees cartographiques avec ces filtres.")
    else:
        m = _build_map(
            snapshot=snapshot,
            weather_df=filtered_weather,
            sectors_df=filtered_sectors,
            hydro_df=hydro_df,
            piezo_df=piezo_df,
            geotech_df=geotech_df,
            rain_col_weather=rain_col_weather,
            sector_rain_col=sector_rain_col,
            min_risk=min_risk,
            show_weather=show_weather,
            show_sectors=show_sectors,
            show_hydro=show_hydro,
            show_piezo=show_piezo,
            show_geotech=show_geotech,
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
                    "sector_count",
                    "note_gc",
                    "commune_risk_level",
                    "avg_sector_score",
                    "max_sector_score",
                    "critical",
                    "high",
                    "moderate",
                    "avg_rain_period_mm",
                    "max_rain_period_mm",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Secteurs filtres")
    if filtered_sectors.empty:
        st.info("Aucun secteur filtre.")
    else:
        view_cols = [
            "sector_id",
            "commune_name",
            "risk_level",
            "score",
            sector_rain_col,
            "geotech_points",
            "piezometers",
            "hydro_stations",
            "under_watch",
        ]
        present_cols = [c for c in view_cols if c in filtered_sectors.columns]
        st.dataframe(filtered_sectors[present_cols].sort_values("score", ascending=False), use_container_width=True, hide_index=True)

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
    st.subheader("Fonctionnement general")
    st.markdown(
        """
        Cette application est construite pour le suivi d'une ligne de **300 km**:
        - **Niveau 1 (mesure actuelle)**: dernier etat meteo/hydro/nappes/geotech.
        - **Niveau 2 (pilotage secteur)**: scoring par secteurs le long de la LGV.
        - **Niveau 3 (vision historique)**: comparaison mensuelle multi-annees entre secteurs.
        - **Niveau 4 (decision territoriale)**: note GC par commune traversee.
        """
    )

    st.subheader("Logique des stations meteo")
    st.markdown(
        """
        - Les points meteo sont associes au corridor LGV, avec distance a la ligne.
        - Modele prioritaire: **Open-Meteo MeteoFrance Seamless** (maillage fin), fallback modele par defaut si indisponible.
        - Chaque point meteo est rattache a la **commune station** (geocodage inverse) pour comparaison territoriale.
        - Cumuls suivis: **24h**, **7 jours**, **30 jours**, **mois courant**.
        - Historique mensuel: **5 dernieres annees** via Open-Meteo Archive, filtrable par periode.
        """
    )

    st.subheader("Score GC et notes")
    st.markdown(
        """
        - **Score secteur (0-4)**:
          - composantes pluie / geotech / nappes / hydro
          - score final = moyenne des composantes
          - classes: faible, modere, eleve, critique.
        - **Note GC commune (0-100)**:
          - base = `(score secteur moyen / 4) * 68`
          - + majoration secteurs critiques/eleves
          - + signal pluie moyen sur la periode
          - borne entre 0 et 100.
        - Seuils de lecture de la note GC:
          - `< 40`: faible
          - `40-59`: modere
          - `60-79`: eleve
          - `>= 80`: critique
        """
    )

    st.subheader("Données ajoutees et comparables")
    st.markdown(
        """
        - Comparaison **inter-secteurs** des pluies mensuelles sur 5 ans.
        - Filtre de date mensuel (debut/fin) pour isoler des saisons ou annees hydrologiques.
        - Tableau des **dernieres mesures meteo proches** par secteur compare.
        """
    )

    st.subheader("Pistes de donnees complementaires (open data)")
    st.markdown(
        """
        Donnees candidates pour renforcer encore la precision de surveillance:
        - Vigilance pluie/inondation et orages (Meteo-France) pour anticipation operationnelle.
        - Donnees radar precipitation (si acces disponible) pour pluie localisee tres fine.
        - Inventaires de desordres geotechniques locaux et inspections OA terrain (SI interne).
        """
    )

    st.subheader("Donnees et fraicheur")
    st.write(
        {
            "snapshot_timestamp_utc": snapshot.get("timestamp_utc"),
            "snapshot_source": snapshot_source,
            "weather_notice": snapshot.get("weather_notice"),
            "history_model": history_payload.get("model"),
            "history_models_compare": history_models,
        }
    )

ts = snapshot.get("timestamp_utc")
if ts:
    st.caption(f"Donnees snapshot: {ts}")
if snapshot_source:
    st.caption(f"Source snapshot: {snapshot_source}")
st.caption(f"Interface update: {datetime.now(timezone.utc).isoformat()}")
