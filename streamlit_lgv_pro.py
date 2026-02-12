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
if not filtered_weather.empty:
    filtered_weather = filtered_weather[filtered_weather["risk_level"].map(lambda x: _risk_rank(str(x))) >= _risk_rank(min_risk)]

filtered_sectors = sectors_df.copy()
if not filtered_sectors.empty and selected_communes:
    filtered_sectors = filtered_sectors[filtered_sectors["commune_name"].astype(str).isin(selected_communes)]
if not filtered_sectors.empty:
    filtered_sectors = filtered_sectors[filtered_sectors["risk_level"].map(lambda x: _risk_rank(str(x))) >= _risk_rank(min_risk)]

commune_df = _aggregate_communes(filtered_sectors, sector_rain_col)
risk_level = str(snapshot.get("risk_level", "INDETERMINE"))
score = float(snapshot.get("score", 0.0) or 0.0)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Risque global", risk_level)
col2.metric("Score global", f"{score:.2f}/4")
col3.metric("Stations meteo", int(len(filtered_weather)))
col4.metric("Secteurs filtres", int(len(filtered_sectors)))
col5.metric("Communes suivies", int(len(commune_df)))

tabs = st.tabs(["Vue executive", "Carte dynamique", "Tables et alertes"])

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
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
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
                tooltip=["station_id", "source", rain_col_weather, "distance_to_lgv_km", "risk_level"],
            )
        )
        st.altair_chart(chart_st, use_container_width=True)

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

ts = snapshot.get("timestamp_utc")
if ts:
    st.caption(f"Donnees snapshot: {ts}")
if snapshot_source:
    st.caption(f"Source snapshot: {snapshot_source}")
st.caption(f"Interface update: {datetime.now(timezone.utc).isoformat()}")
