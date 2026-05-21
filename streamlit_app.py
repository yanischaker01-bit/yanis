from datetime import datetime

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from lgv_monitoring.config import (
    DB_PATH,
    DEFAULT_STATION_MAX_DISTANCE_KM,
    LGV_COORDINATES_LATLON,
    RIVER_POINTS,
)
from lgv_monitoring.database import init_db, set_station_code
from lgv_monitoring.geo import point_distance_to_line_km, build_lgv_line
from lgv_monitoring.service import LGVMonitoringService


st.set_page_config(page_title="LGV SEA Hydrometeo", page_icon="railway", layout="wide")
st.title("LGV SEA Bordeaux-Tours: pluviometrie et niveaux d'eau")

init_db(DB_PATH, RIVER_POINTS)
service = LGVMonitoringService(DB_PATH, RIVER_POINTS)
lgv_line = build_lgv_line(LGV_COORDINATES_LATLON)

with st.sidebar:
    st.header("Parametres")
    collect_distance_km = st.number_input(
        "Distance max station pluvio a collecter (km)",
        min_value=0.5,
        max_value=100.0,
        value=25.0,
        step=0.5,
    )
    display_distance_km = st.number_input(
        "Distance max station affichee sur la carte (km)",
        min_value=0.5,
        max_value=50.0,
        value=DEFAULT_STATION_MAX_DISTANCE_KM,
        step=0.5,
    )
    if st.button("Collecter maintenant", use_container_width=True):
        result = service.collect_once(max_station_distance_km=float(collect_distance_km))
        if result.get("ok"):
            st.success(
                f"Collecte terminee: {result['pluvio_rows']} observations pluvio "
                f"| hydro: {result['hydro_status']}"
            )
        else:
            st.error(f"Echec collecte: {result.get('error')}")

payload = service.get_map_payload()
pluvio_latest = pd.DataFrame(payload["pluvio_latest"])
rivers_df = pd.DataFrame(payload["rivers"])

if not pluvio_latest.empty:
    pluvio_latest = pluvio_latest[pluvio_latest["distance_to_lgv_km"] <= display_distance_km].copy()

left_col, right_col = st.columns([2.0, 1.0])
with left_col:
    m = folium.Map(location=[46.2, 0.2], zoom_start=7, tiles="CartoDB positron")
    folium.PolyLine(
        locations=[(lat, lon) for lat, lon in LGV_COORDINATES_LATLON],
        color="#005f73",
        weight=4,
        tooltip="Trace LGV SEA",
    ).add_to(m)

    if not pluvio_latest.empty:
        for row in pluvio_latest.to_dict(orient="records"):
            popup = (
                f"<b>Station {row['station_id']}</b><br>"
                f"Pluie: {row['precipitation_mm']} mm<br>"
                f"Distance LGV: {row['distance_to_lgv_km']} km<br>"
                f"Obs: {row['obs_time_utc']}"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=7,
                color="#0077b6",
                fill=True,
                fill_opacity=0.85,
                tooltip=f"Station {row['station_id']}",
                popup=folium.Popup(popup, max_width=320),
            ).add_to(m)

    for row in payload["rivers"]:
        river = row["river"]
        river_obs = payload["hydro_details"].get(river, [])
        trend_text = "N/A"
        level_text = "N/A"
        if len(river_obs) >= 2:
            latest = river_obs[0]
            oldest = river_obs[-1]
            t1 = pd.to_datetime(latest["obs_time_utc"], utc=True, errors="coerce")
            t0 = pd.to_datetime(oldest["obs_time_utc"], utc=True, errors="coerce")
            dt_h = max((t1 - t0).total_seconds() / 3600.0, 0.0)
            trend = (latest["level_m"] - oldest["level_m"]) / dt_h if dt_h > 0 else 0.0
            trend_text = f"{trend:.3f} m/h"
            level_text = f"{latest['level_m']:.3f} m"
        elif len(river_obs) == 1:
            level_text = f"{river_obs[0]['level_m']:.3f} m"

        popup = (
            f"<b>{row['river']}</b><br>{row['name']}<br>"
            f"Station code: {row['station_code'] or 'non configure'}<br>"
            f"Niveau: {level_text}<br>Tendance: {trend_text}<br>"
            f"Seuil: {row['threshold_m'] if row['threshold_m'] is not None else 'N/A'} m"
        )
        color = "#ee9b00"
        if row["station_code"] and len(river_obs) > 0 and row["threshold_m"] is not None:
            latest_level = river_obs[0]["level_m"]
            if latest_level >= row["threshold_m"]:
                color = "#bb3e03"
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            tooltip=f"Cours d'eau: {row['river']}",
            popup=folium.Popup(popup, max_width=340),
            icon=folium.Icon(color="blue", icon_color="white", icon="tint", prefix="fa"),
        ).add_to(m)
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=350,
            color=color,
            fill=True,
            fill_opacity=0.1,
            weight=1,
        ).add_to(m)

    st_data = st_folium(m, width=None, height=620)
    if st_data and st_data.get("last_object_clicked"):
        st.caption(f"Dernier clic carte: {st_data['last_object_clicked']}")

with right_col:
    st.subheader("Synthese")
    st.metric("Stations pluvio visibles", int(len(pluvio_latest)))
    configured_rivers = int((rivers_df["station_code"].notna()).sum()) if not rivers_df.empty else 0
    st.metric("Cours d'eau configures (station_code)", configured_rivers)
    if payload["last_runs"]:
        last_run = payload["last_runs"][0]
        st.write("Derniere collecte")
        st.code(f"{last_run['run_time_utc']} | {last_run['status']}")
    else:
        st.info("Aucune collecte en base pour le moment.")

    st.subheader("Configurer station_code")
    if not rivers_df.empty:
        river_selected = st.selectbox("Cours d'eau", rivers_df["river"].tolist())
        current_station = rivers_df.loc[rivers_df["river"] == river_selected, "station_code"].iloc[0]
        station_code_input = st.text_input(
            "Code station HubEau (code_entite)",
            value=current_station if pd.notna(current_station) else "",
        )
        if st.button("Enregistrer station_code", use_container_width=True):
            set_station_code(DB_PATH, river_selected, station_code_input.strip() or None)
            st.success("Station code enregistre.")
            st.rerun()

st.markdown("---")
st.subheader("Detail station pluvio")
if pluvio_latest.empty:
    st.info("Aucune station pluvio disponible pour le filtre courant.")
else:
    station_id = st.selectbox("Station", pluvio_latest["station_id"].tolist())
    station_row = pluvio_latest.loc[pluvio_latest["station_id"] == station_id].iloc[0]
    st.write(
        {
            "distance_to_lgv_km": station_row["distance_to_lgv_km"],
            "precipitation_mm": station_row["precipitation_mm"],
            "obs_time_utc": station_row["obs_time_utc"],
        }
    )

st.subheader("Detail cours d'eau")
if rivers_df.empty:
    st.info("Pas de cours d'eau en base.")
else:
    river_name = st.selectbox("Cours d'eau ", rivers_df["river"].tolist())
    river_row = rivers_df.loc[rivers_df["river"] == river_name].iloc[0].to_dict()
    hydro_df = pd.DataFrame(payload["hydro_details"].get(river_name, []))
    station_radius = river_row.get("station_code") or "non configure"
    distance_km = point_distance_to_line_km(river_row["latitude"], river_row["longitude"], lgv_line)
    st.write(
        {
            "station_code": station_radius,
            "distance_point_to_lgv_km": round(distance_km, 3),
            "threshold_m": river_row.get("threshold_m"),
            "rapid_rise_mph": river_row.get("rapid_rise_mph"),
        }
    )
    if hydro_df.empty:
        st.info("Aucune observation hydrometrique disponible.")
    else:
        hydro_df["obs_time_utc"] = pd.to_datetime(hydro_df["obs_time_utc"], utc=True, errors="coerce")
        hydro_df = hydro_df.sort_values("obs_time_utc")
        st.line_chart(hydro_df.set_index("obs_time_utc")["level_m"])
        st.dataframe(hydro_df.sort_values("obs_time_utc", ascending=False).head(30), use_container_width=True)

st.caption(f"Derniere mise a jour interface: {datetime.utcnow().isoformat()}Z")

