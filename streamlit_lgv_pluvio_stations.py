from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

SNAPSHOT_URL  = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"

# Départements traversés par la LGV SEA
DEPS = {
    "37": {"nom": "Indre-et-Loire", "lat": 47.38, "lon":  0.69},
    "86": {"nom": "Vienne",          "lat": 46.58, "lon":  0.34},
    "79": {"nom": "Deux-Sèvres",     "lat": 46.32, "lon": -0.46},
    "16": {"nom": "Charente",         "lat": 45.65, "lon":  0.16},
    "17": {"nom": "Charente-Maritime","lat": 45.75, "lon": -0.63},
    "33": {"nom": "Gironde",          "lat": 44.84, "lon": -0.58},
}

def rain_risk(max_mm: float):
    if max_mm >= 60: return "ROUGE",  "#dc2626", "🔴"
    if max_mm >= 30: return "ORANGE", "#ea580c", "🟠"
    if max_mm >= 10: return "JAUNE",  "#eab308", "🟡"
    return               "VERT",   "#16a34a", "🟢"

def rain_color_mm(mm: float) -> str:
    if mm >= 60: return "#dc2626"
    if mm >= 30: return "#ea580c"
    if mm >= 10: return "#3b82f6"
    return "#93c5fd"


@st.cache_data(ttl=900)
def load_snapshot():
    try:
        r = requests.get(SNAPSHOT_URL, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


@st.cache_data(ttl=3600)
def load_forecast_dep(dep: str) -> dict:
    d = DEPS[dep]
    try:
        r = requests.get(FORECAST_URL, params={
            "latitude": d["lat"], "longitude": d["lon"],
            "daily": "precipitation_sum,precipitation_probability_max,weathercode",
            "forecast_days": 7, "timezone": "Europe/Paris",
        }, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def load_forecast_coord(lat: float, lon: float) -> pd.DataFrame:
    try:
        r = requests.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum,precipitation_probability_max",
            "forecast_days": 7, "timezone": "Europe/Paris",
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
        daily = data.get("daily", {})
        return pd.DataFrame({
            "date": daily.get("time", []),
            "pluie_mm": daily.get("precipitation_sum", []),
            "proba_%": daily.get("precipitation_probability_max", []),
        })
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_monthly_rain(lat: float, lon: float) -> pd.DataFrame:
    end   = datetime.now(timezone.utc).date()
    start = (end.replace(day=1) - timedelta(days=365)).replace(day=1)
    try:
        r = requests.get(ARCHIVE_URL, params={
            "latitude": lat, "longitude": lon,
            "start_date": str(start), "end_date": str(end),
            "daily": "precipitation_sum", "timezone": "Europe/Paris",
        }, timeout=20)
        r.raise_for_status()
        data  = r.json()
        dates = data["daily"]["time"]
        rain  = data["daily"]["precipitation_sum"]
        monthly: dict = defaultdict(float)
        for d, v in zip(dates, rain):
            if v is not None:
                monthly[d[:7]] += v
        rows = [{"mois": m, "pluie_mm": round(v, 1)} for m, v in sorted(monthly.items())]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def safe_df(records) -> pd.DataFrame:
    if isinstance(records, list) and records:
        try:
            return pd.DataFrame(records)
        except Exception:
            pass
    return pd.DataFrame()


# ── Config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LGV SEA – Pluviométrie", page_icon="🌧", layout="wide")
st.title("🌧 LGV SEA – Pluviométrie")

if st.button("🔄 Rafraîchir", key="top_refresh"):
    st.cache_data.clear()
    st.rerun()

# ── Snapshot ────────────────────────────────────────────────────────────────
snapshot = load_snapshot()
if "_error" in snapshot:
    st.error(f"Erreur snapshot : {snapshot['_error']}")
    st.stop()

_sec = snapshot.get("sectors")
sectors_df = safe_df(_sec.get("sectors", []) if isinstance(_sec, dict) else [])
for col in ["weather_max_24h_mm","weather_max_7d_mm","weather_max_30d_mm",
            "weather_max_month_mm","latitude","longitude","pk_km"]:
    if col in sectors_df.columns:
        sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")

# ── 1. VIGILANCE PLUVIO PAR DÉPARTEMENT ─────────────────────────────────────
st.subheader("⚡ Vigilance pluvio — prévisions 7 jours par département")
dep_cols = st.columns(len(DEPS))
dep_forecasts = {}
for col_w, (dep, info) in zip(dep_cols, DEPS.items()):
    fc = load_forecast_dep(dep)
    daily = fc.get("daily", {})
    rains = [v for v in daily.get("precipitation_sum", []) if v is not None]
    max_mm = max(rains) if rains else 0.0
    total  = sum(rains)
    lvl, color, emoji = rain_risk(max_mm)
    dep_forecasts[dep] = {"rains": rains, "max_mm": max_mm, "total": total, "color": color}
    col_w.markdown(
        f'<div style="padding:10px 6px;border-radius:8px;border:2px solid {color};'
        f'text-align:center;background:{color}14">'
        f'<div style="font-size:22px">{emoji}</div>'
        f'<b style="font-size:13px">Dép. {dep}</b><br>'
        f'<span style="font-size:11px;color:#666">{info["nom"]}</span><br>'
        f'<b style="color:{color}">{lvl}</b><br>'
        f'<span style="font-size:12px">max {max_mm:.0f} mm/j<br>total {total:.0f} mm</span>'
        f'</div>', unsafe_allow_html=True)

# Lien vigilance officielle
st.caption("💡 Vigilance officielle Météo-France : [vigilance.meteofrance.fr](https://vigilance.meteofrance.fr/)")

# ── Sidebar ──────────────────────────────────────────────────────────────────
communes = sorted(sectors_df["commune_name"].dropna().unique()) if "commune_name" in sectors_df.columns else []
with st.sidebar:
    st.subheader("📍 Communes")
    selected_multi = st.multiselect("Sélectionner communes", communes,
                                     default=communes[:3] if len(communes) >= 3 else communes)
    selected_one = st.selectbox("Commune principale (carte + graphes)", ["— Toutes —"] + list(communes))
    periode = st.selectbox("📅 Période pluvio", ["24h","7 jours","30 jours","Mois courant"])
    rain_col = {"24h":"weather_max_24h_mm","7 jours":"weather_max_7d_mm",
                "30 jours":"weather_max_30d_mm","Mois courant":"weather_max_month_mm"}[periode]

# ── 2. COMPARAISON COMMUNES ─────────────────────────────────────────────────
if selected_multi and not sectors_df.empty and rain_col in sectors_df.columns:
    st.subheader("📊 Comparaison communes — pluie par secteur")
    df_cmp = sectors_df[sectors_df["commune_name"].isin(selected_multi)][["commune_name","pk_km",rain_col]].copy()
    df_cmp = df_cmp.dropna(subset=[rain_col]).sort_values(rain_col, ascending=False)
    if not df_cmp.empty:
        fig = px.bar(
            df_cmp, x="commune_name", y=rain_col, color=rain_col,
            color_continuous_scale=["#bfdbfe","#3b82f6","#ea580c","#dc2626"],
            labels={"commune_name":"Commune", rain_col:"Pluie (mm)"},
            text=rain_col,
            hover_data=["pk_km"] if "pk_km" in df_cmp.columns else None,
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=300,
                          plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(t=20,b=20,l=20,r=20),
                          xaxis=dict(tickangle=-30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas de données pour la sélection.")

# ── 3. PRÉVISIONS 7J ────────────────────────────────────────────────────────
comm_df = sectors_df if selected_one == "— Toutes —" else \
          sectors_df[sectors_df["commune_name"] == selected_one]
map_df = comm_df.dropna(subset=["latitude","longitude"]) if \
         "latitude" in sectors_df.columns and "longitude" in sectors_df.columns else pd.DataFrame()
lat_c = float(map_df["latitude"].mean()) if not map_df.empty else 46.2
lon_c = float(map_df["longitude"].mean()) if not map_df.empty else 0.2

st.subheader(f"🔮 Prévisions 7 jours — {'LGV SEA' if selected_one == '— Toutes —' else selected_one}")
fc_df = load_forecast_coord(lat_c, lon_c)
if not fc_df.empty:
    fc_df["pluie_mm"] = pd.to_numeric(fc_df["pluie_mm"], errors="coerce").fillna(0)
    fc_df["color"] = fc_df["pluie_mm"].apply(rain_color_mm)
    fig2 = go.Figure()
    fig2.add_bar(
        x=fc_df["date"], y=fc_df["pluie_mm"],
        marker_color=fc_df["color"].tolist(),
        text=fc_df["pluie_mm"].apply(lambda v: f"{v:.0f}"),
        textposition="outside",
        name="Pluie mm",
    )
    if "proba_%" in fc_df.columns:
        fig2.add_scatter(
            x=fc_df["date"], y=fc_df["proba_%"],
            mode="lines+markers", name="Proba %",
            yaxis="y2", line=dict(color="#6366f1", dash="dot"), marker=dict(size=5),
        )
    fig2.update_layout(
        yaxis=dict(title="mm"),
        yaxis2=dict(title="%", overlaying="y", side="right", range=[0,110]),
        legend=dict(orientation="h", y=1.1), height=280,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=30,b=20,l=20,r=50), xaxis=dict(tickangle=-20),
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Prévisions indisponibles.")

# ── 4. HISTORIQUE MENSUEL ───────────────────────────────────────────────────
st.subheader("📅 Historique mensuel (12 mois)")
monthly_df = load_monthly_rain(lat_c, lon_c)
if not monthly_df.empty:
    fig3 = px.bar(
        monthly_df, x="mois", y="pluie_mm",
        color="pluie_mm",
        color_continuous_scale=["#bfdbfe","#3b82f6","#1d4ed8","#1e3a8a"],
        labels={"mois":"Mois","pluie_mm":"Pluie (mm)"}, text="pluie_mm",
    )
    fig3.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig3.update_layout(coloraxis_showscale=False, height=260,
                       plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(t=20,b=20,l=20,r=20), xaxis=dict(tickangle=-30))
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Historique indisponible.")

# ── 5. CARTE ────────────────────────────────────────────────────────────────
st.subheader("🗺 Carte — pluie par secteur")
if not map_df.empty and rain_col in map_df.columns:
    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=8 if selected_one == "— Toutes —" else 11,
        tiles="CartoDB positron", control_scale=True,
    )
    vmax = map_df[rain_col].max() if map_df[rain_col].max() > 0 else 1
    for _, row in map_df.dropna(subset=[rain_col]).iterrows():
        rv   = float(row[rain_col])
        col  = rain_color_mm(rv)
        rad  = max(5, min(20, 5 + (rv / vmax) * 15))
        folium.CircleMarker(
            [float(row["latitude"]), float(row["longitude"])],
            radius=rad, color=col, fill=True, fill_opacity=0.8, weight=1.5,
            tooltip=f"{row.get('commune_name','')} — {rv:.1f} mm ({periode})",
            popup=folium.Popup(
                f"<b>{row.get('commune_name','')} PK {row.get('pk_km','')} km</b><br>"
                f"Pluie {periode} : <b>{rv:.1f} mm</b>", max_width=230),
        ).add_to(m)
    lgv_lines = snapshot.get("lgv_lines", [])
    if isinstance(lgv_lines, list):
        for seg in lgv_lines:
            if isinstance(seg, list):
                pts = [[p[0], p[1]] for p in seg if isinstance(p, (list,tuple)) and len(p) >= 2]
                if pts:
                    folium.PolyLine(pts, color="#cc0000", weight=2.5, opacity=0.7).add_to(m)
    st_folium(m, use_container_width=True, height=450, returned_objects=[])
else:
    st.info("Pas de données de localisation.")

# ── 6. TABLEAU ──────────────────────────────────────────────────────────────
st.subheader("📋 Données par secteur")
df_tbl = comm_df.copy()
show_cols = [c for c in ["commune_name","pk_km","weather_max_24h_mm","weather_max_7d_mm",
                          "weather_max_30d_mm","weather_max_month_mm"] if c in df_tbl.columns]
rename = {"commune_name":"Commune","pk_km":"PK (km)","weather_max_24h_mm":"24h (mm)",
          "weather_max_7d_mm":"7j (mm)","weather_max_30d_mm":"30j (mm)","weather_max_month_mm":"Mois (mm)"}
disp = df_tbl[show_cols].rename(columns=rename)
rl = rename.get(rain_col, rain_col)
if rl in disp.columns:
    disp = disp.sort_values(rl, ascending=False, na_position="last")
if not disp.empty:
    st.dataframe(disp, use_container_width=True, hide_index=True, height=320)
else:
    st.info("Aucun secteur.")
