from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

import folium
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_folium import st_folium

SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
LGV_CENTER = (46.2, 0.2)

RISK_COLOR  = {"FAIBLE":"#16a34a","MODERE":"#ea580c","ELEVE":"#dc2626","CRITIQUE":"#7f1d1d","INDETERMINE":"#6b7280"}
RISK_EMOJI  = {"FAIBLE":"🟢","MODERE":"🟠","ELEVE":"🔴","CRITIQUE":"⛔","INDETERMINE":"⚪"}
RISK_RANK   = {"FAIBLE":1,"MODERE":2,"ELEVE":3,"CRITIQUE":4}
ALERT_ICON  = {
    "HYDRO":"🌊","HYDRO_RESEAU":"🌊","HYDRO_RESEAU_CRITIQUE":"🌊","HYDRO_SEUIL_URGENCE":"🌊",
    "GEOTECH":"⛰️","GEOTECH_CRITIQUE":"⛰️",
    "SECTEUR":"⚠️","SECTEURS_ELEVES":"⚠️","SECTEURS_IA_ELEVES":"🤖",
    "SOLS_FRAGILES":"🌱","COUVERTURE":"ℹ️",
}


@st.cache_data(ttl=900)
def load_snapshot():
    try:
        r = requests.get(SNAPSHOT_URL, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


@st.cache_data(ttl=3600)
def load_monthly_rain(lat: float, lon: float) -> pd.DataFrame:
    end = datetime.now(timezone.utc).date()
    start = (end.replace(day=1) - timedelta(days=365)).replace(day=1)
    try:
        r = requests.get(OPEN_METEO_URL, params={
            "latitude": lat, "longitude": lon,
            "start_date": str(start), "end_date": str(end),
            "daily": "precipitation_sum", "timezone": "Europe/Paris",
        }, timeout=20)
        r.raise_for_status()
        data = r.json()
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


# ── Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LGV SEA – Surveillance", page_icon="🌧", layout="wide")
st.title("🌧 LGV SEA – Surveillance hydro-météo")

snapshot = load_snapshot()
if "_error" in snapshot:
    st.error(f"Erreur chargement snapshot : {snapshot['_error']}")
    st.stop()

ts = snapshot.get("timestamp_utc", "")
risk_global = snapshot.get("risk_level", "INDETERMINE")
color_global = RISK_COLOR.get(risk_global, "#6b7280")

col_h, col_btn = st.columns([4, 1])
col_h.markdown(
    f'<div style="padding:10px 16px;border-radius:8px;background:{color_global}18;border-left:5px solid {color_global}">'
    f'<b style="font-size:18px">{RISK_EMOJI.get(risk_global,"⚪")} Risque global : {risk_global}</b>'
    + (f' <span style="font-size:12px;color:#888"> — {ts[:16].replace("T"," ")} UTC</span>' if ts else "")
    + '</div>', unsafe_allow_html=True)
if col_btn.button("🔄 Rafraîchir"):
    st.cache_data.clear()
    st.rerun()

# ── Données ───────────────────────────────────────────────────────────────
_sec = snapshot.get("sectors")
sectors_df = safe_df(_sec.get("sectors", []) if isinstance(_sec, dict) else [])
commune_ranking = safe_df(snapshot.get("commune_ranking", []))
alerts_raw = snapshot.get("alerts", [])

for col in ["weather_max_24h_mm","weather_max_7d_mm","weather_max_30d_mm",
            "weather_max_month_mm","latitude","longitude","pk_km"]:
    if col in sectors_df.columns:
        sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Filtres")
    communes = sorted(sectors_df["commune_name"].dropna().unique()) if "commune_name" in sectors_df.columns else []
    selected = st.selectbox("📍 Commune", ["— Toutes —"] + list(communes))
    periode = st.selectbox("📅 Période pluvio", ["24h","7 jours","30 jours","Mois courant"])
    rain_col = {"24h":"weather_max_24h_mm","7 jours":"weather_max_7d_mm",
                "30 jours":"weather_max_30d_mm","Mois courant":"weather_max_month_mm"}[periode]
    risque_min = st.selectbox("⚠ Risque minimum", ["Tout","FAIBLE","MODERE","ELEVE","CRITIQUE"])
    show_alerts = st.checkbox("Afficher alertes", value=True)

# ── Filtrage ──────────────────────────────────────────────────────────────
df = sectors_df.copy()
if selected != "— Toutes —":
    df = df[df["commune_name"] == selected]
if risque_min != "Tout" and "risk_level" in df.columns:
    min_rank = RISK_RANK.get(risque_min, 0)
    df = df[df["risk_level"].map(lambda x: RISK_RANK.get(str(x), 0)) >= min_rank]

# ── Vue commune ────────────────────────────────────────────────────────────
if selected != "— Toutes —":
    commune_row = {}
    if not commune_ranking.empty and "commune_name" in commune_ranking.columns:
        r = commune_ranking[commune_ranking["commune_name"] == selected]
        if not r.empty:
            commune_row = r.iloc[0].to_dict()
    risk_lvl = str(commune_row.get("commune_risk_level", "INDETERMINE"))
    color = RISK_COLOR.get(risk_lvl, "#6b7280")
    st.markdown(
        f'<div style="padding:10px 14px;border-radius:8px;border-left:5px solid {color};'
        f'background:{color}18;margin-bottom:10px">'
        f'<b style="font-size:17px">{RISK_EMOJI.get(risk_lvl,"⚪")} {selected}</b>'
        f'<span style="margin-left:14px;color:{color};font-weight:600">Risque : {risk_lvl}</span>'
        f'</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for cw, label, cn in [(c1,"☔ 24h","weather_max_24h_mm"),(c2,"🌧 7j","weather_max_7d_mm"),
                           (c3,"🌦 30j","weather_max_30d_mm"),(c4,"📅 Mois","weather_max_month_mm")]:
        val = df[cn].max() if cn in df.columns and not df.empty else None
        cw.metric(label, f"{val:.1f} mm" if val is not None and pd.notna(val) else "—")

# ── Alertes ────────────────────────────────────────────────────────────────
if show_alerts and alerts_raw:
    visible = [a for a in alerts_raw if a.get("level") in ("CRITIQUE","ELEVE","MODERE")]
    visible.sort(key=lambda a: {"CRITIQUE":0,"ELEVE":1,"MODERE":2}.get(a.get("level",""), 9))
    if visible:
        st.subheader("🚨 Alertes actives")
        for a in visible:
            lvl   = a.get("level","")
            atype = a.get("type","")
            msg   = a.get("message","")
            icon  = ALERT_ICON.get(atype, "⚠️")
            color = RISK_COLOR.get(lvl, "#6b7280")
            st.markdown(
                f'<div style="padding:7px 12px;border-radius:6px;border-left:4px solid {color};'
                f'background:{color}12;margin-bottom:5px;font-size:13px">'
                f'<b>{icon} [{lvl}]</b> {msg}</div>',
                unsafe_allow_html=True)

# ── Graphe mensuel ─────────────────────────────────────────────────────────
st.subheader("📊 Suivi pluviométrique mensuel (12 mois)")
map_df = df.dropna(subset=["latitude","longitude"]) if "latitude" in df.columns and "longitude" in df.columns else pd.DataFrame()
lat_c = float(map_df["latitude"].mean()) if not map_df.empty else LGV_CENTER[0]
lon_c = float(map_df["longitude"].mean()) if not map_df.empty else LGV_CENTER[1]

monthly_df = load_monthly_rain(lat_c, lon_c)
if not monthly_df.empty:
    fig = px.bar(
        monthly_df, x="mois", y="pluie_mm",
        labels={"mois":"Mois","pluie_mm":"Pluie (mm)"},
        color="pluie_mm",
        color_continuous_scale=["#bfdbfe","#3b82f6","#1d4ed8","#1e3a8a"],
        text="pluie_mm",
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20,b=20,l=20,r=20), height=270, xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Données historiques indisponibles.")

# ── Carte ──────────────────────────────────────────────────────────────────
st.subheader("🗺 Carte des secteurs")
if not map_df.empty:
    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=9 if selected == "— Toutes —" else 11,
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google", control_scale=True,
    )
    for _, row in map_df.iterrows():
        lvl   = str(row.get("risk_level","INDETERMINE"))
        color = RISK_COLOR.get(lvl,"#6b7280")
        rv    = row.get(rain_col) or 0
        folium.CircleMarker(
            [float(row["latitude"]), float(row["longitude"])],
            radius=7, color=color, fill=True, fill_opacity=0.85, weight=1.5,
            tooltip=f"{row.get('commune_name','')} | {lvl} | {rv:.1f} mm",
            popup=folium.Popup(
                f"<b>{row.get('commune_name','')} – PK {row.get('pk_km','')} km</b><br>"
                f"Risque : <b style='color:{color}'>{lvl}</b><br>"
                f"Pluie {periode} : <b>{rv:.1f} mm</b>", max_width=260),
        ).add_to(m)
    lgv_lines = snapshot.get("lgv_lines", [])
    if isinstance(lgv_lines, list):
        for seg in lgv_lines:
            if isinstance(seg, list) and len(seg) >= 2:
                pts = [[p[0], p[1]] for p in seg if isinstance(p, (list,tuple)) and len(p) >= 2]
                if pts:
                    folium.PolyLine(pts, color="#cc0000", weight=2, opacity=0.6).add_to(m)
    st_folium(m, use_container_width=True, height=440, returned_objects=[])
else:
    st.info("Pas de coordonnées pour la sélection.")

# ── Tableau ────────────────────────────────────────────────────────────────
st.subheader("📋 Secteurs" + (f" — {selected}" if selected != "— Toutes —" else ""))
if df.empty:
    st.info("Aucun secteur pour ces filtres.")
else:
    cols = [c for c in ["commune_name","pk_km","risk_level",rain_col,"ai_pred_risk_level"] if c in df.columns]
    rename = {"commune_name":"Commune","pk_km":"PK (km)","risk_level":"Risque",
              "ai_pred_risk_level":"Risque IA","weather_max_24h_mm":"24h mm",
              "weather_max_7d_mm":"7j mm","weather_max_30d_mm":"30j mm","weather_max_month_mm":"Mois mm"}
    disp = df[cols].rename(columns=rename)
    rl = rename.get(rain_col, rain_col)
    if rl in disp.columns:
        disp = disp.sort_values(rl, ascending=False, na_position="last")
    st.dataframe(disp, use_container_width=True, hide_index=True, height=300)

# ── Classement communes ────────────────────────────────────────────────────
if selected == "— Toutes —" and not commune_ranking.empty:
    st.subheader("🏘 Classement communes")
    cr = commune_ranking.copy()
    if "commune_risk_level" in cr.columns:
        cr["_rank"] = cr["commune_risk_level"].map(lambda x: RISK_RANK.get(str(x), 0))
        cr = cr.sort_values("_rank", ascending=False).drop(columns=["_rank"])
    show = [c for c in ["commune_name","departement_name","commune_risk_level",
                         "commune_note","sector_count","critical","high"] if c in cr.columns]
    rename_cr = {"commune_name":"Commune","departement_name":"Département","commune_risk_level":"Risque",
                 "commune_note":"Note","sector_count":"Secteurs","critical":"Critique","high":"Élevé"}
    st.dataframe(cr[show].rename(columns=rename_cr), use_container_width=True,
                 hide_index=True, height=360)
