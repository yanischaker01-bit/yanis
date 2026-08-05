from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEPS = {
    "37": {"nom": "Indre-et-Loire",   "lat": 47.38, "lon":  0.69},
    "86": {"nom": "Vienne",            "lat": 46.58, "lon":  0.34},
    "79": {"nom": "Deux-Sèvres",       "lat": 46.32, "lon": -0.46},
    "16": {"nom": "Charente",           "lat": 45.65, "lon":  0.16},
    "17": {"nom": "Charente-Maritime", "lat": 45.75, "lon": -0.63},
    "33": {"nom": "Gironde",            "lat": 44.84, "lon": -0.58},
}

ALERT_CFG = {
    "ORAGE":      ("⛈️",  "Orage"),
    "CANICULE":   ("🌡️",  "Canicule"),
    "INCENDIE":   ("🔥",  "Incendie"),
    "INONDATION": ("🌊",  "Inondation"),
    "VENT":       ("💨",  "Vent violent"),
    "VIGICRUE":   ("🏞️",  "Vigilance crue"),
}
LEVEL_COLOR = {"ROUGE":"#dc2626","ORANGE":"#ea580c","JAUNE":"#eab308","VERT":"#16a34a","INFO":"#3b82f6"}
LEVEL_EMOJI = {"ROUGE":"🔴","ORANGE":"🟠","JAUNE":"🟡","VERT":"🟢","INFO":"🔵"}
LEVEL_RANK  = {"ROUGE":4,"ORANGE":3,"JAUNE":2,"VERT":1,"INFO":0}


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


@st.cache_data(ttl=1800)
def load_weather_alerts_all() -> list:
    """Alertes dérivées des prévisions Open-Meteo pour les 6 départements LGV SEA."""
    alerts = []
    for dep, info in DEPS.items():
        try:
            r = requests.get(FORECAST_URL, params={
                "latitude": info["lat"], "longitude": info["lon"],
                "daily": ("precipitation_sum,temperature_2m_max,"
                          "weathercode,wind_speed_10m_max"),
                "forecast_days": 7, "timezone": "Europe/Paris",
            }, timeout=15)
            r.raise_for_status()
            daily   = r.json().get("daily", {})
            dates   = daily.get("time", [])
            precips = daily.get("precipitation_sum",        [0]*7)
            tmaxes  = daily.get("temperature_2m_max",       [0]*7)
            wcodes  = daily.get("weathercode",              [0]*7)
            winds   = daily.get("wind_speed_10m_max",       [0]*7)
            rain7   = sum(p or 0 for p in precips)

            seen_fire = False
            for i, date in enumerate(dates):
                p = precips[i] or 0
                t = tmaxes[i]  or 0
                w = wcodes[i]  or 0
                v = winds[i]   or 0
                d_str = date[5:]  # MM-DD

                # ⛈️ Orages (WMO codes 80-82 averses, 95-99 orages)
                if w >= 99:
                    alerts.append(dict(dep=dep, date=date, type="ORAGE", level="ROUGE",
                        msg=f"Dép.{dep} le {d_str} — Orages violents avec grêle"))
                elif w >= 95:
                    alerts.append(dict(dep=dep, date=date, type="ORAGE", level="ORANGE",
                        msg=f"Dép.{dep} le {d_str} — Orages"))
                elif w in (80, 81, 82):
                    alerts.append(dict(dep=dep, date=date, type="ORAGE", level="JAUNE",
                        msg=f"Dép.{dep} le {d_str} — Averses orageuses"))

                # 🌊 Inondation (précipitations intenses)
                if p >= 60:
                    alerts.append(dict(dep=dep, date=date, type="INONDATION", level="ROUGE",
                        msg=f"Dép.{dep} le {d_str} — Pluies diluviennes : {p:.0f} mm"))
                elif p >= 30:
                    alerts.append(dict(dep=dep, date=date, type="INONDATION", level="ORANGE",
                        msg=f"Dép.{dep} le {d_str} — Pluies intenses : {p:.0f} mm"))
                elif p >= 15:
                    alerts.append(dict(dep=dep, date=date, type="INONDATION", level="JAUNE",
                        msg=f"Dép.{dep} le {d_str} — Pluies soutenues : {p:.0f} mm"))

                # 🌡️ Canicule
                if t >= 40:
                    alerts.append(dict(dep=dep, date=date, type="CANICULE", level="ROUGE",
                        msg=f"Dép.{dep} le {d_str} — Canicule extrême : {t:.0f}°C"))
                elif t >= 36:
                    alerts.append(dict(dep=dep, date=date, type="CANICULE", level="ROUGE",
                        msg=f"Dép.{dep} le {d_str} — Canicule : {t:.0f}°C"))
                elif t >= 33:
                    alerts.append(dict(dep=dep, date=date, type="CANICULE", level="ORANGE",
                        msg=f"Dép.{dep} le {d_str} — Forte chaleur : {t:.0f}°C"))

                # 💨 Vent violent
                if v >= 100:
                    alerts.append(dict(dep=dep, date=date, type="VENT", level="ROUGE",
                        msg=f"Dép.{dep} le {d_str} — Vents très violents : {v:.0f} km/h"))
                elif v >= 80:
                    alerts.append(dict(dep=dep, date=date, type="VENT", level="ORANGE",
                        msg=f"Dép.{dep} le {d_str} — Vents violents : {v:.0f} km/h"))
                elif v >= 60:
                    alerts.append(dict(dep=dep, date=date, type="VENT", level="JAUNE",
                        msg=f"Dép.{dep} le {d_str} — Vents forts : {v:.0f} km/h"))

                # 🔥 Risque incendie (chaleur + sécheresse + vent)
                if not seen_fire and t >= 30 and rain7 < 10 and v >= 25:
                    lvl = "ROUGE" if (t >= 35 and rain7 < 5 and v >= 35) else "ORANGE"
                    alerts.append(dict(dep=dep, date=date, type="INCENDIE", level=lvl,
                        msg=f"Dép.{dep} — Risque incendie : {t:.0f}°C, "
                            f"vent {v:.0f} km/h, pluie 7j {rain7:.0f} mm"))
                    seen_fire = True

        except Exception:
            continue

    return sorted(alerts, key=lambda x: (-LEVEL_RANK.get(x["level"], 0), x["date"]))


@st.cache_data(ttl=1800)
def load_vigicrue() -> list:
    """Vigilance crues via API officielle Vigicrue (XML)."""
    VC_LEVEL = {1: "VERT", 2: "JAUNE", 3: "ORANGE", 4: "ROUGE"}
    results  = []
    deps_ok  = set(DEPS.keys())
    try:
        r = requests.get(
            "https://www.vigicrues.gouv.fr/services/2/InfosCrues.xml",
            params={"TypEntVigiCru": 3}, timeout=15,
        )
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            for elem in root.iter():
                cd  = (elem.get("CdEntVigiCru") or elem.get("CdDep")
                       or elem.get("CDDep") or "")
                niv = elem.get("NivVigiCruHydro") or elem.get("NivVigiCru")
                if cd in deps_ok and niv:
                    lvl = VC_LEVEL.get(int(niv), "VERT")
                    nom = DEPS[cd]["nom"]
                    results.append(dict(dep=cd, level=lvl, type="VIGICRUE",
                        msg=f"Dép.{cd} {nom} — Vigilance crue {lvl.lower()}"))
    except Exception:
        pass
    # If API returned nothing useful, show info
    if not results:
        for dep, info in DEPS.items():
            results.append(dict(dep=dep, level="INFO", type="VIGICRUE",
                msg=f"Dép.{dep} {info['nom']} — Vigicrue : données indisponibles"))
    return [r for r in results if r["level"] != "VERT"]


@st.cache_data(ttl=3600)
def load_forecast_dep(dep: str) -> dict:
    d = DEPS[dep]
    try:
        r = requests.get(FORECAST_URL, params={
            "latitude": d["lat"], "longitude": d["lon"],
            "daily": "precipitation_sum,weathercode",
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
            "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max",
            "forecast_days": 7, "timezone": "Europe/Paris",
        }, timeout=15)
        r.raise_for_status()
        daily = r.json().get("daily", {})
        return pd.DataFrame({
            "date":     daily.get("time", []),
            "pluie_mm": daily.get("precipitation_sum", []),
            "proba_%":  daily.get("precipitation_probability_max", []),
            "tmax":     daily.get("temperature_2m_max", []),
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
        return pd.DataFrame([{"mois": m, "pluie_mm": round(v, 1)}
                              for m, v in sorted(monthly.items())])
    except Exception:
        return pd.DataFrame()


def commune_alerts_from_snapshot(df: pd.DataFrame) -> list:
    """Alertes pluie par commune à partir des données snapshot mesurées."""
    alerts = []
    if df.empty:
        return alerts
    thresholds = [
        ("weather_max_24h_mm",   "24h",   30, 60,  "INONDATION"),
        ("weather_max_7d_mm",    "7 jours", 80, 150, "INONDATION"),
        ("weather_max_30d_mm",   "30j",   200, 350, "INONDATION"),
        ("weather_max_month_mm", "mois",  150, 250, "INONDATION"),
    ]
    for _, row in df.iterrows():
        commune = row.get("commune_name", "?")
        pk      = row.get("pk_km", "")
        pk_str  = f" PK {pk:.1f} km" if pd.notna(pk) else ""
        for col, label, seuil_orange, seuil_rouge, atype in thresholds:
            val = row.get(col)
            if val is None or not isinstance(val, (int, float)) or pd.isna(val):
                continue
            if val >= seuil_rouge:
                alerts.append(dict(dep="commune", date="", type=atype, level="ROUGE",
                    msg=f"{commune}{pk_str} — {label} : {val:.1f} mm ⚠️ seuil critique"))
            elif val >= seuil_orange:
                alerts.append(dict(dep="commune", date="", type=atype, level="ORANGE",
                    msg=f"{commune}{pk_str} — {label} : {val:.1f} mm"))
    return sorted(alerts, key=lambda x: (-LEVEL_RANK.get(x["level"], 0), x.get("msg","")))


def safe_df(records) -> pd.DataFrame:
    if isinstance(records, list) and records:
        try:
            return pd.DataFrame(records)
        except Exception:
            pass
    return pd.DataFrame()


def alert_card(a: dict):
    lvl   = a.get("level", "")
    atype = a.get("type", "")
    color = LEVEL_COLOR.get(lvl, "#6b7280")
    icon  = ALERT_CFG.get(atype, ("⚠️", ""))[0]
    st.markdown(
        f'<div style="padding:7px 14px;border-radius:6px;border-left:4px solid {color};'
        f'background:{color}12;margin-bottom:5px;font-size:13px">'
        f'{LEVEL_EMOJI.get(lvl,"")} {icon} <b>[{lvl}]</b> {a.get("msg","")}'
        f'</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="LGV SEA – Pluviométrie", page_icon="🌧", layout="wide")
st.title("🌧 LGV SEA – Pluviométrie & alertes météo")

if st.button("🔄 Rafraîchir"):
    st.cache_data.clear()
    st.rerun()

snapshot = load_snapshot()
if "_error" in snapshot:
    st.error(f"Erreur snapshot : {snapshot['_error']}")
    st.stop()

_sec       = snapshot.get("sectors")
sectors_df = safe_df(_sec.get("sectors", []) if isinstance(_sec, dict) else [])
for col in ["weather_max_24h_mm","weather_max_7d_mm","weather_max_30d_mm",
            "weather_max_month_mm","latitude","longitude","pk_km"]:
    if col in sectors_df.columns:
        sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")

# ── 1. VIGILANCE PLUIE PAR DÉPARTEMENT ─────────────────────────────────────
st.subheader("📡 Vigilance pluvio — prévisions 7 jours")
dep_cols = st.columns(len(DEPS))
for col_w, (dep, info) in zip(dep_cols, DEPS.items()):
    fc    = load_forecast_dep(dep)
    daily = fc.get("daily", {})
    rains = [v for v in daily.get("precipitation_sum", []) if v is not None]
    max_mm = max(rains) if rains else 0.0
    total  = sum(rains)
    lvl, color, emoji = rain_risk(max_mm)
    col_w.markdown(
        f'<div style="padding:10px 6px;border-radius:8px;border:2px solid {color};'
        f'text-align:center;background:{color}14">'
        f'<div style="font-size:22px">{emoji}</div>'
        f'<b style="font-size:13px">Dép. {dep}</b><br>'
        f'<span style="font-size:11px;color:#666">{info["nom"]}</span><br>'
        f'<b style="color:{color}">{lvl}</b><br>'
        f'<span style="font-size:12px">max {max_mm:.0f} mm/j · {total:.0f} mm total</span>'
        f'</div>', unsafe_allow_html=True)

# ── 2. ALERTES ──────────────────────────────────────────────────────────────
st.subheader("🚨 Alertes surveillance — 7 prochains jours")

met_alerts  = load_weather_alerts_all()
vc_alerts   = load_vigicrue()
comm_alerts = commune_alerts_from_snapshot(sectors_df)
active_met  = [a for a in met_alerts  if a["level"] in ("ROUGE","ORANGE","JAUNE")]
active_vc   = [a for a in vc_alerts   if a["level"] in ("ROUGE","ORANGE","JAUNE")]
all_active  = active_met + active_vc

if not all_active:
    st.success("✅ Aucune alerte active sur les 7 prochains jours.")
else:
    # Summary chips
    by_type: dict = defaultdict(list)
    for a in all_active:
        by_type[a["type"]].append(a)

    chips = ""
    for atype, alist in by_type.items():
        worst = max(alist, key=lambda x: LEVEL_RANK.get(x["level"],0))
        color = LEVEL_COLOR.get(worst["level"],"#6b7280")
        icon, label = ALERT_CFG.get(atype, ("⚠️",""))
        chips += (f'<span style="display:inline-block;margin:3px 4px;padding:3px 10px;'
                  f'border-radius:20px;background:{color};color:white;font-size:12px;font-weight:600">'
                  f'{icon} {label} ({len(alist)})</span>')
    st.markdown(chips, unsafe_allow_html=True)

    tab_labels = []
    tab_data   = []
    order = ["ORAGE","INONDATION","VIGICRUE","CANICULE","INCENDIE","VENT"]
    for atype in order:
        if atype in by_type:
            icon, label = ALERT_CFG[atype]
            tab_labels.append(f"{icon} {label}")
            tab_data.append(by_type[atype])

    # Onglet communes (données snapshot mesurées)
    if comm_alerts:
        tab_labels.append("📍 Communes")
        tab_data.append(comm_alerts)

    if tab_labels:
        tabs = st.tabs(tab_labels)
        for tab, alist in zip(tabs, tab_data):
            with tab:
                for a in alist:
                    alert_card(a)

st.caption("Alertes calculées via prévisions Open-Meteo + API Vigicrue · "
           "Vigilance officielle : [vigilance.meteofrance.fr](https://vigilance.meteofrance.fr/) · "
           "[vigicrues.gouv.fr](https://www.vigicrues.gouv.fr/)")

st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
communes = (sorted(sectors_df["commune_name"].dropna().unique())
            if "commune_name" in sectors_df.columns else [])
with st.sidebar:
    st.subheader("📍 Communes")
    selected_multi = st.multiselect("Comparer communes", communes,
                                     default=communes[:4] if len(communes) >= 4 else communes)
    selected_one   = st.selectbox("Commune principale", ["— Toutes —"] + list(communes))
    periode  = st.selectbox("📅 Période pluvio", ["24h","7 jours","30 jours","Mois courant"])
    rain_col = {"24h":"weather_max_24h_mm","7 jours":"weather_max_7d_mm",
                "30 jours":"weather_max_30d_mm","Mois courant":"weather_max_month_mm"}[periode]

# ── 3. COMPARAISON COMMUNES ─────────────────────────────────────────────────
if selected_multi and not sectors_df.empty and rain_col in sectors_df.columns:
    st.subheader("📊 Comparaison communes — pluie par secteur")
    df_cmp = (sectors_df[sectors_df["commune_name"].isin(selected_multi)]
              [["commune_name","pk_km",rain_col]].copy()
              .dropna(subset=[rain_col])
              .sort_values(rain_col, ascending=False))
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
                          margin=dict(t=20,b=20,l=20,r=20), xaxis=dict(tickangle=-30))
        st.plotly_chart(fig, use_container_width=True)

# ── 4. PRÉVISIONS 7J ────────────────────────────────────────────────────────
comm_df = (sectors_df if selected_one == "— Toutes —"
           else sectors_df[sectors_df["commune_name"] == selected_one])
map_df = (comm_df.dropna(subset=["latitude","longitude"])
          if "latitude" in sectors_df.columns and "longitude" in sectors_df.columns
          else pd.DataFrame())
lat_c = float(map_df["latitude"].mean())  if not map_df.empty else 46.2
lon_c = float(map_df["longitude"].mean()) if not map_df.empty else 0.2

label_loc = "LGV SEA" if selected_one == "— Toutes —" else selected_one
st.subheader(f"🔮 Prévisions 7 jours — {label_loc}")
fc_df = load_forecast_coord(lat_c, lon_c)
if not fc_df.empty:
    fc_df["pluie_mm"] = pd.to_numeric(fc_df["pluie_mm"], errors="coerce").fillna(0)
    fc_df["tmax"]     = pd.to_numeric(fc_df["tmax"],     errors="coerce").fillna(0)
    fc_df["color"]    = fc_df["pluie_mm"].apply(rain_color_mm)
    fig2 = go.Figure()
    fig2.add_bar(x=fc_df["date"], y=fc_df["pluie_mm"],
                 marker_color=fc_df["color"].tolist(),
                 text=fc_df["pluie_mm"].apply(lambda v: f"{v:.0f}"),
                 textposition="outside", name="Pluie (mm)")
    if "proba_%" in fc_df.columns:
        fig2.add_scatter(x=fc_df["date"], y=fc_df["proba_%"],
                         mode="lines+markers", name="Proba pluie %",
                         yaxis="y2", line=dict(color="#6366f1", dash="dot"),
                         marker=dict(size=5))
    if "tmax" in fc_df.columns:
        fig2.add_scatter(x=fc_df["date"], y=fc_df["tmax"],
                         mode="lines+markers", name="T° max (°C)",
                         yaxis="y3", line=dict(color="#f97316", width=2),
                         marker=dict(size=5, symbol="diamond"))
    fig2.update_layout(
        yaxis=dict(title="Pluie (mm)", side="left"),
        yaxis2=dict(title="Proba %",   overlaying="y", side="right",
                    range=[0, 110], showgrid=False),
        yaxis3=dict(title="T°C",       overlaying="y", side="right",
                    position=0.92,     showgrid=False, anchor="free"),
        legend=dict(orientation="h", y=1.12), height=290,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=30, b=20, l=20, r=80), xaxis=dict(tickangle=-20),
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Prévisions indisponibles.")

# ── 5. HISTORIQUE MENSUEL ───────────────────────────────────────────────────
st.subheader("📅 Historique mensuel — 12 mois")
monthly_df = load_monthly_rain(lat_c, lon_c)
if not monthly_df.empty:
    fig3 = px.bar(
        monthly_df, x="mois", y="pluie_mm", color="pluie_mm",
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

# ── 6. CARTE ────────────────────────────────────────────────────────────────
st.subheader("🗺 Carte — pluie par secteur")
if not map_df.empty and rain_col in map_df.columns:
    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=8 if selected_one == "— Toutes —" else 11,
        tiles="CartoDB positron", control_scale=True,
    )
    vmax = map_df[rain_col].max() if map_df[rain_col].max() > 0 else 1
    for _, row in map_df.dropna(subset=[rain_col]).iterrows():
        rv  = float(row[rain_col])
        col = rain_color_mm(rv)
        rad = max(5, min(20, 5 + (rv / vmax) * 15))
        folium.CircleMarker(
            [float(row["latitude"]), float(row["longitude"])],
            radius=rad, color=col, fill=True, fill_opacity=0.8, weight=1.5,
            tooltip=f"{row.get('commune_name','')} — {rv:.1f} mm ({periode})",
            popup=folium.Popup(
                f"<b>{row.get('commune_name','')} PK {row.get('pk_km','')} km</b><br>"
                f"Pluie {periode} : <b>{rv:.1f} mm</b>", max_width=230),
        ).add_to(m)
    for seg in (snapshot.get("lgv_lines") or []):
        if isinstance(seg, list):
            pts = [[p[0], p[1]] for p in seg if isinstance(p, (list,tuple)) and len(p) >= 2]
            if pts:
                folium.PolyLine(pts, color="#cc0000", weight=2.5, opacity=0.7).add_to(m)
    st_folium(m, use_container_width=True, height=450, returned_objects=[])
else:
    st.info("Pas de données de localisation.")

# ── 7. TABLEAU ──────────────────────────────────────────────────────────────
st.subheader("📋 Données par secteur")
show = [c for c in ["commune_name","pk_km","weather_max_24h_mm","weather_max_7d_mm",
                    "weather_max_30d_mm","weather_max_month_mm"] if c in comm_df.columns]
rename = {"commune_name":"Commune","pk_km":"PK (km)","weather_max_24h_mm":"24h (mm)",
          "weather_max_7d_mm":"7j (mm)","weather_max_30d_mm":"30j (mm)",
          "weather_max_month_mm":"Mois (mm)"}
disp = comm_df[show].rename(columns=rename)
rl = rename.get(rain_col, rain_col)
if rl in disp.columns:
    disp = disp.sort_values(rl, ascending=False, na_position="last")
if not disp.empty:
    st.dataframe(disp, use_container_width=True, hide_index=True, height=320)
