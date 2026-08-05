from __future__ import annotations

from typing import Dict

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

# ── Config ──────────────────────────────────────────────────────────────
SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"

RISK_COLOR = {
    "FAIBLE":       "#16a34a",
    "MODERE":       "#ea580c",
    "ELEVE":        "#dc2626",
    "CRITIQUE":     "#7f1d1d",
    "INDETERMINE":  "#6b7280",
}
RISK_EMOJI = {
    "FAIBLE": "🟢", "MODERE": "🟠", "ELEVE": "🔴",
    "CRITIQUE": "⛔", "INDETERMINE": "⚪",
}

# ── Chargement snapshot ──────────────────────────────────────────────────
@st.cache_data(ttl=900)
def load_snapshot() -> Dict:
    try:
        r = requests.get(SNAPSHOT_URL, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Impossible de charger le snapshot : {e}")
        return {}


def safe_df(records) -> pd.DataFrame:
    if isinstance(records, list) and records:
        try:
            return pd.DataFrame(records)
        except Exception:
            pass
    return pd.DataFrame()


# ── App ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LGV SEA – Pluvio communes",
    page_icon="🌧",
    layout="wide",
)

st.title("🌧 LGV SEA – Pluviométrie par commune")

snapshot = load_snapshot()
if not snapshot:
    st.stop()

ts = snapshot.get("timestamp_utc", "")
if ts:
    st.caption(f"Données du {ts[:16].replace('T', ' ')} UTC")

# ── Données ─────────────────────────────────────────────────────────────
sectors_raw = (snapshot.get("sectors") or {}).get("sectors", [])
sectors_df = safe_df(sectors_raw)

commune_ranking = safe_df(snapshot.get("commune_ranking", []))
weather_df = safe_df(snapshot.get("weather", []))

if sectors_df.empty:
    st.warning("Pas de données secteurs dans le snapshot.")
    st.stop()

# Colonnes numériques
for col in ["weather_max_24h_mm", "weather_max_7d_mm", "weather_max_30d_mm",
            "weather_max_month_mm", "latitude", "longitude", "pk_km"]:
    if col in sectors_df.columns:
        sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")

# ── Sidebar : sélection commune ──────────────────────────────────────────
with st.sidebar:
    st.subheader("Filtres")

    if st.button("🔄 Rafraîchir", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    communes = sorted(sectors_df["commune_name"].dropna().unique().tolist()) if "commune_name" in sectors_df.columns else []
    selected = st.selectbox("📍 Commune", ["— Toutes —"] + communes)

    periode = st.selectbox("📅 Période pluvio", ["24h", "7 jours", "30 jours", "Mois courant"])
    rain_col = {
        "24h": "weather_max_24h_mm",
        "7 jours": "weather_max_7d_mm",
        "30 jours": "weather_max_30d_mm",
        "Mois courant": "weather_max_month_mm",
    }[periode]

    risque_min = st.selectbox("⚠ Risque minimum", ["Tout", "FAIBLE", "MODERE", "ELEVE", "CRITIQUE"], index=0)

# ── Filtrage ─────────────────────────────────────────────────────────────
df = sectors_df.copy()

if selected != "— Toutes —":
    df = df[df["commune_name"] == selected]

RISK_RANK = {"FAIBLE": 1, "MODERE": 2, "ELEVE": 3, "CRITIQUE": 4}
if risque_min != "Tout" and "risk_level" in df.columns:
    min_rank = RISK_RANK.get(risque_min, 0)
    df = df[df["risk_level"].map(lambda x: RISK_RANK.get(str(x), 0)) >= min_rank]

# ── Vue commune sélectionnée ─────────────────────────────────────────────
if selected != "— Toutes —":
    # Métriques de la commune
    commune_data = {}
    if not commune_ranking.empty and "commune_name" in commune_ranking.columns:
        row = commune_ranking[commune_ranking["commune_name"] == selected]
        if not row.empty:
            commune_data = row.iloc[0].to_dict()

    risk_lvl = commune_data.get("commune_risk_level", df["risk_level"].max() if "risk_level" in df.columns else "INDETERMINE")
    emoji = RISK_EMOJI.get(str(risk_lvl), "⚪")
    color = RISK_COLOR.get(str(risk_lvl), "#6b7280")

    st.markdown(
        f"""<div style="padding:16px;border-radius:10px;border-left:6px solid {color};
        background:{color}18;margin-bottom:16px">
        <span style="font-size:22px;font-weight:700">{emoji} {selected}</span>
        <span style="margin-left:16px;font-size:16px;color:{color};font-weight:600">
        Risque : {risk_lvl}</span></div>""",
        unsafe_allow_html=True,
    )

    # Pluvio agrégée
    c1, c2, c3, c4 = st.columns(4)
    def rain_metric(col_widget, label, col_name):
        if col_name in df.columns:
            val = df[col_name].max()
            col_widget.metric(label, f"{val:.1f} mm" if pd.notna(val) else "—")
        else:
            col_widget.metric(label, "—")

    rain_metric(c1, "☔ Max 24h", "weather_max_24h_mm")
    rain_metric(c2, "🌧 Max 7 jours", "weather_max_7d_mm")
    rain_metric(c3, "🌦 Max 30 jours", "weather_max_30d_mm")
    rain_metric(c4, "📅 Mois courant", "weather_max_month_mm")

    st.markdown("---")

# ── Tableau des secteurs ─────────────────────────────────────────────────
st.subheader("Secteurs" + (f" — {selected}" if selected != "— Toutes —" else " (toutes communes)"))

if df.empty:
    st.info("Aucun secteur ne correspond aux filtres.")
else:
    show_cols = [c for c in ["commune_name", "pk_km", "risk_level", rain_col,
                              "ai_pred_risk_level", "weather_class"] if c in df.columns]
    rename = {
        "commune_name": "Commune", "pk_km": "PK (km)",
        "risk_level": "Risque", "weather_class": "Météo",
        "ai_pred_risk_level": "Risque IA",
        "weather_max_24h_mm": "24h (mm)", "weather_max_7d_mm": "7j (mm)",
        "weather_max_30d_mm": "30j (mm)", "weather_max_month_mm": "Mois (mm)",
    }
    display = df[show_cols].rename(columns=rename).sort_values(
        rename.get(rain_col, rain_col), ascending=False, na_position="last"
    ) if rename.get(rain_col, rain_col) in df[show_cols].rename(columns=rename).columns else df[show_cols].rename(columns=rename)

    st.dataframe(display, use_container_width=True, hide_index=True, height=350)

# ── Carte ────────────────────────────────────────────────────────────────
st.subheader("Carte")

map_df = df.dropna(subset=["latitude", "longitude"])
if not map_df.empty:
    center_lat = map_df["latitude"].mean()
    center_lon = map_df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10 if selected != "— Toutes —" else 8,
                   tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                   attr="Google", control_scale=True)

    for _, row in map_df.iterrows():
        lvl = str(row.get("risk_level", "INDETERMINE"))
        color = RISK_COLOR.get(lvl, "#6b7280")
        rain_val = row.get(rain_col, 0) or 0
        popup_html = (
            f"<b>{row.get('commune_name','')}</b> — PK {row.get('pk_km','')} km<br>"
            f"Risque : <b style='color:{color}'>{lvl}</b><br>"
            f"Pluie {periode} : <b>{rain_val:.1f} mm</b>"
        )
        folium.CircleMarker(
            [float(row["latitude"]), float(row["longitude"])],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.85,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{row.get('commune_name','')} | {lvl} | {rain_val:.1f} mm",
        ).add_to(m)

    # Ligne LGV
    lgv_lines = snapshot.get("lgv_lines", [])
    if isinstance(lgv_lines, list) and lgv_lines:
        for segment in lgv_lines:
            if isinstance(segment, list) and len(segment) >= 2:
                folium.PolyLine(
                    [[p[0], p[1]] for p in segment if isinstance(p, (list, tuple)) and len(p) >= 2],
                    color="#cc0000", weight=2, opacity=0.6,
                ).add_to(m)

    st_folium(m, use_container_width=True, height=420, returned_objects=[])
else:
    st.info("Pas de coordonnées disponibles pour afficher la carte.")

# ── Résumé communes ──────────────────────────────────────────────────────
if selected == "— Toutes —" and not commune_ranking.empty:
    st.subheader("Classement communes par risque")
    cr = commune_ranking.copy()
    show = [c for c in ["commune_name", "departement_name", "commune_risk_level",
                         "commune_note", "sector_count", "critical", "high"] if c in cr.columns]
    rename_cr = {
        "commune_name": "Commune", "departement_name": "Département",
        "commune_risk_level": "Risque", "commune_note": "Note",
        "sector_count": "Secteurs", "critical": "Critique", "high": "Élevé",
    }
    if "commune_risk_level" in cr.columns:
        cr["_rank"] = cr["commune_risk_level"].map(lambda x: RISK_RANK.get(str(x), 0))
        cr = cr.sort_values("_rank", ascending=False)
    st.dataframe(cr[show].rename(columns=rename_cr), use_container_width=True, hide_index=True, height=400)
