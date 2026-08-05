from __future__ import annotations

import requests
import pandas as pd
import streamlit as st

SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"

RISK_COLOR = {
    "FAIBLE": "#16a34a", "MODERE": "#ea580c",
    "ELEVE": "#dc2626", "CRITIQUE": "#7f1d1d", "INDETERMINE": "#6b7280",
}
RISK_RANK = {"FAIBLE": 1, "MODERE": 2, "ELEVE": 3, "CRITIQUE": 4}
RISK_EMOJI = {"FAIBLE": "🟢", "MODERE": "🟠", "ELEVE": "🔴", "CRITIQUE": "⛔", "INDETERMINE": "⚪"}


@st.cache_data(ttl=900)
def load_snapshot():
    try:
        r = requests.get(SNAPSHOT_URL, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def safe_df(records) -> pd.DataFrame:
    if isinstance(records, list) and records:
        try:
            return pd.DataFrame(records)
        except Exception:
            pass
    return pd.DataFrame()


st.set_page_config(page_title="LGV SEA – Pluvio", page_icon="🌧", layout="wide")
st.title("🌧 LGV SEA – Pluviométrie par commune")

snapshot = load_snapshot()

if "_error" in snapshot:
    st.error(f"Erreur chargement : {snapshot['_error']}")
    st.stop()

ts = snapshot.get("timestamp_utc", "")
if ts:
    st.caption(f"Données : {ts[:16].replace('T', ' ')} UTC")

_sec = snapshot.get("sectors")
sectors_df = safe_df(_sec.get("sectors", []) if isinstance(_sec, dict) else [])
commune_ranking = safe_df(snapshot.get("commune_ranking", []))

if sectors_df.empty:
    st.warning("Aucune donnée secteur dans le snapshot.")
    st.stop()

for col in ["weather_max_24h_mm", "weather_max_7d_mm", "weather_max_30d_mm",
            "weather_max_month_mm", "latitude", "longitude", "pk_km"]:
    if col in sectors_df.columns:
        sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Filtres")
    if st.button("🔄 Rafraîchir", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    communes = sorted(sectors_df["commune_name"].dropna().unique()) if "commune_name" in sectors_df.columns else []
    selected = st.selectbox("📍 Commune", ["— Toutes —"] + list(communes))

    periode = st.selectbox("📅 Période", ["24h", "7 jours", "30 jours", "Mois courant"])
    rain_col = {"24h": "weather_max_24h_mm", "7 jours": "weather_max_7d_mm",
                "30 jours": "weather_max_30d_mm", "Mois courant": "weather_max_month_mm"}[periode]

    risque_min = st.selectbox("⚠ Risque minimum", ["Tout", "FAIBLE", "MODERE", "ELEVE", "CRITIQUE"])

# ── Filtrage ─────────────────────────────────────────────────────────────
df = sectors_df.copy()
if selected != "— Toutes —":
    df = df[df["commune_name"] == selected]
if risque_min != "Tout" and "risk_level" in df.columns:
    min_rank = RISK_RANK.get(risque_min, 0)
    df = df[df["risk_level"].map(lambda x: RISK_RANK.get(str(x), 0)) >= min_rank]

# ── Vue commune ───────────────────────────────────────────────────────────
if selected != "— Toutes —":
    commune_row = {}
    if not commune_ranking.empty and "commune_name" in commune_ranking.columns:
        r = commune_ranking[commune_ranking["commune_name"] == selected]
        if not r.empty:
            commune_row = r.iloc[0].to_dict()

    risk_lvl = str(commune_row.get("commune_risk_level", "INDETERMINE"))
    color = RISK_COLOR.get(risk_lvl, "#6b7280")
    emoji = RISK_EMOJI.get(risk_lvl, "⚪")

    st.markdown(
        f'<div style="padding:14px;border-radius:8px;border-left:6px solid {color};'
        f'background:{color}18;margin-bottom:12px">'
        f'<b style="font-size:20px">{emoji} {selected}</b>'
        f'<span style="margin-left:16px;color:{color};font-weight:600">Risque : {risk_lvl}</span>'
        f'</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col_widget, label, col_name in [
        (c1, "☔ Max 24h", "weather_max_24h_mm"),
        (c2, "🌧 Max 7j", "weather_max_7d_mm"),
        (c3, "🌦 Max 30j", "weather_max_30d_mm"),
        (c4, "📅 Mois", "weather_max_month_mm"),
    ]:
        if col_name in df.columns:
            val = df[col_name].max()
            col_widget.metric(label, f"{val:.1f} mm" if pd.notna(val) else "—")
        else:
            col_widget.metric(label, "—")

    st.markdown("---")

# ── Tableau secteurs ─────────────────────────────────────────────────────
titre = f"Secteurs — {selected}" if selected != "— Toutes —" else "Tous les secteurs"
st.subheader(titre)

if df.empty:
    st.info("Aucun secteur pour ces filtres.")
else:
    show_cols = [c for c in ["commune_name", "pk_km", "risk_level", rain_col,
                              "ai_pred_risk_level"] if c in df.columns]
    rename = {
        "commune_name": "Commune", "pk_km": "PK (km)",
        "risk_level": "Risque", "ai_pred_risk_level": "Risque IA",
        "weather_max_24h_mm": "24h mm", "weather_max_7d_mm": "7j mm",
        "weather_max_30d_mm": "30j mm", "weather_max_month_mm": "Mois mm",
    }
    disp = df[show_cols].rename(columns=rename)
    rain_label = rename.get(rain_col, rain_col)
    if rain_label in disp.columns:
        disp = disp.sort_values(rain_label, ascending=False, na_position="last")
    st.dataframe(disp, use_container_width=True, hide_index=True, height=320)

# ── Classement communes ───────────────────────────────────────────────────
if selected == "— Toutes —" and not commune_ranking.empty:
    st.subheader("Classement communes")
    cr = commune_ranking.copy()
    if "commune_risk_level" in cr.columns:
        cr["_rank"] = cr["commune_risk_level"].map(lambda x: RISK_RANK.get(str(x), 0))
        cr = cr.sort_values("_rank", ascending=False).drop(columns=["_rank"])
    show = [c for c in ["commune_name", "departement_name", "commune_risk_level",
                         "commune_note", "sector_count", "critical", "high"] if c in cr.columns]
    rename_cr = {"commune_name": "Commune", "departement_name": "Département",
                 "commune_risk_level": "Risque", "commune_note": "Note",
                 "sector_count": "Secteurs", "critical": "Critique", "high": "Élevé"}
    st.dataframe(cr[show].rename(columns=rename_cr), use_container_width=True,
                 hide_index=True, height=380)
