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

FACTOR_LABELS = {
    "pluie_24h": "Pluie 24h",
    "cumul_7j": "Cumul pluie 7j",
    "fragilite_sol": "Fragilité du sol",
    "interaction_pluie_sol": "Interaction pluie × sol",
    "signal_geotech": "Signal géotechnique",
    "signal_hydro": "Signal hydro",
    "signal_nappes": "Signal nappes",
    "signal_faible": "Signal faible",
}


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


def safe_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def risk_badge(level: str) -> str:
    color = RISK_COLOR.get(level, "#6b7280")
    emoji = RISK_EMOJI.get(level, "⚪")
    return (f'<span style="background:{color}20;color:{color};border:1px solid {color};'
            f'border-radius:12px;padding:1px 9px;font-size:12px;font-weight:600">{emoji} {level}</span>')


def factor_tags(factors) -> str:
    if not isinstance(factors, list) or not factors:
        return ""
    tags = "".join(
        f'<span style="background:#eef2ff;color:#3730a3;border-radius:10px;'
        f'padding:1px 8px;font-size:11px;margin-right:4px">{FACTOR_LABELS.get(f, f)}</span>'
        for f in factors
    )
    return tags


st.set_page_config(page_title="LGV SEA – Pluvio & glissements", page_icon="🌧", layout="wide")
st.title("🌧 LGV SEA – Pluviométrie & prédiction glissements")

snapshot = load_snapshot()

if "_error" in snapshot:
    st.error(f"Erreur chargement : {snapshot['_error']}")
    st.stop()

ts = snapshot.get("timestamp_utc", "")
if ts:
    st.caption(f"Données : {ts[:16].replace('T', ' ')} UTC")

sectors_payload = safe_dict(snapshot.get("sectors"))
sectors_df = safe_df(sectors_payload.get("sectors", []))
sector_summary = safe_dict(sectors_payload.get("summary"))
sector_alerts = sectors_payload.get("alerts", []) if isinstance(sectors_payload.get("alerts"), list) else []
commune_ranking = safe_df(snapshot.get("commune_ranking", []))

if sectors_df.empty:
    st.warning("Aucune donnée secteur dans le snapshot.")
    st.stop()

for col in ["weather_max_24h_mm", "weather_max_7d_mm", "weather_max_30d_mm",
            "weather_max_month_mm", "latitude", "longitude", "pk_km",
            "ai_pred_probability", "ai_confidence", "ai_soil_fragility"]:
    if col in sectors_df.columns:
        sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")

# ── Vue d'ensemble (lecture rapide) ─────────────────────────────────────────
st.subheader("Vue d'ensemble")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Secteurs sous surveillance", int(sector_summary.get("watch", 0)))
k2.metric("Critiques / Élevés (mesuré)",
          int(sector_summary.get("critical", 0)) + int(sector_summary.get("high", 0)))
k3.metric("Critiques / Élevés (IA)",
          int(sector_summary.get("ai_critical", 0)) + int(sector_summary.get("ai_high", 0)))
k4.metric("Probabilité IA moyenne", f"{float(sector_summary.get('ai_mean_probability', 0.0)) * 100:.0f} %")
k5.metric("Secteurs sol fragile", int(sector_summary.get("fragile_soil_sectors", 0)))

# ── Alertes secteurs (mesuré + prédiction IA) ───────────────────────────────
st.subheader("🚨 Alertes secteurs")
if not sector_alerts:
    st.success("Aucun secteur en alerte actuellement.")
else:
    for a in sector_alerts:
        level = a.get("level", "")
        color = RISK_COLOR.get(level, "#6b7280")
        kind = "🤖 Prédiction IA" if a.get("type") == "SECTEUR_IA" else "📏 Mesure"
        st.markdown(
            f'<div style="padding:6px 12px;border-radius:6px;border-left:4px solid {color};'
            f'background:{color}12;margin-bottom:5px;font-size:13px">'
            f'<b>[{level}]</b> {kind} — {a.get("message", "")}'
            f'</div>', unsafe_allow_html=True)

st.divider()

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
    show_ai_detail = st.checkbox("Colonnes IA détaillées dans le tableau", value=False)

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
    ai_lvl = str(commune_row.get("ai_commune_risk_level", "INDETERMINE"))
    color = RISK_COLOR.get(risk_lvl, "#6b7280")
    emoji = RISK_EMOJI.get(risk_lvl, "⚪")

    st.markdown(
        f'<div style="padding:14px;border-radius:8px;border-left:6px solid {color};'
        f'background:{color}18;margin-bottom:12px">'
        f'<b style="font-size:20px">{emoji} {selected}</b>'
        f'<span style="margin-left:16px;color:{color};font-weight:600">Risque mesuré : {risk_lvl}</span>'
        f'<span style="margin-left:16px">Prédiction IA glissement : {risk_badge(ai_lvl)}</span>'
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

    a1, a2, a3 = st.columns(3)
    a1.metric("Probabilité IA max", f"{float(commune_row.get('ai_max_probability', 0.0)) * 100:.0f} %")
    a2.metric("Fragilité sol moyenne", f"{float(commune_row.get('ai_avg_soil_fragility', 0.0)) * 100:.0f} %")
    a3.metric("Secteurs IA critiques/élevés",
              int(commune_row.get("ai_critical", 0)) + int(commune_row.get("ai_high", 0)))

    st.markdown("---")

    with st.expander(f"🔎 Détail prédiction IA par secteur — {selected}", expanded=False):
        if "ai_pred_probability" in df.columns:
            detail_df = df.sort_values("ai_pred_probability", ascending=False)
        else:
            detail_df = df
        for _, srow in detail_df.iterrows():
            proba = float(srow.get("ai_pred_probability", 0.0) or 0.0)
            conf = float(srow.get("ai_confidence", 0.0) or 0.0)
            st.markdown(
                f'**{srow.get("sector_id", "?")}** · PK {srow.get("pk_km", "—")} km '
                f'&nbsp; {risk_badge(str(srow.get("risk_level", "INDETERMINE")))} '
                f'&nbsp; IA {risk_badge(str(srow.get("ai_pred_risk_level", "INDETERMINE")))}',
                unsafe_allow_html=True)
            st.progress(min(max(proba, 0.0), 1.0), text=f"Probabilité IA glissement : {proba * 100:.0f} % (confiance {conf * 100:.0f} %)")
            st.markdown(
                f'Sol dominant : **{srow.get("ai_dominant_pedology", "—")}** '
                f'({srow.get("ai_dominant_soil_type", "—")}) &nbsp;·&nbsp; '
                f'Facteurs : {factor_tags(srow.get("ai_top_factors"))}',
                unsafe_allow_html=True)
            st.markdown("&nbsp;", unsafe_allow_html=True)

# ── Tableau secteurs ─────────────────────────────────────────────────────
titre = f"Secteurs — {selected}" if selected != "— Toutes —" else "Tous les secteurs"
st.subheader(titre)

if df.empty:
    st.info("Aucun secteur pour ces filtres.")
else:
    base_cols = ["commune_name", "pk_km", "risk_level", rain_col, "ai_pred_risk_level"]
    ai_cols = ["ai_pred_probability", "ai_confidence", "ai_soil_fragility", "ai_dominant_pedology"]
    show_cols = [c for c in base_cols + (ai_cols if show_ai_detail else []) if c in df.columns]
    rename = {
        "commune_name": "Commune", "pk_km": "PK (km)",
        "risk_level": "Risque", "ai_pred_risk_level": "Risque IA",
        "weather_max_24h_mm": "24h mm", "weather_max_7d_mm": "7j mm",
        "weather_max_30d_mm": "30j mm", "weather_max_month_mm": "Mois mm",
        "ai_pred_probability": "Proba IA", "ai_confidence": "Confiance IA",
        "ai_soil_fragility": "Fragilité sol", "ai_dominant_pedology": "Sol dominant",
    }
    disp = df[show_cols].rename(columns=rename)
    for pct_col in ["Proba IA", "Confiance IA", "Fragilité sol"]:
        if pct_col in disp.columns:
            disp[pct_col] = (disp[pct_col] * 100).round(0).astype("Int64").astype(str) + " %"
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
                         "commune_note", "sector_count", "critical", "high",
                         "ai_commune_risk_level", "ai_max_probability"] if c in cr.columns]
    rename_cr = {"commune_name": "Commune", "departement_name": "Département",
                 "commune_risk_level": "Risque", "commune_note": "Note",
                 "sector_count": "Secteurs", "critical": "Critique", "high": "Élevé",
                 "ai_commune_risk_level": "Risque IA", "ai_max_probability": "Proba IA max"}
    disp_cr = cr[show].rename(columns=rename_cr)
    if "Proba IA max" in disp_cr.columns:
        disp_cr["Proba IA max"] = (pd.to_numeric(disp_cr["Proba IA max"], errors="coerce") * 100).round(0).astype("Int64").astype(str) + " %"
    st.dataframe(disp_cr, use_container_width=True, hide_index=True, height=380)
