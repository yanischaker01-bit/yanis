from __future__ import annotations

from datetime import datetime, timedelta, timezone

import folium
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

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


@st.cache_data(ttl=3600)
def load_monthly_rain(lat: float, lon: float) -> pd.DataFrame:
    end = datetime.now(timezone.utc).date()
    start = (end.replace(day=1) - timedelta(days=365)).replace(day=1)
    try:
        r = requests.get(ARCHIVE_URL, params={
            "latitude": lat, "longitude": lon,
            "start_date": str(start), "end_date": str(end),
            "daily": "precipitation_sum", "timezone": "Europe/Paris",
        }, timeout=20)
        r.raise_for_status()
        data = r.json()
        dates = data["daily"]["time"]
        rain = data["daily"]["precipitation_sum"]
        monthly: dict = {}
        for d, v in zip(dates, rain):
            if v is not None:
                monthly[d[:7]] = monthly.get(d[:7], 0.0) + v
        return pd.DataFrame([{"mois": m, "pluie_mm": round(v, 1)} for m, v in sorted(monthly.items())])
    except Exception:
        return pd.DataFrame()


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

# ── Carte des secteurs ───────────────────────────────────────────────────
st.subheader("🗺 Carte des secteurs")
map_df = df.dropna(subset=["latitude", "longitude"]) if {"latitude", "longitude"}.issubset(df.columns) else pd.DataFrame()
if map_df.empty:
    st.info("Pas de coordonnées disponibles pour la carte.")
else:
    lat_c = float(map_df["latitude"].mean())
    lon_c = float(map_df["longitude"].mean())
    m = folium.Map(location=[lat_c, lon_c],
                    zoom_start=8 if selected == "— Toutes —" else 12,
                    tiles="CartoDB positron", control_scale=True)
    for seg in (snapshot.get("lgv_lines") or []):
        if isinstance(seg, list):
            pts = [[p[0], p[1]] for p in seg if isinstance(p, (list, tuple)) and len(p) >= 2]
            if pts:
                folium.PolyLine(pts, color="#1d4ed8", weight=2.5, opacity=0.7, tooltip="Trace LGV SEA").add_to(m)
    for _, row in map_df.iterrows():
        risk_lvl_row = str(row.get("risk_level", "INDETERMINE"))
        ai_lvl_row = str(row.get("ai_pred_risk_level", "INDETERMINE"))
        color_map = RISK_COLOR.get(ai_lvl_row, "#6b7280")
        proba_row = float(row.get("ai_pred_probability", 0.0) or 0.0)
        popup = (
            f"<b>{row.get('sector_id', '')}</b> — {row.get('commune_name', '')} (PK {row.get('pk_km', '')} km)<br>"
            f"Risque mesuré : {risk_lvl_row}<br>"
            f"Prédiction IA : {ai_lvl_row} ({proba_row * 100:.0f} %)<br>"
            f"Sol dominant : {row.get('ai_dominant_pedology', '—')}"
        )
        folium.CircleMarker(
            [float(row["latitude"]), float(row["longitude"])],
            radius=6 + 6 * proba_row, color=color_map, fill=True, fill_opacity=0.8, weight=1.5,
            tooltip=f"{row.get('sector_id', '')} — IA {ai_lvl_row}",
            popup=folium.Popup(popup, max_width=280),
        ).add_to(m)
    st.caption("Couleur = niveau de risque prédit par l'IA (glissement). Taille = probabilité.")
    st_folium(m, use_container_width=True, height=440, returned_objects=[])

# ── Profil du risque le long de la ligne ─────────────────────────────────
st.subheader("📈 Profil du risque le long de la ligne (prédiction IA)")
profile_df = df.dropna(subset=["pk_km"]).sort_values("pk_km") if "pk_km" in df.columns else pd.DataFrame()
if profile_df.empty:
    st.info("Pas de profil PK disponible.")
else:
    bar_colors = profile_df.get("ai_pred_risk_level", pd.Series(dtype=str)).map(
        lambda x: RISK_COLOR.get(str(x), "#6b7280"))
    fig = go.Figure()
    fig.add_bar(
        x=profile_df["pk_km"], y=profile_df.get("ai_pred_probability", 0.0) * 100,
        marker_color=bar_colors, name="Probabilité IA glissement (%)",
        customdata=profile_df[["sector_id", "commune_name"]] if {"sector_id", "commune_name"}.issubset(profile_df.columns) else None,
        hovertemplate="PK %{x} km<br>Proba IA : %{y:.0f} %<extra></extra>",
    )
    if "score" in profile_df.columns:
        fig.add_scatter(
            x=profile_df["pk_km"], y=profile_df["score"] * 25,
            mode="lines+markers", name="Risque mesuré (score ×25)",
            line=dict(color="#0f172a", dash="dot"), marker=dict(size=5),
        )
    fig.add_hline(y=65, line_dash="dash", line_color="#dc2626", annotation_text="Seuil élevé")
    fig.add_hline(y=85, line_dash="dash", line_color="#7f1d1d", annotation_text="Seuil critique")
    fig.update_layout(
        xaxis_title="PK (km)", yaxis_title="Probabilité IA (%) / Score mesuré",
        height=320, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Historique pluviométrique 12 mois ─────────────────────────────────────
hist_label = selected if selected != "— Toutes —" else "LGV SEA (centroïde)"
st.subheader(f"📅 Historique pluviométrique 12 mois — {hist_label}")
if not map_df.empty:
    lat_h = float(map_df["latitude"].mean())
    lon_h = float(map_df["longitude"].mean())
    monthly_df = load_monthly_rain(lat_h, lon_h)
    if monthly_df.empty:
        st.info("Historique indisponible.")
    else:
        fig_hist = go.Figure()
        fig_hist.add_bar(
            x=monthly_df["mois"], y=monthly_df["pluie_mm"],
            marker_color="#3b82f6", text=monthly_df["pluie_mm"], textposition="outside",
        )
        fig_hist.update_layout(
            xaxis_title="Mois", yaxis_title="Pluie (mm)", height=260,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20, l=20, r=20), xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("Pas de localisation pour l'historique.")

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
