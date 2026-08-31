from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import folium
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
STALE_MINUTES = 180

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

CHART_LAYOUT = dict(plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=20, b=20, l=20, r=20))


@st.cache_data(ttl=300, show_spinner=False)
def load_snapshot(refresh_token: int = 0):
    """Charge le dernier snapshot en limitant les caches GitHub/CDN.

    refresh_token change lors d'un rafraichissement manuel afin de forcer
    une nouvelle requete. Cela ne modifie pas la date du snapshot si le
    fichier source n'a pas ete regenere.
    """
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            cache_buster = int(datetime.now(timezone.utc).timestamp())
            r = requests.get(
                SNAPSHOT_URL,
                params={"v": cache_buster, "refresh": refresh_token},
                headers={
                    "Cache-Control": "no-cache, no-store, max-age=0",
                    "Pragma": "no-cache",
                    "User-Agent": "LGV-SEA-Monitoring/1.0",
                },
                timeout=25,
            )
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, dict):
                raise ValueError("Le snapshot JSON n'est pas un objet valide")
            return payload
        except Exception as e:
            last_err = e
            if attempt == 0:
                continue
    return {"_error": str(last_err) if last_err else "erreur inconnue"}


def safe_df(records) -> pd.DataFrame:
    if isinstance(records, list) and records:
        try:
            return pd.DataFrame(records)
        except Exception:
            pass
    return pd.DataFrame()


def safe_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def safe_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if f == f else default  # f == f is False for NaN


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


@st.cache_data(ttl=900)
def load_commune_rain_ometo(lat: float, lon: float, periode: str) -> float:
    """Cumul pluie via Open-Meteo.
    24h  → AROME Météo-France 1,3 km (orages locaux), sinon ERA5.
    7j+  → ERA5 seamless (couverture longue durée).
    """
    today = datetime.now(timezone.utc).date()
    if periode == "24h":
        past_days = 1
    elif periode == "7 jours":
        past_days = 7
    elif periode == "30 jours":
        past_days = 30
    else:
        past_days = today.day - 1
    if past_days <= 0:
        return 0.0

    base = {
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "daily": "precipitation_sum",
        "past_days": past_days, "forecast_days": 0,
        "timezone": "Europe/Paris",
    }
    models = ["meteofrance_arome_france", None] if periode == "24h" else [None]
    for model in models:
        try:
            params = dict(base)
            if model:
                params["models"] = model
            r = requests.get(FORECAST_URL, params=params, timeout=15)
            r.raise_for_status()
            vals = r.json()["daily"]["precipitation_sum"]
            if vals and any(v is not None for v in vals):
                return round(sum(v for v in vals if v is not None), 1)
        except Exception:
            continue
    return float("nan")


def risk_badge(level: str) -> str:
    color = RISK_COLOR.get(level, "#6b7280")
    emoji = RISK_EMOJI.get(level, "⚪")
    return f'<span class="risk-badge" style="background:{color}20;color:{color};border-color:{color}">{emoji} {level}</span>'


def fmt_pct(series: pd.Series) -> pd.Series:
    pct = pd.to_numeric(series, errors="coerce") * 100
    return pct.round(0).apply(lambda v: "—" if pd.isna(v) else f"{int(v)} %")


def factor_tags(factors) -> str:
    if not isinstance(factors, list) or not factors:
        return "—"
    return "".join(f'<span class="factor-tag">{FACTOR_LABELS.get(f, f)}</span>' for f in factors)


def humanize_alert_message(message: str, lookup: dict) -> str:
    if ":" not in message:
        return message
    sid, rest = message.split(":", 1)
    info = lookup.get(sid.strip())
    if not info:
        return message
    pk = info.get("pk_km")
    pk_label = f"PK {pk:.1f} km" if isinstance(pk, (int, float)) and pk == pk else "PK n/a"
    commune = info.get("commune_name") or "commune inconnue"
    return f"{pk_label} — {commune} ·{rest}"


def data_age_minutes(timestamp_utc: str) -> float | None:
    try:
        dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 60.0
        return max(0.0, age)
    except Exception:
        return None


def format_data_age(age_minutes: float | None) -> str:
    if age_minutes is None:
        return "age inconnu"
    if age_minutes < 60:
        return f"il y a {age_minutes:.0f} min"
    age_hours = age_minutes / 60.0
    if age_hours < 48:
        return f"il y a {age_hours:.1f} h"
    return f"il y a {age_hours / 24.0:.1f} jours"


def display_data_freshness(timestamp_utc: str, age_minutes: float | None) -> bool:
    """Affiche la fraicheur et retourne True uniquement si le snapshot est exploitable."""
    if not timestamp_utc or age_minutes is None:
        st.error("Date des donnees inconnue. Les alertes internes sont suspendues.")
        return False

    try:
        dt_utc = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        dt_local = dt_utc.astimezone(ZoneInfo("Europe/Paris"))
        local_label = dt_local.strftime("%d/%m/%Y a %H:%M")
        utc_label = dt_utc.astimezone(timezone.utc).strftime("%d/%m/%Y a %H:%M UTC")
    except Exception:
        local_label = timestamp_utc
        utc_label = timestamp_utc

    age_label = format_data_age(age_minutes)
    if age_minutes <= STALE_MINUTES:
        st.success(f"Donnees a jour : {local_label} heure locale ({utc_label}), {age_label}.")
        return True

    if age_minutes <= 24 * 60:
        st.warning(
            f"Actualisation retardee : derniere donnee le {local_label} heure locale "
            f"({utc_label}), {age_label}. Les resultats doivent etre verifies."
        )
        return False

    st.error(
        f"Donnees obsoletes : derniere actualisation le {local_label} heure locale "
        f"({utc_label}), {age_label}. Les indicateurs, probabilites et alertes internes "
        "issus du snapshot sont suspendus."
    )
    st.info(
        "Le bouton Rafraichir force une nouvelle lecture du fichier JSON. "
        "Si la date ne change pas, le processus qui genere streamlit_snapshot_latest.json "
        "doit etre relance ou repare."
    )
    return False


st.set_page_config(page_title="LGV SEA – Pluvio & glissements", page_icon="🌧", layout="wide")
st.markdown(
    """
    <style>
    .risk-badge { border-radius: 12px; padding: 1px 9px; font-size: 12px; font-weight: 600; border: 1px solid; }
    .factor-tag { background: #eef2ff; color: #3730a3; border-radius: 10px; padding: 1px 8px;
                  font-size: 11px; margin-right: 4px; display: inline-block; margin-bottom: 2px; }
    .alert-card { padding: 6px 12px; border-radius: 6px; border-left: 4px solid; margin-bottom: 5px; font-size: 13px; }
    .commune-banner { padding: 14px; border-radius: 8px; border-left: 6px solid; margin-bottom: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🌧 LGV SEA – Pluviométrie & prédiction glissements")

if "snapshot_refresh_token" not in st.session_state:
    st.session_state.snapshot_refresh_token = 0

snapshot = load_snapshot(st.session_state.snapshot_refresh_token)

if not isinstance(snapshot, dict) or "_error" in snapshot:
    err = snapshot.get("_error", "format inattendu") if isinstance(snapshot, dict) else "format inattendu"
    st.error(f"Erreur chargement des données : {err}")
    st.caption("Réessaie via le bouton Rafraîchir dans la barre latérale, ou reviens dans quelques minutes.")
    st.stop()

ts = snapshot.get("timestamp_utc", "")
age_min = data_age_minutes(ts) if ts else None
snapshot_fresh = display_data_freshness(ts, age_min)

sectors_payload = safe_dict(snapshot.get("sectors"))
sectors_df = safe_df(sectors_payload.get("sectors", []))
sector_summary = safe_dict(sectors_payload.get("summary"))
sector_alerts = sectors_payload.get("alerts", []) if isinstance(sectors_payload.get("alerts"), list) else []
commune_ranking = safe_df(snapshot.get("commune_ranking", []))
ai_model = safe_dict(sectors_payload.get("ai_model"))

if sectors_df.empty:
    st.warning("Aucune donnée secteur dans le snapshot.")
    st.stop()

for col in ["weather_max_24h_mm", "weather_max_7d_mm", "weather_max_30d_mm",
            "weather_max_month_mm", "latitude", "longitude", "pk_km", "score",
            "ai_pred_probability", "ai_confidence", "ai_soil_fragility"]:
    if col in sectors_df.columns:
        sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")

sector_lookup: dict = {}
if {"sector_id", "pk_km", "commune_name"}.issubset(sectors_df.columns):
    sector_lookup = sectors_df.set_index("sector_id")[["pk_km", "commune_name"]].to_dict("index")

# ── Vue d'ensemble (toujours visible) ────────────────────────────────────
st.subheader("Vue d'ensemble")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Secteurs sous surveillance", int(sector_summary.get("watch", 0)))
k2.metric("Critiques / Élevés (mesuré)",
          int(sector_summary.get("critical", 0)) + int(sector_summary.get("high", 0)))
k3.metric("Critiques / Élevés (IA)",
          int(sector_summary.get("ai_critical", 0)) + int(sector_summary.get("ai_high", 0)))
k4.metric("Probabilité IA moyenne", f"{safe_float(sector_summary.get('ai_mean_probability')) * 100:.0f} %")
k5.metric("Secteurs sol fragile", int(sector_summary.get("fragile_soil_sectors", 0)))

st.subheader("🚨 Alertes secteurs")
if not snapshot_fresh:
    st.warning(
        "Alertes internes masquees : le snapshot depasse le seuil de "
        f"{STALE_MINUTES / 60:.0f} h. Les vigilances officielles doivent etre consultees separement."
    )
elif not sector_alerts:
    st.success("Aucun secteur en alerte actuellement.")
else:
    for a in sector_alerts:
        level = a.get("level", "")
        color = RISK_COLOR.get(level, "#6b7280")
        kind = "🤖 Prédiction IA" if a.get("type") == "SECTEUR_IA" else "📏 Mesure"
        msg = humanize_alert_message(a.get("message", ""), sector_lookup)
        st.markdown(
            f'<div class="alert-card" style="border-left-color:{color};background:{color}12">'
            f'<b>[{level}]</b> {kind} — {msg}</div>',
            unsafe_allow_html=True)

st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Filtres")
    if st.button("🔄 Rafraîchir les données", width="stretch"):
        st.session_state.snapshot_refresh_token += 1
        st.cache_data.clear()
        st.rerun()

    communes = sorted(sectors_df["commune_name"].dropna().unique()) if "commune_name" in sectors_df.columns else []
    selected = st.selectbox("📍 Commune", ["— Toutes —"] + list(communes))

    periode = st.selectbox("📅 Période", ["24h", "7 jours", "30 jours", "Mois courant"])

    risque_min = st.selectbox("⚠ Risque minimum", ["Tout", "FAIBLE", "MODERE", "ELEVE", "CRITIQUE"])
    show_ai_detail = st.checkbox("Colonnes IA détaillées dans le tableau", value=False)

    st.divider()
    model_name = ai_model.get("name") or "modèle IA LGV SEA"
    model_version = ai_model.get("version")
    st.caption(f"Modèle : {model_name}" + (f" v{model_version}" if model_version else ""))
    st.caption("Sources : Open-Meteo (prévisions + archives), pédologie/géotechnique BRGM, RGA/MVT Géorisques.")
    st.caption("La prédiction IA est une aide à la priorisation, à confirmer par expertise terrain.")

# ── Filtrage ─────────────────────────────────────────────────────────────
df = sectors_df.copy()
if selected != "— Toutes —":
    df = df[df["commune_name"] == selected]
if risque_min != "Tout" and "risk_level" in df.columns:
    min_rank = RISK_RANK.get(risque_min, 0)
    df = df[df["risk_level"].map(lambda x: RISK_RANK.get(str(x), 0)) >= min_rank]

map_df = df.dropna(subset=["latitude", "longitude"]) if {"latitude", "longitude"}.issubset(df.columns) else pd.DataFrame()

tab_carte, tab_analyses, tab_hist, tab_secteurs, tab_communes = st.tabs(
    ["🗺 Carte", "📊 Analyses", "📅 Historique", "📋 Secteurs", "🏘 Communes"]
)

# ── Carte ────────────────────────────────────────────────────────────────
with tab_carte:
    if map_df.empty:
        st.info("Pas de coordonnées disponibles pour la carte.")
    else:
        try:
            lat_c = float(map_df["latitude"].mean())
            lon_c = float(map_df["longitude"].mean())
            m = folium.Map(location=[lat_c, lon_c],
                            zoom_start=8 if selected == "— Toutes —" else 12,
                            tiles="CartoDB positron", control_scale=True)
            for seg in (snapshot.get("lgv_lines") or []):
                if isinstance(seg, list):
                    pts = [[p[0], p[1]] for p in seg if isinstance(p, (list, tuple)) and len(p) >= 2]
                    if pts:
                        folium.PolyLine(pts, color="#1d4ed8", weight=2.5, opacity=0.7,
                                         tooltip="Trace LGV SEA").add_to(m)
            for row in map_df.itertuples(index=False):
                risk_lvl_row = str(getattr(row, "risk_level", "INDETERMINE"))
                ai_lvl_row = str(getattr(row, "ai_pred_risk_level", "INDETERMINE"))
                color_map = RISK_COLOR.get(ai_lvl_row, "#6b7280")
                proba_row = min(max(safe_float(getattr(row, "ai_pred_probability", 0.0)), 0.0), 1.0)
                popup = (
                    f"<b>{getattr(row, 'sector_id', '')}</b> — {getattr(row, 'commune_name', '')} "
                    f"(PK {getattr(row, 'pk_km', '')} km)<br>"
                    f"Risque mesuré : {risk_lvl_row}<br>"
                    f"Prédiction IA : {ai_lvl_row} ({proba_row * 100:.0f} %)<br>"
                    f"Sol dominant : {getattr(row, 'ai_dominant_pedology', '—')}"
                )
                folium.CircleMarker(
                    [safe_float(row.latitude), safe_float(row.longitude)],
                    radius=6 + 6 * proba_row, color=color_map, fill=True, fill_opacity=0.8, weight=1.5,
                    tooltip=f"{getattr(row, 'sector_id', '')} — IA {ai_lvl_row}",
                    popup=folium.Popup(popup, max_width=280),
                ).add_to(m)
            st.caption("Couleur = niveau de risque prédit par l'IA (glissement). Taille = probabilité.")
            st_folium(m, use_container_width=True, height=480, returned_objects=[])
        except Exception as e:
            st.warning(f"Carte indisponible pour le moment ({e}).")

# ── Analyses (profil PK, répartition, facteurs) ───────────────────────────
with tab_analyses:
    st.markdown("**Profil du risque le long de la ligne (prédiction IA)**")
    profile_df = df.dropna(subset=["pk_km"]).sort_values("pk_km") if "pk_km" in df.columns else pd.DataFrame()
    if profile_df.empty:
        st.info("Pas de profil PK disponible.")
    else:
        try:
            if "ai_pred_risk_level" in profile_df.columns:
                bar_colors = profile_df["ai_pred_risk_level"].map(lambda x: RISK_COLOR.get(str(x), "#6b7280"))
            else:
                bar_colors = pd.Series(["#6b7280"] * len(profile_df), index=profile_df.index)
            proba_pct = (profile_df["ai_pred_probability"].fillna(0.0) * 100
                         if "ai_pred_probability" in profile_df.columns else pd.Series(0.0, index=profile_df.index))
            fig = go.Figure()
            fig.add_bar(
                x=profile_df["pk_km"], y=proba_pct, marker_color=bar_colors,
                name="Probabilité IA glissement (%)",
                hovertemplate="PK %{x} km<br>Proba IA : %{y:.0f} %<extra></extra>",
            )
            if "score" in profile_df.columns:
                fig.add_scatter(
                    x=profile_df["pk_km"], y=profile_df["score"].fillna(0.0) * 25,
                    mode="lines+markers", name="Risque mesuré (score ×25)",
                    line=dict(color="#0f172a", dash="dot"), marker=dict(size=5),
                )
            fig.add_hline(y=65, line_dash="dash", line_color="#dc2626", annotation_text="Seuil élevé")
            fig.add_hline(y=85, line_dash="dash", line_color="#7f1d1d", annotation_text="Seuil critique")
            fig.update_layout(xaxis_title="PK (km)", yaxis_title="Probabilité IA (%) / Score mesuré",
                               height=320, legend=dict(orientation="h", y=1.12), **CHART_LAYOUT)
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"Profil indisponible ({e}).")

    st.markdown("**Répartition du risque — mesuré vs prédiction IA**")
    levels = ["FAIBLE", "MODERE", "ELEVE", "CRITIQUE"]
    if "risk_level" in df.columns or "ai_pred_risk_level" in df.columns:
        try:
            measured_counts = df["risk_level"].value_counts() if "risk_level" in df.columns else pd.Series(dtype=int)
            ai_counts = df["ai_pred_risk_level"].value_counts() if "ai_pred_risk_level" in df.columns else pd.Series(dtype=int)
            fig_dist = go.Figure()
            fig_dist.add_bar(x=levels, y=[int(measured_counts.get(lvl, 0)) for lvl in levels],
                              name="Mesuré", marker_color="#0f172a")
            fig_dist.add_bar(x=levels, y=[int(ai_counts.get(lvl, 0)) for lvl in levels],
                              name="Prédiction IA", marker_color="#3b82f6")
            fig_dist.update_layout(barmode="group", yaxis_title="Nombre de secteurs", height=280,
                                    legend=dict(orientation="h", y=1.15), **CHART_LAYOUT)
            st.plotly_chart(fig_dist, width="stretch")
        except Exception as e:
            st.warning(f"Répartition indisponible ({e}).")
    else:
        st.info("Pas de niveau de risque disponible.")

    st.markdown("**Facteurs de risque les plus fréquents**")
    if "ai_top_factors" in df.columns:
        try:
            factor_counts: dict = {}
            for factors in df["ai_top_factors"]:
                if isinstance(factors, list):
                    for f in factors:
                        if f == "signal_faible":
                            continue
                        factor_counts[f] = factor_counts.get(f, 0) + 1
            if not factor_counts:
                st.info("Aucun facteur de risque marquant sur ce filtre.")
            else:
                factors_df = pd.DataFrame(
                    [{"facteur": FACTOR_LABELS.get(k, k), "secteurs": v} for k, v in factor_counts.items()]
                ).sort_values("secteurs", ascending=True)
                fig_factors = go.Figure()
                fig_factors.add_bar(x=factors_df["secteurs"], y=factors_df["facteur"], orientation="h",
                                     marker_color="#7c3aed")
                fig_factors.update_layout(xaxis_title="Nombre de secteurs concernés", height=280, **CHART_LAYOUT)
                st.plotly_chart(fig_factors, width="stretch")
        except Exception as e:
            st.warning(f"Facteurs indisponibles ({e}).")
    else:
        st.info("Facteurs IA indisponibles dans ce snapshot.")

# ── Historique ───────────────────────────────────────────────────────────
with tab_hist:
    hist_label = selected if selected != "— Toutes —" else "LGV SEA (centroïde)"
    st.markdown(f"**Historique pluviométrique 12 mois — {hist_label}**")
    if map_df.empty:
        st.info("Pas de localisation pour l'historique.")
    else:
        try:
            lat_h = float(map_df["latitude"].mean())
            lon_h = float(map_df["longitude"].mean())
            monthly_df = load_monthly_rain(lat_h, lon_h)
            if monthly_df.empty:
                st.info("Historique indisponible (source externe injoignable pour le moment).")
            else:
                fig_hist = go.Figure()
                fig_hist.add_bar(x=monthly_df["mois"], y=monthly_df["pluie_mm"],
                                  marker_color="#3b82f6", text=monthly_df["pluie_mm"], textposition="outside")
                fig_hist.update_layout(xaxis_title="Mois", yaxis_title="Pluie (mm)", height=300,
                                        xaxis=dict(tickangle=-30), **CHART_LAYOUT)
                st.plotly_chart(fig_hist, width="stretch")
        except Exception as e:
            st.warning(f"Historique indisponible ({e}).")

# ── Secteurs (bandeau commune + détail IA + tableau) ──────────────────────
with tab_secteurs:
    ometo_rain: dict = {}
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
            f'<div class="commune-banner" style="border-left-color:{color};background:{color}18">'
            f'<b style="font-size:20px">{emoji} {selected}</b>'
            f'<span style="margin-left:16px;color:{color};font-weight:600">Risque mesuré : {risk_lvl}</span>'
            f'<span style="margin-left:16px">Prédiction IA glissement : {risk_badge(ai_lvl)}</span>'
            f'</div>', unsafe_allow_html=True)

        _om_lat = _om_lon = None
        _loc = map_df.dropna(subset=["latitude", "longitude"]) if not map_df.empty else pd.DataFrame()
        if not _loc.empty:
            _om_lat = round(float(_loc["latitude"].mean()), 4)
            _om_lon = round(float(_loc["longitude"].mean()), 4)

        if _om_lat is not None:
            for _p in ["24h", "7 jours", "30 jours", "Mois courant"]:
                ometo_rain[_p] = load_commune_rain_ometo(_om_lat, _om_lon, _p)

        c1, c2, c3, c4 = st.columns(4)
        for _cw, _label, _key in [
            (c1, "☔ Cumul 24h",       "24h"),
            (c2, "🌧 Cumul 7j",        "7 jours"),
            (c3, "🌦 Cumul 30j",       "30 jours"),
            (c4, "📅 Mois courant",    "Mois courant"),
        ]:
            _v = ometo_rain.get(_key, float("nan"))
            _cw.metric(_label, f"{_v:.1f} mm" if pd.notna(_v) else "—")
        st.caption("Pluie : Open-Meteo ERA5 (near real-time, lag ~6h)")

        a1, a2, a3 = st.columns(3)
        a1.metric("Probabilité IA max", f"{safe_float(commune_row.get('ai_max_probability')) * 100:.0f} %")
        a2.metric("Fragilité sol moyenne", f"{safe_float(commune_row.get('ai_avg_soil_fragility')) * 100:.0f} %")
        a3.metric("Secteurs IA critiques/élevés",
                  int(commune_row.get("ai_critical", 0)) + int(commune_row.get("ai_high", 0)))

        with st.expander(f"🔎 Détail prédiction IA par secteur — {selected}", expanded=False):
            detail_df = (df.sort_values("ai_pred_probability", ascending=False)
                         if "ai_pred_probability" in df.columns else df)
            for row in detail_df.itertuples(index=False):
                proba = min(max(safe_float(getattr(row, "ai_pred_probability", 0.0)), 0.0), 1.0)
                conf = safe_float(getattr(row, "ai_confidence", 0.0))
                st.markdown(
                    f'**{getattr(row, "sector_id", "?")}** · PK {getattr(row, "pk_km", "—")} km '
                    f'&nbsp; {risk_badge(str(getattr(row, "risk_level", "INDETERMINE")))} '
                    f'&nbsp; IA {risk_badge(str(getattr(row, "ai_pred_risk_level", "INDETERMINE")))}',
                    unsafe_allow_html=True)
                st.progress(proba, text=f"Probabilité IA glissement : {proba * 100:.0f} % (confiance {conf * 100:.0f} %)")
                st.markdown(
                    f'Sol dominant : **{getattr(row, "ai_dominant_pedology", "—")}** '
                    f'({getattr(row, "ai_dominant_soil_type", "—")}) &nbsp;·&nbsp; '
                    f'Facteurs : {factor_tags(getattr(row, "ai_top_factors", None))}',
                    unsafe_allow_html=True)
                st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("---")

    titre = f"Secteurs — {selected}" if selected != "— Toutes —" else "Tous les secteurs"
    st.markdown(f"**{titre}**")
    if df.empty:
        st.info("Aucun secteur pour ces filtres.")
    else:
        base_cols = ["commune_name", "pk_km", "risk_level", "ai_pred_risk_level"]
        ai_cols = ["ai_pred_probability", "ai_confidence", "ai_soil_fragility", "ai_dominant_pedology"]
        show_cols = [c for c in base_cols + (ai_cols if show_ai_detail else []) if c in df.columns]
        rename = {
            "commune_name": "Commune", "pk_km": "PK (km)",
            "risk_level": "Risque", "ai_pred_risk_level": "Risque IA",
            "ai_pred_probability": "Proba IA", "ai_confidence": "Confiance IA",
            "ai_soil_fragility": "Fragilité sol", "ai_dominant_pedology": "Sol dominant",
        }
        disp = df[show_cols].copy().rename(columns=rename)
        # Insérer colonne pluie Open-Meteo (commune sélectionnée uniquement)
        if selected != "— Toutes —" and ometo_rain:
            _rv = ometo_rain.get(periode, float("nan"))
            _pluvio_label = f"Pluie {periode}"
            disp.insert(2, _pluvio_label, f"{_rv:.1f} mm" if pd.notna(_rv) else "—")
        for pct_col in ["Proba IA", "Confiance IA", "Fragilité sol"]:
            if pct_col in disp.columns:
                disp[pct_col] = fmt_pct(disp[pct_col])
        if "Risque IA" in disp.columns:
            disp = disp.sort_values(
                "Risque IA",
                key=lambda s: s.map(lambda x: RISK_RANK.get(str(x), 0)),
                ascending=False, na_position="last")
        elif "Risque" in disp.columns:
            disp = disp.sort_values(
                "Risque",
                key=lambda s: s.map(lambda x: RISK_RANK.get(str(x), 0)),
                ascending=False, na_position="last")
        st.dataframe(disp, width="stretch", hide_index=True, height=360)

# ── Communes ────────────────────────────────────────────────────────────
with tab_communes:
    if selected != "— Toutes —":
        st.info("Sélectionne « — Toutes — » dans le filtre Commune pour voir le classement complet.")
    elif commune_ranking.empty:
        st.info("Classement communes indisponible dans ce snapshot.")
    else:
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
            disp_cr["Proba IA max"] = fmt_pct(disp_cr["Proba IA max"])
        st.markdown("**Classement communes**")
        st.dataframe(disp_cr, width="stretch", hide_index=True, height=380)

        if {"commune_name", "commune_note"}.issubset(cr.columns):
            st.markdown("**Top 10 communes les plus à risque**")
            try:
                top_cr = cr.dropna(subset=["commune_note"]).sort_values("commune_note", ascending=False).head(10)
                if not top_cr.empty:
                    fig_top = go.Figure()
                    fig_top.add_bar(
                        x=top_cr["commune_note"], y=top_cr["commune_name"], orientation="h",
                        marker_color=top_cr.get("commune_risk_level", pd.Series(dtype=str)).map(
                            lambda x: RISK_COLOR.get(str(x), "#6b7280")),
                        text=top_cr["commune_note"], textposition="outside",
                    )
                    fig_top.update_layout(xaxis_title="Note de risque (/100)", yaxis=dict(autorange="reversed"),
                                           height=340, margin=dict(t=20, b=20, l=20, r=40),
                                           plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig_top, width="stretch")
            except Exception as e:
                st.warning(f"Graphe top communes indisponible ({e}).")

with st.expander("ℹ️ À propos de ce tableau de bord", expanded=False):
    st.markdown(
        f"- **Modèle IA** : {ai_model.get('name', 'n/a')} "
        f"(v{ai_model.get('version', '?')}) — {ai_model.get('description', '')}\n"
        "- **Sources** : Open-Meteo (prévisions + archives), pédologie/lithologie BRGM, "
        "retrait-gonflement des argiles (RGA) et mouvements de terrain (MVT) Géorisques.\n"
        "- **Limites** : la prédiction IA est un outil d'aide à la priorisation basé sur pluie + "
        "fragilité des sols ; elle ne remplace pas une expertise géotechnique de terrain.\n"
        f"- **Fraîcheur** : rafraîchi automatiquement toutes les heures ; alerte si les données "
        f"dépassent {STALE_MINUTES / 60:.0f} h."
    )
