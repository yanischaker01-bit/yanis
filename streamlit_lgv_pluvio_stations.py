from __future__ import annotations

import io
import math
import os
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import folium
from folium import plugins
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

# =============================================================================
# CONFIGURATION
# =============================================================================
SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VIGICRUES_URLS = [
    "https://www.vigicrues.gouv.fr/services/1/InfoVigiCru.geojson/",
    "https://www.vigicrues.gouv.fr/services/1/InfoVigiCru.geojson",
]
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{start}"
FIRMS_SOURCES = ["VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"]
FIRMS_BBOX = "-0.7,44.75,1.0,47.5"
FIRMS_MAX_CHUNK_DAYS = 5
FIRMS_MAX_HISTORY_DAYS = 60

HEADERS = {
    "User-Agent": "Mozilla/5.0 LGV-SEA-Surveillance/4.0",
    "Accept": "application/json, application/geo+json;q=0.9, */*;q=0.1",
}
RIVERS = [
    "vienne", "clain", "charente", "boutonne", "seugne", "touvre",
    "dronne", "isle", "dordogne", "garonne", "thouet", "sevre",
    "indre", "cher", "creuse", "ciron", "jalles", "estey",
]
LEVEL_COLOR = {
    "CRITIQUE": "#991b1b", "ÉLEVÉ": "#dc2626", "RENFORCÉ": "#f97316",
    "VIGILANCE": "#eab308", "NORMAL": "#16a34a", "INDÉTERMINÉ": "#64748b",
}

# Seuils internes indicatifs. À calibrer avec les référentiels et retours d'expérience MESEA.
T_RAIN_24 = (10, 20, 40, 60)
T_RAIN_3D = (20, 40, 70, 100)
T_RAIN_7D = (30, 60, 100, 150)
T_ANTECEDENT_7D = (20, 40, 70, 100)

# =============================================================================
# OUTILS
# =============================================================================
def normalize(value: object) -> str:
    text = "" if value is None else str(value).lower()
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def num(value: object, default: float = 0.0) -> float:
    try:
        return default if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default


def get_json(url: str, params: dict | None = None, timeout: int = 35) -> dict:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=(8, timeout))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Réponse JSON inattendue")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(str(last_error))


def threshold_points(value: float, thresholds: tuple[float, float, float, float]) -> int:
    if value >= thresholds[3]: return 25
    if value >= thresholds[2]: return 18
    if value >= thresholds[1]: return 11
    if value >= thresholds[0]: return 5
    return 0


def risk_level(score: float) -> str:
    if score >= 80: return "CRITIQUE"
    if score >= 60: return "ÉLEVÉ"
    if score >= 40: return "RENFORCÉ"
    if score >= 20: return "VIGILANCE"
    return "NORMAL"


def style_plot(fig: go.Figure, height: int = 430, hovermode: str = "x unified") -> None:
    fig.update_layout(
        height=height, hovermode=hovermode, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=65, b=60, l=55, r=80), font=dict(family="Arial", color="#334155"),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#cbd5e1", automargin=True)
    fig.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", automargin=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


# =============================================================================
# DONNÉES MÉTÉO ET SNAPSHOT
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_snapshot() -> dict:
    return get_json(SNAPSHOT_URL, timeout=40)


@st.cache_data(ttl=1800, show_spinner=False)
def forecast_point(lat: float, lon: float) -> pd.DataFrame:
    try:
        daily = get_json(FORECAST_URL, {
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max,weather_code,wind_speed_10m_max",
            "forecast_days": 7, "timezone": "Europe/Paris",
        }).get("daily", {})
        frame = pd.DataFrame({
            "date": daily.get("time", []), "rain": daily.get("precipitation_sum", []),
            "prob": daily.get("precipitation_probability_max", []), "tmax": daily.get("temperature_2m_max", []),
            "wcode": daily.get("weather_code", []), "wind": daily.get("wind_speed_10m_max", []),
        })
        if not frame.empty:
            for column in ["rain", "prob", "tmax", "wcode", "wind"]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        return frame
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def antecedent_rain(lat: float, lon: float, days: int = 7) -> float:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    try:
        daily = get_json(ARCHIVE_URL, {
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "start_date": str(start), "end_date": str(end),
            "daily": "precipitation_sum", "timezone": "Europe/Paris",
        }, timeout=50).get("daily", {})
        return sum(num(v) for v in daily.get("precipitation_sum", []))
    except Exception:
        return float("nan")


@st.cache_data(ttl=21600, show_spinner=False)
def history_point(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    try:
        daily = get_json(ARCHIVE_URL, {
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "start_date": str(start), "end_date": str(end),
            "daily": "precipitation_sum", "timezone": "Europe/Paris",
        }, timeout=75).get("daily", {})
        frame = pd.DataFrame({"date": daily.get("time", []), "rain": daily.get("precipitation_sum", [])})
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"])
            frame["rain"] = pd.to_numeric(frame["rain"], errors="coerce").fillna(0)
        return frame
    except Exception:
        return pd.DataFrame()


def detect_asset_factor(group: pd.DataFrame) -> tuple[float, str]:
    """Utilise une fragilité de terrain si elle existe dans le snapshot, sinon facteur neutre."""
    candidates = ["fragility_score", "fragilite", "risk_score", "soil_risk", "argile_score"]
    for column in candidates:
        if column in group.columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if not values.empty:
                raw = float(values.max())
                normalized = raw / 100 if raw > 1 else raw
                return min(15.0, max(0.0, normalized * 15)), f"fragilité {normalized*100:.0f} %"
    return 0.0, "fragilité non renseignée"


def calculate_sector(commune: str, lat: float, lon: float, asset_points: float, asset_note: str) -> dict:
    forecast = forecast_point(lat, lon)
    if forecast.empty:
        return {"commune": commune, "ok": False, "latitude": lat, "longitude": lon}
    rains = forecast["rain"].tolist()
    rain_24 = max(rains) if rains else 0.0
    rain_3d = max((sum(rains[i:i+3]) for i in range(max(1, len(rains)-2))), default=0.0)
    rain_7d = sum(rains)
    wet_days = sum(1 for value in rains if value >= 2)
    max_wind = float(forecast["wind"].max())
    storm = bool((forecast["wcode"] >= 95).any())
    antecedent = antecedent_rain(lat, lon, 7)
    antecedent_for_score = 0.0 if math.isnan(antecedent) else antecedent

    score = (
        threshold_points(rain_24, T_RAIN_24)
        + threshold_points(rain_3d, T_RAIN_3D)
        + threshold_points(rain_7d, T_RAIN_7D)
        + threshold_points(antecedent_for_score, T_ANTECEDENT_7D)
        + min(10, max(0, wet_days - 2) * 2)
        + min(5, max_wind / 20)
        + (5 if storm else 0)
        + asset_points
    )
    score = min(100.0, score)
    peak_idx = forecast["rain"].idxmax()
    reasons = []
    if rain_3d >= 40: reasons.append(f"cumul glissant 3 j {rain_3d:.1f} mm")
    if rain_7d >= 60: reasons.append(f"cumul 7 j {rain_7d:.1f} mm")
    if antecedent_for_score >= 40: reasons.append(f"sol potentiellement humide, antécédent 7 j {antecedent_for_score:.1f} mm")
    if wet_days >= 4: reasons.append(f"pluie persistante sur {wet_days} jours")
    if rain_24 >= 30: reasons.append(f"pic journalier {rain_24:.1f} mm")
    if storm: reasons.append("signal orageux")
    if asset_points > 0: reasons.append(asset_note)
    return {
        "commune": commune, "ok": True, "latitude": lat, "longitude": lon,
        "score": round(score, 1), "level": risk_level(score),
        "rain_24": round(rain_24, 1), "rain_3d": round(rain_3d, 1),
        "rain_7d": round(rain_7d, 1), "antecedent_7d": round(antecedent, 1) if not math.isnan(antecedent) else None,
        "wet_days": wet_days, "wind": round(max_wind, 1), "storm": storm,
        "peak_date": str(forecast.loc[peak_idx, "date"]),
        "reason": "; ".join(reasons) if reasons else "aucun dépassement notable",
    }


@st.cache_data(ttl=1800, show_spinner=False)
def surveillance_all(sectors: pd.DataFrame) -> pd.DataFrame:
    required = {"commune_name", "latitude", "longitude"}
    if sectors.empty or not required.issubset(sectors.columns):
        return pd.DataFrame()
    jobs = []
    for commune, group in sectors.dropna(subset=list(required)).groupby("commune_name"):
        lat = float(pd.to_numeric(group["latitude"], errors="coerce").mean())
        lon = float(pd.to_numeric(group["longitude"], errors="coerce").mean())
        asset_points, asset_note = detect_asset_factor(group)
        jobs.append((str(commune), lat, lon, asset_points, asset_note))
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(calculate_sector, *job): job[0] for job in jobs}
        for future in as_completed(futures):
            try: results.append(future.result())
            except Exception: results.append({"commune": futures[future], "ok": False})
    frame = pd.DataFrame(results)
    if not frame.empty and "score" in frame:
        frame = frame.sort_values(["ok", "score"], ascending=[False, False])
    return frame


# =============================================================================
# VIGICRUES
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_vigicrues() -> tuple[list[dict], bool]:
    payload = None
    for url in VIGICRUES_URLS:
        try:
            candidate = get_json(url, timeout=50)
            if isinstance(candidate.get("features"), list):
                payload = candidate
                break
        except Exception:
            continue
    if payload is None: return [], False
    mapping = {0: "VERT", 1: "VERT", 2: "JAUNE", 3: "ORANGE", 4: "ROUGE"}
    output = []
    for feature in payload["features"]:
        props = feature.get("properties", {}) if isinstance(feature, dict) else {}
        name = str(props.get("NomEntVigiCru") or props.get("LbEntVigiCru") or props.get("lbentcru") or "").strip()
        if not name or not any(r in normalize(name) for r in RIVERS): continue
        raw = props.get("NivSituVigiCruEnt", props.get("NivInfViCr", props.get("NivVigiCru")))
        try: level = mapping.get(int(float(raw)))
        except (TypeError, ValueError): level = {"vert":"VERT","jaune":"JAUNE","orange":"ORANGE","rouge":"ROUGE"}.get(normalize(raw))
        if level: output.append({"name": name, "level": level})
    unique = {(item["name"], item["level"]): item for item in output}
    return list(unique.values()), True


# =============================================================================
# FIRMS AVEC FILTRE DE DATES
# =============================================================================
def get_firms_key() -> str | None:
    try:
        key = st.secrets.get("FIRMS_MAP_KEY")
        if key: return str(key).strip()
    except Exception: pass
    return os.getenv("FIRMS_MAP_KEY")


def date_chunks(start: date, end: date, size: int = FIRMS_MAX_CHUNK_DAYS) -> list[tuple[date, int]]:
    chunks = []
    cursor = start
    while cursor <= end:
        days = min(size, (end - cursor).days + 1)
        chunks.append((cursor, days))
        cursor += timedelta(days=days)
    return chunks


@st.cache_data(ttl=600, show_spinner=False)
def fetch_firms_chunk(key: str, source: str, start: date, days: int) -> tuple[pd.DataFrame, str | None]:
    url = FIRMS_URL.format(key=key, source=source, bbox=FIRMS_BBOX, days=days, start=start)
    try:
        response = requests.get(url, headers=HEADERS, timeout=(8, 40))
        response.raise_for_status()
        text = response.text.strip()
        if "invalid" in text[:300].lower(): return pd.DataFrame(), "invalid_key"
        if not text or "<html" in text[:300].lower(): return pd.DataFrame(), "unexpected"
        frame = pd.read_csv(io.StringIO(text))
        frame["source"] = source
        return frame, None
    except Exception: return pd.DataFrame(), "fetch_failed"


def load_firms_period(start: date, end: date) -> tuple[pd.DataFrame, str | None, int, int]:
    key = get_firms_key()
    if not key: return pd.DataFrame(), "missing_key", 0, 0
    tasks = [(source, chunk_start, days) for source in FIRMS_SOURCES for chunk_start, days in date_chunks(start, end)]
    frames, successes = [], 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_firms_chunk, key, *task): task for task in tasks}
        for future in as_completed(futures):
            frame, error = future.result()
            if error == "invalid_key": return pd.DataFrame(), error, successes, len(tasks)
            if error is None:
                successes += 1
                if not frame.empty: frames.append(frame)
    if not successes: return pd.DataFrame(), "fetch_failed", 0, len(tasks)
    if not frames: return pd.DataFrame(), None, successes, len(tasks)
    output = pd.concat(frames, ignore_index=True)
    subset = [c for c in ["latitude", "longitude", "acq_date", "acq_time", "satellite"] if c in output]
    if subset: output = output.drop_duplicates(subset=subset)
    if "acq_date" in output:
        dates = pd.to_datetime(output["acq_date"], errors="coerce").dt.date
        output = output[(dates >= start) & (dates <= end)]
    return output, None, successes, len(tasks)


def line_points(lines: object) -> list[tuple[float, float]]:
    if not isinstance(lines, list) or not lines or not isinstance(lines[0], list): return []
    return [(num(p.get("lat")), num(p.get("lon"))) for p in lines[0] if isinstance(p, dict) and p.get("lat") is not None and p.get("lon") is not None]


def distance_to_line_km(lat: float, lon: float, line: list[tuple[float, float]]) -> float:
    if len(line) < 2: return float("inf")
    best = float("inf")
    for (lat1, lon1), (lat2, lon2) in zip(line, line[1:]):
        mid = (lat1 + lat2) / 2
        kx, ky = 111.32 * math.cos(math.radians(mid)), 111.32
        x1, y1, x2, y2, xp, yp = lon1*kx, lat1*ky, lon2*kx, lat2*ky, lon*kx, lat*ky
        dx, dy = x2-x1, y2-y1
        den = dx*dx + dy*dy
        t = 0 if den == 0 else max(0, min(1, ((xp-x1)*dx + (yp-y1)*dy)/den))
        best = min(best, math.hypot(xp-(x1+t*dx), yp-(y1+t*dy)))
    return best


# =============================================================================
# INTERFACE
# =============================================================================
st.set_page_config(page_title="Surveillance LGV SEA", page_icon="🛤️", layout="wide")
st.title("🛤️ Surveillance préventive LGV SEA")
st.caption("Aide à la priorisation maintenance : pluie intense, cumul multi-jours, antécédent humide, persistance, vent, orage, crues et feux. Les seuils internes doivent être validés par l'ingénierie MESEA.")

left, right = st.columns([6, 1])
if right.button("🔄 Actualiser", use_container_width=True):
    st.cache_data.clear(); st.rerun()

try: snapshot = load_snapshot()
except Exception as exc:
    st.error(f"Snapshot LGV indisponible : {exc}"); st.stop()
raw = snapshot.get("sectors", {})
sectors = pd.DataFrame(raw.get("sectors", []) if isinstance(raw, dict) else [])
for col in ["latitude", "longitude", "pk_km"]:
    if col in sectors: sectors[col] = pd.to_numeric(sectors[col], errors="coerce")
communes = sorted(sectors["commune_name"].dropna().astype(str).unique()) if "commune_name" in sectors else []
coords = sectors.dropna(subset=["commune_name","latitude","longitude"]).groupby("commune_name")[["latitude","longitude"]].mean() if communes else pd.DataFrame()

with st.sidebar:
    st.header("Filtres opérationnels")
    minimum_level = st.selectbox("Niveau minimal affiché", ["NORMAL", "VIGILANCE", "RENFORCÉ", "ÉLEVÉ", "CRITIQUE"], index=1)
    selected_commune = st.selectbox("Secteur détaillé", communes if communes else ["Indisponible"])
    st.divider()
    st.subheader("Historique pluvio")
    hist_start = st.date_input("Début historique", date(2021,1,1), min_value=date(2021,1,1), max_value=date.today())
    hist_end = st.date_input("Fin historique", date.today()-timedelta(days=5), min_value=date(2021,1,1), max_value=date.today())
    hist_group = st.radio("Regroupement", ["Mensuel", "Annuel"], horizontal=True)
    st.divider()
    st.subheader("NASA FIRMS")
    firms_mode = st.radio("Période", ["24 h", "3 jours", "5 jours", "Personnalisée"])
    today = datetime.now(timezone.utc).date()
    if firms_mode == "Personnalisée":
        firms_start = st.date_input("Début FIRMS", today-timedelta(days=6), min_value=today-timedelta(days=FIRMS_MAX_HISTORY_DAYS), max_value=today)
        firms_end = st.date_input("Fin FIRMS", today, min_value=today-timedelta(days=FIRMS_MAX_HISTORY_DAYS), max_value=today)
    else:
        period_days = {"24 h":1,"3 jours":3,"5 jours":5}[firms_mode]
        firms_end, firms_start = today, today-timedelta(days=period_days-1)
    firms_distance_m = st.slider("Distance maximale à la LGV", 100, 5000, 500, 100)
    firms_confidence = st.multiselect("Confiance", ["l", "n", "h"], default=["n", "h"], format_func=lambda x: {"l":"Faible","n":"Nominale","h":"Élevée"}[x])

st.subheader("1. Priorisation des secteurs sur les 7 prochains jours")
with st.spinner("Analyse pluie + antécédent humide pour toutes les communes..."):
    watch = surveillance_all(sectors)
if watch.empty or not watch.get("ok", pd.Series(dtype=bool)).any():
    st.error("Analyse indisponible. Vérifie le snapshot et Open-Meteo.")
else:
    valid = watch[watch["ok"] == True].copy()
    levels = ["NORMAL","VIGILANCE","RENFORCÉ","ÉLEVÉ","CRITIQUE"]
    valid = valid[valid["level"].map(levels.index) >= levels.index(minimum_level)]
    counts = watch[watch["ok"] == True]["level"].value_counts()
    cards = st.columns(5)
    for card, level in zip(cards, reversed(levels)):
        card.metric(level, int(counts.get(level, 0)))
    if valid.empty:
        st.success("Aucun secteur au niveau sélectionné.")
    else:
        top = valid.head(30)
        fig = go.Figure()
        fig.add_bar(x=top["commune"], y=top["rain_7d"], name="Cumul prévu 7 j", marker_color="#93c5fd")
        fig.add_scatter(x=top["commune"], y=top["rain_3d"], name="Cumul glissant max 3 j", mode="markers+text",
                        marker=dict(color=[LEVEL_COLOR[v] for v in top["level"]], size=[8 + s/12 for s in top["score"]]),
                        text=[f"{v:.0f}" if s >= 40 else "" for v,s in zip(top["rain_3d"], top["score"])], textposition="top center",
                        customdata=top[["score","level","reason"]],
                        hovertemplate="%{x}<br>3 j: %{y:.1f} mm<br>Score: %{customdata[0]}<br>Niveau: %{customdata[1]}<br>%{customdata[2]}<extra></extra>")
        fig.update_layout(yaxis_title="Pluie (mm)", xaxis_tickangle=-45)
        style_plot(fig, 540, "closest")
        display = valid.rename(columns={
            "commune":"Secteur / commune", "level":"Niveau", "score":"Score /100", "rain_24":"Pic 24 h (mm)",
            "rain_3d":"Max 3 j (mm)", "rain_7d":"Cumul 7 j (mm)", "antecedent_7d":"Antécédent 7 j (mm)",
            "wet_days":"Jours humides", "wind":"Vent max (km/h)", "peak_date":"Date du pic", "reason":"Facteurs de vigilance",
        })
        st.dataframe(display[["Secteur / commune","Niveau","Score /100","Pic 24 h (mm)","Max 3 j (mm)","Cumul 7 j (mm)","Antécédent 7 j (mm)","Jours humides","Vent max (km/h)","Date du pic","Facteurs de vigilance"]], use_container_width=True, hide_index=True, height=440)

st.divider()
st.subheader(f"2. Fiche détaillée du secteur : {selected_commune}")
if selected_commune in coords.index:
    point = coords.loc[selected_commune]
    forecast = forecast_point(float(point.latitude), float(point.longitude))
    sector_row = watch[(watch["commune"] == selected_commune) & (watch["ok"] == True)]
    if not sector_row.empty:
        row = sector_row.iloc[0]
        cols = st.columns(6)
        cols[0].metric("Niveau", row.level)
        cols[1].metric("Score", f"{row.score:.0f}/100")
        cols[2].metric("Pic 24 h", f"{row.rain_24:.1f} mm")
        cols[3].metric("Max 3 jours", f"{row.rain_3d:.1f} mm")
        cols[4].metric("Cumul 7 jours", f"{row.rain_7d:.1f} mm")
        cols[5].metric("Antécédent 7 j", "N/D" if pd.isna(row.antecedent_7d) else f"{row.antecedent_7d:.1f} mm")
        st.info(f"Facteurs principaux : {row.reason}")
    if not forecast.empty:
        fig = go.Figure()
        fig.add_bar(x=forecast.date, y=forecast.rain, name="Pluie", marker_color=[rain if rain else "#93c5fd" for rain in ["#dc2626" if v>=40 else "#f97316" if v>=20 else "#2563eb" for v in forecast.rain]], text=[f"{v:.1f}" for v in forecast.rain], textposition="outside")
        fig.add_scatter(x=forecast.date, y=forecast.prob, name="Probabilité pluie", yaxis="y2", mode="lines+markers", line=dict(color="#6366f1", dash="dot"))
        fig.update_layout(yaxis=dict(title="Pluie (mm)"), yaxis2=dict(title="Probabilité (%)", overlaying="y", side="right", range=[0,100], showgrid=False))
        style_plot(fig)

st.divider()
st.subheader("3. Historique pluviométrique depuis 2021")
if hist_start > hist_end:
    st.error("La date de début doit précéder la date de fin.")
elif selected_commune in coords.index:
    point = coords.loc[selected_commune]
    with st.spinner("Chargement de l'historique..."):
        history = history_point(float(point.latitude), float(point.longitude), hist_start, hist_end)
    if history.empty:
        st.warning("Historique indisponible.")
    else:
        history["period"] = history.date.dt.to_period("M").astype(str) if hist_group == "Mensuel" else history.date.dt.year.astype(str)
        grouped = history.groupby("period", as_index=False).rain.sum()
        peak_day = history.loc[history.rain.idxmax()]
        colors = ["#dc2626" if value == grouped.rain.max() else "#2563eb" for value in grouped.rain]
        fig = go.Figure(go.Bar(x=grouped.period, y=grouped.rain, marker_color=colors, text=[f"{v:.0f}" for v in grouped.rain], textposition="outside"))
        fig.update_layout(yaxis_title="Cumul (mm)")
        style_plot(fig, 450)
        a,b,c = st.columns(3)
        a.metric("Cumul sélectionné", f"{history.rain.sum():.1f} mm")
        b.metric("Pic journalier", f"{peak_day.rain:.1f} mm")
        c.metric("Date du pic", peak_day.date.strftime("%d/%m/%Y"))

st.divider()
st.subheader("4. Vigicrues")
river_alerts, river_ok = load_vigicrues()
active = [a for a in river_alerts if a["level"] in {"JAUNE","ORANGE","ROUGE"}]
if not river_ok: st.warning("Vigicrues injoignable : statut crues non vérifié.")
elif not active: st.success("Vigicrues vérifié : aucune vigilance active sur les cours d'eau suivis.")
else:
    for alert in active: st.warning(f"{alert['name']} : vigilance {alert['level'].lower()}")

st.divider()
st.subheader("5. NASA FIRMS et carte opérationnelle")
if firms_start > firms_end:
    st.error("La date de début FIRMS doit précéder la date de fin.")
    firms_frame, firms_error, firms_ok, firms_total = pd.DataFrame(), "dates", 0, 0
else:
    with st.spinner(f"Recherche FIRMS du {firms_start:%d/%m/%Y} au {firms_end:%d/%m/%Y}..."):
        firms_frame, firms_error, firms_ok, firms_total = load_firms_period(firms_start, firms_end)
line = line_points(snapshot.get("lgv_lines", []))
filtered_fires = []
if not firms_frame.empty and line:
    for _, fire in firms_frame.iterrows():
        confidence = str(fire.get("confidence", "")).lower().strip()
        if firms_confidence and confidence not in firms_confidence: continue
        distance = distance_to_line_km(num(fire.get("latitude")), num(fire.get("longitude")), line)
        if distance * 1000 <= firms_distance_m:
            filtered_fires.append({**fire.to_dict(), "distance_m": round(distance*1000)})

center = [float(coords.loc[selected_commune].latitude), float(coords.loc[selected_commune].longitude)] if selected_commune in coords.index else [46.2, 0.2]
map_object = folium.Map(location=center, zoom_start=10, tiles=None, control_scale=True)
folium.TileLayer("CartoDB positron", name="Carte claire", show=False).add_to(map_object)
folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", name="Satellite", attr="Esri, Maxar, Earthstar Geographics, GIS User Community", show=True, max_zoom=22).add_to(map_object)
folium.TileLayer("https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}", name="Noms et limites", attr="Esri", overlay=True, show=True).add_to(map_object)
if line: folium.PolyLine(line, color="#facc15", weight=4, opacity=.95, tooltip="LGV SEA").add_to(map_object)
if not watch.empty:
    for _, item in watch[watch.get("ok", False) == True].iterrows():
        folium.CircleMarker([item.latitude,item.longitude], radius=5+item.score/20, color=LEVEL_COLOR[item.level], fill=True, fill_opacity=.85, tooltip=f"{item.commune} · {item.level} · score {item.score}/100 · pluie 7 j {item.rain_7d} mm").add_to(map_object)
for fire in filtered_fires:
    folium.CircleMarker([num(fire.get("latitude")),num(fire.get("longitude"))], radius=9, color="#7f1d1d", fill=True, fill_color="#dc2626", fill_opacity=.9, tooltip=f"FIRMS · {fire['distance_m']} m LGV · {fire.get('acq_date','')} · confiance {fire.get('confidence','')}").add_to(map_object)
folium.LayerControl(collapsed=False).add_to(map_object)
plugins.Fullscreen(position="topleft", title="Plein écran", title_cancel="Quitter").add_to(map_object)
st_folium(map_object, use_container_width=True, height=650, returned_objects=[])

if firms_error == "missing_key": st.info("FIRMS non activé : ajoute FIRMS_MAP_KEY dans les secrets Streamlit.")
elif firms_error == "invalid_key": st.error("Clé FIRMS invalide.")
elif firms_error: st.warning("FIRMS non vérifié pour cette période.")
else:
    st.caption(f"FIRMS : {firms_ok}/{firms_total} requêtes réussies · période {firms_start:%d/%m/%Y} au {firms_end:%d/%m/%Y} · rayon {firms_distance_m} m.")
    if filtered_fires:
        fire_table = pd.DataFrame(filtered_fires)
        columns = [c for c in ["acq_date","acq_time","satellite","source","confidence","frp","distance_m"] if c in fire_table]
        st.error(f"{len(filtered_fires)} détection(s) FIRMS dans le périmètre de surveillance.")
        st.dataframe(fire_table[columns], use_container_width=True, hide_index=True)
    else: st.success("Aucune détection FIRMS dans le périmètre et les filtres sélectionnés.")

with st.expander("Méthode de priorisation et actions maintenance"):
    st.markdown("""
- **Pluie 24 h** : risque de ruissellement intense, ravinement, mise en charge des fossés et buses.
- **Cumul glissant sur 3 jours** : signale une sollicitation prolongée des talus, remblais et dispositifs de drainage.
- **Cumul sur 7 jours** : repère les secteurs exposés à une saturation progressive.
- **Antécédent pluvieux sur 7 jours** : proxy d'humidité initiale du terrain avant la pluie prévue.
- **Persistance** : nombre de jours avec au moins 2 mm, utile même sans pic extrême.
- **Compléments** : vent, orage et fragilité du terrain si le snapshot contient un champ de fragilité.

**Actions suggérées pour un niveau renforcé ou supérieur** : vérifier fossés, cunettes, buses, descentes d'eau, exutoires, zones de concentration des écoulements, pieds et crêtes de talus, indices de ravinement, fontis, glissement, affaissement de plateforme, pollution du ballast et venues d'eau. La décision opérationnelle reste celle du gestionnaire et doit s'appuyer sur les référentiels internes, inspections et instrumentation disponibles.
""")

st.caption("Sources : Open-Meteo, Vigicrues, NASA FIRMS, Esri World Imagery et snapshot LGV SEA. Outil d'aide à la surveillance, non substitutif aux procédures de sécurité et maintenance.")
