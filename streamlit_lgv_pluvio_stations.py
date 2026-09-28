from __future__ import annotations

import io
import math
import os
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import folium
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from streamlit_folium import st_folium
from urllib3.util.retry import Retry

# =============================================================================
# CONFIGURATION
# =============================================================================
SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{day_range}/{date}"
FIRMS_SOURCES = ["VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"]
FIRMS_BBOX = "-0.7,44.75,1.0,47.5"
FIRMS_RADIUS_KM = 0.5

DEPS = {
    "37": {"nom": "Indre-et-Loire", "lat": 47.38, "lon": 0.69},
    "86": {"nom": "Vienne", "lat": 46.58, "lon": 0.34},
    "79": {"nom": "Deux-Sèvres", "lat": 46.32, "lon": -0.46},
    "16": {"nom": "Charente", "lat": 45.65, "lon": 0.16},
    "17": {"nom": "Charente-Maritime", "lat": 45.75, "lon": -0.63},
    "33": {"nom": "Gironde", "lat": 44.84, "lon": -0.58},
}

# Zones à forte densité de glissements issues du dossier MESEA.
# Les zones possédant des PK locaux ambigus (FN/MS/FS) ne sont pas injectées ici,
# afin d'éviter de les rattacher à tort au référentiel longitudinal principal.
GLISSEMENT_RISK_ZONES = [
    {"ouvrage": "RBT 0973", "pk_debut": 97.20, "pk_fin": 97.50, "poids": 420.0},
    {"ouvrage": "RBT 1042", "pk_debut": 103.78, "pk_fin": 104.05, "poids": 486.5},
    {"ouvrage": "RBT 1056", "pk_debut": 105.20, "pk_fin": 105.70, "poids": 1095.7},
    {"ouvrage": "RBT 1065", "pk_debut": 106.00, "pk_fin": 106.90, "poids": 1799.0},
    {"ouvrage": "RBT 1094", "pk_debut": 109.00, "pk_fin": 109.25, "poids": 534.3},
    {"ouvrage": "RBT 1204", "pk_debut": 119.30, "pk_fin": 119.90, "poids": 1366.0},
    {"ouvrage": "RBT 1204", "pk_debut": 120.85, "pk_fin": 121.05, "poids": 1291.4},
]

FORECAST_MODELS = {
    "Météo-France": "meteofrance_seamless",
    "ECMWF IFS": "ecmwf_ifs025",
    "ICON": "icon_seamless",
    "GFS": "gfs_seamless",
}
SHORT_RANGE_MODELS = {"Météo-France", "ECMWF IFS", "ICON", "GFS"}
LONG_RANGE_MODELS = {"ECMWF IFS", "ICON", "GFS"}

LEVEL_COLOR = {"ROUGE": "#dc2626", "ORANGE": "#ea580c", "JAUNE": "#eab308", "VERT": "#16a34a"}

RETRY = Retry(
    total=3, connect=3, read=3, status=3, backoff_factor=0.7,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]), respect_retry_after_header=True,
)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MESEA-LGV-Pluviometrie/2.1", "Accept": "application/json"})
SESSION.mount("https://", HTTPAdapter(max_retries=RETRY))


def get_json(url: str, params: dict | None = None, timeout=(5, 25)) -> dict:
    response = SESSION.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def safe_df(records) -> pd.DataFrame:
    try:
        return pd.DataFrame(records) if isinstance(records, list) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def rain_color_mm(value) -> str:
    if pd.isna(value): return "#9ca3af"
    if value >= 60: return "#dc2626"
    if value >= 30: return "#ea580c"
    if value >= 10: return "#3b82f6"
    return "#93c5fd"


# =============================================================================
# SNAPSHOT ET CLASSEMENT GLISSEMENTS
# =============================================================================
@st.cache_data(ttl=900)
def load_snapshot() -> dict:
    return get_json(SNAPSHOT_URL)


def get_default_glissement_communes(sectors: pd.DataFrame, max_communes=6):
    required = {"commune_name", "pk_km"}
    if sectors.empty or not required.issubset(sectors.columns):
        return [], pd.DataFrame()

    work = sectors[["commune_name", "pk_km"]].copy()
    work["pk_km"] = pd.to_numeric(work["pk_km"], errors="coerce")
    work = work.dropna(subset=["commune_name", "pk_km"])
    if work.empty:
        return [], pd.DataFrame()

    scores = defaultdict(float)
    ouvrages = defaultdict(set)
    min_dist = defaultdict(lambda: float("inf"))

    for zone in GLISSEMENT_RISK_ZONES:
        start, end = zone["pk_debut"], zone["pk_fin"]
        center = (start + end) / 2
        best = None
        for commune, group in work.groupby("commune_name"):
            pks = group["pk_km"].to_numpy(float)
            dist_interval = min(0.0 if start <= pk <= end else min(abs(pk-start), abs(pk-end)) for pk in pks)
            dist_center = min(abs(pk-center) for pk in pks)
            candidate = (dist_interval, dist_center, str(commune))
            if best is None or candidate < best:
                best = candidate
        if best:
            dist, _, commune = best
            scores[commune] += zone["poids"] / (1.0 + dist)
            ouvrages[commune].add(zone["ouvrage"])
            min_dist[commune] = min(min_dist[commune], dist)

    ranking = pd.DataFrame([
        {"Commune": c, "Score glissements": round(score, 1),
         "Nombre de zones": len(ouvrages[c]), "Ouvrages": ", ".join(sorted(ouvrages[c])),
         "Distance minimale (km)": round(min_dist[c], 2)}
        for c, score in scores.items()
    ])
    if ranking.empty:
        return [], ranking
    ranking = ranking.sort_values(["Score glissements", "Nombre de zones", "Commune"],
                                  ascending=[False, False, True]).reset_index(drop=True)
    return ranking.head(max_communes)["Commune"].tolist(), ranking


# =============================================================================
# PREVISIONS MULTI-MODELES
# =============================================================================
@st.cache_data(ttl=1800)
def fetch_forecast_model(lat: float, lon: float, model_code: str) -> dict:
    return get_json(FORECAST_URL, {
        "latitude": round(lat, 4), "longitude": round(lon, 4), "models": model_code,
        "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max,wind_speed_10m_max,weather_code",
        "forecast_days": 7, "timezone": "Europe/Paris",
    })


def model_to_df(payload: dict, label: str) -> pd.DataFrame:
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    if not dates: return pd.DataFrame()
    n = len(dates)
    def vals(key):
        v = daily.get(key, [])
        return v if len(v) == n else [None] * n
    df = pd.DataFrame({
        "date": dates, "pluie_mm": vals("precipitation_sum"),
        "proba_%": vals("precipitation_probability_max"),
        "tmax": vals("temperature_2m_max"), "vent_max": vals("wind_speed_10m_max"),
        "weather_code": vals("weather_code"), "modele": label,
    })
    for col in ["pluie_mm", "proba_%", "tmax", "vent_max", "weather_code"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_forecast_coord(lat: float, lon: float):
    frames, failures = [], []
    for label, code in FORECAST_MODELS.items():
        try:
            df = model_to_df(fetch_forecast_model(lat, lon, code), label)
            if df.empty: failures.append(label)
            else: frames.append(df)
        except Exception:
            failures.append(label)
    if not frames:
        return pd.DataFrame(), {"quality": "indisponible", "models_ok": [], "models_failed": failures}

    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    first = raw["date"].min()
    raw["echeance_j"] = (raw["date"] - first).dt.days
    groups = []
    for _, group in raw.groupby("date"):
        allowed = SHORT_RANGE_MODELS if int(group["echeance_j"].iloc[0]) <= 3 else LONG_RANGE_MODELS
        filtered = group[group["modele"].isin(allowed)]
        groups.append(filtered if not filtered.empty else group)
    selected = pd.concat(groups, ignore_index=True)

    rows = []
    for date, group in selected.groupby("date", sort=True):
        rain = group["pluie_mm"].dropna()
        probability = group["proba_%"].dropna()
        temp = group["tmax"].dropna()
        wind = group["vent_max"].dropna()
        models = sorted(group.loc[group["pluie_mm"].notna(), "modele"].unique())
        rows.append({
            "date": date.date().isoformat(),
            "pluie_mm": rain.median() if not rain.empty else math.nan,
            "pluie_min_mm": rain.min() if not rain.empty else math.nan,
            "pluie_max_mm": rain.max() if not rain.empty else math.nan,
            "proba_%": probability.max() if not probability.empty else math.nan,
            "tmax": temp.median() if not temp.empty else math.nan,
            "vent_max": wind.max() if not wind.empty else math.nan,
            "nb_modeles": len(models), "modeles": ", ".join(models),
        })
    result = pd.DataFrame(rows)
    quality = "bonne" if len(result) == 7 and (result["nb_modeles"] >= 2).all() else "dégradée"
    return result, {"quality": quality, "models_ok": sorted(raw["modele"].unique()), "models_failed": failures}


@st.cache_data(ttl=1800)
def load_dept_forecast(dep: str) -> pd.DataFrame:
    d = DEPS[dep]
    df, _ = load_forecast_coord(d["lat"], d["lon"])
    return df


# =============================================================================
# HISTORIQUE ERA5
# =============================================================================
@st.cache_data(ttl=3600)
def load_daily_history(lat: float, lon: float, days: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=days-1)
    try:
        data = get_json(ARCHIVE_URL, {
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "start_date": str(start), "end_date": str(end),
            "daily": "precipitation_sum", "timezone": "Europe/Paris",
        })
        daily = data.get("daily", {})
        df = pd.DataFrame({"date": daily.get("time", []), "pluie_mm": daily.get("precipitation_sum", [])})
        df["pluie_mm"] = pd.to_numeric(df["pluie_mm"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_all_communes_rain(_sectors: pd.DataFrame, days=30) -> pd.DataFrame:
    if _sectors.empty: return pd.DataFrame()
    coords = (_sectors.dropna(subset=["latitude", "longitude"])
              .groupby("commune_name")[["latitude", "longitude"]].mean())
    frames = []
    for commune, row in coords.iterrows():
        df = load_daily_history(float(row["latitude"]), float(row["longitude"]), days)
        if not df.empty:
            df["commune_name"] = commune
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_monthly_rain(lat: float, lon: float) -> pd.DataFrame:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = (end.replace(day=1) - timedelta(days=365)).replace(day=1)
    try:
        data = get_json(ARCHIVE_URL, {
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "start_date": str(start), "end_date": str(end),
            "daily": "precipitation_sum", "timezone": "Europe/Paris",
        })
        monthly = defaultdict(float)
        for date, value in zip(data["daily"]["time"], data["daily"]["precipitation_sum"]):
            if value is not None: monthly[date[:7]] += value
        return pd.DataFrame([{"mois": m, "pluie_mm": round(v, 1)} for m, v in sorted(monthly.items())])
    except Exception:
        return pd.DataFrame()


# =============================================================================
# FIRMS
# =============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(min(1, math.sqrt(a)))


def build_polyline(lines):
    if not lines: return []
    segment = lines[0]
    points = [(float(p["lat"]), float(p["lon"])) for p in segment if isinstance(p, dict) and "lat" in p and "lon" in p]
    if len(points) < 2: return []
    output, total = [(points[0][0], points[0][1], 0.0)], 0.0
    for a, b in zip(points, points[1:]):
        total += haversine_km(a[0], a[1], b[0], b[1])
        output.append((b[0], b[1], total))
    return output


def pk_distance(lat, lon, line):
    best = (None, None)
    for (lat1, lon1, pk1), (lat2, lon2, pk2) in zip(line, line[1:]):
        kx, ky = 111.320*math.cos(math.radians((lat1+lat2)/2)), 111.320
        x1, y1, x2, y2, xp, yp = lon1*kx, lat1*ky, lon2*kx, lat2*ky, lon*kx, lat*ky
        dx, dy = x2-x1, y2-y1
        den = dx*dx + dy*dy
        t = 0 if den == 0 else max(0, min(1, ((xp-x1)*dx+(yp-y1)*dy)/den))
        dist = math.hypot(xp-(x1+t*dx), yp-(y1+t*dy))
        if best[1] is None or dist < best[1]: best = (pk1+t*(pk2-pk1), dist)
    return best


@st.cache_data(ttl=300)
def load_firms(_line, day_range=1):
    key = os.environ.get("FIRMS_MAP_KEY")
    try: key = st.secrets.get("FIRMS_MAP_KEY", key)
    except Exception: pass
    if not key: return [], "missing_key"
    frames, ok = [], 0
    date = datetime.now(timezone.utc).date().isoformat()
    for source in FIRMS_SOURCES:
        try:
            url = FIRMS_AREA_URL.format(key=key, source=source, area=FIRMS_BBOX, day_range=day_range, date=date)
            response = SESSION.get(url, timeout=(5, 25)); response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            if "latitude" in df.columns:
                ok += 1; frames.append(df)
        except Exception: pass
    if ok == 0: return [], "fetch_failed"
    if not frames: return [], None
    alerts, seen = [], set()
    for _, row in pd.concat(frames, ignore_index=True).iterrows():
        try: lat, lon = float(row["latitude"]), float(row["longitude"])
        except Exception: continue
        dedup = (round(lat, 4), round(lon, 4))
        if dedup in seen: continue
        pk, dist = pk_distance(lat, lon, _line)
        if pk is not None and dist is not None and dist <= FIRMS_RADIUS_KM:
            seen.add(dedup)
            alerts.append({"lat": lat, "lon": lon, "pk": round(pk, 1), "distance_m": round(dist*1000),
                           "date": row.get("acq_date", ""), "satellite": row.get("satellite", "")})
    return alerts, None


# =============================================================================
# INTERFACE
# =============================================================================
st.set_page_config(page_title="LGV SEA - Pluviométrie", page_icon="🌧️", layout="wide")
st.title("🌧️ LGV SEA - Pluviométrie et surveillance des glissements")
st.caption("Prévisions consolidées multi-modèles Open-Meteo. Historique ERA5. Les données affichées sont des indicateurs et ne remplacent pas les vigilances officielles.")

if st.button("🔄 Rafraîchir les données"):
    st.cache_data.clear(); st.rerun()

try:
    snapshot = load_snapshot()
except Exception as exc:
    st.error(f"Snapshot indisponible : {exc}"); st.stop()

raw_sectors = snapshot.get("sectors", {})
sectors_df = safe_df(raw_sectors.get("sectors", []) if isinstance(raw_sectors, dict) else [])
for col in ["latitude", "longitude", "pk_km"]:
    if col in sectors_df.columns: sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")

required = {"commune_name", "latitude", "longitude", "pk_km"}
if not required.issubset(sectors_df.columns):
    st.error("Le snapshot ne contient pas toutes les colonnes requises : commune_name, latitude, longitude, pk_km.")
    st.stop()

communes = sorted(sectors_df["commune_name"].dropna().unique())
default_communes, risk_ranking = get_default_glissement_communes(sectors_df, 6)
if not default_communes: default_communes = communes[:6]

with st.sidebar:
    st.header("Paramètres")
    selected_multi = st.multiselect("Communes à comparer", communes, default=default_communes)
    options = ["— Toutes —"] + communes
    default_main = options.index(default_communes[0]) if default_communes and default_communes[0] in options else 0
    selected_one = st.selectbox("Commune principale", options, index=default_main,
                                help="Par défaut : commune la mieux classée à proximité des secteurs de glissements connus.")
    if not risk_ranking.empty:
        with st.expander("Classement glissements utilisé"):
            st.dataframe(risk_ranking, hide_index=True, use_container_width=True)
    firms_days = st.slider("Fenêtre FIRMS (jours)", 1, 10, 1)

# Cartes départementales consolidées
st.subheader("🌦️ Prévision consolidée à 7 jours par département")
cols = st.columns(len(DEPS))
for col, (dep, info) in zip(cols, DEPS.items()):
    try: df = load_dept_forecast(dep)
    except Exception: df = pd.DataFrame()
    with col.container(border=True):
        st.caption(f"Dép. {dep} · {info['nom']}")
        if df.empty:
            st.metric("Cumul 7 j", "Indisponible")
        else:
            total = df["pluie_mm"].sum(min_count=1)
            maximum = df["pluie_mm"].max()
            st.metric("Cumul 7 j", f"{total:.1f} mm" if pd.notna(total) else "N/A")
            st.caption(f"Maximum journalier : {maximum:.1f} mm" if pd.notna(maximum) else "Maximum indisponible")

# TOP pluviométrie historique
st.subheader("🌧️ TOP 20 communes - 30 derniers jours complets")
with st.spinner("Chargement de l'historique ERA5..."):
    all_rain = load_all_communes_rain(sectors_df, 30)
if all_rain.empty:
    st.warning("Historique pluvio indisponible. Aucune valeur nulle n'est supposée.")
else:
    totals = all_rain.groupby("commune_name")["pluie_mm"].sum(min_count=1).dropna().nlargest(20).sort_values()
    fig = go.Figure(go.Bar(x=totals.values, y=totals.index, orientation="h",
                           marker_color=[rain_color_mm(v) for v in totals.values],
                           text=[f"{v:.1f} mm" for v in totals.values], textposition="outside"))
    fig.update_layout(height=560, xaxis_title="Cumul (mm)", yaxis_title=None, margin=dict(l=20, r=80, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Comparaison communes à risque
st.subheader("📊 Comparaison des communes sélectionnées - 30 jours complets")
rows = []
for commune in selected_multi[:12]:
    loc = sectors_df[sectors_df["commune_name"] == commune].dropna(subset=["latitude", "longitude"])
    if loc.empty: continue
    history = load_daily_history(float(loc["latitude"].mean()), float(loc["longitude"].mean()), 30)
    value = history["pluie_mm"].sum(min_count=1) if not history.empty else math.nan
    rows.append({"Commune": commune, "Cumul (mm)": value})
cmp = pd.DataFrame(rows).dropna(subset=["Cumul (mm)"]).sort_values("Cumul (mm)")
if cmp.empty:
    st.info("Aucune donnée disponible pour les communes sélectionnées.")
else:
    fig = go.Figure(go.Bar(x=cmp["Cumul (mm)"], y=cmp["Commune"], orientation="h",
                           marker_color=[rain_color_mm(v) for v in cmp["Cumul (mm)"]],
                           text=[f"{v:.1f} mm" for v in cmp["Cumul (mm)"]], textposition="outside"))
    fig.update_layout(height=max(300, len(cmp)*38+80), xaxis_title="Cumul ERA5 (mm)", margin=dict(r=80))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Localisation principale
comm_df = sectors_df if selected_one == "— Toutes —" else sectors_df[sectors_df["commune_name"] == selected_one]
map_df = comm_df.dropna(subset=["latitude", "longitude"])
lat_c = float(map_df["latitude"].mean()) if not map_df.empty else 46.2
lon_c = float(map_df["longitude"].mean()) if not map_df.empty else 0.2
label = "LGV SEA" if selected_one == "— Toutes —" else selected_one

# Prévision détaillée
st.subheader(f"🔮 Prévisions 7 jours - {label}")
forecast, meta = load_forecast_coord(lat_c, lon_c)
if forecast.empty:
    st.error("Aucun modèle météo n'a répondu. Aucun zéro de pluie n'est supposé.")
else:
    forecast["color"] = forecast["pluie_mm"].apply(rain_color_mm)
    err_minus = (forecast["pluie_mm"] - forecast["pluie_min_mm"]).clip(lower=0).fillna(0)
    err_plus = (forecast["pluie_max_mm"] - forecast["pluie_mm"]).clip(lower=0).fillna(0)
    fig = go.Figure()
    fig.add_bar(x=forecast["date"], y=forecast["pluie_mm"], marker_color=forecast["color"],
                name="Pluie médiane", text=forecast["pluie_mm"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"),
                textposition="outside",
                error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus, color="#64748b"),
                customdata=forecast[["pluie_min_mm", "pluie_max_mm", "nb_modeles", "modeles"]].to_numpy(),
                hovertemplate="<b>%{x}</b><br>Médiane : %{y:.1f} mm<br>Fourchette : %{customdata[0]:.1f} à %{customdata[1]:.1f} mm<br>%{customdata[2]} modèle(s) : %{customdata[3]}<extra></extra>")
    fig.add_scatter(x=forecast["date"], y=forecast["proba_%"], mode="lines+markers", name="Probabilité max (%)",
                    yaxis="y2", line=dict(color="#6366f1", dash="dot"))
    fig.add_scatter(x=forecast["date"], y=forecast["tmax"], mode="lines+markers", name="T° max médiane",
                    yaxis="y3", line=dict(color="#f97316"))
    fig.update_layout(height=420, yaxis=dict(title="Pluie (mm)"),
                      yaxis2=dict(title="Probabilité %", overlaying="y", side="right", range=[0, 110], showgrid=False),
                      yaxis3=dict(title="T° C", overlaying="y", side="right", anchor="free", position=0.93, showgrid=False),
                      legend=dict(orientation="h", y=1.15), margin=dict(r=90))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if meta["quality"] == "bonne": st.success("Prévision consolidée : au moins deux modèles par journée.")
    else: st.warning("Prévision dégradée : certaines journées reposent sur moins de deux modèles.")
    st.caption("Modèles reçus : " + (", ".join(meta["models_ok"]) or "aucun"))
    if meta["models_failed"]: st.caption("Modèles indisponibles : " + ", ".join(meta["models_failed"]))

# Historique mensuel
st.subheader(f"📅 Historique mensuel - {label}")
monthly = load_monthly_rain(lat_c, lon_c)
if monthly.empty:
    st.info("Historique mensuel indisponible.")
else:
    fig = go.Figure(go.Bar(x=monthly["mois"], y=monthly["pluie_mm"], marker_color="#2563eb",
                           text=monthly["pluie_mm"].map(lambda v: f"{v:.0f}"), textposition="outside"))
    fig.update_layout(height=350, yaxis_title="Pluie (mm)", xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# FIRMS et carte
line = build_polyline(snapshot.get("lgv_lines") or [])
firms, firms_error = load_firms(line, firms_days)
st.subheader("🔥 Détections NASA FIRMS à moins de 500 m de la LGV")
if firms_error == "missing_key": st.info("Clé FIRMS absente : renseigner FIRMS_MAP_KEY dans les secrets Streamlit.")
elif firms_error: st.warning("FIRMS injoignable : statut non vérifié.")
elif firms: st.error(f"{len(firms)} détection(s) satellite à proximité de la LGV.")
else: st.success("Aucune détection FIRMS à proximité sur la période interrogée.")

st.subheader("🗺️ Carte des secteurs LGV SEA")
if map_df.empty:
    st.info("Aucune localisation disponible.")
else:
    m = folium.Map(location=[lat_c, lon_c], zoom_start=8 if selected_one == "— Toutes —" else 11,
                   tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                   attr="Esri World Imagery", control_scale=True)
    for _, row in map_df.iterrows():
        folium.CircleMarker([float(row["latitude"]), float(row["longitude"])], radius=5, color="#2563eb",
                            fill=True, fill_opacity=.8,
                            tooltip=f"{row['commune_name']} - PK {row['pk_km']}").add_to(m)
    for segment in snapshot.get("lgv_lines") or []:
        points = [[p["lat"], p["lon"]] for p in segment if isinstance(p, dict) and "lat" in p and "lon" in p]
        if points: folium.PolyLine(points, color="#dc2626", weight=3, opacity=.8).add_to(m)
    for alert in firms:
        folium.Marker([alert["lat"], alert["lon"]], icon=folium.Icon(color="red", icon="fire", prefix="fa"),
                      tooltip=f"FIRMS PK {alert['pk']} - {alert['distance_m']} m").add_to(m)
    st_folium(m, use_container_width=True, height=500, returned_objects=[])

st.subheader("📋 Secteurs affichés")
st.dataframe(comm_df[["commune_name", "pk_km"]].rename(columns={"commune_name": "Commune", "pk_km": "PK (km)"})
             .sort_values("PK (km)"), hide_index=True, use_container_width=True, height=320)
