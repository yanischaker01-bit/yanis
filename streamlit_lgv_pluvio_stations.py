from __future__ import annotations

import io
import math
import os
import unicodedata
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="LGV SEA - Surveillance météo des glissements",
    page_icon="🌧️",
    layout="wide",
)

SNAPSHOT_URL = (
    "https://yanischaker01-bit.github.io/yanis/"
    "reports/streamlit_snapshot_latest.json"
)
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MF_VIGILANCE_URL = "https://public.opendatasoft.com/api/records/1.0/search/"
MF_VIGILANCE_DATASET = "weatherref-france-vigilance-meteo-departement"
VIGICRUES_URL = "https://www.vigicrues.gouv.fr/services/v1.1/TerEntVigiCru.json"
FIRMS_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{key}/{source}/{area}/{day_range}/{date}"
)

TIMEZONE = "Europe/Paris"
HTTP_TIMEOUT = (5, 25)
CACHE_FORECAST_SECONDS = 1800
CACHE_ALERT_SECONDS = 900

# Un seul modèle est présenté à l'utilisateur : ECMWF IFS.
# Le second identifiant est uniquement un alias de compatibilité Open-Meteo,
# jamais un second modèle combiné.
ECMWF_MODEL_ALIASES = ("ecmwf_ifs", "ecmwf_ifs025")

# Zones documentées de forte densité de glissements.
# Les raccordements ne sont utilisés que si le snapshot contient un champ d'axe.
RISK_ZONES = [
    {"ouvrage": "RBT 0973", "axe": "LGV", "pk_d": 97.20, "pk_f": 97.50},
    {"ouvrage": "RBT 1042", "axe": "LGV", "pk_d": 103.78, "pk_f": 104.05},
    {"ouvrage": "RBT 1056", "axe": "LGV", "pk_d": 105.20, "pk_f": 105.70},
    {"ouvrage": "RBT 1065", "axe": "LGV", "pk_d": 106.00, "pk_f": 106.90},
    {"ouvrage": "RBT 1094", "axe": "LGV", "pk_d": 109.00, "pk_f": 109.25},
    {"ouvrage": "RBT 1204", "axe": "LGV", "pk_d": 119.30, "pk_f": 119.90},
    {"ouvrage": "RBT 1204", "axe": "LGV", "pk_d": 120.85, "pk_f": 121.05},
    {"ouvrage": "RBT 0202", "axe": "LGV", "pk_d": 19.90, "pk_f": 20.40},
    {"ouvrage": "RBT FN1 0021", "axe": "FN1", "pk_d": 2.00, "pk_f": 2.40},
    {"ouvrage": "RBT FN2 0017", "axe": "FN2", "pk_d": 1.60, "pk_f": 2.00},
    {"ouvrage": "DBT MS1 0025", "axe": "MS1", "pk_d": 2.80, "pk_f": 3.30},
    {"ouvrage": "DBT MS2 0029", "axe": "MS2", "pk_d": 2.80, "pk_f": 3.30},
    {"ouvrage": "RBT FS1 0030", "axe": "FS1", "pk_d": 2.80, "pk_f": 3.25},
    {"ouvrage": "RBT FS2 0030", "axe": "FS2", "pk_d": 3.10, "pk_f": 3.30},
]

DEPS = {
    "37": "Indre-et-Loire",
    "86": "Vienne",
    "79": "Deux-Sèvres",
    "16": "Charente",
    "17": "Charente-Maritime",
    "33": "Gironde",
}

FIRMS_SOURCES = ("VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT")
FIRMS_BBOX = "-0.7,44.75,1.0,47.5"
FIRMS_RADIUS_KM = 0.5

LEVEL_RANK = {"ROUGE": 4, "ORANGE": 3, "JAUNE": 2, "VERT": 1, "INFO": 0}
LEVEL_COLOR = {
    "ROUGE": "#b91c1c", "ORANGE": "#c2410c", "JAUNE": "#a16207",
    "VERT": "#15803d", "INFO": "#475569",
}
LEVEL_ICON = {
    "ROUGE": "🔴", "ORANGE": "🟠", "JAUNE": "🟡",
    "VERT": "🟢", "INFO": "🔵",
}

# Seuils simples d'aide à la surveillance. À faire valider par l'expert GC.
RAIN_THRESHOLDS = {
    "ROUGE": {"day": 50.0, "week": 120.0},
    "ORANGE": {"day": 30.0, "week": 80.0},
    "JAUNE": {"day": 15.0, "week": 50.0},
}

RIVER_NAMES = (
    "vienne", "clain", "charente", "boutonne", "seugne", "touvre",
    "dronne", "isle", "dordogne", "garonne", "thouet", "sevre",
    "indre", "cher", "creuse", "ciron", "jalles", "estey",
)

# =============================================================================
# STYLE
# =============================================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1450px;}
    [data-testid="stMetric"] {background:#ffffff; border:1px solid #e2e8f0;
        border-radius:14px; padding:14px 16px; box-shadow:0 3px 12px rgba(15,23,42,.04);}
    .hero {padding:22px 24px; border-radius:18px; color:white;
        background:linear-gradient(120deg,#073763,#0f6b78); margin-bottom:18px;}
    .hero h1 {margin:0; font-size:1.8rem;}
    .hero p {margin:.5rem 0 0; color:#dbeafe;}
    .status-card {border:1px solid #e2e8f0; border-left:5px solid var(--c);
        border-radius:14px; padding:14px 16px; background:white; margin-bottom:9px;}
    .small-muted {font-size:.86rem; color:#64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# HTTP
# =============================================================================
def build_session() -> requests.Session:
    retry = Retry(
        total=4, connect=4, read=4, status=4, backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]), respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "MESEA-LGV-Surveillance/4.0"})
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    return session

HTTP = build_session()

def get_json(url: str, params: dict | None = None) -> dict:
    response = HTTP.get(url, params=params, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("error"):
        raise RuntimeError(str(payload.get("reason") if isinstance(payload, dict) else payload))
    return payload

# =============================================================================
# DONNÉES DE RÉFÉRENCE ET LOCALISATION
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_snapshot() -> dict:
    return get_json(SNAPSHOT_URL)

def pick_column(columns, candidates: tuple[str, ...]) -> str | None:
    lookup = {str(col).lower(): col for col in columns}
    return next((lookup[name.lower()] for name in candidates if name.lower() in lookup), None)

def normalize_axis(value) -> str:
    text = str(value or "LGV").upper().replace(" ", "").strip()
    return {"LGVSEA": "LGV", "LIGNE": "LGV", "V1": "LGV", "V2": "LGV"}.get(text, text)

def prepare_sectors(snapshot: dict) -> tuple[pd.DataFrame, bool]:
    source = snapshot.get("sectors", {})
    records = source.get("sectors", []) if isinstance(source, dict) else source
    frame = pd.DataFrame(records if isinstance(records, list) else [])
    if frame.empty:
        return frame, False

    commune = pick_column(frame.columns, ("commune_name", "commune", "nom_commune"))
    pk = pick_column(frame.columns, ("pk_km", "pk", "pk_decimal"))
    lat = pick_column(frame.columns, ("latitude", "lat", "y"))
    lon = pick_column(frame.columns, ("longitude", "lon", "lng", "x"))
    axis = pick_column(frame.columns, ("axe", "ligne", "line", "branch", "raccordement"))
    if not all((commune, pk, lat, lon)):
        return pd.DataFrame(), False

    rename = {commune: "commune", pk: "pk_km", lat: "latitude", lon: "longitude"}
    if axis:
        rename[axis] = "axe"
    frame = frame.rename(columns=rename).copy()
    frame["axe"] = frame["axe"].map(normalize_axis) if "axe" in frame else "LGV"
    frame["commune"] = frame["commune"].astype(str).str.strip()
    for col in ("pk_km", "latitude", "longitude"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["commune", "pk_km", "latitude", "longitude"])
    return frame, axis is not None

def interval_distance(pk: float, start: float, end: float) -> float:
    return 0.0 if start <= pk <= end else min(abs(pk - start), abs(pk - end))

def locate_risk_communes(sectors: pd.DataFrame, has_axis: bool) -> tuple[pd.DataFrame, list[str]]:
    found, skipped = [], []
    for zone in RISK_ZONES:
        if zone["axe"] != "LGV" and not has_axis:
            skipped.append(zone["ouvrage"])
            continue
        candidates = sectors[sectors["axe"] == zone["axe"]] if has_axis else sectors
        if candidates.empty:
            skipped.append(zone["ouvrage"])
            continue
        distances = candidates["pk_km"].apply(
            lambda pk: interval_distance(float(pk), zone["pk_d"], zone["pk_f"])
        )
        idx = distances.idxmin()
        max_gap = 3.0 if zone["axe"] == "LGV" else 1.0
        if float(distances.loc[idx]) > max_gap:
            skipped.append(zone["ouvrage"])
            continue
        point = candidates.loc[idx]
        found.append({
            **zone,
            "commune": point["commune"],
            "latitude": float(point["latitude"]),
            "longitude": float(point["longitude"]),
        })
    return pd.DataFrame(found), skipped

def aggregate_communes(zones: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for commune, group in zones.groupby("commune", sort=True):
        rows.append({
            "commune": commune,
            "latitude": float(group["latitude"].mean()),
            "longitude": float(group["longitude"].mean()),
            "ouvrages": ", ".join(sorted(group["ouvrage"].unique())),
        })
    return pd.DataFrame(rows)

# =============================================================================
# PRÉVISION UNIQUE ECMWF IFS
# =============================================================================
@st.cache_data(ttl=CACHE_FORECAST_SECONDS, show_spinner=False)
def fetch_ecmwf(lat: float, lon: float) -> tuple[dict, str]:
    errors = []
    for model_code in ECMWF_MODEL_ALIASES:
        try:
            payload = get_json(FORECAST_URL, {
                "latitude": round(float(lat), 4),
                "longitude": round(float(lon), 4),
                "models": model_code,
                "daily": (
                    "precipitation_sum,precipitation_probability_max,"
                    "weather_code,temperature_2m_max,wind_gusts_10m_max"
                ),
                "forecast_days": 7,
                "timezone": TIMEZONE,
                "cell_selection": "land",
            })
            return payload, model_code
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("ECMWF indisponible : " + " | ".join(errors))

def forecast_frame(lat: float, lon: float) -> tuple[pd.DataFrame, str | None]:
    try:
        payload, model_code = fetch_ecmwf(lat, lon)
    except Exception:
        return pd.DataFrame(), None
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        return pd.DataFrame(), None
    n = len(dates)
    def vals(name):
        content = daily.get(name, [])
        return content if len(content) == n else [None] * n
    frame = pd.DataFrame({
        "date": pd.to_datetime(dates, errors="coerce"),
        "pluie_mm": vals("precipitation_sum"),
        "proba_pct": vals("precipitation_probability_max"),
        "weather_code": vals("weather_code"),
        "tmax_c": vals("temperature_2m_max"),
        "rafale_kmh": vals("wind_gusts_10m_max"),
    }).dropna(subset=["date"])
    for col in ("pluie_mm", "proba_pct", "weather_code", "tmax_c", "rafale_kmh"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame, model_code

def surveillance_level(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty or frame["pluie_mm"].dropna().empty:
        return "INFO", "Prévision non vérifiée"
    max_day = float(frame["pluie_mm"].max())
    total = float(frame["pluie_mm"].sum())
    for level in ("ROUGE", "ORANGE", "JAUNE"):
        threshold = RAIN_THRESHOLDS[level]
        if max_day >= threshold["day"] or total >= threshold["week"]:
            return level, {
                "ROUGE": "Surveillance renforcée",
                "ORANGE": "Surveillance élevée",
                "JAUNE": "Vigilance météo",
            }[level]
    return "VERT", "Surveillance normale"

# =============================================================================
# ALERTES ORAGE / INONDATION
# =============================================================================
def derived_weather_alerts(commune: str, frame: pd.DataFrame) -> list[dict]:
    alerts = []
    for row in frame.itertuples(index=False):
        date_label = row.date.strftime("%d/%m")
        weather_code = int(row.weather_code) if pd.notna(row.weather_code) else -1
        rain = float(row.pluie_mm) if pd.notna(row.pluie_mm) else 0.0
        if weather_code >= 99:
            alerts.append({"type": "ORAGE", "level": "ROUGE", "title": commune,
                           "message": f"Orages violents possibles le {date_label}."})
        elif weather_code >= 95:
            alerts.append({"type": "ORAGE", "level": "ORANGE", "title": commune,
                           "message": f"Orages possibles le {date_label}."})
        if rain >= 50:
            alerts.append({"type": "INONDATION", "level": "ROUGE", "title": commune,
                           "message": f"Pluie très forte prévue le {date_label}."})
        elif rain >= 30:
            alerts.append({"type": "INONDATION", "level": "ORANGE", "title": commune,
                           "message": f"Pluie forte prévue le {date_label}."})
        elif rain >= 15:
            alerts.append({"type": "INONDATION", "level": "JAUNE", "title": commune,
                           "message": f"Pluie soutenue prévue le {date_label}."})
    return alerts

# =============================================================================
# VIGILANCE MÉTÉO-FRANCE
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_mf_vigilance() -> tuple[list[dict], bool]:
    query = " OR ".join(f"domain_id:{dep}" for dep in DEPS)
    try:
        payload = get_json(MF_VIGILANCE_URL, {
            "dataset": MF_VIGILANCE_DATASET, "q": query, "rows": 100,
        })
    except Exception:
        return [], False
    colors = {"jaune": "JAUNE", "orange": "ORANGE", "rouge": "ROUGE"}
    alerts = []
    for record in payload.get("records", []):
        fields = record.get("fields", {})
        dep = str(fields.get("domain_id", ""))
        level = colors.get(str(fields.get("color", "")).lower())
        if dep in DEPS and level:
            phenomenon = str(fields.get("phenomenon") or "phénomène météo").capitalize()
            alerts.append({
                "type": "MÉTÉO-FRANCE", "level": level, "title": f"{DEPS[dep]} ({dep})",
                "message": f"{phenomenon} : vigilance {level.lower()}.",
            })
    return alerts, True

# =============================================================================
# VIGICRUES
# =============================================================================
def normalize_text(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", str(value).lower())
        if unicodedata.category(char) != "Mn"
    )

def walk_json(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)

@st.cache_data(ttl=1800, show_spinner=False)
def load_vigicrues() -> tuple[list[dict], bool]:
    levels_num = {2: "JAUNE", 3: "ORANGE", 4: "ROUGE"}
    levels_name = {"jaune": "JAUNE", "orange": "ORANGE", "rouge": "ROUGE"}
    try:
        root = get_json(VIGICRUES_URL)
    except Exception:
        return [], False
    territories = root.get("ListEntVigiCru", [])
    alerts, seen, successful = [], set(), 1
    for territory in territories if isinstance(territories, list) else []:
        code = territory.get("CdEntVigiCru") if isinstance(territory, dict) else None
        kind = territory.get("TypEntVigiCru", "5") if isinstance(territory, dict) else "5"
        if not code:
            continue
        try:
            payload = get_json(VIGICRUES_URL, {"CdEntVigiCru": code, "TypEntVigiCru": kind})
            successful += 1
        except Exception:
            continue
        for item in walk_json(payload):
            name = next((item.get(key) for key in (
                "LbEntVigiCru", "LibEntVigiCru", "NomEntVigiCru", "LibTroncon",
                "NomTroncon", "NomCoursDeau", "Nom"
            ) if item.get(key)), None)
            if not name or not any(river in normalize_text(name) for river in RIVER_NAMES):
                continue
            raw = next((item.get(key) for key in (
                "NivVigiCru", "NivVigiCruHydro", "NivVig", "NiveauVigilance",
                "CdCouleur", "Couleur", "couleur"
            ) if item.get(key) not in (None, "")), None)
            level = levels_name.get(normalize_text(raw))
            if not level:
                try:
                    level = levels_num.get(int(float(raw)))
                except Exception:
                    level = None
            key = (normalize_text(name), level)
            if level and key not in seen:
                seen.add(key)
                alerts.append({"type": "VIGICRUE", "level": level,
                               "title": str(name), "message": f"Vigilance crue {level.lower()}."})
    return alerts, successful > 0

# =============================================================================
# FIRMS
# =============================================================================
def get_firms_key() -> str | None:
    try:
        value = st.secrets.get("FIRMS_MAP_KEY")
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get("FIRMS_MAP_KEY")

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))

def snapshot_polyline(snapshot: dict) -> list[tuple[float, float]]:
    points = []
    for segment in snapshot.get("lgv_lines", []) or []:
        if not isinstance(segment, list):
            continue
        for point in segment:
            if isinstance(point, dict) and "lat" in point and "lon" in point:
                points.append((float(point["lat"]), float(point["lon"])))
    return points

@st.cache_data(ttl=300, show_spinner=False)
def load_firms_raw(key: str, source: str, date_text: str) -> pd.DataFrame:
    url = FIRMS_URL.format(key=key, source=source, area=FIRMS_BBOX, day_range=1, date=date_text)
    response = HTTP.get(url, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    text = response.text.strip()
    if not text or "invalid" in text.lower()[:200]:
        raise RuntimeError("Réponse FIRMS invalide")
    return pd.read_csv(io.StringIO(text))

def load_firms(snapshot: dict) -> tuple[list[dict], str | None]:
    key = get_firms_key()
    if not key:
        return [], "Clé FIRMS absente"
    line = snapshot_polyline(snapshot)
    if not line:
        return [], "Tracé LGV indisponible"
    date_text = datetime.now(timezone.utc).date().isoformat()
    alerts, success, seen = [], 0, set()
    for source in FIRMS_SOURCES:
        try:
            frame = load_firms_raw(key, source, date_text)
            success += 1
        except Exception:
            continue
        for row in frame.itertuples(index=False):
            try:
                lat, lon = float(row.latitude), float(row.longitude)
            except Exception:
                continue
            marker = (round(lat, 4), round(lon, 4))
            if marker in seen:
                continue
            distance = min(haversine_km(lat, lon, p_lat, p_lon) for p_lat, p_lon in line)
            if distance <= FIRMS_RADIUS_KM:
                seen.add(marker)
                alerts.append({
                    "type": "FIRMS", "level": "ROUGE", "title": "Détection satellite",
                    "message": f"Point chaud détecté à environ {distance*1000:.0f} m de la LGV.",
                })
    if success == 0:
        return [], "FIRMS injoignable"
    return alerts, None

# =============================================================================
# RENDU
# =============================================================================
def render_alert(alert: dict) -> None:
    level = alert.get("level", "INFO")
    color = LEVEL_COLOR.get(level, LEVEL_COLOR["INFO"])
    icon = {
        "ORAGE": "⛈️", "INONDATION": "🌊", "VIGICRUE": "🏞️",
        "FIRMS": "🔥", "MÉTÉO-FRANCE": "🛡️",
    }.get(alert.get("type"), "ℹ️")
    st.markdown(
        f"<div class='status-card' style='--c:{color}'>"
        f"<b>{icon} {alert.get('title','')}</b> · {alert.get('message','')}"
        f"</div>", unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
      <h1>Surveillance météo des zones de glissements</h1>
      <p>Prévisions à 7 jours sur les communes à fort risque · LGV SEA</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Surveillance")
    if st.button("🔄 Actualiser", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Prévision : ECMWF IFS · mise en cache 30 min")

try:
    snapshot = load_snapshot()
except Exception as exc:
    st.error(f"Référentiel LGV indisponible : {exc}")
    st.stop()

sectors, has_axis = prepare_sectors(snapshot)
if sectors.empty:
    st.error("Le snapshot ne contient pas les champs commune, PK, latitude et longitude attendus.")
    st.stop()

zones, skipped = locate_risk_communes(sectors, has_axis)
communes = aggregate_communes(zones)
if communes.empty:
    st.error("Aucune commune à fort risque n'a pu être localisée dans le référentiel.")
    st.stop()

with st.sidebar:
    selected = st.multiselect(
        "Communes suivies",
        options=communes["commune"].tolist(),
        default=communes["commune"].tolist(),
    )
    st.divider()
    st.info(
        "Une seule prévision est affichée par commune afin de conserver une lecture simple."
    )

communes = communes[communes["commune"].isin(selected)]
if communes.empty:
    st.info("Sélectionne au moins une commune.")
    st.stop()

forecast_by_commune = {}
model_codes = set()
summary_rows = []
derived_alerts = []

with st.spinner("Chargement de la prévision de surveillance…"):
    for row in communes.itertuples(index=False):
        frame, model_code = forecast_frame(row.latitude, row.longitude)
        forecast_by_commune[row.commune] = frame
        if model_code:
            model_codes.add(model_code)
        level, label = surveillance_level(frame)
        if frame.empty:
            total = math.nan
        else:
            total = float(frame["pluie_mm"].sum(min_count=1))
            derived_alerts.extend(derived_weather_alerts(row.commune, frame))
        summary_rows.append({
            "commune": row.commune, "ouvrages": row.ouvrages,
            "level": level, "label": label, "total": total,
        })

mf_alerts, mf_ok = load_mf_vigilance()
vc_alerts, vc_ok = load_vigicrues()
firms_alerts, firms_error = load_firms(snapshot)

# En-tête de situation
valid = [row for row in summary_rows if pd.notna(row["total"])]
if valid:
    worst = max(valid, key=lambda row: LEVEL_RANK[row["level"]])
    color = LEVEL_COLOR[worst["level"]]
    st.markdown(
        f"<div class='status-card' style='--c:{color}'>"
        f"<b>{LEVEL_ICON[worst['level']]} Situation générale : {worst['label']}</b><br>"
        f"<span class='small-muted'>Niveau le plus élevé observé sur les communes suivies.</span>"
        f"</div>", unsafe_allow_html=True,
    )
else:
    st.warning("Prévision météo non vérifiée actuellement.")

# Graphe principal unique
st.subheader("Prévision de pluie à 7 jours")
fig = go.Figure()
palette = ["#0f6b78", "#2563eb", "#7c3aed", "#c2410c", "#15803d", "#be123c", "#475569"]
for index, row in communes.reset_index(drop=True).iterrows():
    frame = forecast_by_commune.get(row["commune"], pd.DataFrame())
    if frame.empty:
        continue
    fig.add_trace(go.Scatter(
        x=frame["date"], y=frame["pluie_mm"], mode="lines+markers",
        name=row["commune"],
        line=dict(width=3, color=palette[index % len(palette)]),
        marker=dict(size=7),
        hovertemplate="<b>%{fullData.name}</b><br>%{x|%d/%m}<br>%{y:.1f} mm<extra></extra>",
    ))
fig.update_layout(
    height=470, hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(t=46, b=45, l=55, r=25),
    yaxis_title="Pluie prévue (mm/jour)", xaxis_title=None,
    legend=dict(orientation="h", y=1.03, x=0),
    font=dict(family="Arial, sans-serif", color="#334155"),
)
fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", rangemode="tozero")
fig.update_xaxes(showgrid=False, tickformat="%d/%m")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
model_label = "ECMWF IFS" if model_codes else "non vérifié"
st.caption(
    f"Une seule prévision opérationnelle par commune · modèle affiché : {model_label}. "
    "La valeur 0 mm n'est utilisée que lorsque le modèle renvoie explicitement zéro."
)

# Synthèse volontairement sobre
st.subheader("Communes sous surveillance")
columns = st.columns(min(3, max(1, len(summary_rows))))
for index, item in enumerate(sorted(summary_rows, key=lambda x: (-LEVEL_RANK[x['level']], x['commune']))):
    with columns[index % len(columns)].container(border=True):
        st.markdown(f"**{LEVEL_ICON[item['level']]} {item['commune']}**")
        st.write(item["label"])
        if pd.notna(item["total"]):
            st.caption(f"Cumul prévu sur 7 jours : {item['total']:.0f} mm")
        else:
            st.caption("Prévision non vérifiée")
        st.caption(f"Zones : {item['ouvrages']}")

# Alertes conservées
st.subheader("Alertes et vigilances")
tab_meteo, tab_crues, tab_firms = st.tabs([
    "⛈️ Orage et inondation", "🏞️ Vigicrue", "🔥 FIRMS",
])

with tab_meteo:
    combined_weather = mf_alerts + derived_alerts
    if not mf_ok:
        st.warning("Vigilance Météo-France non vérifiée.")
    if combined_weather:
        for alert in sorted(combined_weather, key=lambda x: -LEVEL_RANK[x["level"]]):
            render_alert(alert)
    elif mf_ok:
        st.success("Aucune vigilance ou alerte météo significative détectée.")

with tab_crues:
    if not vc_ok:
        st.warning("Vigicrue non vérifié actuellement.")
    elif vc_alerts:
        for alert in sorted(vc_alerts, key=lambda x: -LEVEL_RANK[x["level"]]):
            render_alert(alert)
    else:
        st.success("Aucune vigilance crue significative détectée sur les cours d'eau suivis.")

with tab_firms:
    if firms_error:
        st.warning(f"Statut FIRMS non vérifié : {firms_error}.")
    elif firms_alerts:
        for alert in firms_alerts:
            render_alert(alert)
    else:
        st.success("Aucune détection FIRMS à moins de 500 m de la LGV.")

if skipped:
    with st.expander("Qualité du référentiel", expanded=False):
        st.write(
            "Zones non associées automatiquement : " + ", ".join(skipped) + ". "
            "Elles ne sont pas rattachées à une commune afin d'éviter une localisation erronée."
        )

st.divider()
st.caption(
    "Outil d'aide à la surveillance. La prévision météo ne remplace pas les vigilances officielles, "
    "les inspections terrain ni l'avis de l'expert Génie Civil."
)
