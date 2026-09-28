from __future__ import annotations

import math
import time
from collections import defaultdict
from datetime import datetime, timezone

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
TIMEZONE = "Europe/Paris"
REQUEST_TIMEOUT = (5, 25)

# Zones de forte densité documentées dans le dossier glissements MESEA.
# L'axe est indispensable pour ne pas confondre un PK de raccordement avec le PK LGV.
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

# Modèles déterministes explicitement tracés.
# AROME HD est privilégié à courte échéance. Les modèles globaux complètent J4-J7.
MODEL_CANDIDATES = [
    {
        "label": "Météo-France AROME HD",
        "codes": ["meteofrance_arome_france_hd", "meteofrance_arome_france"],
        "max_day": 3,
        "weight": 1.45,
    },
    {
        "label": "Météo-France ARPEGE Europe",
        "codes": ["meteofrance_arpege_europe", "meteofrance_seamless"],
        "max_day": 4,
        "weight": 1.20,
    },
    {
        "label": "ECMWF IFS",
        "codes": ["ecmwf_ifs", "ecmwf_ifs025"],
        "max_day": 6,
        "weight": 1.35,
    },
    {
        "label": "DWD ICON Europe",
        "codes": ["icon_eu", "icon_seamless"],
        "max_day": 6,
        "weight": 1.00,
    },
]

LEVEL_RANK = {"ROUGE": 4, "ORANGE": 3, "JAUNE": 2, "VERT": 1, "INDISPONIBLE": 0}
LEVEL_COLOR = {
    "ROUGE": "#dc2626",
    "ORANGE": "#ea580c",
    "JAUNE": "#eab308",
    "VERT": "#16a34a",
    "INDISPONIBLE": "#64748b",
}
LEVEL_ICON = {
    "ROUGE": "🔴",
    "ORANGE": "🟠",
    "JAUNE": "🟡",
    "VERT": "🟢",
    "INDISPONIBLE": "⚪",
}

# Seuils météo internes d'aide à la surveillance, à valider par l'expert GC.
# Ils ne constituent ni une probabilité de glissement ni une consigne réglementaire.
THRESHOLDS = {
    "ROUGE": {"h24": 50.0, "h72": 80.0, "d7": 120.0},
    "ORANGE": {"h24": 30.0, "h72": 50.0, "d7": 80.0},
    "JAUNE": {"h24": 15.0, "h72": 30.0, "d7": 50.0},
}

# =============================================================================
# HTTP ROBUSTE
# =============================================================================


def build_http_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers.update({
        "User-Agent": "MESEA-LGV-Glissements-Meteo/3.0",
        "Accept": "application/json",
    })
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


HTTP = build_http_session()


def get_json(url: str, params: dict | None = None) -> dict:
    response = HTTP.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Réponse JSON inattendue")
    if payload.get("error"):
        raise RuntimeError(str(payload.get("reason") or payload))
    return payload


# =============================================================================
# SNAPSHOT ET COMMUNES À RISQUE
# =============================================================================


@st.cache_data(ttl=900, show_spinner=False)
def fetch_snapshot() -> dict:
    return get_json(SNAPSHOT_URL)


def safe_dataframe(value) -> pd.DataFrame:
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict) and isinstance(value.get("sectors"), list):
        return pd.DataFrame(value["sectors"])
    return pd.DataFrame()


def first_existing(columns, candidates: list[str]) -> str | None:
    lowered = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def normalise_axis(value) -> str:
    text = str(value or "LGV").upper().strip().replace(" ", "")
    aliases = {
        "LIGNE": "LGV",
        "LGVSEA": "LGV",
        "V1": "LGV",
        "V2": "LGV",
    }
    return aliases.get(text, text)


def prepare_sectors(snapshot: dict) -> tuple[pd.DataFrame, dict]:
    sectors = safe_dataframe(snapshot.get("sectors"))
    if sectors.empty:
        return sectors, {}

    commune_col = first_existing(
        sectors.columns,
        ["commune_name", "commune", "nom_commune", "libelle_commune"],
    )
    pk_col = first_existing(
        sectors.columns,
        ["pk_km", "pk", "pk_decimal", "pk_moyen"],
    )
    lat_col = first_existing(sectors.columns, ["latitude", "lat", "y"])
    lon_col = first_existing(sectors.columns, ["longitude", "lon", "lng", "x"])
    axis_col = first_existing(
        sectors.columns,
        ["axe", "line", "ligne", "voie", "branch", "raccordement"],
    )

    required = [commune_col, pk_col, lat_col, lon_col]
    if any(col is None for col in required):
        return pd.DataFrame(), {
            "error": "Le snapshot doit contenir commune, PK, latitude et longitude."
        }

    rename = {
        commune_col: "commune",
        pk_col: "pk_km",
        lat_col: "latitude",
        lon_col: "longitude",
    }
    if axis_col:
        rename[axis_col] = "axe"

    work = sectors.rename(columns=rename).copy()
    if "axe" not in work.columns:
        work["axe"] = "LGV"

    work["commune"] = work["commune"].astype(str).str.strip()
    work["axe"] = work["axe"].map(normalise_axis)
    for col in ["pk_km", "latitude", "longitude"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["commune", "pk_km", "latitude", "longitude"])
    work = work[(work["latitude"].between(40, 52)) & (work["longitude"].between(-6, 10))]

    return work, {"axis_field_found": axis_col is not None}


def distance_to_interval(pk: float, start: float, end: float) -> float:
    if start <= pk <= end:
        return 0.0
    return min(abs(pk - start), abs(pk - end))


def identify_risk_locations(sectors: pd.DataFrame, axis_available: bool) -> tuple[pd.DataFrame, list[str]]:
    """Associe chaque zone documentée à sa commune la plus proche sur le même axe.

    Une zone de raccordement n'est jamais rabattue sur la LGV si le snapshot ne porte
    pas l'information d'axe. Cela évite une commune faussement attribuée au PK 2 ou 3 LGV.
    """
    rows = []
    skipped = []

    for zone in RISK_ZONES:
        axis = zone["axe"]
        if axis != "LGV" and not axis_available:
            skipped.append(f"{zone['ouvrage']} ({axis})")
            continue

        candidates = sectors[sectors["axe"] == axis] if axis_available else sectors
        if candidates.empty:
            skipped.append(f"{zone['ouvrage']} ({axis})")
            continue

        distances = candidates["pk_km"].apply(
            lambda pk: distance_to_interval(float(pk), zone["pk_d"], zone["pk_f"])
        )
        nearest_idx = distances.idxmin()
        nearest = candidates.loc[nearest_idx]
        distance = float(distances.loc[nearest_idx])

        # Garde-fous pour ne pas associer une zone à une commune très éloignée.
        max_distance = 3.0 if axis == "LGV" else 1.0
        if distance > max_distance:
            skipped.append(f"{zone['ouvrage']} ({axis})")
            continue

        rows.append({
            "ouvrage": zone["ouvrage"],
            "axe": axis,
            "pk_d": zone["pk_d"],
            "pk_f": zone["pk_f"],
            "pk_centre": (zone["pk_d"] + zone["pk_f"]) / 2,
            "commune": nearest["commune"],
            "latitude": float(nearest["latitude"]),
            "longitude": float(nearest["longitude"]),
            "distance_pk_km": distance,
        })

    mapped = pd.DataFrame(rows)
    return mapped, skipped


# =============================================================================
# PRÉVISIONS MULTI-MODÈLES
# =============================================================================


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_model_forecast(lat: float, lon: float, model_code: str) -> dict:
    params = {
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "models": model_code,
        "daily": (
            "precipitation_sum,precipitation_probability_max,"
            "temperature_2m_max,wind_gusts_10m_max,weather_code"
        ),
        "forecast_days": 7,
        "timezone": TIMEZONE,
        "cell_selection": "land",
    }
    return get_json(FORECAST_URL, params=params)


def payload_to_frame(payload: dict, label: str, code: str, weight: float, max_day: int) -> pd.DataFrame:
    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    if not dates:
        return pd.DataFrame()

    n = len(dates)

    def values(name: str):
        data = daily.get(name) or []
        return data if len(data) == n else [None] * n

    frame = pd.DataFrame({
        "date": pd.to_datetime(dates, errors="coerce"),
        "pluie_mm": values("precipitation_sum"),
        "proba_pct": values("precipitation_probability_max"),
        "tmax_c": values("temperature_2m_max"),
        "rafale_kmh": values("wind_gusts_10m_max"),
        "weather_code": values("weather_code"),
    })
    frame["model_label"] = label
    frame["model_code"] = code
    frame["weight"] = weight
    frame["max_day"] = max_day

    for col in ["pluie_mm", "proba_pct", "tmax_c", "rafale_kmh", "weather_code"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame.dropna(subset=["date"])


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    data = pd.DataFrame({"value": values, "weight": weights}).dropna()
    data = data[data["weight"] > 0].sort_values("value")
    if data.empty:
        return math.nan
    cutoff = data["weight"].sum() / 2.0
    return float(data.loc[data["weight"].cumsum() >= cutoff, "value"].iloc[0])


def fetch_first_working_model(lat: float, lon: float, config: dict) -> tuple[pd.DataFrame, str | None, list[str]]:
    errors = []
    for code in config["codes"]:
        try:
            payload = fetch_model_forecast(lat, lon, code)
            frame = payload_to_frame(
                payload,
                config["label"],
                code,
                config["weight"],
                config["max_day"],
            )
            if not frame.empty:
                return frame, code, errors
        except Exception as exc:
            errors.append(f"{code}: {exc}")
    return pd.DataFrame(), None, errors


@st.cache_data(ttl=1800, show_spinner=False)
def load_consensus_forecast(lat: float, lon: float) -> tuple[pd.DataFrame, dict]:
    frames = []
    models_used = []
    model_errors = []

    for config in MODEL_CANDIDATES:
        frame, code, errors = fetch_first_working_model(lat, lon, config)
        model_errors.extend(errors)
        if not frame.empty and code:
            frames.append(frame)
            models_used.append({"label": config["label"], "code": code})

    if not frames:
        return pd.DataFrame(), {
            "quality": "indisponible",
            "models_used": [],
            "errors": model_errors,
        }

    raw = pd.concat(frames, ignore_index=True)
    first_date = raw["date"].min()
    raw["day_index"] = (raw["date"] - first_date).dt.days
    raw = raw[raw["day_index"] <= raw["max_day"]]

    rows = []
    for date, group in raw.groupby("date", sort=True):
        rain_group = group.dropna(subset=["pluie_mm"])
        if rain_group.empty:
            continue

        rain = weighted_median(rain_group["pluie_mm"], rain_group["weight"])
        rain_min = float(rain_group["pluie_mm"].min())
        rain_max = float(rain_group["pluie_mm"].max())
        probability = group["proba_pct"].dropna()
        temperature = group["tmax_c"].dropna()
        gust = group["rafale_kmh"].dropna()
        labels = sorted(rain_group["model_label"].unique())

        rows.append({
            "date": date.date(),
            "pluie_mm": rain,
            "pluie_min_mm": rain_min,
            "pluie_max_mm": rain_max,
            "dispersion_mm": rain_max - rain_min,
            "proba_pct": float(probability.max()) if not probability.empty else math.nan,
            "tmax_c": float(temperature.median()) if not temperature.empty else math.nan,
            "rafale_kmh": float(gust.max()) if not gust.empty else math.nan,
            "nb_modeles": len(labels),
            "modeles": ", ".join(labels),
        })

    result = pd.DataFrame(rows).head(7)
    if result.empty:
        quality = "indisponible"
    elif len(result) < 7:
        quality = "partielle"
    elif int((result["nb_modeles"] >= 2).sum()) >= 5:
        quality = "consolidée"
    else:
        quality = "fragile"

    return result, {
        "quality": quality,
        "models_used": models_used,
        "errors": model_errors,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# INDICATEURS DE SURVEILLANCE
# =============================================================================


def rolling_max(series: pd.Series, window: int) -> float:
    if series.empty:
        return math.nan
    return float(series.rolling(window=window, min_periods=1).sum().max())


def classify_rain(h24: float, h72: float, d7: float) -> tuple[str, list[str]]:
    reasons = []
    for level in ["ROUGE", "ORANGE", "JAUNE"]:
        cfg = THRESHOLDS[level]
        if h24 >= cfg["h24"]:
            reasons.append(f"max 24 h ≥ {cfg['h24']:.0f} mm")
        if h72 >= cfg["h72"]:
            reasons.append(f"max 72 h ≥ {cfg['h72']:.0f} mm")
        if d7 >= cfg["d7"]:
            reasons.append(f"cumul 7 j ≥ {cfg['d7']:.0f} mm")
        if reasons:
            return level, reasons
    return "VERT", ["aucun seuil météo interne dépassé"]


def summarize_forecast(frame: pd.DataFrame) -> dict:
    if frame.empty or frame["pluie_mm"].dropna().empty:
        return {"ok": False, "level": "INDISPONIBLE"}

    rain = frame["pluie_mm"].fillna(0.0)
    high = frame["pluie_max_mm"].fillna(frame["pluie_mm"]).fillna(0.0)
    h24 = float(rain.max())
    h72 = rolling_max(rain, 3)
    d7 = float(rain.sum())
    high7 = float(high.sum())
    level, reasons = classify_rain(h24, h72, d7)
    worst_idx = rain.idxmax()
    worst = frame.loc[worst_idx]

    return {
        "ok": True,
        "level": level,
        "reasons": reasons,
        "h24": h24,
        "h72": h72,
        "d7": d7,
        "high7": high7,
        "worst_date": worst["date"],
        "worst_rain": float(worst["pluie_mm"]),
        "worst_probability": worst.get("proba_pct", math.nan),
    }


def aggregate_communes(mapped: pd.DataFrame) -> pd.DataFrame:
    if mapped.empty:
        return pd.DataFrame()
    rows = []
    for commune, group in mapped.groupby("commune", sort=True):
        rows.append({
            "commune": commune,
            "latitude": float(group["latitude"].mean()),
            "longitude": float(group["longitude"].mean()),
            "ouvrages": ", ".join(sorted(group["ouvrage"].unique())),
            "axes": ", ".join(sorted(group["axe"].unique())),
            "pk_min": float(group["pk_d"].min()),
            "pk_max": float(group["pk_f"].max()),
            "nb_zones": int(len(group)),
        })
    return pd.DataFrame(rows)


# =============================================================================
# INTERFACE
# =============================================================================


st.title("🌧️ Prévisions 7 jours dans les zones à fort risque de glissement")
st.caption(
    "Surveillance météo LGV SEA · prévision multi-modèles · valeurs centrales, "
    "fourchette des modèles et qualité de la prévision."
)

with st.sidebar:
    st.header("Paramètres")
    if st.button("🔄 Actualiser les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Cache météo : 30 min · snapshot : 15 min")
    st.divider()
    st.warning(
        "Les couleurs sont des seuils météo internes d'aide à la surveillance. "
        "Elles ne représentent pas une probabilité de glissement."
    )

try:
    snapshot = fetch_snapshot()
except Exception as exc:
    st.error(f"Snapshot LGV indisponible : {exc}")
    st.stop()

sectors_df, sector_meta = prepare_sectors(snapshot)
if sectors_df.empty:
    st.error(sector_meta.get("error", "Aucun secteur LGV exploitable dans le snapshot."))
    st.stop()

mapped_zones, skipped_zones = identify_risk_locations(
    sectors_df,
    bool(sector_meta.get("axis_field_found")),
)
communes_df = aggregate_communes(mapped_zones)

if communes_df.empty:
    st.error(
        "Aucune zone à risque n'a pu être associée aux communes du snapshot. "
        "Vérifie les champs PK, commune, latitude, longitude et axe."
    )
    st.stop()

# Toutes les communes identifiées sont chargées par défaut, sans classement exposé.
all_communes = communes_df["commune"].tolist()
with st.sidebar:
    selected_communes = st.multiselect(
        "Communes à surveiller",
        options=all_communes,
        default=all_communes,
        help="Toutes les communes associées aux zones de forte densité sont sélectionnées par défaut.",
    )

if skipped_zones:
    with st.expander("⚠️ Zones non localisées automatiquement", expanded=False):
        st.write(
            "Ces zones n'ont pas été attribuées afin d'éviter une commune erronée : "
            + ", ".join(skipped_zones)
            + ". Ajoute un champ d'axe au snapshot pour intégrer les raccordements."
        )

selected_df = communes_df[communes_df["commune"].isin(selected_communes)].copy()
if selected_df.empty:
    st.info("Sélectionne au moins une commune à surveiller.")
    st.stop()

forecasts: dict[str, pd.DataFrame] = {}
results = []
progress = st.progress(0, text="Chargement des prévisions multi-modèles…")

for index, row in enumerate(selected_df.itertuples(index=False), start=1):
    frame, meta = load_consensus_forecast(row.latitude, row.longitude)
    summary = summarize_forecast(frame)
    forecasts[row.commune] = frame
    results.append({
        "commune": row.commune,
        "ouvrages": row.ouvrages,
        "axes": row.axes,
        "nb_zones": row.nb_zones,
        "latitude": row.latitude,
        "longitude": row.longitude,
        "quality": meta.get("quality", "indisponible"),
        "models_used": meta.get("models_used", []),
        **summary,
    })
    progress.progress(index / len(selected_df), text=f"Prévisions : {row.commune}")

progress.empty()

available = [r for r in results if r.get("ok")]
unavailable = [r for r in results if not r.get("ok")]
available.sort(key=lambda r: (-LEVEL_RANK[r["level"]], -r["d7"], r["commune"]))

if available:
    worst_level = max(available, key=lambda r: LEVEL_RANK[r["level"]])["level"]
    if worst_level == "ROUGE":
        st.error("Surveillance météo renforcée sur au moins une commune.")
    elif worst_level == "ORANGE":
        st.warning("Surveillance météo élevée sur au moins une commune.")
    elif worst_level == "JAUNE":
        st.warning("Vigilance météo sur au moins une commune.")
    else:
        st.success("Aucun seuil météo interne dépassé sur les communes actuellement vérifiées.")

if unavailable:
    st.warning(
        "Prévision non vérifiée pour : "
        + ", ".join(r["commune"] for r in unavailable)
        + ". Une donnée indisponible n'est jamais interprétée comme 0 mm."
    )

# Synthèse compacte
if available:
    cols = st.columns(4)
    cols[0].metric("Communes vérifiées", len(available))
    cols[1].metric("Cumul 7 j maximal", f"{max(r['d7'] for r in available):.1f} mm")
    cols[2].metric("Maximum 24 h", f"{max(r['h24'] for r in available):.1f} mm")
    cols[3].metric("Maximum 72 h", f"{max(r['h72'] for r in available):.1f} mm")

st.subheader("Synthèse par commune")
for item in available:
    color = LEVEL_COLOR[item["level"]]
    icon = LEVEL_ICON[item["level"]]
    with st.container(border=True):
        head, quality_col = st.columns([5, 2])
        head.markdown(f"### {icon} {item['commune']}")
        quality_col.markdown(
            f"<div style='text-align:right;color:{color};font-weight:700'>"
            f"{item['level']}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Zones : {item['ouvrages']} · Axe(s) : {item['axes']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cumul central 7 j", f"{item['d7']:.1f} mm")
        c2.metric("Scénario haut 7 j", f"{item['high7']:.1f} mm")
        c3.metric("Maximum 24 h", f"{item['h24']:.1f} mm")
        c4.metric("Maximum 72 h", f"{item['h72']:.1f} mm")

        probability = item.get("worst_probability")
        probability_text = f"{probability:.0f} %" if pd.notna(probability) else "non disponible"
        st.write(
            f"**Journée la plus pluvieuse :** {item['worst_date'].strftime('%d/%m/%Y')} · "
            f"{item['worst_rain']:.1f} mm · probabilité maximale {probability_text}."
        )
        st.caption(
            f"Qualité : {item['quality']} · Motif : " + "; ".join(item["reasons"])
        )

st.subheader("Évolution journalière sur 7 jours")
fig = go.Figure()
palette = ["#2563eb", "#dc2626", "#16a34a", "#f97316", "#7c3aed", "#0891b2", "#be123c", "#4f46e5"]

for idx, item in enumerate(available):
    frame = forecasts.get(item["commune"], pd.DataFrame())
    if frame.empty:
        continue
    custom = frame[["pluie_min_mm", "pluie_max_mm", "proba_pct", "nb_modeles", "modeles"]].to_numpy()
    fig.add_trace(go.Scatter(
        x=frame["date"],
        y=frame["pluie_mm"],
        mode="lines+markers",
        name=item["commune"],
        line=dict(color=palette[idx % len(palette)], width=2.5),
        marker=dict(size=7),
        customdata=custom,
        hovertemplate=(
            "<b>%{fullData.name}</b><br>%{x|%d/%m/%Y}<br>"
            "Valeur centrale : %{y:.1f} mm<br>"
            "Fourchette modèles : %{customdata[0]:.1f} à %{customdata[1]:.1f} mm<br>"
            "Probabilité max : %{customdata[2]:.0f} %<br>"
            "Nombre de modèles : %{customdata[3]}<br>"
            "%{customdata[4]}<extra></extra>"
        ),
    ))

fig.update_layout(
    height=480,
    hovermode="x unified",
    yaxis_title="Précipitations prévues (mm/j)",
    xaxis_title=None,
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(orientation="h", y=1.03, x=0),
    margin=dict(t=90, b=40, l=55, r=25),
)
fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", rangemode="tozero")
fig.update_xaxes(showgrid=False)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.caption(
    "Valeur centrale = médiane pondérée des modèles disponibles. "
    "Le détail au survol affiche la fourchette minimale-maximale et les modèles reçus."
)

st.subheader("Tableau de surveillance")
table_rows = []
for item in available:
    table_rows.append({
        "Commune": item["commune"],
        "Zones / ouvrages": item["ouvrages"],
        "Niveau météo": f"{LEVEL_ICON[item['level']]} {item['level']}",
        "Max 24 h (mm)": round(item["h24"], 1),
        "Max 72 h (mm)": round(item["h72"], 1),
        "Cumul central 7 j (mm)": round(item["d7"], 1),
        "Scénario haut 7 j (mm)": round(item["high7"], 1),
        "Qualité": item["quality"],
    })
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

with st.expander("Méthode et limites"):
    st.markdown(
        """
- **J0 à J3** : priorité à AROME haute résolution lorsqu'il est disponible.
- **J0 à J4** : complément ARPEGE Europe.
- **J0 à J7** : consolidation avec ECMWF IFS et ICON Europe.
- La valeur centrale est une **médiane pondérée**, plus robuste qu'un modèle unique.
- Le scénario haut est la somme des valeurs journalières maximales des modèles disponibles. Il s'agit d'un scénario prudent, pas d'une prévision probabiliste.
- Une journée avec un seul modèle reste visible, mais la qualité globale est dégradée.
- Les seuils colorés sont des seuils météo d'aide à la surveillance et doivent être validés par l'expert Génie Civil.
- Les prévisions ne remplacent ni l'inspection terrain, ni le suivi des fissures, suintements, drains et niveaux piézométriques.
        """
    )

st.caption(
    "Sources météo : Open-Meteo, modèles Météo-France AROME/ARPEGE, ECMWF IFS et DWD ICON selon disponibilité. "
    "Référentiel des zones : dossier glissements MESEA."
)
