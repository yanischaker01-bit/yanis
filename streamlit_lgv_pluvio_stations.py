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
FIRMS_AREA_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{key}/{source}/{bbox}/{days}/{start_date}"
)
FIRMS_SOURCES = ["VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT"]
FIRMS_BBOX = "-0.7,44.75,1.0,47.5"
FIRMS_MAX_CHUNK_DAYS = 5
FIRMS_LOOKBACK_DAYS = 60
HISTORY_MIN_DATE = date(2021, 1, 1)

HEADERS = {
    "User-Agent": "LGV-SEA-Surveillance/5.0 (Streamlit)",
    "Accept": "application/json, application/geo+json;q=0.9, text/csv;q=0.8, */*;q=0.1",
}

RIVERS_LGV = [
    "vienne", "clain", "charente", "boutonne", "seugne", "touvre",
    "dronne", "isle", "dordogne", "garonne", "thouet", "sevre",
    "indre", "cher", "creuse", "ciron", "jalles", "estey",
]

LEVELS = ["NORMAL", "À SURVEILLER", "RENFORCÉ", "PRIORITAIRE", "CRITIQUE"]
LEVEL_COLORS = {
    "NORMAL": "#16a34a",
    "À SURVEILLER": "#eab308",
    "RENFORCÉ": "#f97316",
    "PRIORITAIRE": "#dc2626",
    "CRITIQUE": "#7f1d1d",
    "INDÉTERMINÉ": "#64748b",
}

# Seuils indicatifs pour l'aide à la priorisation. À valider avec les règles MESEA.
THRESHOLDS = {
    "past_24h": (10, 20, 35, 50),
    "past_3d": (20, 40, 65, 90),
    "past_7d": (30, 55, 90, 130),
    "past_30d": (80, 130, 190, 260),
    "future_24h": (10, 20, 35, 50),
    "future_3d": (20, 40, 65, 90),
    "future_7d": (30, 55, 90, 130),
}

# =============================================================================
# UTILITAIRES
# =============================================================================
def normalize(value: object) -> str:
    text = "" if value is None else str(value).lower()
    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return default if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default


def get_json(url: str, params: dict | None = None, timeout: int = 40) -> dict:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                url, params=params, headers=HEADERS,
                timeout=(8, timeout), allow_redirects=True,
            )
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


def threshold_score(value: float, limits: tuple[float, float, float, float], weight: float) -> float:
    if value >= limits[3]:
        return weight
    if value >= limits[2]:
        return weight * 0.75
    if value >= limits[1]:
        return weight * 0.48
    if value >= limits[0]:
        return weight * 0.22
    return 0.0


def level_from_score(score: float) -> str:
    if score >= 80:
        return "CRITIQUE"
    if score >= 60:
        return "PRIORITAIRE"
    if score >= 40:
        return "RENFORCÉ"
    if score >= 20:
        return "À SURVEILLER"
    return "NORMAL"


def max_rolling(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    if len(values) <= window:
        return sum(values)
    return max(sum(values[index:index + window]) for index in range(len(values) - window + 1))


def longest_wet_sequence(values: list[float], minimum: float = 2.0) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value >= minimum else 0
        longest = max(longest, current)
    return longest


def style_chart(fig: go.Figure, height: int = 440, hovermode: str = "x unified") -> None:
    fig.update_layout(
        height=height,
        hovermode=hovermode,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=70, b=65, l=60, r=90),
        font=dict(family="Arial, sans-serif", color="#334155"),
        legend=dict(orientation="h", y=1.13, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#cbd5e1", automargin=True)
    fig.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", automargin=True)
    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, sep=";", decimal=",", encoding="utf-8-sig").encode("utf-8-sig")

# =============================================================================
# SNAPSHOT ET MÉTÉO
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_snapshot() -> dict:
    return get_json(SNAPSHOT_URL, timeout=45)


@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast(lat: float, lon: float) -> pd.DataFrame:
    try:
        daily = get_json(
            FORECAST_URL,
            params={
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "daily": (
                    "precipitation_sum,precipitation_probability_max,"
                    "temperature_2m_max,weather_code,wind_speed_10m_max"
                ),
                "forecast_days": 7,
                "timezone": "Europe/Paris",
            },
        ).get("daily", {})
        frame = pd.DataFrame({
            "date": daily.get("time", []),
            "rain": daily.get("precipitation_sum", []),
            "probability": daily.get("precipitation_probability_max", []),
            "temperature": daily.get("temperature_2m_max", []),
            "weather_code": daily.get("weather_code", []),
            "wind": daily.get("wind_speed_10m_max", []),
        })
        for column in ["rain", "probability", "temperature", "weather_code", "wind"]:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        return frame
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_recent_rain(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)
    return load_history(lat, lon, start_date, end_date)


@st.cache_data(ttl=21600, show_spinner=False)
def load_history(lat: float, lon: float, start_date: date, end_date: date) -> pd.DataFrame:
    try:
        daily = get_json(
            ARCHIVE_URL,
            params={
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "start_date": str(start_date),
                "end_date": str(end_date),
                "daily": "precipitation_sum",
                "timezone": "Europe/Paris",
            },
            timeout=80,
        ).get("daily", {})
        frame = pd.DataFrame({
            "date": daily.get("time", []),
            "rain": daily.get("precipitation_sum", []),
        })
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"])
            frame["rain"] = pd.to_numeric(frame["rain"], errors="coerce").fillna(0)
        return frame
    except Exception:
        return pd.DataFrame()


def fragility_factor(group: pd.DataFrame) -> tuple[float, str]:
    candidates = [
        "fragility_score", "fragilite", "risk_score", "soil_risk",
        "argile_score", "geotechnical_risk", "vulnerability",
    ]
    for column in candidates:
        if column not in group.columns:
            continue
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        if values.empty:
            continue
        raw = float(values.max())
        ratio = raw / 100 if raw > 1 else raw
        ratio = min(1.0, max(0.0, ratio))
        return ratio * 10, f"fragilité patrimoniale {ratio * 100:.0f} %"
    return 0.0, "fragilité patrimoniale non renseignée"


def assess_sector(
    commune: str,
    latitude: float,
    longitude: float,
    asset_score: float,
    asset_note: str,
) -> dict:
    forecast = load_forecast(latitude, longitude)
    recent = load_recent_rain(latitude, longitude, 30)
    if forecast.empty and recent.empty:
        return {"commune": commune, "ok": False, "latitude": latitude, "longitude": longitude}

    future_values = forecast["rain"].tolist() if not forecast.empty else []
    past_values = recent["rain"].tolist() if not recent.empty else []

    metrics = {
        "past_24h": past_values[-1] if past_values else 0.0,
        "past_3d": sum(past_values[-3:]),
        "past_7d": sum(past_values[-7:]),
        "past_30d": sum(past_values[-30:]),
        "future_24h": max(future_values) if future_values else 0.0,
        "future_3d": max_rolling(future_values, 3),
        "future_7d": sum(future_values),
    }

    persistence_past = longest_wet_sequence(past_values[-14:])
    persistence_future = longest_wet_sequence(future_values)
    combined_14d = metrics["past_7d"] + metrics["future_7d"]
    storm = bool((forecast["weather_code"] >= 95).any()) if not forecast.empty else False
    max_wind = float(forecast["wind"].max()) if not forecast.empty else 0.0
    max_probability = float(forecast["probability"].max()) if not forecast.empty else 0.0

    score = (
        threshold_score(metrics["past_7d"], THRESHOLDS["past_7d"], 13)
        + threshold_score(metrics["past_30d"], THRESHOLDS["past_30d"], 12)
        + threshold_score(metrics["future_24h"], THRESHOLDS["future_24h"], 15)
        + threshold_score(metrics["future_3d"], THRESHOLDS["future_3d"], 20)
        + threshold_score(metrics["future_7d"], THRESHOLDS["future_7d"], 18)
        + min(7, persistence_future * 1.5)
        + min(5, persistence_past)
        + min(5, max_wind / 20)
        + (5 if storm else 0)
        + asset_score
    )
    score = min(100.0, score)
    level = level_from_score(score)

    if not forecast.empty:
        peak_index = forecast["rain"].idxmax()
        peak_date = str(forecast.loc[peak_index, "date"])
    else:
        peak_date = "indisponible"

    reasons = []
    if metrics["past_7d"] >= 40:
        reasons.append(f"sol déjà sollicité : {metrics['past_7d']:.1f} mm sur 7 j")
    if metrics["past_30d"] >= 130:
        reasons.append(f"antécédent 30 j élevé : {metrics['past_30d']:.1f} mm")
    if metrics["future_3d"] >= 40:
        reasons.append(f"fort cumul glissant prévu sur 3 j : {metrics['future_3d']:.1f} mm")
    if metrics["future_7d"] >= 55:
        reasons.append(f"cumul prévu sur 7 j : {metrics['future_7d']:.1f} mm")
    if metrics["future_24h"] >= 20:
        reasons.append(f"pic journalier prévu : {metrics['future_24h']:.1f} mm")
    if persistence_future >= 4:
        reasons.append(f"pluie persistante prévue : {persistence_future} jours")
    if storm:
        reasons.append("signal orageux")
    if asset_score > 0:
        reasons.append(asset_note)

    mechanism = []
    if metrics["future_24h"] >= 20:
        mechanism.extend(["ruissellement", "ravinement", "mise en charge drainage"])
    if metrics["past_7d"] + metrics["future_7d"] >= 90:
        mechanism.extend(["saturation progressive", "perte de portance", "instabilité de talus"])
    if persistence_future >= 4:
        mechanism.extend(["colmatage des évacuations", "venues d'eau"])
    mechanism_text = ", ".join(dict.fromkeys(mechanism)) or "sollicitation faible"

    return {
        "commune": commune,
        "ok": True,
        "latitude": latitude,
        "longitude": longitude,
        "score": round(score, 1),
        "level": level,
        **{key: round(value, 1) for key, value in metrics.items()},
        "combined_14d": round(combined_14d, 1),
        "persistence_past": persistence_past,
        "persistence_future": persistence_future,
        "probability": round(max_probability),
        "wind": round(max_wind, 1),
        "storm": storm,
        "peak_date": peak_date,
        "reasons": "; ".join(reasons) if reasons else "aucun seuil notable",
        "mechanisms": mechanism_text,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def assess_all_sectors(sectors: pd.DataFrame) -> pd.DataFrame:
    required = {"commune_name", "latitude", "longitude"}
    if sectors.empty or not required.issubset(sectors.columns):
        return pd.DataFrame()

    jobs = []
    clean = sectors.dropna(subset=list(required))
    for commune, group in clean.groupby("commune_name"):
        latitude = float(pd.to_numeric(group["latitude"], errors="coerce").mean())
        longitude = float(pd.to_numeric(group["longitude"], errors="coerce").mean())
        asset_score, asset_note = fragility_factor(group)
        jobs.append((str(commune), latitude, longitude, asset_score, asset_note))

    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(assess_sector, *job): job[0] for job in jobs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                results.append({"commune": futures[future], "ok": False})

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
    if payload is None:
        return [], False

    levels_number = {0: "VERT", 1: "VERT", 2: "JAUNE", 3: "ORANGE", 4: "ROUGE"}
    levels_name = {"vert": "VERT", "jaune": "JAUNE", "orange": "ORANGE", "rouge": "ROUGE"}
    alerts = []
    for feature in payload["features"]:
        properties = feature.get("properties", {}) if isinstance(feature, dict) else {}
        name = str(
            properties.get("NomEntVigiCru")
            or properties.get("LbEntVigiCru")
            or properties.get("lbentcru")
            or ""
        ).strip()
        if not name or not any(river in normalize(name) for river in RIVERS_LGV):
            continue
        raw_level = properties.get(
            "NivSituVigiCruEnt",
            properties.get("NivInfViCr", properties.get("NivVigiCru")),
        )
        try:
            level = levels_number.get(int(float(raw_level)))
        except (TypeError, ValueError):
            level = levels_name.get(normalize(raw_level))
        if level:
            alerts.append({"name": name, "level": level})
    unique = {(item["name"], item["level"]): item for item in alerts}
    return list(unique.values()), True

# =============================================================================
# NASA FIRMS
# =============================================================================
def get_firms_key() -> str | None:
    try:
        key = st.secrets.get("FIRMS_MAP_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass
    key = os.getenv("FIRMS_MAP_KEY") or os.getenv("FIRMS_KEY")
    return key.strip() if key else None


def date_chunks(start_date: date, end_date: date) -> list[tuple[date, int]]:
    chunks = []
    cursor = start_date
    while cursor <= end_date:
        days = min(FIRMS_MAX_CHUNK_DAYS, (end_date - cursor).days + 1)
        chunks.append((cursor, days))
        cursor += timedelta(days=days)
    return chunks


@st.cache_data(ttl=600, show_spinner=False)
def fetch_firms_chunk(
    key: str,
    source: str,
    start_date: date,
    days: int,
) -> tuple[pd.DataFrame, str | None, str]:
    url = FIRMS_AREA_URL.format(
        key=key, source=source, bbox=FIRMS_BBOX,
        days=days, start_date=start_date,
    )
    try:
        response = requests.get(url, headers=HEADERS, timeout=(10, 50))
        status = f"HTTP {response.status_code}"
        response.raise_for_status()
        text = response.text.strip()
        preview = text[:300].lower()
        if "invalid" in preview or "map key" in preview and "error" in preview:
            return pd.DataFrame(), "invalid_key", status
        if not text or "<html" in preview:
            return pd.DataFrame(), "unexpected_response", status
        frame = pd.read_csv(io.StringIO(text))
        if frame.empty:
            return pd.DataFrame(), None, status
        if not {"latitude", "longitude"}.issubset(frame.columns):
            return pd.DataFrame(), "unexpected_response", status
        frame["source"] = source
        return frame, None, status
    except requests.Timeout:
        return pd.DataFrame(), "timeout", "délai dépassé"
    except requests.HTTPError:
        return pd.DataFrame(), "http_error", f"HTTP {response.status_code}"
    except Exception as exc:
        return pd.DataFrame(), "fetch_failed", str(exc)[:100]


def load_firms_period(
    start_date: date,
    end_date: date,
    sources: tuple[str, ...],
) -> tuple[pd.DataFrame, str | None, list[dict]]:
    key = get_firms_key()
    if not key:
        return pd.DataFrame(), "missing_key", []

    tasks = [
        (source, chunk_start, days)
        for source in sources
        for chunk_start, days in date_chunks(start_date, end_date)
    ]
    frames = []
    diagnostics = []
    with ThreadPoolExecutor(max_workers=min(4, len(tasks) or 1)) as pool:
        futures = {
            pool.submit(fetch_firms_chunk, key, source, chunk_start, days):
            (source, chunk_start, days)
            for source, chunk_start, days in tasks
        }
        for future in as_completed(futures):
            source, chunk_start, days = futures[future]
            frame, error, detail = future.result()
            diagnostics.append({
                "source": source,
                "début": str(chunk_start),
                "jours": days,
                "statut": "OK" if error is None else error,
                "détail": detail,
                "lignes": len(frame),
            })
            if error == "invalid_key":
                return pd.DataFrame(), "invalid_key", diagnostics
            if error is None and not frame.empty:
                frames.append(frame)

    succeeded = sum(item["statut"] == "OK" for item in diagnostics)
    if succeeded == 0:
        return pd.DataFrame(), "fetch_failed", diagnostics
    if not frames:
        return pd.DataFrame(), None, diagnostics

    output = pd.concat(frames, ignore_index=True)
    subset = [
        column for column in
        ["latitude", "longitude", "acq_date", "acq_time", "satellite", "source"]
        if column in output.columns
    ]
    if subset:
        output = output.drop_duplicates(subset=subset)
    if "acq_date" in output:
        acquisition_dates = pd.to_datetime(output["acq_date"], errors="coerce").dt.date
        output = output[(acquisition_dates >= start_date) & (acquisition_dates <= end_date)]
    return output, None, diagnostics


def confidence_category(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"l", "low", "faible"}:
        return "Faible"
    if text in {"n", "nominal", "nominale"}:
        return "Nominale"
    if text in {"h", "high", "élevée", "elevee"}:
        return "Élevée"
    try:
        numeric = float(text)
        if numeric >= 80:
            return "Élevée"
        if numeric >= 30:
            return "Nominale"
        return "Faible"
    except ValueError:
        return "Non renseignée"


def lgv_line(lines: object) -> list[tuple[float, float]]:
    if not isinstance(lines, list) or not lines or not isinstance(lines[0], list):
        return []
    return [
        (to_float(point.get("lat")), to_float(point.get("lon")))
        for point in lines[0]
        if isinstance(point, dict)
        and point.get("lat") is not None
        and point.get("lon") is not None
    ]


def distance_to_line_km(latitude: float, longitude: float, line: list[tuple[float, float]]) -> float:
    if len(line) < 2:
        return float("inf")
    best = float("inf")
    for (lat1, lon1), (lat2, lon2) in zip(line, line[1:]):
        middle_latitude = (lat1 + lat2) / 2
        kx = 111.32 * math.cos(math.radians(middle_latitude))
        ky = 111.32
        x1, y1 = lon1 * kx, lat1 * ky
        x2, y2 = lon2 * kx, lat2 * ky
        xp, yp = longitude * kx, latitude * ky
        dx, dy = x2 - x1, y2 - y1
        denominator = dx * dx + dy * dy
        ratio = 0.0 if denominator == 0 else ((xp - x1) * dx + (yp - y1) * dy) / denominator
        ratio = max(0.0, min(1.0, ratio))
        nearest_x, nearest_y = x1 + ratio * dx, y1 + ratio * dy
        best = min(best, math.hypot(xp - nearest_x, yp - nearest_y))
    return best

# =============================================================================
# INTERFACE
# =============================================================================
st.set_page_config(
    page_title="Surveillance LGV SEA",
    page_icon="🛤️",
    layout="wide",
)
st.title("🛤️ Surveillance hydrométéorologique LGV SEA")
st.caption(
    "Outil d'aide à la priorisation des inspections : fortes pluies passées et prévues, "
    "persistance, saturation potentielle, crues et détections thermiques."
)

header_left, header_right = st.columns([6, 1])
if header_right.button("🔄 Actualiser", use_container_width=True):
    st.cache_data.clear()
    st.session_state.pop("firms_result", None)
    st.rerun()

try:
    snapshot = load_snapshot()
except Exception as exc:
    st.error(f"Snapshot LGV indisponible : {exc}")
    st.stop()

sector_payload = snapshot.get("sectors", {})
sector_records = sector_payload.get("sectors", []) if isinstance(sector_payload, dict) else []
sectors = pd.DataFrame(sector_records)
for column in ["latitude", "longitude", "pk_km"]:
    if column in sectors:
        sectors[column] = pd.to_numeric(sectors[column], errors="coerce")

required_columns = {"commune_name", "latitude", "longitude"}
if sectors.empty or not required_columns.issubset(sectors.columns):
    st.error("Le snapshot ne contient pas les colonnes commune_name, latitude et longitude.")
    st.stop()

communes = sorted(sectors["commune_name"].dropna().astype(str).unique())
coordinates = (
    sectors.dropna(subset=list(required_columns))
    .groupby("commune_name")[["latitude", "longitude"]]
    .mean()
)

with st.sidebar:
    st.header("Pilotage")
    minimum_level = st.selectbox(
        "Niveau minimal",
        LEVELS,
        index=1,
    )
    view_mode = st.radio(
        "Vue des secteurs",
        ["Tous", "Top 10", "Communes sélectionnées"],
    )
    selected_communes = st.multiselect(
        "Communes à comparer",
        communes,
        default=communes[: min(6, len(communes))],
    )
    detailed_commune = st.selectbox("Fiche détaillée", communes)

    st.divider()
    st.subheader("Historique depuis 2021")
    history_start = st.date_input(
        "Début historique", HISTORY_MIN_DATE,
        min_value=HISTORY_MIN_DATE, max_value=date.today(),
    )
    history_end = st.date_input(
        "Fin historique", date.today() - timedelta(days=5),
        min_value=HISTORY_MIN_DATE, max_value=date.today(),
    )
    history_group = st.radio("Regroupement", ["Mensuel", "Annuel"], horizontal=True)

    st.divider()
    st.subheader("NASA FIRMS")
    firms_mode = st.radio("Période FIRMS", ["24 h", "3 jours", "5 jours", "Personnalisée"])
    today_utc = datetime.now(timezone.utc).date()
    if firms_mode == "Personnalisée":
        firms_start = st.date_input(
            "Début FIRMS", today_utc - timedelta(days=6),
            min_value=today_utc - timedelta(days=FIRMS_LOOKBACK_DAYS),
            max_value=today_utc,
        )
        firms_end = st.date_input(
            "Fin FIRMS", today_utc,
            min_value=today_utc - timedelta(days=FIRMS_LOOKBACK_DAYS),
            max_value=today_utc,
        )
    else:
        period_days = {"24 h": 1, "3 jours": 3, "5 jours": 5}[firms_mode]
        firms_end = today_utc
        firms_start = today_utc - timedelta(days=period_days - 1)

    firms_sources = st.multiselect(
        "Satellites",
        FIRMS_SOURCES,
        default=FIRMS_SOURCES,
        format_func=lambda value: value.replace("VIIRS_", "").replace("_NRT", ""),
    )
    firms_distance = st.slider("Distance à la LGV", 100, 5000, 500, 100)
    firms_confidences = st.multiselect(
        "Confiance",
        ["Faible", "Nominale", "Élevée", "Non renseignée"],
        default=["Nominale", "Élevée"],
    )
    run_firms = st.button("🔥 Lancer la recherche FIRMS", use_container_width=True)

# Analyse générale
st.subheader("1. Secteurs ayant subi ou susceptibles de subir de fortes pluies")
with st.spinner("Analyse des communes : pluie passée, prévision et persistance..."):
    assessment = assess_all_sectors(sectors)

if assessment.empty or not assessment.get("ok", pd.Series(dtype=bool)).any():
    st.error("Analyse indisponible. Vérifie Open-Meteo et le snapshot.")
    st.stop()

valid_all = assessment[assessment["ok"] == True].copy()
summary_columns = st.columns(5)
for column, level in zip(summary_columns, reversed(LEVELS)):
    column.metric(level, int((valid_all["level"] == level).sum()))

filtered = valid_all[
    valid_all["level"].map(LEVELS.index) >= LEVELS.index(minimum_level)
].copy()
if view_mode == "Top 10":
    filtered = filtered.head(10)
elif view_mode == "Communes sélectionnées":
    filtered = filtered[filtered["commune"].isin(selected_communes)]

if filtered.empty:
    st.success("Aucun secteur ne correspond aux filtres sélectionnés.")
else:
    chart_data = filtered.head(40)
    figure = go.Figure()
    figure.add_bar(
        x=chart_data["commune"],
        y=chart_data["past_7d"],
        name="Pluie passée 7 j",
        marker_color="#64748b",
    )
    figure.add_bar(
        x=chart_data["commune"],
        y=chart_data["future_7d"],
        name="Pluie prévue 7 j",
        marker_color="#60a5fa",
    )
    figure.add_scatter(
        x=chart_data["commune"],
        y=chart_data["future_3d"],
        name="Max glissant prévu 3 j",
        mode="markers+text",
        marker=dict(
            color=[LEVEL_COLORS[level] for level in chart_data["level"]],
            size=[10 + score / 10 for score in chart_data["score"]],
            line=dict(color="white", width=1),
        ),
        text=[f"{value:.0f}" if score >= 40 else "" for value, score in zip(chart_data["future_3d"], chart_data["score"])],
        textposition="top center",
        customdata=chart_data[["score", "level", "reasons", "mechanisms"]],
        hovertemplate=(
            "%{x}<br>Max prévu 3 j : %{y:.1f} mm"
            "<br>Score : %{customdata[0]}/100"
            "<br>Niveau : %{customdata[1]}"
            "<br>Facteurs : %{customdata[2]}"
            "<br>Mécanismes : %{customdata[3]}<extra></extra>"
        ),
    )
    figure.update_layout(barmode="stack", yaxis_title="Pluie (mm)", xaxis_tickangle=-45)
    style_chart(figure, 570, "closest")

    display = filtered.rename(columns={
        "commune": "Secteur / commune",
        "level": "Niveau",
        "score": "Indice /100",
        "past_24h": "Passé 24 h",
        "past_3d": "Passé 3 j",
        "past_7d": "Passé 7 j",
        "past_30d": "Passé 30 j",
        "future_24h": "Pic prévu 24 h",
        "future_3d": "Max prévu 3 j",
        "future_7d": "Prévu 7 j",
        "combined_14d": "Passé 7 j + futur 7 j",
        "persistence_future": "Persistance future",
        "peak_date": "Date du pic",
        "reasons": "Facteurs",
        "mechanisms": "Mécanismes possibles",
    })
    table_columns = [
        "Secteur / commune", "Niveau", "Indice /100",
        "Passé 24 h", "Passé 3 j", "Passé 7 j", "Passé 30 j",
        "Pic prévu 24 h", "Max prévu 3 j", "Prévu 7 j",
        "Passé 7 j + futur 7 j", "Persistance future", "Date du pic",
        "Facteurs", "Mécanismes possibles",
    ]
    st.dataframe(display[table_columns], use_container_width=True, hide_index=True, height=470)
    st.download_button(
        "⬇️ Exporter l'analyse CSV",
        data=csv_bytes(display[table_columns]),
        file_name=f"surveillance_lgv_{date.today()}.csv",
        mime="text/csv",
    )

# Comparaison
st.divider()
st.subheader("2. Comparaison des communes")
comparison_names = selected_communes if selected_communes else [detailed_commune]
comparison = valid_all[valid_all["commune"].isin(comparison_names)].copy()
if comparison.empty:
    st.info("Sélectionne au moins une commune dans la barre latérale.")
else:
    metric_choice = st.radio(
        "Indicateur comparé",
        ["Pluie passée", "Pluie prévue", "Sollicitation combinée"],
        horizontal=True,
    )
    metric_columns = {
        "Pluie passée": [("past_7d", "Passé 7 j"), ("past_30d", "Passé 30 j")],
        "Pluie prévue": [("future_3d", "Prévu max 3 j"), ("future_7d", "Prévu 7 j")],
        "Sollicitation combinée": [("combined_14d", "Passé 7 j + futur 7 j"), ("score", "Indice /100")],
    }[metric_choice]
    compare_figure = go.Figure()
    for column_name, label in metric_columns:
        compare_figure.add_bar(
            x=comparison["commune"],
            y=comparison[column_name],
            name=label,
            text=[f"{value:.1f}" for value in comparison[column_name]],
            textposition="outside",
        )
    compare_figure.update_layout(barmode="group", yaxis_title="Valeur")
    style_chart(compare_figure, 430, "closest")

# Fiche détaillée et historique
st.divider()
st.subheader(f"3. Fiche secteur : {detailed_commune}")
sector_result = valid_all[valid_all["commune"] == detailed_commune]
if not sector_result.empty:
    item = sector_result.iloc[0]
    cards = st.columns(7)
    cards[0].metric("Niveau", item["level"])
    cards[1].metric("Indice", f"{item['score']:.0f}/100")
    cards[2].metric("Passé 7 j", f"{item['past_7d']:.1f} mm")
    cards[3].metric("Passé 30 j", f"{item['past_30d']:.1f} mm")
    cards[4].metric("Prévu 3 j", f"{item['future_3d']:.1f} mm")
    cards[5].metric("Prévu 7 j", f"{item['future_7d']:.1f} mm")
    cards[6].metric("Pic prévu", f"{item['future_24h']:.1f} mm")
    st.info(f"Facteurs : {item['reasons']}")
    st.warning(f"Mécanismes à contrôler : {item['mechanisms']}")

    point = coordinates.loc[detailed_commune]
    daily_forecast = load_forecast(float(point["latitude"]), float(point["longitude"]))
    if not daily_forecast.empty:
        detail_figure = go.Figure()
        detail_figure.add_bar(
            x=daily_forecast["date"], y=daily_forecast["rain"],
            name="Pluie prévue (mm)",
            marker_color=[
                "#dc2626" if value >= 35 else "#f97316" if value >= 20 else "#2563eb"
                for value in daily_forecast["rain"]
            ],
            text=[f"{value:.1f}" for value in daily_forecast["rain"]],
            textposition="outside",
        )
        detail_figure.add_scatter(
            x=daily_forecast["date"], y=daily_forecast["probability"],
            name="Probabilité (%)", yaxis="y2", mode="lines+markers",
            line=dict(color="#6366f1", dash="dot"),
        )
        detail_figure.update_layout(
            yaxis=dict(title="Pluie (mm)"),
            yaxis2=dict(title="Probabilité (%)", overlaying="y", side="right", range=[0, 100], showgrid=False),
        )
        style_chart(detail_figure)

st.subheader("Historique pluviométrique depuis 2021")
if history_start > history_end:
    st.error("La date de début historique doit précéder la date de fin.")
else:
    point = coordinates.loc[detailed_commune]
    with st.spinner("Chargement de l'historique..."):
        history = load_history(
            float(point["latitude"]), float(point["longitude"]),
            history_start, history_end,
        )
    if history.empty:
        st.warning("Historique indisponible.")
    else:
        history["period"] = (
            history["date"].dt.to_period("M").astype(str)
            if history_group == "Mensuel"
            else history["date"].dt.year.astype(str)
        )
        grouped_history = history.groupby("period", as_index=False)["rain"].sum()
        peak_day = history.loc[history["rain"].idxmax()]
        history_figure = go.Figure(go.Bar(
            x=grouped_history["period"],
            y=grouped_history["rain"],
            marker_color=[
                "#dc2626" if value == grouped_history["rain"].max() else "#2563eb"
                for value in grouped_history["rain"]
            ],
            text=[f"{value:.0f}" for value in grouped_history["rain"]],
            textposition="outside",
        ))
        history_figure.update_layout(yaxis_title="Cumul de pluie (mm)")
        style_chart(history_figure, 450)
        history_cards = st.columns(3)
        history_cards[0].metric("Cumul période", f"{history['rain'].sum():.1f} mm")
        history_cards[1].metric("Pic journalier", f"{peak_day['rain']:.1f} mm")
        history_cards[2].metric("Date du pic", peak_day["date"].strftime("%d/%m/%Y"))

# Crues
st.divider()
st.subheader("4. Vigicrues")
river_alerts, river_ok = load_vigicrues()
active_rivers = [item for item in river_alerts if item["level"] in {"JAUNE", "ORANGE", "ROUGE"}]
if not river_ok:
    st.warning("Vigicrues injoignable : statut des crues non vérifié.")
elif not active_rivers:
    st.success("Vigicrues vérifié : aucune vigilance active sur les cours d'eau suivis.")
else:
    for alert in active_rivers:
        st.warning(f"{alert['name']} : vigilance {alert['level'].lower()}")

# FIRMS, exécuté uniquement au clic
st.divider()
st.subheader("5. NASA FIRMS et carte opérationnelle")
if run_firms:
    if firms_start > firms_end:
        st.error("La date de début FIRMS doit précéder la date de fin.")
    elif not firms_sources:
        st.error("Sélectionne au moins un satellite FIRMS.")
    else:
        with st.spinner("Interrogation de NASA FIRMS..."):
            st.session_state["firms_result"] = {
                "frame_error_diag": load_firms_period(
                    firms_start, firms_end, tuple(firms_sources)
                ),
                "start": firms_start,
                "end": firms_end,
                "distance": firms_distance,
                "confidences": tuple(firms_confidences),
            }

firms_saved = st.session_state.get("firms_result")
firms_frame = pd.DataFrame()
firms_error = None
firms_diagnostics: list[dict] = []
filtered_fires: list[dict] = []

if firms_saved:
    firms_frame, firms_error, firms_diagnostics = firms_saved["frame_error_diag"]
    saved_start = firms_saved["start"]
    saved_end = firms_saved["end"]
    saved_distance = firms_saved["distance"]
    saved_confidences = firms_saved["confidences"]
    line = lgv_line(snapshot.get("lgv_lines", []))

    if not firms_frame.empty and line:
        for _, detection in firms_frame.iterrows():
            category = confidence_category(detection.get("confidence", ""))
            if saved_confidences and category not in saved_confidences:
                continue
            latitude = to_float(detection.get("latitude"), float("nan"))
            longitude = to_float(detection.get("longitude"), float("nan"))
            if math.isnan(latitude) or math.isnan(longitude):
                continue
            distance_m = distance_to_line_km(latitude, longitude, line) * 1000
            if distance_m <= saved_distance:
                filtered_fires.append({
                    **detection.to_dict(),
                    "confidence_label": category,
                    "distance_m": round(distance_m),
                })

    if firms_error == "missing_key":
        st.info("FIRMS non activé : ajoute FIRMS_MAP_KEY dans les secrets Streamlit.")
    elif firms_error == "invalid_key":
        st.error("Clé FIRMS refusée. Vérifie FIRMS_MAP_KEY sans guillemets supplémentaires.")
    elif firms_error:
        st.warning("FIRMS n'a pas pu être vérifié. Consulte le diagnostic ci-dessous.")
    elif filtered_fires:
        st.error(f"{len(filtered_fires)} détection(s) thermique(s) dans le périmètre filtré.")
    else:
        st.success("FIRMS vérifié : aucune détection correspondant aux filtres.")

    if firms_diagnostics:
        with st.expander("Diagnostic FIRMS"):
            st.dataframe(pd.DataFrame(firms_diagnostics), use_container_width=True, hide_index=True)
else:
    st.info("Configure les filtres dans la barre latérale, puis clique sur Lancer la recherche FIRMS.")

# Carte
line = lgv_line(snapshot.get("lgv_lines", []))
selected_point = coordinates.loc[detailed_commune]
map_object = folium.Map(
    location=[float(selected_point["latitude"]), float(selected_point["longitude"])],
    zoom_start=10,
    tiles=None,
    control_scale=True,
)
folium.TileLayer("CartoDB positron", name="Carte claire", show=False).add_to(map_object)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    name="Satellite",
    attr="Esri, Maxar, Earthstar Geographics, GIS User Community",
    show=True,
    max_zoom=22,
).add_to(map_object)
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    name="Noms et limites",
    attr="Esri",
    overlay=True,
    show=True,
).add_to(map_object)

if line:
    folium.PolyLine(line, color="#facc15", weight=4, opacity=0.95, tooltip="LGV SEA").add_to(map_object)

priority_group = folium.FeatureGroup(name="Sollicitation hydrique", show=True)
for _, item in valid_all.iterrows():
    folium.CircleMarker(
        location=[item["latitude"], item["longitude"]],
        radius=5 + item["score"] / 18,
        color=LEVEL_COLORS[item["level"]],
        fill=True,
        fill_color=LEVEL_COLORS[item["level"]],
        fill_opacity=0.85,
        tooltip=(
            f"{item['commune']} | {item['level']} | indice {item['score']}/100 | "
            f"passé 7 j {item['past_7d']} mm | prévu 7 j {item['future_7d']} mm"
        ),
    ).add_to(priority_group)
priority_group.add_to(map_object)

if filtered_fires:
    fire_group = folium.FeatureGroup(name="Détections FIRMS", show=True)
    for detection in filtered_fires:
        folium.CircleMarker(
            location=[to_float(detection.get("latitude")), to_float(detection.get("longitude"))],
            radius=9,
            color="#7f1d1d",
            fill=True,
            fill_color="#dc2626",
            fill_opacity=0.9,
            tooltip=(
                f"FIRMS | {detection['distance_m']} m de la LGV | "
                f"{detection.get('acq_date', '')} | {detection['confidence_label']}"
            ),
        ).add_to(fire_group)
    fire_group.add_to(map_object)

folium.LayerControl(collapsed=False).add_to(map_object)
plugins.Fullscreen(position="topleft", title="Plein écran", title_cancel="Quitter").add_to(map_object)
st_folium(map_object, use_container_width=True, height=650, returned_objects=[])

if filtered_fires:
    fire_table = pd.DataFrame(filtered_fires)
    columns = [
        column for column in [
            "acq_date", "acq_time", "satellite", "source", "confidence_label",
            "frp", "distance_m", "latitude", "longitude",
        ] if column in fire_table.columns
    ]
    st.dataframe(fire_table[columns], use_container_width=True, hide_index=True)

# Aide à la décision
with st.expander("Lecture génie civil et contrôles suggérés", expanded=False):
    st.markdown("""
### Ce que l'indice cherche à repérer

- **Intensité sur 24 h** : ruissellement concentré, ravinement, débordement ou mise en charge des fossés, buses et descentes d'eau.
- **Cumul glissant sur 3 jours** : sollicitation prolongée des talus, remblais, déblais et dispositifs de drainage.
- **Cumul sur 7 et 30 jours** : humidification et saturation progressive, perte de portance et augmentation possible des pressions interstitielles.
- **Pluie passée + pluie future** : secteur déjà humide confronté à un nouvel épisode.
- **Persistance** : plusieurs jours pluvieux peuvent être importants même sans pic extrême.

### Contrôles prioritaires pour un secteur renforcé ou supérieur

1. Fossés, cunettes, buses, traversées hydrauliques, descentes d'eau et exutoires.
2. Crêtes et pieds de talus, ravinement, fissures, bombements, glissements et matériaux déplacés.
3. Plateforme et voie : affaissement, défaut géométrique, ballast pollué, pompage, zones humides et venues d'eau.
4. Ouvrages : obstruction, mise en charge, affouillement en entrée ou sortie et érosion localisée.
5. Croisement avec les tournées, incidents antérieurs, instrumentation, géologie, argiles et patrimoine de drainage.

L'application signale une **sollicitation hydrique** et non la certitude d'un effondrement. Toute décision de sécurité ou d'exploitation doit suivre les procédures internes et l'expertise terrain.
""")

st.caption(
    "Sources : Open-Meteo, Vigicrues, NASA FIRMS, Esri World Imagery et snapshot LGV SEA. "
    "Outil d'aide à la surveillance, non substitutif aux procédures de maintenance et de sécurité."
)
