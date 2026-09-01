from __future__ import annotations

import io
import math
import os
import time
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium


# =============================================================================
# CONFIGURATION
# =============================================================================

SNAPSHOT_URL = (
    "https://yanischaker01-bit.github.io/yanis/"
    "reports/streamlit_snapshot_latest.json"
)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

VIGICRUES_GEOJSON_URL = (
    "https://www.vigicrues.gouv.fr/services/1/InfoVigiCru.geojson/"
)

FIRMS_AREA_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{key}/{source}/{area}/{day_range}/{date}"
)
FIRMS_SOURCES = [
    "VIIRS_NOAA21_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_SNPP_NRT",
]
FIRMS_BBOX = "-0.7,44.75,1.0,47.5"
FIRMS_RADIUS_KM = 0.5
FIRMS_MAX_DAY_RANGE = 10
FIRMS_MAX_LOOKBACK_DAYS = 60
FIRMS_CONF_LABELS = {"l": "Faible", "n": "Nominale", "h": "Élevée"}

DEPS = {
    "37": {"nom": "Indre-et-Loire", "lat": 47.38, "lon": 0.69},
    "86": {"nom": "Vienne", "lat": 46.58, "lon": 0.34},
    "79": {"nom": "Deux-Sèvres", "lat": 46.32, "lon": -0.46},
    "16": {"nom": "Charente", "lat": 45.65, "lon": 0.16},
    "17": {"nom": "Charente-Maritime", "lat": 45.75, "lon": -0.63},
    "33": {"nom": "Gironde", "lat": 44.84, "lon": -0.58},
}

LEVEL_COLOR = {
    "ROUGE": "#dc2626",
    "ORANGE": "#ea580c",
    "JAUNE": "#eab308",
    "VERT": "#16a34a",
    "INFO": "#3b82f6",
}
LEVEL_RANK = {"ROUGE": 4, "ORANGE": 3, "JAUNE": 2, "VERT": 1, "INFO": 0}
LEVEL_LABEL = {
    "ROUGE": "Rouge",
    "ORANGE": "Orange",
    "JAUNE": "Jaune",
    "VERT": "Vert",
    "INFO": "Info",
}
LEVEL_BADGE = {
    "ROUGE": "red",
    "ORANGE": "orange",
    "JAUNE": "yellow",
    "VERT": "green",
    "INFO": "gray",
}
ALERT_CFG = {
    "ORAGE": ("⛈️", "Orage"),
    "CANICULE": ("🌡️", "Canicule"),
    "INCENDIE": ("🔥", "Incendie"),
    "INONDATION": ("🌊", "Inondation"),
    "VENT": ("💨", "Vent violent"),
    "VIGICRUE": ("🏞️", "Vigilance crue"),
    "FEU_FIRMS": ("🔥", "Détection FIRMS"),
}

_RIVERS_RAW = [
    "vienne", "clain", "charente", "boutonne", "seugne", "touvre",
    "dronne", "isle", "dordogne", "garonne", "thouet", "sevre",
    "indre", "cher", "creuse", "ciron", "jalles", "estey",
    "leyre", "midouze", "brion", "anglin",
]

CHART_COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "teal": "#0f766e",
    "orange": "#f97316",
    "red": "#dc2626",
    "slate": "#475569",
}

HTTP_HEADERS = {
    "Accept": "application/json, application/geo+json;q=0.9, */*;q=0.1",
    "User-Agent": (
        "Mozilla/5.0 LGV-PluvioStations/2.0 "
        "(https://lgvpluviostations.streamlit.app/)"
    ),
}


# =============================================================================
# OUTILS
# =============================================================================

def _normalize(value: object) -> str:
    text = "" if value is None else str(value)
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(char) != "Mn"
    )


RIVERS_LGV = [_normalize(river) for river in _RIVERS_RAW]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_df(records: object) -> pd.DataFrame:
    if isinstance(records, list):
        try:
            return pd.DataFrame(records)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def request_json(url: str, *, params: dict | None = None, timeout: int = 25) -> dict:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HTTP_HEADERS,
                timeout=(8, timeout),
                allow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Réponse JSON inattendue")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Service indisponible : {last_error}")


def rain_risk(max_mm: float) -> tuple[str, str, str]:
    if max_mm >= 60:
        return "ROUGE", "#dc2626", "🔴"
    if max_mm >= 30:
        return "ORANGE", "#ea580c", "🟠"
    if max_mm >= 10:
        return "JAUNE", "#eab308", "🟡"
    return "VERT", "#16a34a", "🟢"


def rain_color_mm(mm: float) -> str:
    return rain_risk(safe_float(mm))[1]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    value = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.asin(min(1.0, math.sqrt(value)))


def alert_card(alert: dict) -> None:
    level = alert.get("level", "INFO")
    alert_type = alert.get("type", "")
    icon, label = ALERT_CFG.get(alert_type, ("ℹ️", alert_type or "Information"))
    color = LEVEL_COLOR.get(level, LEVEL_COLOR["INFO"])
    message = str(alert.get("msg", ""))
    st.markdown(
        f"""
        <div style="border-left:5px solid {color};padding:0.65rem 0.85rem;
        margin:0.35rem 0;background:#f8fafc;border-radius:0.4rem">
        <strong>{icon} {label} · {LEVEL_LABEL.get(level, level)}</strong><br>
        {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_weather_chart(
    fig: go.Figure,
    *,
    height: int = 340,
    hovermode: str = "x unified",
) -> go.Figure:
    fig.update_layout(
        height=height,
        hovermode=hovermode,
        font={"family": "Arial, sans-serif", "size": 12, "color": "#334155"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"t": 55, "b": 45, "l": 55, "r": 90},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#cbd5e1")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
    return fig


def show_weather_chart(
    fig: go.Figure,
    *,
    height: int = 340,
    hovermode: str = "x unified",
) -> None:
    style_weather_chart(fig, height=height, hovermode=hovermode)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


# =============================================================================
# SNAPSHOT
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_snapshot_raw() -> dict:
    return request_json(SNAPSHOT_URL, timeout=30)


def load_snapshot() -> dict:
    try:
        return _fetch_snapshot_raw()
    except Exception as exc:
        return {"_error": str(exc)}


# =============================================================================
# OPEN-METEO
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_dept_forecast_raw(dep: str) -> dict:
    info = DEPS[dep]
    return request_json(
        FORECAST_URL,
        params={
            "latitude": info["lat"],
            "longitude": info["lon"],
            "daily": (
                "precipitation_sum,temperature_2m_max,weather_code,"
                "wind_speed_10m_max"
            ),
            "forecast_days": 7,
            "timezone": "Europe/Paris",
        },
    ).get("daily", {})


def load_forecast_dep(dep: str) -> dict:
    try:
        return {"daily": _fetch_dept_forecast_raw(dep)}
    except Exception:
        return {}


def load_weather_alerts_all() -> tuple[list[dict], int, int]:
    alerts: list[dict] = []
    ok_count = 0

    for dep in DEPS:
        try:
            daily = _fetch_dept_forecast_raw(dep)
            if not daily or not daily.get("time"):
                continue
            ok_count += 1
        except Exception:
            continue

        dates = daily.get("time", [])
        rains = daily.get("precipitation_sum", [])
        temps = daily.get("temperature_2m_max", [])
        codes = daily.get("weather_code", daily.get("weathercode", []))
        winds = daily.get("wind_speed_10m_max", [])
        rain7 = sum(safe_float(value) for value in rains)
        seen_fire = False

        for index, current_date in enumerate(dates):
            rain = safe_float(rains[index]) if index < len(rains) else 0.0
            temp = safe_float(temps[index]) if index < len(temps) else 0.0
            code = int(safe_float(codes[index])) if index < len(codes) else 0
            wind = safe_float(winds[index]) if index < len(winds) else 0.0
            date_label = str(current_date)[5:]

            if code >= 95:
                level = "ROUGE" if code >= 99 else "ORANGE"
                alerts.append({"dep": dep, "date": current_date, "type": "ORAGE", "level": level,
                               "msg": f"Dép. {dep} le {date_label} : risque orageux (code WMO {code})."})
            elif code >= 80:
                alerts.append({"dep": dep, "date": current_date, "type": "ORAGE", "level": "JAUNE",
                               "msg": f"Dép. {dep} le {date_label} : averses potentiellement orageuses."})

            if rain >= 60:
                rain_level = "ROUGE"
            elif rain >= 30:
                rain_level = "ORANGE"
            elif rain >= 15:
                rain_level = "JAUNE"
            else:
                rain_level = ""
            if rain_level:
                alerts.append({"dep": dep, "date": current_date, "type": "INONDATION", "level": rain_level,
                               "msg": f"Dép. {dep} le {date_label} : {rain:.1f} mm prévus."})

            if temp >= 40:
                heat_level = "ROUGE"
            elif temp >= 35:
                heat_level = "ORANGE"
            elif temp >= 30:
                heat_level = "JAUNE"
            else:
                heat_level = ""
            if heat_level:
                alerts.append({"dep": dep, "date": current_date, "type": "CANICULE", "level": heat_level,
                               "msg": f"Dép. {dep} le {date_label} : température maximale {temp:.1f} °C."})

            if wind >= 100:
                wind_level = "ROUGE"
            elif wind >= 80:
                wind_level = "ORANGE"
            elif wind >= 60:
                wind_level = "JAUNE"
            else:
                wind_level = ""
            if wind_level:
                alerts.append({"dep": dep, "date": current_date, "type": "VENT", "level": wind_level,
                               "msg": f"Dép. {dep} le {date_label} : vent maximal {wind:.0f} km/h."})

            if not seen_fire and temp >= 30 and rain7 < 10 and wind >= 25:
                fire_level = "ROUGE" if temp >= 35 and rain7 < 5 and wind >= 35 else "ORANGE"
                alerts.append({"dep": dep, "date": current_date, "type": "INCENDIE", "level": fire_level,
                               "msg": f"Dép. {dep} le {date_label} : chaleur, sécheresse et vent combinés."})
                seen_fire = True

    return alerts, ok_count, len(DEPS)


@st.cache_data(ttl=3600, show_spinner=False)
def load_forecast_coord(lat: float, lon: float) -> pd.DataFrame:
    try:
        daily = request_json(
            FORECAST_URL,
            params={
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "daily": (
                    "precipitation_sum,precipitation_probability_max,"
                    "temperature_2m_max"
                ),
                "forecast_days": 7,
                "timezone": "Europe/Paris",
            },
        ).get("daily", {})
        return pd.DataFrame({
            "date": daily.get("time", []),
            "pluie_mm": daily.get("precipitation_sum", []),
            "proba_%": daily.get("precipitation_probability_max", []),
            "tmax": daily.get("temperature_2m_max", []),
        })
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_commune_daily_series(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=max(1, days) - 1)
    try:
        daily = request_json(
            ARCHIVE_URL,
            params={
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "start_date": str(start_date),
                "end_date": str(end_date),
                "daily": "precipitation_sum",
                "timezone": "Europe/Paris",
            },
            timeout=35,
        ).get("daily", {})
        frame = pd.DataFrame({
            "date": daily.get("time", []),
            "pluie_mm": daily.get("precipitation_sum", []),
        })
        if not frame.empty:
            frame["pluie_mm"] = pd.to_numeric(frame["pluie_mm"], errors="coerce").fillna(0.0)
        return frame
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_communes_daily_rain(sectors: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    required = {"commune_name", "latitude", "longitude"}
    if sectors.empty or not required.issubset(sectors.columns):
        return pd.DataFrame()
    coordinates = (
        sectors.dropna(subset=["commune_name", "latitude", "longitude"])
        .groupby("commune_name")[["latitude", "longitude"]]
        .mean()
    )
    frames: list[pd.DataFrame] = []
    for commune, row in coordinates.iterrows():
        frame = load_commune_daily_series(float(row["latitude"]), float(row["longitude"]), days)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["commune_name"] = commune
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_commune_rain_openmeteo(lat: float, lon: float, period: str) -> float:
    days_by_period = {"24 h": 1, "7 jours": 7, "30 jours": 30}
    if period in days_by_period:
        frame = load_commune_daily_series(lat, lon, days_by_period[period])
        return float(frame["pluie_mm"].sum()) if not frame.empty else float("nan")

    today = datetime.now(timezone.utc).date()
    days = max(1, (today - today.replace(day=1)).days)
    frame = load_commune_daily_series(lat, lon, days)
    return float(frame["pluie_mm"].sum()) if not frame.empty else float("nan")


@st.cache_data(ttl=3600, show_spinner=False)
def load_monthly_rain(lat: float, lon: float) -> pd.DataFrame:
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_date = (end_date.replace(day=1) - timedelta(days=365)).replace(day=1)
    try:
        daily = request_json(
            ARCHIVE_URL,
            params={
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "start_date": str(start_date),
                "end_date": str(end_date),
                "daily": "precipitation_sum",
                "timezone": "Europe/Paris",
            },
            timeout=40,
        ).get("daily", {})
        monthly: dict[str, float] = defaultdict(float)
        for current_date, rain in zip(daily.get("time", []), daily.get("precipitation_sum", [])):
            monthly[str(current_date)[:7]] += safe_float(rain)
        return pd.DataFrame([
            {"mois": month, "pluie_mm": round(value, 1)}
            for month, value in sorted(monthly.items())[-12:]
        ])
    except Exception:
        return pd.DataFrame()


# =============================================================================
# VIGICRUES
# =============================================================================


def _vigicrues_level(raw_level: object) -> str | None:
    if raw_level in (None, ""):
        return None
    by_name = {
        "vert": "VERT", "green": "VERT",
        "jaune": "JAUNE", "yellow": "JAUNE",
        "orange": "ORANGE", "rouge": "ROUGE", "red": "ROUGE",
    }
    normalized = _normalize(raw_level).strip()
    if normalized in by_name:
        return by_name[normalized]
    try:
        number = int(float(normalized))
    except (TypeError, ValueError):
        return None
    return {0: "VERT", 1: "VERT", 2: "JAUNE", 3: "ORANGE", 4: "ROUGE"}.get(number)


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_vigicrues_geojson_raw() -> dict:
    payload = request_json(VIGICRUES_GEOJSON_URL, timeout=45)
    if not isinstance(payload.get("features"), list):
        raise RuntimeError("Flux Vigicrues sans liste de tronçons")
    return payload


def load_vigicrue_rivers() -> tuple[list[dict], bool]:
    try:
        payload = _fetch_vigicrues_geojson_raw()
    except Exception:
        return [], False

    results: list[dict] = []
    for feature in payload.get("features", []):
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties", {})
        if not isinstance(properties, dict):
            continue

        name = str(
            properties.get("NomEntVigiCru")
            or properties.get("LbEntVigiCru")
            or properties.get("lbentcru")
            or ""
        ).strip()
        if not name:
            continue
        normalized_name = _normalize(name)
        if not any(river in normalized_name for river in RIVERS_LGV):
            continue

        raw_level = properties.get("NivSituVigiCruEnt")
        if raw_level is None:
            raw_level = properties.get("NivInfViCr")
        if raw_level is None:
            raw_level = properties.get("NivVigiCru")
        level = _vigicrues_level(raw_level)
        if not level:
            continue

        code = str(properties.get("CdEntVigiCru") or "").strip()
        results.append({
            "riviere": name,
            "code": code,
            "level": level,
            "type": "VIGICRUE",
            "msg": f"{name} : vigilance {LEVEL_LABEL.get(level, level).lower()}.",
        })

    seen: set[tuple[str, str, str]] = set()
    deduplicated: list[dict] = []
    for item in results:
        key = (item.get("code", ""), _normalize(item["riviere"]), item["level"])
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
    deduplicated.sort(key=lambda item: (-LEVEL_RANK.get(item["level"], 0), item["riviere"]))
    return deduplicated, True


# =============================================================================
# NASA FIRMS
# =============================================================================


def get_firms_map_key() -> str | None:
    try:
        key = st.secrets.get("FIRMS_MAP_KEY")
        if key:
            return str(key).strip()
        for section_name in ("firms", "FIRMS", "default"):
            if section_name in st.secrets:
                section_key = st.secrets[section_name].get("FIRMS_MAP_KEY")
                if section_key:
                    return str(section_key).strip()
    except Exception:
        pass
    key = os.environ.get("FIRMS_MAP_KEY") or os.environ.get("FIRMS_KEY")
    return key.strip() if key else None


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_firms_source_raw(
    key: str,
    source: str,
    day_range: int,
    date_str: str,
) -> pd.DataFrame:
    url = FIRMS_AREA_URL.format(
        key=key,
        source=source,
        area=FIRMS_BBOX,
        day_range=day_range,
        date=date_str,
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HTTP_HEADERS, timeout=(8, 30))
            response.raise_for_status()
            text = response.text.strip()
            if "invalid" in text[:300].lower():
                raise ValueError("invalid_key")
            if not text or "<html" in text[:300].lower():
                raise RuntimeError("unexpected_response")
            frame = pd.read_csv(io.StringIO(text))
            if "latitude" not in frame.columns or "longitude" not in frame.columns:
                raise RuntimeError("unexpected_response")
            frame["source"] = source
            return frame
        except ValueError:
            raise
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"FIRMS indisponible : {last_error}")


def load_firms_hotspots(
    day_range: int = 1,
    end_date: date | None = None,
) -> tuple[pd.DataFrame, str | None]:
    key = get_firms_map_key()
    if not key:
        return pd.DataFrame(), "missing_key"

    day_range = max(1, min(int(day_range), FIRMS_MAX_DAY_RANGE))
    selected_date = end_date or datetime.now(timezone.utc).date()
    frames: list[pd.DataFrame] = []
    success_count = 0

    for source in FIRMS_SOURCES:
        try:
            frame = _fetch_firms_source_raw(key, source, day_range, str(selected_date))
            success_count += 1
            if not frame.empty:
                frames.append(frame)
        except ValueError as exc:
            if str(exc) == "invalid_key":
                return pd.DataFrame(), "invalid_key"
        except Exception:
            continue

    if success_count == 0:
        return pd.DataFrame(), "fetch_failed"
    if not frames:
        return pd.DataFrame(), None

    frame = pd.concat(frames, ignore_index=True)
    dedup_columns = [column for column in ("latitude", "longitude", "acq_date", "acq_time") if column in frame.columns]
    if dedup_columns:
        frame = frame.drop_duplicates(subset=dedup_columns)
    return frame, None


@st.cache_data(ttl=3600, show_spinner=False)
def build_lgv_pk_polyline(lgv_lines: object) -> list[tuple[float, float, float]]:
    if not isinstance(lgv_lines, list) or not lgv_lines:
        return []
    segment = lgv_lines[0]
    if not isinstance(segment, list):
        return []
    points = [
        (safe_float(point.get("lat")), safe_float(point.get("lon")))
        for point in segment
        if isinstance(point, dict) and point.get("lat") is not None and point.get("lon") is not None
    ]
    if len(points) < 2:
        return []
    output = [(points[0][0], points[0][1], 0.0)]
    cumulative = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        cumulative += _haversine_km(lat1, lon1, lat2, lon2)
        output.append((lat2, lon2, cumulative))
    return output


def pk_and_distance(
    lat: float,
    lon: float,
    polyline: list[tuple[float, float, float]],
) -> tuple[float | None, float | None]:
    if len(polyline) < 2:
        return None, None
    best_distance_squared: float | None = None
    best_pk: float | None = None
    for (lat1, lon1, pk1), (lat2, lon2, pk2) in zip(polyline, polyline[1:]):
        latitude_middle = (lat1 + lat2) / 2.0
        kx = 111.320 * math.cos(math.radians(latitude_middle))
        ky = 111.320
        x1, y1 = lon1 * kx, lat1 * ky
        x2, y2 = lon2 * kx, lat2 * ky
        xp, yp = lon * kx, lat * ky
        dx, dy = x2 - x1, y2 - y1
        length_squared = dx * dx + dy * dy
        projection = 0.0 if length_squared == 0 else ((xp - x1) * dx + (yp - y1) * dy) / length_squared
        projection = max(0.0, min(1.0, projection))
        closest_x = x1 + projection * dx
        closest_y = y1 + projection * dy
        distance_squared = (xp - closest_x) ** 2 + (yp - closest_y) ** 2
        if best_distance_squared is None or distance_squared < best_distance_squared:
            best_distance_squared = distance_squared
            best_pk = pk1 + projection * (pk2 - pk1)
    distance = math.sqrt(best_distance_squared) if best_distance_squared is not None else None
    return best_pk, distance


def load_firms_alerts(
    polyline: list[tuple[float, float, float]],
    day_range: int = 1,
    end_date: date | None = None,
    radius_km: float = FIRMS_RADIUS_KM,
) -> tuple[list[dict], str | None]:
    frame, error = load_firms_hotspots(day_range, end_date)
    if error:
        return [], error
    if frame.empty or not polyline:
        return [], None

    alerts: list[dict] = []
    for _, row in frame.iterrows():
        latitude = safe_float(row.get("latitude"), float("nan"))
        longitude = safe_float(row.get("longitude"), float("nan"))
        if math.isnan(latitude) or math.isnan(longitude):
            continue
        pk, distance_km = pk_and_distance(latitude, longitude, polyline)
        if distance_km is None or distance_km > radius_km:
            continue
        confidence_raw = row.get("confidence", "")
        confidence = FIRMS_CONF_LABELS.get(str(confidence_raw).strip().lower(), str(confidence_raw))
        acquisition_time = str(row.get("acq_time", "")).zfill(4)
        formatted_time = f"{acquisition_time[:2]}:{acquisition_time[2:]}" if acquisition_time else ""
        alerts.append({
            "latitude": latitude,
            "longitude": longitude,
            "pk_km": round(pk, 3) if pk is not None else None,
            "distance_m": round(distance_km * 1000),
            "date": str(row.get("acq_date", "")),
            "heure": formatted_time,
            "confidence": confidence,
            "frp": safe_float(row.get("frp"), float("nan")),
            "satellite": str(row.get("satellite") or row.get("source") or ""),
            "level": "ROUGE" if distance_km <= 0.2 else "ORANGE",
            "type": "FEU_FIRMS",
            "msg": f"Détection satellite à {distance_km * 1000:.0f} m de la LGV, vers le PK {pk:.3f}.",
        })
    alerts.sort(key=lambda item: item["distance_m"])
    return alerts, None


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="LGV SEA - Pluviométrie et vigilance",
    page_icon="🌧️",
    layout="wide",
)

st.title("🌧️ LGV SEA - Pluviométrie et vigilance")
header_left, header_right = st.columns([5, 1])
header_left.caption(
    "Météo : Open-Meteo · Crues : Vigicrues · Incendies : NASA FIRMS. "
    "Les indicateurs calculés ne remplacent pas les vigilances officielles."
)
if header_right.button("🔄 Rafraîchir", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

snapshot = load_snapshot()
if "_error" in snapshot:
    st.error(f"Snapshot LGV indisponible : {snapshot['_error']}")
    st.stop()

sector_container = snapshot.get("sectors", {})
sector_records = sector_container.get("sectors", []) if isinstance(sector_container, dict) else []
sectors_df = safe_df(sector_records)
for numeric_column in (
    "weather_max_24h_mm", "weather_max_7d_mm", "weather_max_30d_mm",
    "weather_max_month_mm", "latitude", "longitude", "pk_km",
):
    if numeric_column in sectors_df.columns:
        sectors_df[numeric_column] = pd.to_numeric(sectors_df[numeric_column], errors="coerce")

communes = (
    sorted(sectors_df["commune_name"].dropna().astype(str).unique())
    if "commune_name" in sectors_df.columns else []
)

with st.sidebar:
    st.header("Paramètres")
    selected_commune = st.selectbox("Commune principale", ["Toutes les communes"] + communes)
    selected_comparison = st.multiselect("Comparer des communes", communes, default=communes[:6])
    rain_period = st.selectbox("Période pluviométrique", ["24 h", "7 jours", "30 jours", "Mois courant"])
    st.divider()
    st.subheader("NASA FIRMS")
    firms_days = st.slider("Fenêtre de recherche", 1, FIRMS_MAX_DAY_RANGE, 1, format="%d jour(s)")
    firms_end_date = st.date_input(
        "Date de fin",
        value=datetime.now(timezone.utc).date(),
        min_value=datetime.now(timezone.utc).date() - timedelta(days=FIRMS_MAX_LOOKBACK_DAYS),
        max_value=datetime.now(timezone.utc).date(),
    )

# Prévisions par département
st.subheader("Pluie prévue sur 7 jours par département")
dep_columns = st.columns(len(DEPS))
dep_rain_data: dict[str, dict] = {}
for department in DEPS:
    forecast = load_forecast_dep(department)
    values = forecast.get("daily", {}).get("precipitation_sum", []) if forecast else []
    rain_values = [safe_float(value) for value in values if value is not None]
    is_ok = bool(forecast and rain_values)
    maximum = max(rain_values) if rain_values else 0.0
    total = sum(rain_values)
    level, color, emoji = rain_risk(maximum) if is_ok else ("INFO", "#9ca3af", "❓")
    dep_rain_data[department] = {"ok": is_ok, "max": maximum, "total": total, "level": level, "color": color}

for column, (department, info) in zip(dep_columns, DEPS.items()):
    data = dep_rain_data[department]
    with column.container(border=True):
        st.caption(f"Dép. {department} · {info['nom']}")
        if data["ok"]:
            st.metric("Cumul 7 j", f"{data['total']:.0f} mm", delta=f"max {data['max']:.0f} mm/j", delta_color="off")
        else:
            st.metric("Cumul 7 j", "Indisponible")

st.divider()

# Alertes météo et Vigicrues
st.subheader("📊 Indicateurs météo et crues")
weather_alerts, weather_ok, weather_total = load_weather_alerts_all()
vigicrues_alerts, vigicrues_ok = load_vigicrue_rivers()
active_weather = [item for item in weather_alerts if item.get("level") in {"JAUNE", "ORANGE", "ROUGE"}]
active_vigicrues = [item for item in vigicrues_alerts if item.get("level") in {"JAUNE", "ORANGE", "ROUGE"}]

status_columns = st.columns(2)
with status_columns[0]:
    if weather_ok == 0:
        st.warning("Open-Meteo injoignable. Les indicateurs météo ne sont pas vérifiés.")
    elif weather_ok < weather_total:
        st.warning(f"Open-Meteo vérifié pour {weather_ok}/{weather_total} départements.")
    elif active_weather:
        st.warning(f"{len(active_weather)} indicateur(s) météo significatif(s).")
    else:
        st.success("Open-Meteo vérifié : aucun indicateur significatif.")

with status_columns[1]:
    if not vigicrues_ok:
        st.warning(
            "Vigicrues est temporairement injoignable. "
            "Le statut des crues n'a pas pu être vérifié."
        )
    elif active_vigicrues:
        st.warning(f"{len(active_vigicrues)} tronçon(s) Vigicrues en vigilance active.")
    else:
        st.success("Vigicrues vérifié : aucune vigilance active sur les tronçons sélectionnés.")

all_active = active_weather + active_vigicrues
if all_active:
    with st.expander("Détail des alertes", expanded=True):
        for alert in sorted(all_active, key=lambda item: -LEVEL_RANK.get(item.get("level", "INFO"), 0)):
            alert_card(alert)

st.divider()

# FIRMS
st.subheader(f"🔥 Détections NASA FIRMS à moins de {FIRMS_RADIUS_KM * 1000:.0f} m de la LGV")
lgv_polyline = build_lgv_pk_polyline(snapshot.get("lgv_lines", []))
firms_alerts, firms_error = load_firms_alerts(
    lgv_polyline,
    day_range=firms_days,
    end_date=firms_end_date,
)
if firms_error == "missing_key":
    st.warning(
        "Clé FIRMS manquante. Ajoute `FIRMS_MAP_KEY = \"...\"` dans "
        "`.streamlit/secrets.toml` ou dans les secrets Streamlit Cloud."
    )
elif firms_error == "invalid_key":
    st.error("Clé FIRMS invalide. Vérifie la valeur de `FIRMS_MAP_KEY`.")
elif firms_error == "fetch_failed":
    st.warning("NASA FIRMS injoignable. Le statut incendie n'a pas pu être vérifié.")
elif firms_alerts:
    st.error(f"{len(firms_alerts)} détection(s) satellite à proximité de la LGV.")
    firms_frame = pd.DataFrame(firms_alerts).rename(columns={
        "pk_km": "PK (km)", "distance_m": "Distance LGV (m)",
        "date": "Date", "heure": "Heure UTC", "confidence": "Confiance",
        "frp": "FRP (MW)", "satellite": "Satellite",
    })
    st.dataframe(
        firms_frame[["PK (km)", "Distance LGV (m)", "Date", "Heure UTC", "Confiance", "FRP (MW)", "Satellite"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success("NASA FIRMS vérifié : aucune détection dans le rayon sélectionné.")

st.divider()

# Top communes
st.subheader("🌧️ Comparaison des communes")
if selected_comparison and not sectors_df.empty:
    comparison_source = sectors_df[sectors_df["commune_name"].isin(selected_comparison)]
    if not comparison_source.empty:
        grouped = comparison_source.groupby("commune_name")[["latitude", "longitude"]].mean().dropna()
        comparison_rows = []
        for commune, row in grouped.iterrows():
            rainfall = load_commune_rain_openmeteo(float(row["latitude"]), float(row["longitude"]), rain_period)
            if not math.isnan(rainfall):
                comparison_rows.append({"Commune": commune, "Cumul (mm)": round(rainfall, 1)})
        comparison_frame = pd.DataFrame(comparison_rows).sort_values("Cumul (mm)", ascending=False) if comparison_rows else pd.DataFrame()
        if not comparison_frame.empty:
            comparison_figure = px.bar(
                comparison_frame,
                x="Commune",
                y="Cumul (mm)",
                color="Cumul (mm)",
                color_continuous_scale=["#bfdbfe", "#2563eb", "#1e3a8a"],
                text="Cumul (mm)",
            )
            comparison_figure.update_layout(coloraxis_showscale=False)
            show_weather_chart(comparison_figure, height=400, hovermode="closest")
        else:
            st.info("Données de comparaison indisponibles.")
else:
    st.info("Sélectionne au moins une commune dans la barre latérale.")

st.divider()

# Sélection principale
if selected_commune == "Toutes les communes":
    commune_df = sectors_df.copy()
else:
    commune_df = sectors_df[sectors_df["commune_name"] == selected_commune].copy()

map_df = (
    commune_df.dropna(subset=["latitude", "longitude"])
    if {"latitude", "longitude"}.issubset(commune_df.columns)
    else pd.DataFrame()
)
latitude_center = float(map_df["latitude"].mean()) if not map_df.empty else 46.2
longitude_center = float(map_df["longitude"].mean()) if not map_df.empty else 0.2
location_label = "LGV SEA" if selected_commune == "Toutes les communes" else selected_commune

# Prévisions 7 jours
st.subheader(f"🔮 Prévisions sur 7 jours · {location_label}")
forecast_frame = load_forecast_coord(latitude_center, longitude_center)
if not forecast_frame.empty:
    for column in ("pluie_mm", "proba_%", "tmax"):
        forecast_frame[column] = pd.to_numeric(forecast_frame[column], errors="coerce")
    forecast_frame["pluie_mm"] = forecast_frame["pluie_mm"].fillna(0.0)
    forecast_figure = go.Figure()
    forecast_figure.add_bar(
        x=forecast_frame["date"],
        y=forecast_frame["pluie_mm"],
        name="Pluie (mm)",
        marker_color=[rain_color_mm(value) for value in forecast_frame["pluie_mm"]],
        text=[f"{value:.1f}" for value in forecast_frame["pluie_mm"]],
        textposition="outside",
        yaxis="y",
    )
    forecast_figure.add_scatter(
        x=forecast_frame["date"], y=forecast_frame["proba_%"],
        name="Probabilité pluie (%)", mode="lines+markers", yaxis="y2",
        line={"color": "#6366f1", "dash": "dot"},
    )
    forecast_figure.add_scatter(
        x=forecast_frame["date"], y=forecast_frame["tmax"],
        name="Température max. (°C)", mode="lines+markers", yaxis="y3",
        line={"color": "#f97316", "width": 2},
    )
    forecast_figure.update_layout(
        yaxis={"title": "Pluie (mm)", "side": "left", "rangemode": "tozero"},
        yaxis2={"title": "Probabilité (%)", "overlaying": "y", "side": "right", "range": [0, 100], "showgrid": False},
        yaxis3={"title": "Température (°C)", "overlaying": "y", "side": "right", "anchor": "free", "position": 0.92, "showgrid": False},
    )
    show_weather_chart(forecast_figure, height=400)
else:
    st.info("Prévisions indisponibles.")

# Historique mensuel
st.subheader(f"📅 Historique mensuel · {location_label}")
monthly_frame = load_monthly_rain(latitude_center, longitude_center)
if not monthly_frame.empty:
    monthly_figure = px.bar(
        monthly_frame,
        x="mois",
        y="pluie_mm",
        color="pluie_mm",
        color_continuous_scale=["#bfdbfe", "#3b82f6", "#1e3a8a"],
        labels={"mois": "Mois", "pluie_mm": "Pluie (mm)"},
        text="pluie_mm",
    )
    monthly_figure.update_layout(coloraxis_showscale=False)
    average = float(monthly_frame["pluie_mm"].mean())
    monthly_figure.add_hline(
        y=average,
        line_dash="dash",
        line_color=CHART_COLORS["teal"],
        annotation_text=f"Moyenne : {average:.1f} mm",
        annotation_position="top left",
    )
    show_weather_chart(monthly_figure, height=360, hovermode="closest")
else:
    st.info("Historique mensuel indisponible.")

# Carte
st.subheader("🗺️ Carte des secteurs LGV SEA")
if not map_df.empty:
    map_object = folium.Map(
        location=[latitude_center, longitude_center],
        zoom_start=8 if selected_commune == "Toutes les communes" else 11,
        tiles="CartoDB positron",
        control_scale=True,
    )
    for _, row in map_df.iterrows():
        department = min(
            DEPS.keys(),
            key=lambda dep: (DEPS[dep]["lat"] - float(row["latitude"])) ** 2 + (DEPS[dep]["lon"] - float(row["longitude"])) ** 2,
        )
        color = dep_rain_data.get(department, {}).get("color", "#64748b")
        commune_name = str(row.get("commune_name", "Secteur"))
        pk_value = row.get("pk_km")
        pk_label = f"{safe_float(pk_value):.3f}" if pd.notna(pk_value) else "indisponible"
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=f"{commune_name} · PK {pk_label}",
        ).add_to(map_object)

    for alert in firms_alerts:
        folium.CircleMarker(
            location=[alert["latitude"], alert["longitude"]],
            radius=8,
            color="#dc2626",
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.9,
            tooltip=f"FIRMS · PK {alert['pk_km']} · {alert['distance_m']} m de la LGV",
        ).add_to(map_object)

    st_folium(map_object, use_container_width=True, height=500, returned_objects=[])
else:
    st.info("Aucune coordonnée disponible pour cette sélection.")

# Tableau
st.subheader("📋 Secteurs LGV SEA")
columns_to_show = [column for column in ("commune_name", "pk_km", "latitude", "longitude") if column in commune_df.columns]
display_frame = commune_df[columns_to_show].rename(columns={
    "commune_name": "Commune", "pk_km": "PK (km)",
    "latitude": "Latitude", "longitude": "Longitude",
}) if columns_to_show else pd.DataFrame()
if not display_frame.empty:
    if "PK (km)" in display_frame.columns:
        display_frame = display_frame.sort_values("PK (km)")
    st.dataframe(display_frame, use_container_width=True, hide_index=True, height=350)
else:
    st.info("Aucun secteur disponible.")

st.caption(
    "Sources : Open-Meteo, Vigicrues, NASA FIRMS et snapshot LGV SEA. "
    "Les indicateurs internes sont informatifs et ne remplacent pas les consignes officielles."
)
