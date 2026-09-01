#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tableau de bord LGV SEA – Pluviométrie, glissements, vigilance météo, incendies, crues.

Sources :
- Snapshot local/distant (secteurs, IA)
- Open-Meteo (prévisions et archives)
- Météo-France (vigilance officielle)
- Vigicrues (crues)
- NASA FIRMS (incendies)
"""

from __future__ import annotations

import io
import os
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from folium.plugins import MarkerCluster
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# CONSTANTES ET CONFIGURATION
# -----------------------------------------------------------------------------

# Snapshots
SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
STALE_MINUTES = 180

# FIRMS (NASA)
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{day_range}/{date}"
FIRMS_SOURCES = ["VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"]
FIRMS_BBOX = "-0.7,44.75,1.0,47.5"  # corridor LGV SEA
FIRMS_RADIUS_KM = 0.5
FIRMS_MAX_DAY_RANGE = 10
FIRMS_MAX_LOOKBACK_DAYS = 60
FIRMS_CONF_LABELS = {"l": "faible", "n": "moyenne", "h": "élevée"}

# Météo-France vigilance
MF_VIGILANCE_URL = "https://public.opendatasoft.com/api/records/1.0/search/"
MF_VIGILANCE_DATASET = "weatherref-france-vigilance-meteo-departement"
MF_PHENOMENON_LABELS = {
    "pluie-inondation": "🌊 Pluie-inondation",
    "vent": "💨 Vent",
    "neige-verglas": "❄️ Neige/verglas",
    "orages": "⛈️ Orages",
    "grand-froid": "🥶 Grand froid",
    "vagues-submersion": "🌊 Vagues-submersion",
    "canicule": "☀️ Canicule",
}
LEVEL_COLOR = {"ROUGE": "#dc2626", "ORANGE": "#ea580c", "JAUNE": "#eab308", "VERT": "#16a34a", "INFO": "#3b82f6"}
LEVEL_RANK = {"ROUGE": 4, "ORANGE": 3, "JAUNE": 2, "VERT": 1, "INFO": 0}
LEVEL_LABEL = {"ROUGE": "Rouge", "ORANGE": "Orange", "JAUNE": "Jaune", "VERT": "Vert", "INFO": "Info"}
LEVEL_BADGE = {"ROUGE": "red", "ORANGE": "orange", "JAUNE": "yellow", "VERT": "green", "INFO": "gray"}

# Départements LGV SEA
DEPS = {
    "37": {"nom": "Indre-et-Loire", "lat": 47.39, "lon": 0.69},
    "86": {"nom": "Vienne", "lat": 46.58, "lon": 0.34},
    "79": {"nom": "Deux-Sèvres", "lat": 46.33, "lon": -0.46},
    "16": {"nom": "Charente", "lat": 45.67, "lon": 0.33},
    "17": {"nom": "Charente-Maritime", "lat": 45.75, "lon": -0.63},
    "33": {"nom": "Gironde", "lat": 44.83, "lon": -0.58},
    "24": {"nom": "Dordogne", "lat": 45.18, "lon": 0.72},
    "47": {"nom": "Lot-et-Garonne", "lat": 44.32, "lon": 0.65},
    "40": {"nom": "Landes", "lat": 43.89, "lon": -0.50},
    "49": {"nom": "Maine-et-Loire", "lat": 47.47, "lon": -0.55},
    "85": {"nom": "Vendée", "lat": 46.67, "lon": -1.43},
    "36": {"nom": "Indre", "lat": 46.81, "lon": 1.69},
}

# Cours d'eau LGV SEA (normalisés)
_RIVERS_RAW = [
    "vienne", "clain", "charente", "boutonne", "seugne", "touvre",
    "dronne", "isle", "dordogne", "garonne", "thouet", "sevre",
    "indre", "cher", "creuse", "ciron", "jalles", "estey",
    "leyre", "midouze", "brion", "anglin",
]
RIVERS_LGV = [unicodedata.normalize("NFD", r.lower()).encode("ascii", "ignore").decode() for r in _RIVERS_RAW]
DEP_OK = {"37", "86", "79", "16", "17", "33", "24", "47", "40", "49", "85", "36"}

# Risques glissements
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

ALERT_CFG = {
    "FEU_FIRMS": ("🔥", "Incendie"),
    "ORAGE": ("⛈️", "Orage"),
    "INONDATION": ("🌊", "Inondation"),
    "CANICULE": ("☀️", "Canicule"),
    "VENT": ("💨", "Vent"),
    "INCENDIE": ("🔥", "Incendie"),
    "VIGICRUE": ("🏞️", "Crue"),
}

CHART_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=20, b=20, l=20, r=20),
)

# -----------------------------------------------------------------------------
# UTILITAIRES
# -----------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Normalise une chaîne : minuscules, sans accents."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if f == f else default  # NaN check
    except (TypeError, ValueError):
        return default


def safe_df(records) -> pd.DataFrame:
    if isinstance(records, list) and records:
        try:
            return pd.DataFrame(records)
        except Exception:
            pass
    return pd.DataFrame()


def safe_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def fmt_pct(series: pd.Series) -> pd.Series:
    pct = pd.to_numeric(series, errors="coerce") * 100
    return pct.round(0).apply(lambda v: "—" if pd.isna(v) else f"{int(v)} %")


def risk_badge(level: str) -> str:
    color = RISK_COLOR.get(level, "#6b7280")
    emoji = RISK_EMOJI.get(level, "⚪")
    return f'<span class="risk-badge" style="background:{color}20;color:{color};border-color:{color}">{emoji} {level}</span>'


def data_age_minutes(timestamp_utc: str) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return None


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


def nearest_dep(lat: float, lon: float) -> str:
    """Retourne le code département le plus proche (pour la carte)."""
    return min(DEPS.keys(),
               key=lambda d: (DEPS[d]["lat"] - lat) ** 2 + (DEPS[d]["lon"] - lon) ** 2)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# -----------------------------------------------------------------------------
# CHARGEMENT DES DONNÉES
# -----------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_snapshot() -> dict:
    """Charge le snapshot le plus récent (local ou distant)."""
    candidates = []
    errors = []

    # Tentative locale
    local_paths = [
        Path(__file__).resolve().parent / "reports" / "streamlit_snapshot_latest.json",
        Path.cwd() / "reports" / "streamlit_snapshot_latest.json",
    ]
    for path in local_paths:
        try:
            if path.is_file() and path.stat().st_size > 0:
                with path.open("r", encoding="utf-8") as f:
                    data = pd.read_json(f).to_dict()
                if data:
                    candidates.append(("local", str(path), data))
                    break
        except Exception as e:
            errors.append(f"Local {path}: {e}")

    # Tentative distante
    try:
        resp = requests.get(
            SNAPSHOT_URL,
            timeout=(10, 30),
            headers={"Accept": "application/json", "Cache-Control": "no-cache"},
            params={"v": int(datetime.now(timezone.utc).timestamp() // 300)},
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            candidates.append(("distant", SNAPSHOT_URL, data))
    except Exception as e:
        errors.append(f"Distant {SNAPSHOT_URL}: {e}")

    if not candidates:
        return {"_error": "Aucun snapshot valide", "_details": errors}

    # Choisir le plus récent
    def ts_val(c):
        raw = c[2].get("timestamp_utc", "")
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    source_type, source_name, selected = max(candidates, key=ts_val)
    selected = dict(selected)
    selected["_snapshot_source"] = source_type
    selected["_snapshot_location"] = source_name
    selected["_load_warnings"] = errors
    return selected


@st.cache_data(ttl=3600)
def load_monthly_rain(lat: float, lon: float) -> pd.DataFrame:
    """Historique mensuel des pluies depuis 2021."""
    end = datetime.now(timezone.utc).date()
    start = datetime(2021, 1, 1).date()
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
        monthly = {}
        for d, v in zip(dates, rain):
            if v is not None:
                monthly[d[:7]] = monthly.get(d[:7], 0.0) + v
        return pd.DataFrame([{"mois": m, "pluie_mm": round(v, 1)} for m, v in sorted(monthly.items())])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def load_commune_rain_ometo(lat: float, lon: float, periode: str) -> float:
    """Cumul pluie via Open-Meteo (prévision ou archive selon période)."""
    today = datetime.now(timezone.utc).date()
    if periode == "24h":
        past_days = 1
    elif periode == "7 jours":
        past_days = 7
    elif periode == "30 jours":
        past_days = 30
    else:  # mois courant
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


@st.cache_data(ttl=1800)
def load_vigicrue_rivers() -> Tuple[List[dict], bool]:
    """Charge la vigilance crues officielle (Vigicrues) pour les cours d'eau LGV."""
    api_url = "https://www.vigicrues.gouv.fr/services/v1.1/TerEntVigiCru.json"
    headers = {
        "Accept": "application/json",
        "User-Agent": "LGV-PluvioStations/1.0",
    }
    level_map = {0: "VERT", 1: "JAUNE", 2: "ORANGE", 3: "ROUGE"}

    def walk(value):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    def first(item, keys):
        for key in keys:
            val = item.get(key)
            if val not in (None, ""):
                return val
        return None

    try:
        r = requests.get(api_url, headers=headers, timeout=(5, 25))
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return [], False

    territories = payload.get("ListEntVigiCru", [])
    if not isinstance(territories, list):
        return [], False

    results = []
    for territory in territories:
        code = territory.get("CdEntVigiCru")
        etype = territory.get("TypEntVigiCru", "5")
        if not code:
            continue
        try:
            r2 = requests.get(api_url, params={"CdEntVigiCru": code, "TypEntVigiCru": etype},
                              headers=headers, timeout=(5, 25))
            r2.raise_for_status()
            detail = r2.json()
        except Exception:
            continue

        for item in walk(detail):
            name = first(item, ("LbEntVigiCru", "LibEntVigiCru", "NomEntVigiCru",
                                "LibTroncon", "NomTroncon", "NomCoursDeau", "Nom", "lib"))
            if not name:
                continue
            name = str(name).strip()
            name_norm = normalize(name)
            if not any(r in name_norm for r in RIVERS_LGV):
                continue

            level_raw = first(item, ("NivVigiCru", "NivVigiCruHydro", "NivVig",
                                     "NiveauVigilance", "CdCouleur", "Couleur", "couleur"))
            level = None
            if level_raw is not None:
                norm = normalize(str(level_raw))
                if norm in {"vert", "green"}: level = "VERT"
                elif norm in {"jaune", "yellow"}: level = "JAUNE"
                elif norm == "orange": level = "ORANGE"
                elif norm in {"rouge", "red"}: level = "ROUGE"
                else:
                    try:
                        level = level_map.get(int(float(norm)))
                    except (TypeError, ValueError):
                        pass
            if not level:
                continue

            dep_raw = first(item, ("CdDep", "CDDep", "CodeDepartement"))
            dep = str(dep_raw).zfill(2) if dep_raw not in (None, "") else ""
            if dep and dep not in DEP_OK:
                continue

            results.append({
                "riviere": name,
                "dep": dep,
                "level": level,
                "type": "VIGICRUE",
                "msg": f"{name} — vigilance {level.lower()}",
            })

    # Déduplication
    seen = set()
    dedup = []
    for item in results:
        key = (normalize(item["riviere"]), item["level"])
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    dedup.sort(key=lambda x: (-LEVEL_RANK.get(x["level"], 0), x["riviere"]))
    return dedup, True


def get_firms_map_key() -> Optional[str]:
    """Récupère la clé FIRMS depuis les secrets ou l'environnement."""
    try:
        key = st.secrets.get("FIRMS_MAP_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("FIRMS_MAP_KEY")


@st.cache_data(ttl=3600)
def build_lgv_pk_polyline(_lgv_lines) -> List[Tuple[float, float, float]]:
    """Construit une polyligne avec PK cumulé à partir du tracé LGV."""
    if not _lgv_lines:
        return []
    seg = _lgv_lines[0]
    pts = []
    cum = 0.0
    for i, (lat, lon) in enumerate(seg):
        if i == 0:
            pts.append((lat, lon, 0.0))
        else:
            prev = seg[i-1]
            cum += haversine_km(prev[0], prev[1], lat, lon)
            pts.append((lat, lon, cum))
    return pts


def pk_and_distance(lat: float, lon: float, polyline: list) -> Tuple[Optional[float], Optional[float]]:
    """Retourne (pk_km, distance_km) du point le plus proche sur la LGV."""
    if len(polyline) < 2:
        return None, None
    best_dist2 = None
    best_pk = None
    # Projection sur chaque segment en métrique locale
    for i in range(len(polyline) - 1):
        lat1, lon1, pk1 = polyline[i]
        lat2, lon2, pk2 = polyline[i+1]
        dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2))
        dy = (lat2 - lat1) * 110.574
        seg_len2 = dx*dx + dy*dy
        if seg_len2 == 0:
            continue
        tx = ((lat - lat1)*110.574*dy + (lon - lon1)*111.32*math.cos(math.radians(lat1))*dx) / seg_len2
        tx = max(0.0, min(1.0, tx))
        proj_lat = lat1 + tx * (lat2 - lat1)
        proj_lon = lon1 + tx * (lon2 - lon1)
        dist2 = (lat - proj_lat)**2 * 110.574**2 + (lon - proj_lon)**2 * (111.32*math.cos(math.radians(lat)))**2
        if best_dist2 is None or dist2 < best_dist2:
            best_dist2 = dist2
            best_pk = pk1 + tx * (pk2 - pk1)
    return best_pk, math.sqrt(best_dist2) if best_dist2 is not None else None


@st.cache_data(ttl=3600)
def load_firms_hotspots(day_range: int = 1, end_date=None) -> Tuple[pd.DataFrame, Optional[str]]:
    """Détections FIRMS (VIIRS NRT) sur la bbox LGV."""
    key = get_firms_map_key()
    if not key:
        return pd.DataFrame(), "missing_key"
    if end_date is None:
        end_date = datetime.now(timezone.utc).date()
    if day_range > FIRMS_MAX_DAY_RANGE:
        day_range = FIRMS_MAX_DAY_RANGE
    date_str = end_date.strftime("%Y-%m-%d")

    all_dfs = []
    for source in FIRMS_SOURCES:
        try:
            url = FIRMS_AREA_URL.format(key=key, source=source, area=FIRMS_BBOX,
                                        day_range=day_range, date=date_str)
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            txt = r.text.strip()
            if "invalid" in txt.lower()[:200]:
                return pd.DataFrame(), "invalid_key"
            if not txt or "<html" in txt.lower()[:200]:
                continue
            df = pd.read_csv(io.StringIO(txt))
            if "latitude" not in df.columns:
                continue
            df["source"] = source
            all_dfs.append(df)
        except Exception:
            continue
    if not all_dfs:
        return pd.DataFrame(), "fetch_failed"
    return pd.concat(all_dfs, ignore_index=True), None


def load_firms_alerts(_polyline: list, day_range: int = 1, end_date=None,
                      radius_km: float = FIRMS_RADIUS_KM) -> Tuple[List[dict], Optional[str]]:
    """Filtre les hotspots FIRMS à proximité de la LGV."""
    df, err = load_firms_hotspots(day_range, end_date)
    if err:
        return [], err
    if df.empty:
        return [], None

    poly = _polyline if _polyline else []
    alerts = []
    for _, row in df.iterrows():
        lat = safe_float(row.get("latitude"))
        lon = safe_float(row.get("longitude"))
        if not poly:
            pk, dist = None, None
        else:
            pk, dist = pk_and_distance(lat, lon, poly)
        if pk is None or dist is None or dist > radius_km:
            continue
        conf = row.get("confidence", "")
        frp = row.get("frp", 0)
        date = row.get("acq_date", "")
        time_ = row.get("acq_time", "")
        if isinstance(time_, (int, float)):
            time_ = f"{int(time_):04d}"
        alerts.append({
            "type": "FEU_FIRMS",
            "level": "ORANGE" if frp > 50 else "JAUNE",
            "msg": (f"🔥 Feu à {dist*1000:.0f} m de la LGV (PK {pk:.1f} km) — "
                    f"{date} {time_} UTC, confiance {conf}, FRP {frp:.0f} MW"),
            "lat": lat, "lon": lon,
            "pk": pk, "distance": dist,
            "satellite": row.get("source", ""),
            "confiance": conf,
            "frp": frp,
            "date": date,
            "heure": time_,
        })
    return alerts, None


@st.cache_data(ttl=3600)
def load_commune_daily_series(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    """Série journalière de pluie sur les `days` derniers jours (archive ERA5)."""
    try:
        params = {
            "latitude": round(lat, 4), "longitude": round(lon, 4),
            "daily": "precipitation_sum",
            "start_date": (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat(),
            "end_date": datetime.now(timezone.utc).date().isoformat(),
            "timezone": "Europe/Paris",
        }
        r = requests.get(ARCHIVE_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        dates = data["daily"]["time"]
        rain = data["daily"]["precipitation_sum"]
        return pd.DataFrame({
            "date": pd.to_datetime(dates),
            "pluie_mm": [v if v is not None else 0.0 for v in rain],
        })
    except Exception:
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# FONCTIONS D'AFFICHAGE (UI)
# -----------------------------------------------------------------------------

def alert_card(a: dict) -> None:
    """Affiche une carte d'alerte (uniformisée)."""
    lvl = a.get("level", "")
    atype = a.get("type", "")
    icon, label = ALERT_CFG.get(atype, ("", atype))

    if atype == "FEU_FIRMS":
        c1, c2 = st.columns([1, 7], vertical_alignment="center")
        with c1:
            st.badge(LEVEL_LABEL.get(lvl, lvl or "Info"), color=LEVEL_BADGE.get(lvl, "gray"))
        with c2:
            st.markdown(f"{icon}  {a.get('msg','')}")
        return

    color = LEVEL_COLOR.get(lvl, "#6b7280")
    bg = f"{color}18"
    st.markdown(
        f'<div style="padding:6px 12px;border-radius:6px;border-left:4px solid {color};background:{bg};margin-bottom:4px;">'
        f'<b>{icon} {label}</b> — {a.get("msg", "")}'
        f'</div>',
        unsafe_allow_html=True,
    )


def style_weather_chart(fig: go.Figure, height: int = 340, hovermode: str = "x unified") -> go.Figure:
    fig.update_layout(
        height=height,
        hovermode=hovermode,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickangle=-20),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", rangemode="tozero"),
    )
    return fig


def show_weather_chart(fig: go.Figure, height: int = 340, hovermode: str = "x unified") -> None:
    style_weather_chart(fig, height, hovermode)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# -----------------------------------------------------------------------------
# APPLICATION STREAMLIT
# -----------------------------------------------------------------------------

st.set_page_config(page_title="LGV SEA – Pluvio & glissements", page_icon="🌧", layout="wide")

# CSS personnalisé
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

# ---- Chargement du snapshot ----
snapshot = load_snapshot()
if "_error" in snapshot:
    st.error(f"Erreur : {snapshot['_error']}")
    st.stop()

ts = snapshot.get("timestamp_utc", "")
age_min = data_age_minutes(ts) if ts else None
if snapshot.get("_load_warnings"):
    with st.expander("État des sources", expanded=False):
        for w in snapshot["_load_warnings"]:
            st.warning(w)

if ts:
    caption = f"Données : {ts[:16].replace('T', ' ')} UTC"
    if age_min is not None:
        caption += f" (il y a {age_min:.0f} min)" if age_min < 120 else f" (il y a {age_min/60:.1f} h)"
    st.caption(caption)

# ---- Extraction des données ----
sectors_payload = safe_dict(snapshot.get("sectors"))
sectors_df = safe_df(sectors_payload.get("sectors", []))
sector_summary = safe_dict(sectors_payload.get("summary"))
sector_alerts = sectors_payload.get("alerts", []) if isinstance(sectors_payload.get("alerts"), list) else []
commune_ranking = safe_df(snapshot.get("commune_ranking", []))
ai_model = safe_dict(sectors_payload.get("ai_model"))
lgv_lines = snapshot.get("lgv_lines", [])

if sectors_df.empty:
    st.warning("Aucun secteur disponible.")
    st.stop()

# Nettoyage numérique
for col in ["weather_max_24h_mm", "weather_max_7d_mm", "weather_max_30d_mm",
            "weather_max_month_mm", "latitude", "longitude", "pk_km", "score",
            "ai_pred_probability", "ai_confidence", "ai_soil_fragility"]:
    if col in sectors_df.columns:
        sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")

sector_lookup = {}
if {"sector_id", "pk_km", "commune_name"}.issubset(sectors_df.columns):
    sector_lookup = sectors_df.set_index("sector_id")[["pk_km", "commune_name"]].to_dict("index")

# ---- Vue d'ensemble ----
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
if not sector_alerts:
    st.success("Aucun secteur en alerte.")
else:
    for a in sector_alerts:
        level = a.get("level", "")
        color = RISK_COLOR.get(level, "#6b7280")
        kind = "🤖 IA" if a.get("type") == "SECTEUR_IA" else "📏 Mesure"
        msg = humanize_alert_message(a.get("message", ""), sector_lookup)
        st.markdown(
            f'<div class="alert-card" style="border-left-color:{color};background:{color}12">'
            f'<b>[{level}]</b> {kind} — {msg}</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ---- Sidebar ----
with st.sidebar:
    st.subheader("Filtres")
    if st.button("🔄 Rafraîchir", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    communes = sorted(sectors_df["commune_name"].dropna().unique()) if "commune_name" in sectors_df.columns else []
    selected = st.selectbox("📍 Commune", ["— Toutes —"] + list(communes))

    periode = st.selectbox("📅 Période", ["24h", "7 jours", "30 jours", "Mois courant"])

    risque_min = st.selectbox("⚠ Risque minimum", ["Tout", "FAIBLE", "MODERE", "ELEVE", "CRITIQUE"])
    show_ai_detail = st.checkbox("Colonnes IA détaillées", value=False)

    st.divider()
    st.caption(f"Modèle : {ai_model.get('name', 'modèle IA')} v{ai_model.get('version', '?')}")
    st.caption("Sources : Open-Meteo, BRGM, Géorisques.")
    st.caption("Prédiction IA à confirmer par expertise terrain.")

# ---- Filtrage des données ----
df = sectors_df.copy()
if selected != "— Toutes —":
    df = df[df["commune_name"] == selected]
if risque_min != "Tout" and "risk_level" in df.columns:
    min_rank = RISK_RANK.get(risque_min, 0)
    df = df[df["risk_level"].map(lambda x: RISK_RANK.get(str(x), 0)) >= min_rank]

map_df = df.dropna(subset=["latitude", "longitude"]) if {"latitude", "longitude"}.issubset(df.columns) else pd.DataFrame()

# ---- Chargement des alertes externes ----
# Vigicrue
vc_alerts, vc_ok = load_vigicrue_rivers()
vc_active = [a for a in vc_alerts if a["level"] in ("JAUNE", "ORANGE", "ROUGE")]

# Météo-France vigilance (simplifiée ici, on pourrait implémenter l'appel)
mf_alerts = []  # à remplir si on implémente l'appel
mf_ok = True   # placeholder

# FIRMS
polyline = build_lgv_pk_polyline(lgv_lines)
firms_end_date = datetime.now(timezone.utc).date()
firms_alerts, firms_err = load_firms_alerts(polyline, day_range=1, end_date=firms_end_date)
_firms_unverified = (firms_err is not None and firms_err != "missing_key")

# Prévisions météo Open-Meteo (exemple pour les départements)
def load_forecast_dep(dep: str) -> dict:
    lat, lon = DEPS[dep]["lat"], DEPS[dep]["lon"]
    try:
        r = requests.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon,
            "daily": ["precipitation_sum", "temperature_2m_max", "windspeed_10m_max", "weathercode"],
            "past_days": 7, "forecast_days": 7,
            "timezone": "Europe/Paris",
        }, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def rain_risk(max_mm):
    if max_mm >= 40:
        return "ROUGE", "#dc2626", "🔴"
    elif max_mm >= 25:
        return "ORANGE", "#ea580c", "🟠"
    elif max_mm >= 10:
        return "JAUNE", "#eab308", "🟡"
    elif max_mm >= 0:
        return "VERT", "#16a34a", "🟢"
    return "INDETERMINE", "#9ca3af", "❓"

dep_rain_data = {}
for d in DEPS:
    fc = load_forecast_dep(d)
    ok = bool(fc)
    if ok:
        rain = fc.get("daily", {}).get("precipitation_sum", [])
        if rain:
            max_mm = max([v for v in rain if v is not None], default=0)
            total = sum([v for v in rain if v is not None], 0.0)
            lvl, col, emoji = rain_risk(max_mm)
        else:
            lvl, col, emoji = "INDETERMINE", "#9ca3af", "❓"
            max_mm = total = 0.0
    else:
        lvl, col, emoji = "INDETERMINE", "#9ca3af", "❓"
        max_mm = total = 0.0
    dep_rain_data[d] = {"max": max_mm, "total": total, "lvl": lvl, "color": col, "emoji": emoji, "ok": ok}

# ---- Onglets ----
tab_carte, tab_analyses, tab_hist, tab_secteurs, tab_communes, tab_vigilance = st.tabs(
    ["🗺 Carte", "📊 Analyses", "📅 Historique", "📋 Secteurs", "🏘 Communes", "🛡️ Vigilance"]
)

# ---------- CARTE ----------
with tab_carte:
    if map_df.empty:
        st.info("Pas de coordonnées pour la carte.")
    else:
        try:
            lat_c = float(map_df["latitude"].mean())
            lon_c = float(map_df["longitude"].mean())
            m = folium.Map(location=[lat_c, lon_c],
                           zoom_start=8 if selected == "— Toutes —" else 12,
                           tiles="CartoDB positron", control_scale=True)

            # Tracé LGV
            for seg in lgv_lines:
                if isinstance(seg, list):
                    pts = [[p[0], p[1]] for p in seg if isinstance(p, (list, tuple)) and len(p) >= 2]
                    if pts:
                        folium.PolyLine(pts, color="#1d4ed8", weight=2.5, opacity=0.7,
                                        tooltip="LGV SEA").add_to(m)

            # Marqueurs secteurs
            cluster = MarkerCluster().add_to(m)
            for row in map_df.itertuples(index=False):
                risk_lvl = str(getattr(row, "risk_level", "INDETERMINE"))
                ai_lvl = str(getattr(row, "ai_pred_risk_level", "INDETERMINE"))
                color = RISK_COLOR.get(ai_lvl, "#6b7280")
                proba = min(max(safe_float(getattr(row, "ai_pred_probability", 0.0)), 0.0), 1.0)
                popup = (
                    f"<b>{getattr(row, 'sector_id', '')}</b> — {getattr(row, 'commune_name', '')} "
                    f"(PK {getattr(row, 'pk_km', '')} km)<br>"
                    f"Risque mesuré : {risk_lvl}<br>"
                    f"Prédiction IA : {ai_lvl} ({proba*100:.0f} %)<br>"
                    f"Sol : {getattr(row, 'ai_dominant_pedology', '—')}"
                )
                folium.CircleMarker(
                    [safe_float(row.latitude), safe_float(row.longitude)],
                    radius=6 + 6 * proba,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    weight=1.5,
                    tooltip=f"{getattr(row, 'sector_id', '')} — IA {ai_lvl}",
                    popup=folium.Popup(popup, max_width=280),
                ).add_to(cluster)

            st.caption("Couleur = risque IA, taille = probabilité.")
            st_folium(m, width=1400, height=520, returned_objects=[])
        except Exception as e:
            st.warning(f"Carte indisponible ({e}).")

# ---------- ANALYSES ----------
with tab_analyses:
    st.markdown("**Profil du risque le long de la ligne (IA)**")
    profile_df = df.dropna(subset=["pk_km"]).sort_values("pk_km") if "pk_km" in df.columns else pd.DataFrame()
    if profile_df.empty:
        st.info("Pas de profil PK disponible.")
    else:
        try:
            bar_colors = profile_df["ai_pred_risk_level"].map(lambda x: RISK_COLOR.get(str(x), "#6b7280"))
            proba_pct = (profile_df["ai_pred_probability"].fillna(0.0) * 100
                         if "ai_pred_probability" in profile_df.columns else pd.Series(0.0, index=profile_df.index))
            fig = go.Figure()
            fig.add_bar(x=profile_df["pk_km"], y=proba_pct, marker_color=bar_colors,
                        name="Probabilité IA (%)",
                        hovertemplate="PK %{x} km<br>Proba : %{y:.0f} %<extra></extra>")
            if "score" in profile_df.columns:
                fig.add_scatter(x=profile_df["pk_km"], y=profile_df["score"].fillna(0.0) * 25,
                                mode="lines+markers", name="Score mesuré (×25)",
                                line=dict(color="#0f172a", dash="dot"))
            fig.add_hline(y=65, line_dash="dash", line_color="#dc2626", annotation_text="Élevé")
            fig.add_hline(y=85, line_dash="dash", line_color="#7f1d1d", annotation_text="Critique")
            fig.update_layout(xaxis_title="PK (km)", yaxis_title="Probabilité / Score",
                              height=320, legend=dict(orientation="h", y=1.12), **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Profil indisponible ({e}).")

    st.markdown("**Répartition du risque — mesuré vs IA**")
    levels = ["FAIBLE", "MODERE", "ELEVE", "CRITIQUE"]
    if "risk_level" in df.columns or "ai_pred_risk_level" in df.columns:
        try:
            measured = df["risk_level"].value_counts() if "risk_level" in df.columns else pd.Series(dtype=int)
            ai = df["ai_pred_risk_level"].value_counts() if "ai_pred_risk_level" in df.columns else pd.Series(dtype=int)
            fig2 = go.Figure()
            fig2.add_bar(x=levels, y=[int(measured.get(lvl, 0)) for lvl in levels], name="Mesuré", marker_color="#0f172a")
            fig2.add_bar(x=levels, y=[int(ai.get(lvl, 0)) for lvl in levels], name="Prédiction IA", marker_color="#3b82f6")
            fig2.update_layout(barmode="group", yaxis_title="Secteurs", height=280,
                               legend=dict(orientation="h", y=1.15), **CHART_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Répartition indisponible ({e}).")
    else:
        st.info("Pas de niveau de risque.")

    st.markdown("**Facteurs de risque les plus fréquents**")
    if "ai_top_factors" in df.columns:
        try:
            factor_counts = defaultdict(int)
            for factors in df["ai_top_factors"]:
                if isinstance(factors, list):
                    for f in factors:
                        if f != "signal_faible":
                            factor_counts[f] += 1
            if factor_counts:
                fact_df = pd.DataFrame(
                    [{"facteur": FACTOR_LABELS.get(k, k), "secteurs": v}
                     for k, v in factor_counts.items()]
                ).sort_values("secteurs", ascending=True)
                fig3 = go.Figure()
                fig3.add_bar(x=fact_df["secteurs"], y=fact_df["facteur"], orientation="h", marker_color="#7c3aed")
                fig3.update_layout(xaxis_title="Secteurs concernés", height=280, **CHART_LAYOUT)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Aucun facteur marquant.")
        except Exception as e:
            st.warning(f"Facteurs indisponibles ({e}).")
    else:
        st.info("Facteurs IA non disponibles.")

# ---------- HISTORIQUE ----------
with tab_hist:
    hist_label = selected if selected != "— Toutes —" else "LGV SEA (centroïde)"
    st.markdown(f"**Historique pluviométrique depuis 2021 — {hist_label}**")
    if map_df.empty:
        st.info("Pas de localisation.")
    else:
        try:
            lat_h = float(map_df["latitude"].mean())
            lon_h = float(map_df["longitude"].mean())
            monthly = load_monthly_rain(lat_h, lon_h)
            if monthly.empty:
                st.info("Historique indisponible.")
            else:
                fig = go.Figure()
                fig.add_bar(x=monthly["mois"], y=monthly["pluie_mm"],
                            marker_color="#3b82f6", text=monthly["pluie_mm"], textposition="outside")
                fig.update_layout(xaxis_title="Mois", yaxis_title="Pluie (mm)", height=300,
                                  xaxis=dict(tickangle=-30), **CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Historique indisponible ({e}).")

# ---------- SECTEURS ----------
with tab_secteurs:
    ometo_rain = {}
    if selected != "— Toutes —":
        # Bandeau commune
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
            f'<span style="margin-left:16px">IA : {risk_badge(ai_lvl)}</span>'
            f'</div>', unsafe_allow_html=True)

        # Cumuls pluie
        loc = map_df.dropna(subset=["latitude", "longitude"]) if not map_df.empty else pd.DataFrame()
        if not loc.empty:
            lat_o = round(float(loc["latitude"].mean()), 4)
            lon_o = round(float(loc["longitude"].mean()), 4)
            for p in ["24h", "7 jours", "30 jours", "Mois courant"]:
                ometo_rain[p] = load_commune_rain_ometo(lat_o, lon_o, p)

        cols = st.columns(4)
        labels = ["☔ 24h", "🌧 7j", "🌦 30j", "📅 Mois"]
        for col, lab, key in zip(cols, labels, ["24h", "7 jours", "30 jours", "Mois courant"]):
            v = ometo_rain.get(key, float("nan"))
            col.metric(lab, f"{v:.1f} mm" if pd.notna(v) else "—")

        st.caption("Pluie : Open-Meteo ERA5 (near real-time)")

        # Métriques IA
        a1, a2, a3 = st.columns(3)
        a1.metric("Probabilité IA max", f"{safe_float(commune_row.get('ai_max_probability')) * 100:.0f} %")
        a2.metric("Fragilité sol moyenne", f"{safe_float(commune_row.get('ai_avg_soil_fragility')) * 100:.0f} %")
        a3.metric("Secteurs IA critiques/élevés",
                  int(commune_row.get("ai_critical", 0)) + int(commune_row.get("ai_high", 0)))

        # Détail IA par secteur
        with st.expander(f"🔎 Détail IA par secteur — {selected}", expanded=False):
            detail_df = df.sort_values("ai_pred_probability", ascending=False) if "ai_pred_probability" in df.columns else df
            for row in detail_df.itertuples(index=False):
                proba = safe_float(getattr(row, "ai_pred_probability", 0.0))
                conf = safe_float(getattr(row, "ai_confidence", 0.0))
                st.markdown(
                    f'**{getattr(row, "sector_id", "?")}** · PK {getattr(row, "pk_km", "—")} km '
                    f'&nbsp; {risk_badge(str(getattr(row, "risk_level", "INDETERMINE")))} '
                    f'&nbsp; IA {risk_badge(str(getattr(row, "ai_pred_risk_level", "INDETERMINE")))}',
                    unsafe_allow_html=True)
                st.progress(proba, text=f"Probabilité : {proba*100:.0f} % (confiance {conf*100:.0f} %)")
                factors = getattr(row, "ai_top_factors", None)
                st.markdown(
                    f'Sol : **{getattr(row, "ai_dominant_pedology", "—")}** '
                    f'({getattr(row, "ai_dominant_soil_type", "—")}) &nbsp;·&nbsp; '
                    f'Facteurs : {factor_tags(factors)}',
                    unsafe_allow_html=True)
                st.markdown("---")

        st.divider()

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
        if selected != "— Toutes —" and ometo_rain:
            rv = ometo_rain.get(periode, float("nan"))
            disp.insert(2, f"Pluie {periode}", f"{rv:.1f} mm" if pd.notna(rv) else "—")
        for pct_col in ["Proba IA", "Confiance IA", "Fragilité sol"]:
            if pct_col in disp.columns:
                disp[pct_col] = fmt_pct(disp[pct_col])
        if "Risque IA" in disp.columns:
            disp = disp.sort_values("Risque IA",
                                    key=lambda s: s.map(lambda x: RISK_RANK.get(str(x), 0)),
                                    ascending=False, na_position="last")
        elif "Risque" in disp.columns:
            disp = disp.sort_values("Risque",
                                    key=lambda s: s.map(lambda x: RISK_RANK.get(str(x), 0)),
                                    ascending=False, na_position="last")
        st.dataframe(disp, use_container_width=True, hide_index=True, height=360)

# ---------- COMMUNES ----------
with tab_communes:
    if selected != "— Toutes —":
        st.info("Sélectionne « — Toutes — » pour voir le classement complet.")
    elif commune_ranking.empty:
        st.info("Classement communes indisponible.")
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
        st.dataframe(disp_cr, use_container_width=True, hide_index=True, height=380)

        if {"commune_name", "commune_note"}.issubset(cr.columns):
            st.markdown("**Top 10 communes les plus à risque**")
            try:
                top = cr.dropna(subset=["commune_note"]).sort_values("commune_note", ascending=False).head(10)
                if not top.empty:
                    fig = go.Figure()
                    fig.add_bar(x=top["commune_note"], y=top["commune_name"], orientation="h",
                                marker_color=top["commune_risk_level"].map(lambda x: RISK_COLOR.get(str(x), "#6b7280")),
                                text=top["commune_note"], textposition="outside")
                    fig.update_layout(xaxis_title="Note (/100)", yaxis=dict(autorange="reversed"),
                                      height=340, margin=dict(t=20, b=20, l=20, r=40),
                                      plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Graphe top communes indisponible ({e}).")

# ---------- VIGILANCE (onglet supplémentaire) ----------
with tab_vigilance:
    st.markdown("## 🛡️ Vigilances en cours")

    # Météo-France
    st.markdown("### Météo-France (officielle)")
    if mf_ok:
        if mf_alerts:
            for a in mf_alerts:
                alert_card(a)
        else:
            st.success("Aucune vigilance particulière en cours.")
    else:
        st.warning("⚠️ Vigilance Météo-France non vérifiée (API injoignable).")

    # Prévisions météo Open-Meteo
    st.markdown("### 🌦️ Prévisions météo (Open-Meteo)")
    met_ok = sum(1 for d in dep_rain_data.values() if d["ok"])
    if met_ok == 0:
        st.warning("Prévisions météo non disponibles (Open-Meteo injoignable).")
    else:
        # Grouper par type d'alerte
        met_alerts = []
        for dep, info in dep_rain_data.items():
            if not info["ok"]:
                continue
            lvl = info["lvl"]
            if lvl in ("ORANGE", "ROUGE"):
                met_alerts.append({
                    "type": "INONDATION",
                    "level": lvl,
                    "msg": f"Département {dep} – cumul 7j de {info['total']:.0f} mm (max {info['max']:.0f} mm/j)",
                })
        if met_alerts:
            for a in met_alerts:
                alert_card(a)
        else:
            st.success("Aucune alerte météo significative sur les 7 prochains jours.")

    # Vigicrue
    st.markdown("### 🏞️ Vigicrues")
    if vc_ok:
        if vc_active:
            for a in vc_active:
                alert_card(a)
        else:
            st.success("Aucune vigilance crue pour les cours d'eau LGV.")
    else:
        st.warning("⚠️ Vigicrues non vérifié (API injoignable).")

    # FIRMS
    st.markdown("### 🔥 Incendies (FIRMS)")
    if firms_err == "missing_key":
        st.warning("Clé FIRMS manquante. [Obtenez une clé gratuite]"
                   "(https://firms.modaps.eosdis.nasa.gov/api/map_key/)")
    elif firms_err == "invalid_key":
        st.error("Clé FIRMS invalide.")
    elif firms_err == "fetch_failed":
        st.warning("FIRMS injoignable – statut non vérifié.")
    else:
        if firms_alerts:
            for a in firms_alerts:
                alert_card(a)
        else:
            st.success(f"Aucun feu détecté à moins de {FIRMS_RADIUS_KM*1000:.0f} m de la LGV.")

# ---- Pied de page ----
with st.expander("ℹ️ À propos", expanded=False):
    st.markdown(
        f"- **Modèle IA** : {ai_model.get('name', 'n/a')} v{ai_model.get('version', '?')}\n"
        "- **Sources** : Open-Meteo, BRGM, Géorisques, Météo-France, Vigicrues, NASA FIRMS.\n"
        "- **Limites** : prédiction IA à confirmer par expertise terrain.\n"
        f"- **Fraîcheur** : données rafraîchies automatiquement toutes les heures ; "
        f"alerte si données > {STALE_MINUTES/60:.0f} h."
    )
