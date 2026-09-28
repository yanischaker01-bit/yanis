from __future__ import annotations

import math
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

# =============================================================================
# CONFIGURATION
# =============================================================================
SNAPSHOT_URLS = [
    "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json",
    "https://yanischaker01-bit.github.io/yanis/dashboard/reports/streamlit_snapshot_latest.json",
]
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PIEZO_STATIONS_URL = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations"
PIEZO_TR_URL = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques_tr"
PIEZO_HISTORY_URL = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques"
VIGICRUES_URL = "https://www.vigicrues.gouv.fr/services/v1.1/TerEntVigiCru.json"
MF_URL = "https://public.opendatasoft.com/api/records/1.0/search/"
MF_DATASET = "weatherref-france-vigilance-meteo-departement"
DEPARTMENTS = ["37", "86", "79", "16", "17", "33"]
RIVERS = ["vienne", "clain", "charente", "boutonne", "seugne", "touvre", "dronne", "isle", "dordogne", "garonne", "thouet", "sevre", "indre", "cher", "creuse", "ciron", "jalles"]

LEVEL_RANK = {"INDETERMINE": -1, "VERT": 0, "JAUNE": 1, "ORANGE": 2, "ROUGE": 3}
LEVEL_COLOR = {"INDETERMINE": "#64748b", "VERT": "#16a34a", "JAUNE": "#eab308", "ORANGE": "#ea580c", "ROUGE": "#dc2626"}
LEVEL_ACTION = {
    "INDETERMINE": "Données insuffisantes, contrôle manuel nécessaire",
    "VERT": "Surveillance courante",
    "JAUNE": "Surveillance renforcée et contrôle de la fraîcheur des données",
    "ORANGE": "Inspection ciblée et contrôle du drainage",
    "ROUGE": "Contrôle prioritaire et application des consignes métier",
}

st.set_page_config(page_title="LGV SEA - Surveillance optimisée", page_icon="⚠️", layout="wide")

# =============================================================================
# OUTILS
# =============================================================================
def norm(value: Any) -> str:
    value = unicodedata.normalize("NFD", str(value or "").lower())
    return "".join(c for c in value if unicodedata.category(c) != "Mn")


def safe_float(value: Any, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_json(url: str, params=None, timeout=(5, 30), headers=None) -> dict:
    response = requests.get(url, params=params, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.json()


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1, math.sqrt(a)))


# =============================================================================
# SNAPSHOT ET SECTEURS (robuste)
# =============================================================================

ALIASES = {
    "lat": "latitude", "y": "latitude", "latitude_wgs84": "latitude",
    "lon": "longitude", "lng": "longitude", "long": "longitude",
    "x": "longitude", "longitude_wgs84": "longitude",
    "pk": "pk_km", "pk_gps": "pk_km", "pk_theorique": "pk_km",
    "commune": "commune_name", "nom_commune": "commune_name", "libelle_commune": "commune_name",
}
REQUIRED = {"commune_name", "latitude", "longitude", "pk_km"}


def _normalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    replacements = {" ": "_", "-": "_", ".": "_", "/": "_"}
    for source, target in replacements.items():
        text = text.replace(source, target)
    while "__" in text:
        text = text.replace("__", "_")
    return ALIASES.get(text, text)


def _extract_sector_records(payload: dict) -> list:
    if not isinstance(payload, dict):
        return []
    candidates = [payload.get("sectors"), payload.get("data"), payload.get("snapshot")]
    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            return candidate
        if not isinstance(candidate, dict):
            continue
        for key in ("sectors", "records", "items", "features"):
            records = candidate.get(key)
            if not isinstance(records, list) or not records:
                continue
            if key != "features":
                return records
            normalized = []
            for feature in records:
                if not isinstance(feature, dict):
                    continue
                props = dict(feature.get("properties") or {})
                geometry = feature.get("geometry") or {}
                coords = geometry.get("coordinates") or []
                if geometry.get("type") == "Point" and len(coords) >= 2:
                    props.setdefault("longitude", coords[0])
                    props.setdefault("latitude", coords[1])
                normalized.append(props)
            if normalized:
                return normalized
    return []


def _normalize_sectors(records: list) -> pd.DataFrame:
    if not isinstance(records, list) or not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame = frame.copy()
    frame.columns = [_normalize_column_name(c) for c in frame.columns]
    # rename aliases once more
    for src, tgt in ALIASES.items():
        if src in frame.columns and tgt not in frame.columns:
            frame = frame.rename(columns={src: tgt})
    if "pk_km" not in frame.columns and "pk_m" in frame.columns:
        frame["pk_km"] = pd.to_numeric(frame["pk_m"], errors="coerce") / 1000.0
    for col in ("latitude", "longitude", "pk_km"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _snapshot_is_usable(payload: dict) -> tuple[bool, str]:
    records = _extract_sector_records(payload)
    if not records:
        return False, "aucun secteur"
    frame = _normalize_sectors(records)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        return False, "colonnes manquantes : " + ", ".join(missing)
    localized = frame.dropna(subset=["latitude", "longitude", "pk_km"]) if all(c in frame.columns for c in ("latitude","longitude","pk_km")) else pd.DataFrame()
    if localized.empty:
        return False, "aucun secteur avec latitude, longitude et PK valides"
    return True, "ok"


@st.cache_data(ttl=900, show_spinner=False)
def load_snapshot():
    # try local first (reports/*)
    local_paths = [
        "reports/streamlit_snapshot_latest.json",
        "reports/streamlit_snapshot_last_valid.json",
    ]
    for p in local_paths:
        try:
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as fh:
                    data = json.load(fh)
                valid, reason = _snapshot_is_usable(data)
                if valid:
                    data["_snapshot_source"] = p
                    return data
        except Exception:
            pass
    # remote fetch with anti-cache slot (5 minutes)
    slot = int(datetime.now(timezone.utc).timestamp() // 300)
    headers = {"Accept": "application/json", "Cache-Control": "no-cache", "Pragma": "no-cache"}
    errors = []
    for url in SNAPSHOT_URLS:
        try:
            r = requests.get(url, params={"v": slot}, headers=headers, timeout=(5,25))
            r.raise_for_status()
            data = r.json()
            valid, reason = _snapshot_is_usable(data)
            if not valid:
                errors.append(f"{url}: {reason}")
                continue
            data["_snapshot_source"] = url
            return data
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(" | ".join(errors))


def load_points(snapshot):
    # extract and normalize sectors into DataFrame with required cols
    payload = snapshot if not isinstance(snapshot.get("sectors"), (dict, list)) else snapshot
    records = _extract_sector_records(snapshot)
    df = _normalize_sectors(records)
    # convert some optional columns
    for col in ["latitude", "longitude", "pk_km", "ai_pred_probability", "ai_soil_fragility", "score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "commune_name" not in df.columns:
        df["commune_name"] = "Commune inconnue"
    if not all(c in df.columns for c in ("latitude","longitude","pk_km")):
        raise RuntimeError(f"Snapshot indisponible : colonnes manquantes dans les secteurs")
    return df.dropna(subset=["latitude", "longitude", "pk_km"])


def make_sectors(points, width=10):
    work = points.copy()
    work["pk_start"] = (work["pk_km"] // width).astype(int) * width
    rows = []
    for pk_start, group in work.groupby("pk_start"):
        ai = pd.to_numeric(group.get("ai_pred_probability"), errors="coerce").dropna()
        soil = pd.to_numeric(group.get("ai_soil_fragility"), errors="coerce").dropna()
        measured = pd.to_numeric(group.get("score"), errors="coerce").dropna()
        base = float(ai.max()) if not ai.empty else 0.40
        fragility = float(soil.mean()) if not soil.empty else 0.40
        signal = min(1.0, float(measured.max()) / 4) if not measured.empty else 0.20
        quick_score = round(100 * (0.55 * base + 0.30 * fragility + 0.15 * signal), 1)
        rows.append({
            "sector_id": f"PK_{pk_start:03d}_{pk_start+width:03d}",
            "name": f"PK {pk_start:03d}-{pk_start+width:03d}",
            "pk_start": float(pk_start), "pk_end": float(pk_start + width),
            "latitude": float(group["latitude"].mean()), "longitude": float(group["longitude"].mean()),
            "communes": ", ".join(sorted(group["commune_name"].astype(str).unique())),
            "static_score": quick_score, "static_level": risk_level(quick_score),
            "susceptibility": base, "soil_fragility": fragility, "signal": signal,
        })
    return pd.DataFrame(rows).sort_values("pk_start")

# =============================================================================
# ... remainder of file unchanged ...
# =============================================================================
