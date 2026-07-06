from __future__ import annotations

from datetime import date, datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import unicodedata

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_folium import st_folium

SNAPSHOT_LATEST = Path("reports/streamlit_snapshot_latest.json")
REMOTE_SNAPSHOT_URLS = [
    "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json",
    "https://raw.githubusercontent.com/yanischaker01-bit/yanis/main/reports/streamlit_snapshot_latest.json",
]

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_MODEL_METEOFRANCE = "meteofrance_seamless"
OPEN_METEO_ARCHIVE_LOOKBACK_DAYS = 35
OPEN_METEO_ARCHIVE_BATCH_SIZE = 40
OPEN_METEO_ARCHIVE_TIMEOUT_S = 20
OPEN_METEO_ARCHIVE_UNITARY_FALLBACK_MAX_BATCH = 3
METEO_FRANCE_TOKEN_URL_DEFAULT = "https://portail-api.meteofrance.fr/token"
METEO_FRANCE_DP_OBS_BASE_URL = "https://public-api.meteofrance.fr/public/DPObs"
METEO_FRANCE_VIGILANCE_URL = "https://public-api.meteofrance.fr/public/DPVigilance/v1/cartevigilance/encours"
PUBLIC_VIGILANCE_DEPARTEMENT_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "weatherref-france-vigilance-meteo-departement/records"
)
METEO_FRANCE_SOURCE_LABEL = "Meteo-France Portail API"
LGV_VIGILANCE_DEPARTMENTS = {
    "16": "Charente",
    "17": "Charente-Maritime",
    "33": "Gironde",
    "37": "Indre-et-Loire",
    "79": "Deux-Sevres",
    "86": "Vienne",
}
VIGILANCE_PHENOMENON_LABELS = {
    "1": "Vent violent",
    "2": "Pluie-inondation",
    "3": "Orages (dont foudre)",
    "4": "Crues",
    "5": "Neige-verglas",
    "6": "Canicule",
    "7": "Grand froid",
    "8": "Avalanches",
    "9": "Vagues-submersion",
}
VIGILANCE_COLOR_LABELS = {
    "1": "Vert",
    "2": "Jaune",
    "3": "Orange",
    "4": "Rouge",
}
VIGILANCE_COLOR_HEX = {
    "1": "#31aa35",
    "2": "#fff600",
    "3": "#ffb82b",
    "4": "#CC0000",
}
OPEN_METEO_SOURCE_LABEL = "Open-Meteo MeteoFrance"
OPEN_METEO_ARCHIVE_SOURCE_LABEL = "Open-Meteo archive (communes LGV)"
SOURCE_MODE_METEOFRANCE = "Meteo-France Portail API"
SOURCE_MODE_OPEN = "Open-Meteo MeteoFrance"
SOURCE_MODE_MIX = "Meteo-France + Open-Meteo + InfoClimat"
INFOCLIMAT_PRIORITY_MATCH_KM = 25.0
INFOCLIMAT_STRICT_RADIUS_KM = 10.0
INFOCLIMAT_MIN_STATIONS_COVERAGE = 4
INFOCLIMAT_ADAPTIVE_RADII_KM = [10.0, 20.0, 30.0, 40.0, 50.0]
OPEN_METEO_COMMUNE_OVERRIDES = {
    "openmeteo_ref_07412": "Cognac",
    "openmeteo_ref_07510": "Merignac",
}
INFOCLIMAT_PRIORITY_STATIONS = [
    {"name": "ST GERVAIS", "commune": "Saint-Gervais", "lat": 45.03, "lon": -0.47, "aliases": ["MF33415001", "ST GERVAIS"]},
    {"name": "MONTLIEU_SAPC", "commune": "Montlieu-la-Garde", "lat": 45.22, "lon": -0.29, "aliases": ["MF17243002", "MONTLIEU"]},
    {"name": "PASSIRAC", "commune": "Passirac", "lat": 45.33, "lon": -0.08, "aliases": ["MF16256001", "PASSIRAC"]},
    {"name": "LA COURONNE", "commune": "La Couronne", "lat": 45.63, "lon": 0.10, "aliases": ["MF16113001", "LA COURONNE"]},
    {"name": "Angouleme - Brie-Champnier", "commune": "Brie", "lat": 45.73, "lon": 0.22, "aliases": ["07420", "MF16078001", "ANGOULEME"]},
    {"name": "BRUX_SAPC", "commune": "Brux", "lat": 46.28, "lon": 0.19, "aliases": ["MF86039001", "BRUX"]},
    {"name": "JOUE-LES-TOURS OB", "commune": "Joue-les-Tours", "lat": 47.33, "lon": 0.66, "aliases": ["MF37122001", "JOUE LES TOURS"]},
    {"name": "SAINT-EPAIN", "commune": "Saint-Epain", "lat": 47.16, "lon": 0.60, "aliases": ["MF37216003", "SAINT-EPAIN", "ST EPAIN"]},
    {"name": "Les Ormes", "commune": "Les Ormes", "lat": 46.97, "lon": 0.60, "aliases": ["LES ORMES"]},
    {"name": "Naintre", "commune": "Naintre", "lat": 46.76, "lon": 0.48, "aliases": ["NAINTRE"]},
    {"name": "Poitiers-Biard", "commune": "Poitiers", "lat": 46.59, "lon": 0.31, "aliases": ["07335", "MF86027001", "POITIERS-BIARD"]},
]

HISTORY_MIN_DATE = date(2026, 1, 1)
ENABLE_RAIN_30D = True
RAIN_METRICS = {
    "24h": "rain_24h_mm",
    "7 jours": "rain_7d_mm",
    **({"30 jours": "rain_30d_mm"} if ENABLE_RAIN_30D else {}),
    "Mois courant": "rain_month_mm",
}
INFOCLIMAT_HISTORY_SOURCE = "InfoClimat SYNOP (local)"
HISTORY_SOURCES = [
    INFOCLIMAT_HISTORY_SOURCE,
]
HISTORY_SOURCE_COLORS = {
    INFOCLIMAT_HISTORY_SOURCE: "#0f766e",
    METEO_FRANCE_SOURCE_LABEL: "#0b3d91",
    OPEN_METEO_SOURCE_LABEL: "#1d4ed8",
}
MONTHLY_ALERT_COLORS = {
    "FAIBLE": "#16a34a",
    "MODERE": "#65a30d",
    "ELEVE": "#ea580c",
    "CRITIQUE": "#dc2626",
    "INCONNU": "#64748b",
}
RELIABILITY_BORDER_COLORS = {
    "FIABLE": "#0f766e",
    "SURVEILLER": "#b45309",
    "A_VERIFIER": "#7f1d1d",
}
OPEN_METEO_LOCAL_MAX_AGE_H = 72.0
WEATHER_STALE_ALERT_H = 72.0
SOURCE_RELIABILITY_HINTS = {
    "SYNOP": 95.0,
    "INFOCLIMAT": 95.0,
    "METEOFRANCE": 93.0,
    "OPEN_METEO": 84.0,
}
COMMUNE_MATCH_MODE_LABELS = {
    "code_exact": "Code commune exact",
    "commune_name": "Nom de commune",
    "nearest_station": "Station la plus proche",
    "no_data": "Sans donnee",
}
LOCAL_INFOCLIMAT_RADIUS_KM = 250.0
MAP_TILE_STYLES = {
    "Google Hybrid": {
        "tiles": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        "attr": "Google",
    },
    "Google Satellite": {
        "tiles": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attr": "Google",
    },
    "OpenStreetMap": {
        "tiles": "OpenStreetMap",
        "attr": "OpenStreetMap",
    },
}


def _http_get_with_retry(
    url: str,
    params: Dict[str, object],
    timeout: int = 30,
    max_attempts: int = 2,
    headers: Dict[str, str] | None = None,
) -> requests.Response:
    last_exc: Exception | None = None
    session = requests.Session()
    session.trust_env = False
    for _ in range(max(1, int(max_attempts))):
        try:
            return session.get(url, params=params, timeout=int(timeout), headers=headers)
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("HTTP request failed")


def _extract_lgv_lines(payload: Dict[str, object]) -> List[List[Tuple[float, float]]]:
    lines: List[List[Tuple[float, float]]] = []
    raw_lines = payload.get("lgv_lines", []) if isinstance(payload, dict) else []
    for line in raw_lines if isinstance(raw_lines, list) else []:
        coords: List[Tuple[float, float]] = []
        for pt in line if isinstance(line, list) else []:
            if not isinstance(pt, dict):
                continue
            lat = pd.to_numeric(pt.get("lat"), errors="coerce")
            lon = pd.to_numeric(pt.get("lon"), errors="coerce")
            if pd.isna(lat) or pd.isna(lon):
                continue
            coords.append((float(lat), float(lon)))
        if len(coords) >= 2:
            lines.append(coords)
    return lines


def _safe_weather_df(payload: Dict[str, object]) -> pd.DataFrame:
    df = pd.DataFrame(payload.get("weather") or [])
    if df.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "station_display",
                "station_commune_name",
                "source",
                "date_obs_raw",
                "latitude",
                "longitude",
                "distance_to_lgv_km",
                "rain_24h_mm",
                "rain_7d_mm",
                "rain_30d_mm",
                "rain_month_mm",
                "selection_mode",
                "rain_calc_method",
                "_obs_ts",
            ]
        )
    for col in [
        "distance_to_lgv_km",
        "latitude",
        "longitude",
        "rain_24h_mm",
        "rain_7d_mm",
        "rain_30d_mm",
        "rain_month_mm",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, default in [
        ("station_id", "station_inconnue"),
        ("source", "source_inconnue"),
        ("station_commune_name", "Inconnue"),
        ("date_obs_raw", ""),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default).astype(str)
    station_name_candidates = []
    for name_col in ["station_name", "name", "nom", "libelle_station", "libelle", "nom_station"]:
        if name_col in df.columns:
            station_name_candidates.append(df[name_col].fillna("").astype(str).str.strip())
    if station_name_candidates:
        station_name = station_name_candidates[0].copy()
        for ser in station_name_candidates[1:]:
            station_name = station_name.where(station_name.astype(str).str.len() > 0, ser)
    else:
        station_name = pd.Series("", index=df.index, dtype=str)
    commune_name = df["station_commune_name"].fillna("Inconnue").astype(str).str.strip()
    station_id = df["station_id"].fillna("station_inconnue").astype(str).str.strip()
    station_name = station_name.fillna("").astype(str).str.strip()

    # Hard overrides requested for specific Open-Meteo reference ids.
    for idx in df.index:
        sid = str(station_id.loc[idx]).strip().lower()
        forced_commune = OPEN_METEO_COMMUNE_OVERRIDES.get(sid, "")
        if forced_commune:
            commune_name.loc[idx] = str(forced_commune)

    # Enrich commune names: priority station mapping first, then nearest LGV commune by coordinates.
    for idx in df.index:
        if not _is_unknown_commune(commune_name.loc[idx]):
            continue
        inferred = _priority_commune_from_row(
            station_id=station_id.loc[idx],
            station_name=station_name.loc[idx],
            commune_name=commune_name.loc[idx],
        )
        if inferred:
            commune_name.loc[idx] = inferred

    for idx in df.index:
        if not _is_unknown_commune(commune_name.loc[idx]):
            continue
        lat = pd.to_numeric(df.loc[idx, "latitude"], errors="coerce") if "latitude" in df.columns else np.nan
        lon = pd.to_numeric(df.loc[idx, "longitude"], errors="coerce") if "longitude" in df.columns else np.nan
        if pd.isna(lat) or pd.isna(lon):
            continue
        nearest = _nearest_lgv_commune_name(float(lat), float(lon), max_km=45.0)
        if nearest:
            commune_name.loc[idx] = nearest

    commune_name = commune_name.fillna("Inconnue").astype(str).str.strip()
    df["station_commune_name"] = commune_name

    invalid_name = (
        (station_name.str.len() <= 0)
        | (station_name.str.lower() == station_id.str.lower())
        | (station_name.str.lower().isin({"station_inconnue", "inconnue", "unknown"}))
    )
    station_name = station_name.where(~invalid_name, commune_name)
    station_name = station_name.where(station_name.astype(str).str.len() > 0, commune_name)
    df["station_name"] = station_name.astype(str)

    same_as_commune = station_name.str.lower() == commune_name.str.lower()
    df["station_display"] = np.where(
        same_as_commune,
        commune_name + " (" + station_id + ")",
        station_name + " (" + commune_name + " - " + station_id + ")",
    )
    if "obs_ts_utc" in df.columns:
        obs_ts = pd.to_datetime(df["obs_ts_utc"], utc=True, errors="coerce")
    else:
        obs_ts = pd.to_datetime(df["date_obs_raw"], utc=True, errors="coerce")
    df["_obs_ts"] = obs_ts
    df = df.sort_values("_obs_ts", ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["station_id", "source"], keep="first")
    return df.reset_index(drop=True)


def _multiselect_all(label: str, options: List[str], key: str) -> List[str]:
    clean = sorted({str(v).strip() for v in options if str(v).strip()})
    if not clean:
        st.multiselect(label, ["Tout"], default=["Tout"], key=key, disabled=True)
        return []
    ui_opts = ["Tout"] + clean
    picked = st.multiselect(label, ui_opts, default=["Tout"], key=key)
    if not picked or "Tout" in picked:
        return clean
    return [v for v in clean if v in picked]


def _is_infoclimat_source(source: object) -> bool:
    txt = str(source or "").strip().lower()
    return ("infoclimat" in txt) or ("info_climat" in txt) or ("synop" in txt)


def _is_open_meteo_source(source: object) -> bool:
    txt = str(source or "").strip().lower()
    return ("open" in txt) and ("meteo" in txt)


def _is_infoclimat_row(row: Dict[str, object]) -> bool:
    src_txt = str(row.get("source") or "").strip().lower()
    sel_txt = str(row.get("selection_mode") or "").strip().lower()
    calc_txt = str(row.get("rain_calc_method") or "").strip().lower()
    return (
        _is_infoclimat_source(src_txt)
        or ("info_climat" in sel_txt)
        or ("infoclimat" in sel_txt)
        or ("synop" in sel_txt)
        or ("synop" in calc_txt)
    )


def _create_base_map(location: List[float], zoom_start: int, map_style: str) -> folium.Map:
    style = MAP_TILE_STYLES.get(str(map_style), MAP_TILE_STYLES["Google Hybrid"])
    tile_value = str(style.get("tiles") or "")
    if tile_value.lower().startswith("http"):
        fmap = folium.Map(location=location, zoom_start=zoom_start, tiles=None)
        folium.TileLayer(
            tiles=tile_value,
            attr=str(style.get("attr") or "Map"),
            name=str(map_style),
            overlay=False,
            control=False,
        ).add_to(fmap)
        return fmap
    return folium.Map(location=location, zoom_start=zoom_start, tiles=tile_value)


def _find_latest_file(patterns: List[str]) -> Path | None:
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend([p for p in Path(".").glob(pat) if p.is_file()])
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _normalize_commune_name(txt: object) -> str:
    raw = str(txt or "").strip().lower()
    if not raw:
        return ""
    normalized = unicodedata.normalize("NFKD", raw)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("-", " ").replace("'", " ")
    return " ".join(normalized.split())


@st.cache_data(show_spinner=False, ttl=86400)
def _load_lgv_communes_catalog() -> pd.DataFrame:
    src = _find_latest_file(
        [
            "data/lgv_communes_cache.json",
            "data/lgv_communes_*.json",
            "seed_data/lgv_communes_seed.json",
            "seed_data/lgv_communes_*.json",
        ]
    )
    if src is None:
        return pd.DataFrame()
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    communes: List[Dict[str, object]] = []
    if isinstance(payload, dict):
        raw = payload.get("communes", [])
        if isinstance(raw, list):
            communes = [r for r in raw if isinstance(r, dict)]
    elif isinstance(payload, list):
        communes = [r for r in payload if isinstance(r, dict)]
    if not communes:
        return pd.DataFrame()

    df = pd.DataFrame(communes)
    if df.empty:
        return pd.DataFrame()
    for col in ["commune_code", "commune_name"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()
    for col in ["centroid_latitude", "centroid_longitude"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["centroid_latitude", "centroid_longitude"])
    if df.empty:
        return pd.DataFrame()
    df["commune_name_norm"] = df["commune_name"].map(_normalize_commune_name)
    return (
        df[["commune_code", "commune_name", "commune_name_norm", "centroid_latitude", "centroid_longitude"]]
        .drop_duplicates(subset=["commune_code", "commune_name_norm"], keep="first")
        .reset_index(drop=True)
    )


def _resolve_history_reference_point(
    station_row: Dict[str, object],
    reference_mode: str,
) -> Tuple[float | None, float | None, str, str]:
    station_lat = pd.to_numeric(station_row.get("latitude"), errors="coerce")
    station_lon = pd.to_numeric(station_row.get("longitude"), errors="coerce")
    station_display = str(station_row.get("station_display") or station_row.get("station_id") or "station")

    if str(reference_mode) != "Commune LGV (aligne app PRO)":
        if pd.isna(station_lat) or pd.isna(station_lon):
            return None, None, "Point station indisponible", "Coordonnees station invalides."
        return (
            float(station_lat),
            float(station_lon),
            f"Point station: {station_display}",
            "",
        )

    catalog = _load_lgv_communes_catalog()
    if catalog.empty:
        if pd.isna(station_lat) or pd.isna(station_lon):
            return None, None, "Commune LGV indisponible", "Catalogue communes LGV introuvable."
        return (
            float(station_lat),
            float(station_lon),
            f"Fallback point station: {station_display}",
            "Catalogue communes LGV indisponible, bascule sur le point station.",
        )

    code_candidates = []
    for key in ["station_commune_code", "commune_code", "insee_code"]:
        val = str(station_row.get(key) or "").strip()
        if val:
            code_candidates.append(val)
    code_candidates = [c for i, c in enumerate(code_candidates) if c and c not in code_candidates[:i]]

    match = pd.DataFrame()
    if code_candidates:
        match = catalog[catalog["commune_code"].isin(code_candidates)].copy()
    if match.empty:
        commune_name = str(station_row.get("station_commune_name") or "").strip()
        commune_norm = _normalize_commune_name(commune_name)
        if commune_norm:
            match = catalog[catalog["commune_name_norm"] == commune_norm].copy()

    if not match.empty:
        best = match.iloc[0]
        lat = pd.to_numeric(best.get("centroid_latitude"), errors="coerce")
        lon = pd.to_numeric(best.get("centroid_longitude"), errors="coerce")
        if pd.notna(lat) and pd.notna(lon):
            name = str(best.get("commune_name") or "Inconnue")
            code = str(best.get("commune_code") or "")
            return float(lat), float(lon), f"Centroide commune LGV: {name} ({code})", ""

    if pd.isna(station_lat) or pd.isna(station_lon):
        return None, None, "Commune LGV non resolue", "Impossible de resoudre la commune LGV et coordonnees station invalides."
    return (
        float(station_lat),
        float(station_lon),
        f"Fallback point station: {station_display}",
        "Commune LGV non resolue, bascule sur le point station.",
    )


def _clean_station_label(txt: object) -> str:
    out = str(txt or "").strip()
    if not out:
        return ""
    out = out.replace("_", " ").replace(";", " ")
    out = " ".join(out.split())
    return out


def _infer_commune_from_station_name(station_name: object) -> str:
    name = _clean_station_label(station_name)
    if not name:
        return "Inconnue"
    for sep in [" [", " / ", " - "]:
        if sep in name:
            name = name.split(sep)[0].strip()
    if "-" in name:
        name = name.split("-")[0].strip()
    return name.title() if name else "Inconnue"


@st.cache_data(show_spinner=False, ttl=86400)
def _load_synop_station_name_lookup(station_ids: Tuple[str, ...]) -> Dict[str, str]:
    targets = {str(s).strip() for s in station_ids if str(s).strip()}
    if not targets:
        return {}
    cache_path = _find_latest_file(["data/synop_cache/synop_*.csv.gz", "data/synop_cache/synop.*.csv"])
    if cache_path is None:
        return {}

    out: Dict[str, str] = {}
    try:
        usecols_candidates = {"geo_id_wmo", "numer_sta", "station_id", "name", "nom_station", "libelle_station"}
        stream = pd.read_csv(
            cache_path,
            sep=";",
            dtype=str,
            usecols=lambda c: c in usecols_candidates,
            chunksize=250000,
            low_memory=False,
            compression="infer",
        )
        for chunk in stream:
            if chunk.empty:
                continue
            id_col = next((c for c in ["geo_id_wmo", "numer_sta", "station_id"] if c in chunk.columns), None)
            name_col = next((c for c in ["name", "libelle_station", "nom_station"] if c in chunk.columns), None)
            if id_col is None or name_col is None:
                continue
            sub = chunk[[id_col, name_col]].dropna(subset=[id_col])
            if sub.empty:
                continue
            sub[id_col] = sub[id_col].astype(str).str.strip()
            sub = sub[sub[id_col].isin(targets)]
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                sid = str(row[id_col]).strip()
                name = _clean_station_label(row[name_col])
                if sid and name and sid not in out:
                    out[sid] = name
            if len(out) >= len(targets):
                break
    except Exception:
        return out
    return out


@st.cache_data(show_spinner=False, ttl=1800)
def _load_infoclimat_synop_local(max_distance_km: float = LOCAL_INFOCLIMAT_RADIUS_KM) -> Tuple[pd.DataFrame, str]:
    synop_path = _find_latest_file(
        [
            "data/synop_all_stations_*.csv",
            "seed_data/synop_all_stations_seed.csv",
        ]
    )
    if synop_path is None:
        return pd.DataFrame(), (
            "InfoClimat/SYNOP local: aucun fichier synop_all_stations_*.csv disponible "
            "(ni seed_data/synop_all_stations_seed.csv)."
        )
    try:
        df = pd.read_csv(synop_path, dtype=str)
    except Exception as exc:
        return pd.DataFrame(), f"InfoClimat/SYNOP local: lecture impossible ({exc})."
    if df.empty:
        return pd.DataFrame(), f"InfoClimat/SYNOP local: fichier vide ({synop_path.name})."
    if "station_id" not in df.columns:
        return pd.DataFrame(), f"InfoClimat/SYNOP local: colonne station_id absente ({synop_path.name})."

    for col in [
        "distance_to_lgv_km",
        "latitude",
        "longitude",
        "rain_24h_mm",
        "rain_7d_mm",
        "rain_30d_mm",
        "rain_month_mm",
        "rain_12h_mm",
        "rain_instant_mm",
        "rain_forecast_mm",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "distance_to_lgv_km" in df.columns:
        df = df[pd.to_numeric(df["distance_to_lgv_km"], errors="coerce").fillna(9999.0) <= float(max_distance_km)].copy()
    if df.empty:
        return pd.DataFrame(), f"InfoClimat/SYNOP local: 0 station <= {float(max_distance_km):.0f} km LGV."

    station_ids = tuple(sorted(df["station_id"].astype(str).str.strip().dropna().unique().tolist()))
    name_lookup = _load_synop_station_name_lookup(station_ids)
    df["station_name"] = df["station_id"].astype(str).map(name_lookup).fillna("")
    df["station_name"] = df["station_name"].astype(str).map(_clean_station_label)

    if "station_commune_name" not in df.columns:
        df["station_commune_name"] = ""
    df["station_commune_name"] = df["station_commune_name"].fillna("").astype(str).str.strip()
    df["station_commune_name"] = np.where(
        df["station_commune_name"].astype(str).str.len() > 0,
        df["station_commune_name"],
        df["station_name"].map(_infer_commune_from_station_name),
    )
    df["station_commune_name"] = df["station_commune_name"].fillna("Inconnue").astype(str)

    if "source" not in df.columns:
        df["source"] = "infoclimat_synop"
    else:
        df["source"] = "infoclimat_synop"
    if "selection_mode" not in df.columns:
        df["selection_mode"] = "infoclimat_synop_local_cache"
    if "date_obs_raw" not in df.columns:
        if "date" in df.columns:
            df["date_obs_raw"] = df["date"].fillna("").astype(str)
        else:
            df["date_obs_raw"] = ""
    if "rain_class" not in df.columns:
        df["rain_class"] = "NORMAL"

    keep_cols = [
        "station_id",
        "station_name",
        "station_commune_name",
        "date_obs_raw",
        "latitude",
        "longitude",
        "distance_to_lgv_km",
        "precipitation_mm",
        "rain_24h_mm",
        "rain_7d_mm",
        "rain_30d_mm",
        "rain_month_mm",
        "rain_12h_mm",
        "rain_instant_mm",
        "rain_forecast_mm",
        "rain_class",
        "source",
        "selection_mode",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    notice = (
        f"InfoClimat/SYNOP local: {len(out)} stations proches chargees "
        f"(<= {float(max_distance_km):.0f} km LGV) depuis {synop_path.name}."
    )
    return out, notice


def _build_open_meteo_reference_key(stations_df: pd.DataFrame) -> Tuple[Tuple[str, str, str, float, float, float], ...]:
    if stations_df.empty:
        return tuple()
    work = stations_df.copy()
    work["latitude"] = pd.to_numeric(work.get("latitude"), errors="coerce")
    work["longitude"] = pd.to_numeric(work.get("longitude"), errors="coerce")
    if "distance_to_lgv_km" not in work.columns:
        work["distance_to_lgv_km"] = np.nan
    work["distance_to_lgv_km"] = pd.to_numeric(work.get("distance_to_lgv_km"), errors="coerce")
    work = work.dropna(subset=["latitude", "longitude"]).copy()
    work = work.drop_duplicates(subset=["station_id"], keep="first")
    if work.empty:
        return tuple()
    rows: List[Tuple[str, str, str, float, float, float]] = []
    for _, row in work.iterrows():
        sid = str(row.get("station_id") or "").strip()
        if not sid:
            continue
        sname = _clean_station_label(row.get("station_name") or "")
        commune = str(row.get("station_commune_name") or "").strip()
        lat = float(pd.to_numeric(row.get("latitude"), errors="coerce"))
        lon = float(pd.to_numeric(row.get("longitude"), errors="coerce"))
        dist = pd.to_numeric(row.get("distance_to_lgv_km"), errors="coerce")
        dist_val = float(dist) if pd.notna(dist) else -1.0
        rows.append((sid, sname, commune, lat, lon, dist_val))
    rows.sort(key=lambda x: x[0])
    return tuple(rows)


def _build_open_meteo_key_from_lgv_communes() -> Tuple[Tuple[str, str, str, float, float, float], ...]:
    communes_df = _load_lgv_communes_catalog()
    if communes_df.empty:
        return tuple()
    work = communes_df.copy()
    work["centroid_latitude"] = pd.to_numeric(work.get("centroid_latitude"), errors="coerce")
    work["centroid_longitude"] = pd.to_numeric(work.get("centroid_longitude"), errors="coerce")
    work = work.dropna(subset=["centroid_latitude", "centroid_longitude"]).copy()
    if work.empty:
        return tuple()

    rows: List[Tuple[str, str, str, float, float, float]] = []
    for _, row in work.iterrows():
        code = str(row.get("commune_code") or "").strip()
        commune = str(row.get("commune_name") or "").strip()
        if not commune:
            continue
        sid = code if code else f"LGVCOMM_{len(rows)+1:03d}"
        lat = float(pd.to_numeric(row.get("centroid_latitude"), errors="coerce"))
        lon = float(pd.to_numeric(row.get("centroid_longitude"), errors="coerce"))
        # Commune traversed by LGV SEA -> considered as on-corridor reference.
        rows.append((sid, commune, commune, lat, lon, 0.0))
    rows.sort(key=lambda x: x[0])
    return tuple(rows)


@st.cache_data(show_spinner=False, ttl=1800)
def _load_open_meteo_grid_local() -> Tuple[pd.DataFrame, str]:
    src = _find_latest_file(
        [
            "data/open_meteo_lgv_grid_*.csv",
            "seed_data/open_meteo_lgv_grid_seed.csv",
            "seed_data/open_meteo_lgv_grid_*.csv",
        ]
    )
    if src is None:
        return pd.DataFrame(), (
            "Open-Meteo grille locale: aucun fichier open_meteo_lgv_grid_*.csv disponible "
            "(ni seed_data/open_meteo_lgv_grid_seed.csv)."
        )
    try:
        df = pd.read_csv(src, dtype=str)
    except Exception as exc:
        return pd.DataFrame(), f"Open-Meteo grille locale: lecture impossible ({exc})."
    if df.empty:
        return pd.DataFrame(), f"Open-Meteo grille locale: fichier vide ({src.name})."

    work = df.copy()
    for col in [
        "distance_to_lgv_km",
        "latitude",
        "longitude",
        "precipitation_mm",
        "rain_24h_mm",
        "rain_7d_mm",
        "rain_30d_mm",
        "rain_month_mm",
        "rain_12h_mm",
        "rain_instant_mm",
        "rain_forecast_mm",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["latitude", "longitude"]).copy()
    if work.empty:
        return pd.DataFrame(), f"Open-Meteo grille locale: coordonnees invalides ({src.name})."

    # Map each grid point to nearest traversed LGV commune for professional naming.
    catalog = _load_lgv_communes_catalog()
    if not catalog.empty:
        cat = catalog.copy()
        cat["centroid_latitude"] = pd.to_numeric(cat.get("centroid_latitude"), errors="coerce")
        cat["centroid_longitude"] = pd.to_numeric(cat.get("centroid_longitude"), errors="coerce")
        cat = cat.dropna(subset=["centroid_latitude", "centroid_longitude"]).reset_index(drop=True)
    else:
        cat = pd.DataFrame()

    nearest_codes: List[str] = []
    nearest_names: List[str] = []
    nearest_distances: List[float] = []
    for _, row in work.iterrows():
        lat = pd.to_numeric(row.get("latitude"), errors="coerce")
        lon = pd.to_numeric(row.get("longitude"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon) or cat.empty:
            nearest_codes.append("")
            nearest_names.append("")
            nearest_distances.append(np.nan)
            continue
        dvals = [
            _haversine_km(
                float(lat),
                float(lon),
                float(crow.get("centroid_latitude")),
                float(crow.get("centroid_longitude")),
            )
            for _, crow in cat.iterrows()
        ]
        best_idx = int(np.argmin(dvals))
        best_row = cat.iloc[best_idx]
        nearest_codes.append(str(best_row.get("commune_code") or "").strip())
        nearest_names.append(str(best_row.get("commune_name") or "").strip())
        nearest_distances.append(float(dvals[best_idx]))

    work["nearest_lgv_commune_code"] = nearest_codes
    work["nearest_lgv_commune_name"] = nearest_names
    work["nearest_lgv_commune_km"] = nearest_distances

    if "station_commune_name" not in work.columns:
        work["station_commune_name"] = ""
    work["station_commune_name"] = work["station_commune_name"].fillna("").astype(str).str.strip()

    unknown_mask = work["station_commune_name"].map(_is_unknown_commune) | (work["station_commune_name"].astype(str).str.len() == 0)
    near_mask = pd.to_numeric(work.get("nearest_lgv_commune_km"), errors="coerce").fillna(9999.0) <= 8.0
    work["station_commune_name"] = np.where(
        unknown_mask & near_mask,
        work["nearest_lgv_commune_name"].fillna("").astype(str),
        work["station_commune_name"],
    )
    work["station_commune_name"] = np.where(
        work["station_commune_name"].astype(str).str.len() > 0,
        work["station_commune_name"],
        work["nearest_lgv_commune_name"].fillna("").astype(str),
    )
    work["station_commune_name"] = work["station_commune_name"].fillna("Inconnue").astype(str)

    if "station_ref_id" not in work.columns:
        work["station_ref_id"] = ""
    work["station_ref_id"] = work["station_ref_id"].fillna("").astype(str).str.strip()
    work["station_ref_id"] = np.where(
        work["station_ref_id"].astype(str).str.len() > 0,
        work["station_ref_id"],
        work["nearest_lgv_commune_code"].fillna("").astype(str),
    )

    if "station_id" not in work.columns:
        work["station_id"] = ""
    work["station_id"] = work["station_id"].fillna("").astype(str).str.strip()
    has_ref = work["station_ref_id"].astype(str).str.len() > 0
    work["station_id"] = np.where(has_ref, "openmeteo_ref_" + work["station_ref_id"], work["station_id"])
    missing_id = work["station_id"].astype(str).str.len() == 0
    if missing_id.any():
        replacement = [f"openmeteo_grid_{idx+1:04d}" for idx in range(int(missing_id.sum()))]
        work.loc[missing_id, "station_id"] = replacement

    if "station_name" not in work.columns:
        work["station_name"] = ""
    work["station_name"] = work["station_name"].fillna("").astype(str).map(_clean_station_label)
    work["station_name"] = np.where(
        work["station_name"].astype(str).str.len() > 0,
        work["station_name"],
        work["station_commune_name"].fillna("Inconnue").astype(str),
    )

    if "source" not in work.columns:
        work["source"] = OPEN_METEO_SOURCE_LABEL
    else:
        work["source"] = OPEN_METEO_SOURCE_LABEL
    if "selection_mode" not in work.columns:
        work["selection_mode"] = "open_meteo_grid_lgv_seed"
    else:
        mode_text = work["selection_mode"].fillna("").astype(str)
        work["selection_mode"] = np.where(mode_text.str.len() > 0, mode_text + ";seed_fallback", "open_meteo_grid_lgv_seed")
    if "meteo_model" not in work.columns:
        work["meteo_model"] = OPEN_METEO_MODEL_METEOFRANCE
    else:
        work["meteo_model"] = work["meteo_model"].fillna("").astype(str).replace("", OPEN_METEO_MODEL_METEOFRANCE)
    if "date_obs_raw" not in work.columns:
        if "date" in work.columns:
            work["date_obs_raw"] = work["date"].fillna("").astype(str)
        else:
            work["date_obs_raw"] = ""
    if "rain_class" not in work.columns:
        work["rain_class"] = "NORMAL"

    work["_obs_ts"] = pd.to_datetime(work.get("date_obs_raw", pd.Series("", index=work.index)), utc=True, errors="coerce")
    latest_obs_ts = pd.to_datetime(work["_obs_ts"], utc=True, errors="coerce").dropna().max()
    if pd.notna(latest_obs_ts):
        now_utc = pd.Timestamp.now(tz="UTC")
        fallback_age_h = float((now_utc - latest_obs_ts).total_seconds() / 3600.0)
        if fallback_age_h > float(OPEN_METEO_LOCAL_MAX_AGE_H):
            return pd.DataFrame(), (
                "Open-Meteo grille locale ignoree: fichier trop ancien pour un usage exploitation "
                + f"(obs={latest_obs_ts.isoformat()}, age={fallback_age_h:.1f} h, seuil={OPEN_METEO_LOCAL_MAX_AGE_H:.0f} h)."
            )
    work = work.sort_values(["_obs_ts", "distance_to_lgv_km"], ascending=[False, True], na_position="last")
    work = work.drop_duplicates(subset=["station_id"], keep="first")

    keep_cols = [
        "station_id",
        "station_name",
        "station_commune_name",
        "date_obs_raw",
        "latitude",
        "longitude",
        "distance_to_lgv_km",
        "precipitation_mm",
        "rain_24h_mm",
        "rain_7d_mm",
        "rain_30d_mm",
        "rain_month_mm",
        "rain_12h_mm",
        "rain_instant_mm",
        "rain_forecast_mm",
        "rain_class",
        "source",
        "selection_mode",
        "meteo_model",
        "station_ref_id",
        "date",
    ]
    keep_cols = [c for c in keep_cols if c in work.columns]
    out = work[keep_cols].copy()
    return out, f"Open-Meteo grille locale: {len(out)} points charges depuis {src.name}."


def _runtime_secret_or_env(*keys: str) -> str:
    for key in keys:
        k = str(key or "").strip()
        if not k:
            continue
        try:
            secret_val = st.secrets.get(k, "")
            if isinstance(secret_val, str) and secret_val.strip():
                return secret_val.strip()
        except Exception:
            pass
        env_val = os.getenv(k, "").strip()
        if env_val:
            return env_val
    return ""


def _meteo_france_portal_is_configured() -> bool:
    direct_token = _runtime_secret_or_env("METEOFRANCE_ACCESS_TOKEN", "METEO_FRANCE_ACCESS_TOKEN")
    if direct_token:
        return True
    client_id = _runtime_secret_or_env("METEOFRANCE_CLIENT_ID", "METEO_FRANCE_CLIENT_ID")
    client_secret = _runtime_secret_or_env("METEOFRANCE_CLIENT_SECRET", "METEO_FRANCE_CLIENT_SECRET")
    return bool(client_id and client_secret)


def _normalize_col_key(txt: object) -> str:
    return "".join(ch for ch in _ascii_norm(txt) if ch.isalnum())


def _parse_csv_flexible(text: str) -> pd.DataFrame:
    raw = str(text or "").strip()
    if not raw:
        return pd.DataFrame()
    for sep in [None, ";", ","]:
        try:
            if sep is None:
                df = pd.read_csv(StringIO(raw), sep=None, engine="python", dtype=str)
            else:
                df = pd.read_csv(StringIO(raw), sep=sep, engine="python", dtype=str)
            if not df.empty and len(df.columns) >= 2:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def _find_column_by_keys(df: pd.DataFrame, exact_keys: List[str], contains_tokens: List[str] | None = None) -> str:
    if df.empty:
        return ""
    norm_map: Dict[str, str] = {}
    for col in df.columns:
        key = _normalize_col_key(col)
        if key and key not in norm_map:
            norm_map[key] = str(col)
    for key in exact_keys:
        k = _normalize_col_key(key)
        if k in norm_map:
            return norm_map[k]
    if contains_tokens:
        for col in df.columns:
            nk = _normalize_col_key(col)
            if nk and all(_normalize_col_key(tok) in nk for tok in contains_tokens):
                return str(col)
    return ""


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.fillna("").astype(str).str.replace(",", ".", regex=False), errors="coerce")


@st.cache_data(show_spinner=False, ttl=3000)
def _request_meteo_france_portal_token() -> Tuple[str, str]:
    direct_token = _runtime_secret_or_env("METEOFRANCE_ACCESS_TOKEN", "METEO_FRANCE_ACCESS_TOKEN")
    if direct_token:
        return direct_token, "Meteo-France token: fourni via variable d'environnement/secrets."

    client_id = _runtime_secret_or_env("METEOFRANCE_CLIENT_ID", "METEO_FRANCE_CLIENT_ID")
    client_secret = _runtime_secret_or_env("METEOFRANCE_CLIENT_SECRET", "METEO_FRANCE_CLIENT_SECRET")
    if not client_id or not client_secret:
        return "", (
            "Meteo-France Portail API: configure METEOFRANCE_CLIENT_ID et METEOFRANCE_CLIENT_SECRET "
            "(ou METEOFRANCE_ACCESS_TOKEN) dans Streamlit secrets."
        )

    token_url = _runtime_secret_or_env("METEOFRANCE_TOKEN_URL", "METEO_FRANCE_TOKEN_URL")
    if not token_url:
        token_url = METEO_FRANCE_TOKEN_URL_DEFAULT

    session = requests.Session()
    session.trust_env = False
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    attempts = [
        {"grant_type": "client_credentials"},
        {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
    ]
    last_err = ""
    for payload in attempts:
        try:
            if "client_id" in payload:
                resp = session.post(token_url, data=payload, headers=headers, timeout=25)
            else:
                resp = session.post(token_url, data=payload, headers=headers, auth=(client_id, client_secret), timeout=25)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
                continue
            parsed = resp.json() if "application/json" in str(resp.headers.get("Content-Type", "")).lower() else {}
            access_token = str((parsed or {}).get("access_token") or "").strip()
            if access_token:
                return access_token, "Meteo-France token OAuth2 recupere."
            last_err = "access_token absent"
        except Exception as exc:
            last_err = str(exc)

    return "", f"Meteo-France token indisponible ({last_err})."


@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_meteo_france_synop_station_catalog(token: str) -> Tuple[pd.DataFrame, str]:
    tk = str(token or "").strip()
    if not tk:
        return pd.DataFrame(), "Meteo-France catalog stations: token manquant."
    headers = {"Authorization": f"Bearer {tk}", "accept": "text/csv"}
    url = f"{METEO_FRANCE_DP_OBS_BASE_URL}/liste-stations-synop"
    try:
        resp = _http_get_with_retry(url, params={"format": "csv"}, timeout=35, max_attempts=2, headers=headers)
    except Exception as exc:
        return pd.DataFrame(), f"Meteo-France catalog stations: erreur HTTP ({exc})."
    if resp.status_code != 200:
        return pd.DataFrame(), f"Meteo-France catalog stations: HTTP {resp.status_code}."

    df = _parse_csv_flexible(resp.text)
    if df.empty:
        return pd.DataFrame(), "Meteo-France catalog stations: CSV vide."

    id_col = _find_column_by_keys(
        df,
        exact_keys=["geo_id_wmo", "id_station", "id", "station_id", "numer_sta"],
        contains_tokens=["id", "station"],
    )
    if not id_col:
        id_col = _find_column_by_keys(df, exact_keys=["indicatif", "omm", "wmo"], contains_tokens=["wmo"])
    name_col = _find_column_by_keys(
        df,
        exact_keys=["nom_station", "station_name", "nom", "libelle_station", "station"],
        contains_tokens=["nom"],
    )
    lat_col = _find_column_by_keys(df, exact_keys=["latitude", "lat"], contains_tokens=["lat"])
    lon_col = _find_column_by_keys(df, exact_keys=["longitude", "lon", "long"], contains_tokens=["lon"])
    if not lon_col:
        lon_col = _find_column_by_keys(df, exact_keys=["longitude", "lon", "long"], contains_tokens=["long"])
    if not id_col:
        return pd.DataFrame(), "Meteo-France catalog stations: colonne id station introuvable."

    out = pd.DataFrame()
    out["station_api_id"] = df[id_col].fillna("").astype(str).str.strip()
    out["station_api_id"] = out["station_api_id"].apply(lambda v: "".join(ch for ch in str(v) if ch.isdigit())).astype(str)
    out["station_api_id"] = np.where(
        out["station_api_id"].astype(str).str.len() > 0,
        out["station_api_id"].astype(str).str.zfill(5),
        "",
    )
    out["station_name"] = (
        df[name_col].fillna("").astype(str).map(_clean_station_label) if name_col else pd.Series("", index=df.index, dtype=str)
    )
    out["latitude"] = _to_num(df[lat_col]) if lat_col else pd.Series(np.nan, index=df.index)
    out["longitude"] = _to_num(df[lon_col]) if lon_col else pd.Series(np.nan, index=df.index)
    out = out[(out["station_api_id"].astype(str).str.len() > 0)].copy()
    out = out.dropna(subset=["latitude", "longitude"], how="any")
    out = out.drop_duplicates(subset=["station_api_id"], keep="first")
    return out.reset_index(drop=True), f"Meteo-France catalog stations: {len(out)} stations geolocalisees."


@st.cache_data(show_spinner=False, ttl=900)
def _fetch_meteo_france_portal_commune_weather() -> Tuple[pd.DataFrame, str]:
    token, token_note = _request_meteo_france_portal_token()
    if not token:
        return pd.DataFrame(), token_note

    stations_df, stations_note = _fetch_meteo_france_synop_station_catalog(token)
    headers = {"Authorization": f"Bearer {token}", "accept": "text/csv"}
    synop_url = f"{METEO_FRANCE_DP_OBS_BASE_URL}/v1/synop"
    try:
        synop_resp = _http_get_with_retry(synop_url, params={"format": "csv"}, timeout=40, max_attempts=2, headers=headers)
    except Exception as exc:
        return pd.DataFrame(), f"{token_note} | SYNOP portail erreur HTTP ({exc})."
    if synop_resp.status_code != 200:
        return pd.DataFrame(), f"{token_note} | SYNOP portail HTTP {synop_resp.status_code}."

    synop_raw = _parse_csv_flexible(synop_resp.text)
    if synop_raw.empty:
        return pd.DataFrame(), f"{token_note} | SYNOP portail vide."

    id_col = _find_column_by_keys(
        synop_raw,
        exact_keys=["geo_id_wmo", "id_station", "station_id", "numer_sta", "id"],
        contains_tokens=["station"],
    )
    if not id_col:
        id_col = _find_column_by_keys(synop_raw, exact_keys=["wmo", "omm"], contains_tokens=["wmo"])
    dt_col = _find_column_by_keys(
        synop_raw,
        exact_keys=["validity_time", "reference_time", "date_obs_raw", "date"],
        contains_tokens=["date"],
    )
    lat_col = _find_column_by_keys(synop_raw, exact_keys=["latitude", "lat"], contains_tokens=["lat"])
    lon_col = _find_column_by_keys(synop_raw, exact_keys=["longitude", "lon", "long"], contains_tokens=["lon"])
    if not lon_col:
        lon_col = _find_column_by_keys(synop_raw, exact_keys=["longitude", "lon", "long"], contains_tokens=["long"])

    if not id_col:
        return pd.DataFrame(), f"{token_note} | SYNOP portail: colonne id station introuvable."

    synop = synop_raw.copy()
    synop["station_api_id"] = synop[id_col].fillna("").astype(str).str.strip()
    synop["station_api_id"] = synop["station_api_id"].apply(lambda v: "".join(ch for ch in str(v) if ch.isdigit())).astype(str)
    synop["station_api_id"] = np.where(
        synop["station_api_id"].astype(str).str.len() > 0,
        synop["station_api_id"].astype(str).str.zfill(5),
        "",
    )
    synop = synop[synop["station_api_id"].astype(str).str.len() > 0].copy()
    if synop.empty:
        return pd.DataFrame(), f"{token_note} | SYNOP portail: aucune station exploitable."

    synop["date_obs_raw"] = synop[dt_col].fillna("").astype(str) if dt_col else ""
    synop["_obs_ts"] = pd.to_datetime(synop["date_obs_raw"], utc=True, errors="coerce")
    if synop["_obs_ts"].isna().all():
        synop["_obs_ts"] = pd.Timestamp(datetime.now(timezone.utc))
        synop["date_obs_raw"] = synop["_obs_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if lat_col:
        synop["latitude"] = _to_num(synop[lat_col])
    else:
        synop["latitude"] = np.nan
    if lon_col:
        synop["longitude"] = _to_num(synop[lon_col])
    else:
        synop["longitude"] = np.nan

    rr1_col = _find_column_by_keys(synop, exact_keys=["rr1"], contains_tokens=["rr1"])
    rr3_col = _find_column_by_keys(synop, exact_keys=["rr3"], contains_tokens=["rr3"])
    rr6_col = _find_column_by_keys(synop, exact_keys=["rr6"], contains_tokens=["rr6"])
    rr12_col = _find_column_by_keys(synop, exact_keys=["rr12"], contains_tokens=["rr12"])
    rr24_col = _find_column_by_keys(synop, exact_keys=["rr24"], contains_tokens=["rr24"])
    rr_col = _find_column_by_keys(synop, exact_keys=["rr"], contains_tokens=None)
    pr_col = _find_column_by_keys(
        synop,
        exact_keys=["precipitation", "precip", "pluie"],
        contains_tokens=["precip"],
    )

    synop["rr1_mm"] = _to_num(synop[rr1_col]) if rr1_col else np.nan
    synop["rr3_mm"] = _to_num(synop[rr3_col]) if rr3_col else np.nan
    synop["rr6_mm"] = _to_num(synop[rr6_col]) if rr6_col else np.nan
    synop["rr12_mm"] = _to_num(synop[rr12_col]) if rr12_col else np.nan
    synop["rr24_mm"] = _to_num(synop[rr24_col]) if rr24_col else np.nan
    synop["rr_mm"] = _to_num(synop[rr_col]) if rr_col else np.nan
    synop["precip_mm"] = _to_num(synop[pr_col]) if pr_col else np.nan
    synop["step_mm"] = synop["rr1_mm"]
    for col in ["rr3_mm", "rr6_mm", "rr12_mm", "rr24_mm", "rr_mm", "precip_mm"]:
        synop["step_mm"] = synop["step_mm"].where(synop["step_mm"].notna(), synop[col])
    synop["step_mm"] = pd.to_numeric(synop["step_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)

    if not stations_df.empty:
        st_lookup = stations_df.copy()
        st_lookup["station_api_id"] = st_lookup["station_api_id"].astype(str).str.zfill(5)
        synop = synop.merge(
            st_lookup[["station_api_id", "station_name", "latitude", "longitude"]].rename(
                columns={"latitude": "lat_station", "longitude": "lon_station"}
            ),
            how="left",
            on="station_api_id",
        )
        synop["latitude"] = synop["latitude"].fillna(pd.to_numeric(synop.get("lat_station"), errors="coerce"))
        synop["longitude"] = synop["longitude"].fillna(pd.to_numeric(synop.get("lon_station"), errors="coerce"))
        synop["station_name"] = synop.get("station_name", pd.Series("", index=synop.index)).fillna("").astype(str).map(_clean_station_label)
    else:
        synop["station_name"] = ""

    synop = synop.dropna(subset=["latitude", "longitude"]).copy()
    if synop.empty:
        return pd.DataFrame(), f"{token_note} | SYNOP portail sans coordonnees station exploitables."

    now_utc = pd.to_datetime(synop["_obs_ts"], utc=True, errors="coerce").dropna().max()
    if pd.isna(now_utc):
        now_utc = pd.Timestamp(datetime.now(timezone.utc))
    lower_12h = now_utc - pd.Timedelta(hours=12)
    lower_24h = now_utc - pd.Timedelta(hours=24)
    lower_7d = now_utc - pd.Timedelta(days=7)
    lower_30d = now_utc - pd.Timedelta(days=30)
    month_start = pd.Timestamp(year=int(now_utc.year), month=int(now_utc.month), day=1, tz="UTC")

    station_rows: List[Dict[str, object]] = []
    for station_id, grp in synop.groupby("station_api_id"):
        g = grp.copy()
        g["_obs_ts"] = pd.to_datetime(g["_obs_ts"], utc=True, errors="coerce")
        g = g.dropna(subset=["_obs_ts"]).sort_values("_obs_ts")
        if g.empty:
            continue
        latest = g.iloc[-1]
        rain_24h_direct = pd.to_numeric(latest.get("rr24_mm"), errors="coerce")
        rain_12h_direct = pd.to_numeric(latest.get("rr12_mm"), errors="coerce")
        rain_instant = pd.to_numeric(latest.get("rr1_mm"), errors="coerce")
        if pd.isna(rain_instant):
            rain_instant = pd.to_numeric(latest.get("step_mm"), errors="coerce")

        step = pd.to_numeric(g.get("step_mm"), errors="coerce").fillna(0.0).clip(lower=0.0)
        ts = pd.to_datetime(g["_obs_ts"], utc=True, errors="coerce")
        span_h = float((ts.max() - ts.min()).total_seconds() / 3600.0) if len(ts) >= 2 else 0.0
        rain_12h = float(step[ts > lower_12h].sum()) if span_h >= 6.0 else np.nan
        rain_24h = float(step[ts > lower_24h].sum()) if span_h >= 18.0 else np.nan
        rain_7d = float(step[ts > lower_7d].sum()) if span_h >= 5.0 * 24.0 else np.nan
        rain_30d = float(step[ts > lower_30d].sum()) if span_h >= 25.0 * 24.0 else np.nan
        month_has_coverage = ts.min() <= month_start if not ts.empty else False
        rain_month = float(step[ts >= month_start].sum()) if month_has_coverage else np.nan

        rain_12h = max(rain_12h, float(rain_12h_direct) if pd.notna(rain_12h_direct) else 0.0)
        rain_24h = max(rain_24h, float(rain_24h_direct) if pd.notna(rain_24h_direct) else 0.0)
        rain_7d = max(rain_7d, rain_24h) if pd.notna(rain_7d) else np.nan
        rain_30d = max(rain_30d, rain_7d) if pd.notna(rain_30d) and pd.notna(rain_7d) else np.nan
        rain_month = max(rain_month, rain_24h) if pd.notna(rain_month) else np.nan
        rain_inst = max(0.0, float(rain_instant) if pd.notna(rain_instant) else 0.0)

        station_rows.append(
            {
                "station_api_id": str(station_id).zfill(5),
                "provider_station_name": _clean_station_label(latest.get("station_name") or f"Station {station_id}"),
                "date_obs_raw": str(latest.get("date_obs_raw") or latest.get("_obs_ts") or ""),
                "_obs_ts": pd.to_datetime(latest.get("_obs_ts"), utc=True, errors="coerce"),
                "latitude": float(pd.to_numeric(latest.get("latitude"), errors="coerce")),
                "longitude": float(pd.to_numeric(latest.get("longitude"), errors="coerce")),
                "precipitation_mm": round(float(rain_24h), 3),
                "rain_24h_mm": round(float(rain_24h), 3),
                "rain_7d_mm": round(float(rain_7d), 3),
                "rain_30d_mm": round(float(rain_30d), 3),
                "rain_month_mm": round(float(rain_month), 3),
                "rain_12h_mm": round(float(rain_12h), 3),
                "rain_instant_mm": round(float(rain_inst), 3),
                "rain_forecast_mm": 0.0,
                "rain_class": "NORMAL",
                "source": METEO_FRANCE_SOURCE_LABEL,
                "selection_mode": "meteo_france_portail_synop_station",
                "meteo_model": "mf_dpobs_synop",
                "history_span_h": round(float(span_h), 1),
                "rain_calc_method": "mf_portal_live_rr24_only_if_short_history",
            }
        )

    station_agg = pd.DataFrame(station_rows)
    if station_agg.empty:
        return pd.DataFrame(), f"{token_note} | SYNOP portail: station aggregates vides."

    communes = _load_lgv_communes_catalog()
    if communes.empty:
        out = station_agg.copy()
        out["station_id"] = "meteofrance_station_" + out["station_api_id"].astype(str)
        out["station_name"] = out["provider_station_name"]
        out["station_commune_name"] = out["provider_station_name"]
        out["distance_to_lgv_km"] = np.nan
        out["station_ref_id"] = out["station_api_id"]
        keep_cols = [
            "station_id",
            "station_name",
            "station_commune_name",
            "date_obs_raw",
            "latitude",
            "longitude",
            "distance_to_lgv_km",
            "precipitation_mm",
            "rain_24h_mm",
            "rain_7d_mm",
            "rain_30d_mm",
            "rain_month_mm",
            "rain_12h_mm",
            "rain_instant_mm",
            "rain_forecast_mm",
            "rain_class",
            "source",
            "selection_mode",
            "meteo_model",
            "station_ref_id",
        ]
        return out[keep_cols].copy(), (
            f"{token_note} | {stations_note} | SYNOP portail: {len(out)} stations (mode station, "
            "catalogue communes LGV indisponible)."
        )

    comm = communes.copy()
    comm["centroid_latitude"] = pd.to_numeric(comm.get("centroid_latitude"), errors="coerce")
    comm["centroid_longitude"] = pd.to_numeric(comm.get("centroid_longitude"), errors="coerce")
    comm = comm.dropna(subset=["centroid_latitude", "centroid_longitude"]).copy()
    if comm.empty:
        return pd.DataFrame(), f"{token_note} | {stations_note} | communes LGV sans centroide."

    proxy_rows: List[Dict[str, object]] = []
    for _, crow in comm.iterrows():
        clat = float(pd.to_numeric(crow.get("centroid_latitude"), errors="coerce"))
        clon = float(pd.to_numeric(crow.get("centroid_longitude"), errors="coerce"))
        dists = [
            _haversine_km(clat, clon, float(srow["latitude"]), float(srow["longitude"]))
            for _, srow in station_agg.iterrows()
        ]
        if not dists:
            continue
        best_i = int(np.argmin(dists))
        src_row = station_agg.iloc[best_i]
        ccode = str(crow.get("commune_code") or "").strip()
        cname = str(crow.get("commune_name") or "").strip()
        sid = f"meteofrance_ref_{ccode}" if ccode else f"meteofrance_ref_{best_i+1:03d}"
        proxy_rows.append(
            {
                "station_id": sid,
                "station_ref_id": ccode or str(src_row.get("station_api_id") or ""),
                "station_name": cname or str(src_row.get("provider_station_name") or sid),
                "station_commune_name": cname or str(src_row.get("provider_station_name") or "Inconnue"),
                "date_obs_raw": str(src_row.get("date_obs_raw") or ""),
                "latitude": clat,
                "longitude": clon,
                "distance_to_lgv_km": 0.0,
                "precipitation_mm": float(pd.to_numeric(src_row.get("precipitation_mm"), errors="coerce") or 0.0),
                "rain_24h_mm": float(pd.to_numeric(src_row.get("rain_24h_mm"), errors="coerce") or 0.0),
                "rain_7d_mm": float(pd.to_numeric(src_row.get("rain_7d_mm"), errors="coerce") or 0.0),
                "rain_30d_mm": float(pd.to_numeric(src_row.get("rain_30d_mm"), errors="coerce") or 0.0),
                "rain_month_mm": float(pd.to_numeric(src_row.get("rain_month_mm"), errors="coerce") or 0.0),
                "rain_12h_mm": float(pd.to_numeric(src_row.get("rain_12h_mm"), errors="coerce") or 0.0),
                "rain_instant_mm": float(pd.to_numeric(src_row.get("rain_instant_mm"), errors="coerce") or 0.0),
                "rain_forecast_mm": 0.0,
                "rain_class": str(src_row.get("rain_class") or "NORMAL"),
                "source": METEO_FRANCE_SOURCE_LABEL,
                "selection_mode": "meteo_france_portail_synop_commune_proxy",
                "meteo_model": "mf_dpobs_synop",
                "provider_station_id": str(src_row.get("station_api_id") or ""),
                "provider_station_name": str(src_row.get("provider_station_name") or ""),
                "provider_station_dist_km": round(float(dists[best_i]), 3),
                "rain_calc_method": "mf_portal_synop_nearest_station",
            }
        )
    out = pd.DataFrame(proxy_rows)
    if out.empty:
        return pd.DataFrame(), f"{token_note} | {stations_note} | SYNOP portail: aucune commune LGV projetee."

    out = out.sort_values("station_id").reset_index(drop=True)
    notice = (
        f"{token_note} | {stations_note} | Meteo-France Portail: {len(out)} communes LGV projetees "
        f"depuis {len(station_agg)} stations SYNOP."
    )
    return out, notice


def _find_vigilance_domain_ids(node: object) -> List[Dict[str, object]]:
    found: List[Dict[str, object]] = []
    if isinstance(node, dict):
        if isinstance(node.get("domain_ids"), list):
            found.extend([d for d in node["domain_ids"] if isinstance(d, dict)])
        for value in node.values():
            found.extend(_find_vigilance_domain_ids(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_find_vigilance_domain_ids(item))
    return found


@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_meteo_france_vigilance(token: str) -> Tuple[pd.DataFrame, str]:
    tk = str(token or "").strip()
    if not tk:
        return pd.DataFrame(), "Meteo-France Vigilance: token indisponible."

    try:
        resp = _http_get_with_retry(
            METEO_FRANCE_VIGILANCE_URL,
            params={"format": "json"},
            timeout=25,
            headers={"Authorization": f"Bearer {tk}", "Accept": "application/json"},
        )
    except Exception as exc:
        return pd.DataFrame(), f"Meteo-France Vigilance: erreur reseau ({exc})."
    if resp.status_code != 200:
        return pd.DataFrame(), f"Meteo-France Vigilance: HTTP {resp.status_code} (verifier l'abonnement API Vigilance)."
    try:
        payload = resp.json()
    except Exception as exc:
        return pd.DataFrame(), f"Meteo-France Vigilance: reponse invalide ({exc})."

    domain_entries = [
        d for d in _find_vigilance_domain_ids(payload) if isinstance(d.get("phenomenon_items"), list)
    ]
    rows: List[Dict[str, object]] = []
    for entry in domain_entries:
        domain_id_raw = str(entry.get("domain_id") or "").strip()
        domain_id = domain_id_raw.zfill(2) if domain_id_raw.isdigit() else domain_id_raw
        if domain_id not in LGV_VIGILANCE_DEPARTMENTS:
            domain_id = domain_id_raw.lstrip("0") or domain_id_raw
        if domain_id not in LGV_VIGILANCE_DEPARTMENTS:
            continue
        for item in entry.get("phenomenon_items", []):
            if not isinstance(item, dict):
                continue
            phen_id = str(item.get("phenomenon_id") or "").strip()
            color_id = str(item.get("phenomenon_max_color_id") or "").strip()
            if not phen_id or not color_id:
                continue
            rows.append(
                {
                    "domain_id": domain_id,
                    "department_name": LGV_VIGILANCE_DEPARTMENTS.get(domain_id, domain_id),
                    "phenomenon_id": phen_id,
                    "phenomenon_name": VIGILANCE_PHENOMENON_LABELS.get(phen_id, f"Phenomene {phen_id}"),
                    "color_id": color_id,
                    "color_name": VIGILANCE_COLOR_LABELS.get(color_id, "Inconnu"),
                    "color_hex": VIGILANCE_COLOR_HEX.get(color_id, "#94a3b8"),
                }
            )

    if not rows:
        return pd.DataFrame(), "Meteo-France Vigilance: aucune donnee exploitable pour les departements LGV."

    out = pd.DataFrame(rows).drop_duplicates(subset=["domain_id", "phenomenon_id"], keep="first").reset_index(drop=True)
    max_color_id = int(pd.to_numeric(out["color_id"], errors="coerce").max())
    notice = (
        f"Meteo-France Vigilance: {out['domain_id'].nunique()} departement(s) LGV, "
        f"niveau max {VIGILANCE_COLOR_LABELS.get(str(max_color_id), '?')}."
    )
    return out, notice


@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_public_vigilance_departemental() -> Tuple[pd.DataFrame, str]:
    dept_clause = ",".join(f'"{d}"' for d in LGV_VIGILANCE_DEPARTMENTS)
    params = {"where": f'echeance="J" and domain_id in ({dept_clause})', "limit": 100}
    try:
        resp = _http_get_with_retry(PUBLIC_VIGILANCE_DEPARTEMENT_URL, params=params, timeout=20)
    except Exception as exc:
        return pd.DataFrame(), f"Vigilance (source publique): erreur reseau ({exc})."
    if resp.status_code != 200:
        return pd.DataFrame(), f"Vigilance (source publique): HTTP {resp.status_code}."
    try:
        payload = resp.json()
    except Exception as exc:
        return pd.DataFrame(), f"Vigilance (source publique): reponse invalide ({exc})."

    records = payload.get("results", []) if isinstance(payload, dict) else []
    if not records and isinstance(payload, dict):
        records = [r.get("fields", r) for r in payload.get("records", []) if isinstance(r, dict)]

    rows: List[Dict[str, object]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        domain_id = str(rec.get("domain_id") or "").strip()
        if domain_id not in LGV_VIGILANCE_DEPARTMENTS:
            continue
        phen_id = str(rec.get("phenomenon_id") or "").strip()
        color_id = str(rec.get("color_id") or "").strip()
        if not phen_id or not color_id:
            continue
        rows.append(
            {
                "domain_id": domain_id,
                "department_name": LGV_VIGILANCE_DEPARTMENTS.get(domain_id, domain_id),
                "phenomenon_id": phen_id,
                "phenomenon_name": VIGILANCE_PHENOMENON_LABELS.get(phen_id, f"Phenomene {phen_id}"),
                "color_id": color_id,
                "color_name": VIGILANCE_COLOR_LABELS.get(color_id, "Inconnu"),
                "color_hex": VIGILANCE_COLOR_HEX.get(color_id, "#94a3b8"),
            }
        )

    if not rows:
        return pd.DataFrame(), "Vigilance (source publique): aucune donnee exploitable."

    out = pd.DataFrame(rows)
    out["_color_num"] = pd.to_numeric(out["color_id"], errors="coerce")
    out = (
        out.sort_values("_color_num", ascending=False)
        .drop_duplicates(subset=["domain_id", "phenomenon_id"], keep="first")
        .drop(columns=["_color_num"])
        .reset_index(drop=True)
    )
    notice = f"Vigilance (source publique Opendatasoft, sans cle): {out['domain_id'].nunique()} departement(s) LGV."
    return out, notice


@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_open_meteo_reference_points(
    points_key: Tuple[Tuple[str, str, str, float, float, float], ...],
    model: str = OPEN_METEO_MODEL_METEOFRANCE,
) -> Tuple[pd.DataFrame, str]:
    if not points_key:
        return pd.DataFrame(), "Open-Meteo reference: aucun point station."

    rows: List[Dict[str, object]] = []
    notices: List[str] = []
    batch_size = 20
    now_utc = datetime.now(timezone.utc)
    for i in range(0, len(points_key), batch_size):
        batch = list(points_key[i : i + batch_size])
        lats = ",".join(f"{float(x[3]):.6f}" for x in batch)
        lons = ",".join(f"{float(x[4]):.6f}" for x in batch)
        params: Dict[str, object] = {
            "latitude": lats,
            "longitude": lons,
            "hourly": "precipitation",
            "past_days": 35,
            "forecast_days": 1,
            "timezone": "UTC",
            "models": model,
        }
        used_model = str(model)
        try:
            resp = _http_get_with_retry(OPEN_METEO_FORECAST_URL, params=params, timeout=35, max_attempts=2)
            if resp.status_code != 200:
                notices.append(f"Open-Meteo reference batch HTTP {resp.status_code}")
                continue
            payload = resp.json()
            entries = payload if isinstance(payload, list) else [payload]
            if not entries:
                continue
            for idx, entry in enumerate(entries):
                if idx >= len(batch):
                    break
                sid, sname, commune, lat, lon, dist = batch[idx]
                hourly = entry.get("hourly") if isinstance(entry, dict) else {}
                times = hourly.get("time") if isinstance(hourly, dict) else []
                precs = hourly.get("precipitation") if isinstance(hourly, dict) else []
                points: List[Tuple[datetime, float, str]] = []
                for t, p in zip(times if isinstance(times, list) else [], precs if isinstance(precs, list) else []):
                    dt = pd.to_datetime(t, utc=True, errors="coerce")
                    mm = pd.to_numeric(p, errors="coerce")
                    if pd.isna(dt) or pd.isna(mm):
                        continue
                    points.append((dt.to_pydatetime(), max(0.0, float(mm)), str(t)))

                past = [x for x in points if x[0] <= now_utc]
                future = [x for x in points if x[0] > now_utc]
                past.sort(key=lambda x: x[0])
                future.sort(key=lambda x: x[0])
                if past:
                    dt_obs, rain_instant, dt_str = past[-1]
                else:
                    dt_obs = now_utc
                    rain_instant = 0.0
                    dt_str = now_utc.strftime("%Y-%m-%dT%H:%M")
                lower_12h = now_utc - pd.Timedelta(hours=12)
                lower_24h = now_utc - pd.Timedelta(hours=24)
                lower_7d = now_utc - pd.Timedelta(days=7)
                lower_30d = now_utc - pd.Timedelta(days=30)
                month_start = datetime(now_utc.year, now_utc.month, 1, tzinfo=timezone.utc)
                rain_12h = float(sum(v for dt, v, _ in past if dt > lower_12h))
                rain_24h = float(sum(v for dt, v, _ in past if dt > lower_24h))
                rain_7d = float(sum(v for dt, v, _ in past if dt > lower_7d))
                rain_30d = float(sum(v for dt, v, _ in past if dt > lower_30d))
                rain_month = float(sum(v for dt, v, _ in past if dt >= month_start))
                rain_forecast = float(sum(v for dt, v, _ in future[:12])) if future else 0.0
                station_name = _clean_station_label(sname) or f"Open-Meteo ref {sid}"
                rows.append(
                    {
                        "station_id": f"openmeteo_ref_{sid}",
                        "station_name": station_name,
                        "station_commune_name": commune or _infer_commune_from_station_name(station_name),
                        "date_obs_raw": dt_str,
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "distance_to_lgv_km": None if float(dist) < 0 else float(dist),
                        "precipitation_mm": round(rain_24h, 3),
                        "rain_24h_mm": round(rain_24h, 3),
                        "rain_7d_mm": round(max(rain_24h, rain_7d), 3),
                        "rain_30d_mm": round(max(rain_7d, rain_30d), 3),
                        "rain_month_mm": round(max(rain_24h, rain_month), 3),
                        "rain_12h_mm": round(max(rain_instant, rain_12h), 3),
                        "rain_instant_mm": round(rain_instant, 3),
                        "rain_forecast_mm": round(max(0.0, rain_forecast), 3),
                        "rain_class": "NORMAL",
                        "source": "open_meteo_reference",
                        "selection_mode": "open_meteo_grid_lgv_commune",
                        "meteo_model": used_model,
                        "station_ref_id": sid,
                        "date": dt_obs.isoformat(),
                    }
                )
        except Exception as exc:
            notices.append(f"Open-Meteo reference batch erreur: {exc}")
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        detail = " | ".join(notices[:2]) if notices else "aucune donnee retournee"
        return out, f"Open-Meteo reference: {detail}."
    out = out.sort_values("distance_to_lgv_km", na_position="last").reset_index(drop=True)
    model_values = out.get("meteo_model", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    model_name = str(model_values[0]) if model_values else "open_meteo_unknown"
    notice = f"Open-Meteo reference: {len(out)} communes LGV calculees ({model_name})."
    if notices:
        notice = notice + " | " + " | ".join(notices[:2])
    return out, notice


def _open_meteo_archive_metrics_from_entry(entry: object) -> Dict[str, object]:
    if not isinstance(entry, dict):
        return {}
    daily = entry.get("daily", {}) if isinstance(entry.get("daily"), dict) else {}
    times = daily.get("time", []) or []
    vals = daily.get("precipitation_sum", []) or []
    if not times or not vals:
        return {}

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(times, utc=True, errors="coerce"),
            "precip_mm": pd.to_numeric(vals, errors="coerce"),
        }
    )
    df = df.dropna(subset=["date"])
    if df.empty:
        return {}
    df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        return {}

    last_date = pd.Timestamp(df["date"].max()).tz_convert("UTC")
    lower_7d = last_date - pd.Timedelta(days=6)
    lower_30d = last_date - pd.Timedelta(days=29)
    month_start = pd.Timestamp(year=int(last_date.year), month=int(last_date.month), day=1, tz="UTC")
    rain_24h = float(df.loc[df["date"] == last_date, "precip_mm"].sum())
    rain_7d = float(df.loc[df["date"] >= lower_7d, "precip_mm"].sum())
    rain_30d = float(df.loc[df["date"] >= lower_30d, "precip_mm"].sum())
    rain_month = float(df.loc[df["date"] >= month_start, "precip_mm"].sum())

    rain_24h = max(0.0, rain_24h)
    rain_7d = max(rain_24h, rain_7d)
    rain_30d = max(rain_7d, rain_30d)
    rain_month = max(rain_24h, rain_month)

    return {
        "date_obs_raw": last_date.strftime("%Y-%m-%d"),
        "date": last_date.isoformat(),
        "rain_24h_mm": round(rain_24h, 3),
        "rain_7d_mm": round(rain_7d, 3),
        "rain_30d_mm": round(rain_30d, 3),
        "rain_month_mm": round(rain_month, 3),
    }


@st.cache_data(show_spinner=False, ttl=1800)
def _fetch_open_meteo_archive_reference_points(
    points_key: Tuple[Tuple[str, str, str, float, float, float], ...],
    model: str = OPEN_METEO_MODEL_METEOFRANCE,
    lookback_days: int = 45,
) -> Tuple[pd.DataFrame, str]:
    if not points_key:
        return pd.DataFrame(), "Open-Meteo archive: aucun point station."

    rows: List[Dict[str, object]] = []
    notices: List[str] = []
    batch_size = 20
    now_utc = datetime.now(timezone.utc)
    lookback = max(32, min(int(lookback_days), 120))
    start_date = (now_utc - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")
    end_date = now_utc.strftime("%Y-%m-%d")

    def _build_row(
        ref_point: Tuple[str, str, str, float, float, float],
        metrics: Dict[str, object],
        used_model: str,
    ) -> Dict[str, object]:
        sid, sname, commune, lat, lon, dist = ref_point
        station_name = _clean_station_label(sname) or commune or f"Open-Meteo archive {sid}"
        station_commune = commune or _infer_commune_from_station_name(station_name)
        commune_ref = station_commune or "commune LGV"
        return {
            "station_id": f"openmeteo_ref_{sid}",
            "station_name": station_name,
            "station_commune_name": station_commune,
            "date_obs_raw": metrics.get("date_obs_raw"),
            "latitude": float(lat),
            "longitude": float(lon),
            "distance_to_lgv_km": None if float(dist) < 0 else float(dist),
            "precipitation_mm": metrics.get("rain_24h_mm"),
            "rain_24h_mm": metrics.get("rain_24h_mm"),
            "rain_7d_mm": metrics.get("rain_7d_mm"),
            "rain_30d_mm": metrics.get("rain_30d_mm"),
            "rain_month_mm": metrics.get("rain_month_mm"),
            "rain_12h_mm": np.nan,
            "rain_instant_mm": np.nan,
            "rain_forecast_mm": np.nan,
            "rain_class": "NORMAL",
            "source": OPEN_METEO_ARCHIVE_SOURCE_LABEL,
            "selection_mode": "open_meteo_archive_lgv_commune",
            "meteo_model": used_model,
            "station_ref_id": sid,
            "date": metrics.get("date"),
            "history_backfill_source": OPEN_METEO_ARCHIVE_SOURCE_LABEL,
            "history_backfill_station": commune_ref,
            "history_backfill_obs_date": metrics.get("date_obs_raw"),
            "rain_calc_method": "open_meteo_archive_daily_precipitation_sum",
        }

    def _fetch_single_point(
        ref_point: Tuple[str, str, str, float, float, float],
    ) -> Tuple[Dict[str, object], str, str]:
        _, _, _, lat, lon, _ = ref_point
        params: Dict[str, object] = {
            "latitude": f"{float(lat):.6f}",
            "longitude": f"{float(lon):.6f}",
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum",
            "timezone": "UTC",
            "models": model,
        }
        used_model = str(model)
        try:
            resp = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=params, timeout=35, max_attempts=2)
            if resp.status_code != 200:
                fallback_params = dict(params)
                fallback_params.pop("models", None)
                resp = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=fallback_params, timeout=35, max_attempts=2)
                used_model = "open_meteo_default"
            if resp.status_code != 200 or not resp.text.strip():
                return {}, used_model, f"Open-Meteo archive point HTTP {resp.status_code}"
            payload = resp.json()
            entry = payload[0] if isinstance(payload, list) and payload else payload
            metrics = _open_meteo_archive_metrics_from_entry(entry)
            if not metrics:
                return {}, used_model, "Open-Meteo archive point: serie vide"
            return metrics, used_model, ""
        except Exception as exc:
            return {}, used_model, f"Open-Meteo archive point erreur: {exc}"

    for i in range(0, len(points_key), batch_size):
        batch = list(points_key[i : i + batch_size])
        lats = ",".join(f"{float(x[3]):.6f}" for x in batch)
        lons = ",".join(f"{float(x[4]):.6f}" for x in batch)
        params: Dict[str, object] = {
            "latitude": lats,
            "longitude": lons,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum",
            "timezone": "UTC",
            "models": model,
        }
        used_model = str(model)
        batch_entries: List[object] = []
        try:
            resp = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=params, timeout=35, max_attempts=2)
            if resp.status_code != 200:
                fallback_params = dict(params)
                fallback_params.pop("models", None)
                resp = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=fallback_params, timeout=35, max_attempts=2)
                used_model = "open_meteo_default"
            if resp.status_code == 200 and resp.text.strip():
                payload = resp.json()
                if isinstance(payload, list):
                    batch_entries = payload
                elif isinstance(payload, dict) and len(batch) == 1:
                    batch_entries = [payload]
        except Exception as exc:
            notices.append(f"Open-Meteo archive batch erreur: {exc}")

        if len(batch_entries) == len(batch):
            for idx, entry in enumerate(batch_entries):
                metrics = _open_meteo_archive_metrics_from_entry(entry)
                if not metrics:
                    if len(notices) < 6:
                        notices.append(f"Open-Meteo archive: serie vide pour {batch[idx][2] or batch[idx][0]}")
                    continue
                rows.append(_build_row(batch[idx], metrics, used_model))
            continue

        if len(batch) > 1 and len(notices) < 6:
            notices.append("Open-Meteo archive: fallback unitaire sur certains points LGV.")
        for ref_point in batch:
            metrics, point_model, point_notice = _fetch_single_point(ref_point)
            if point_notice and len(notices) < 6:
                notices.append(point_notice)
            if metrics:
                rows.append(_build_row(ref_point, metrics, point_model))

    out = pd.DataFrame(rows)
    if out.empty:
        detail = " | ".join(notices[:3]) if notices else "aucune donnee retournee"
        return out, f"Open-Meteo archive: {detail}."
    out = out.sort_values("distance_to_lgv_km", na_position="last").reset_index(drop=True)
    model_values = out.get("meteo_model", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    model_name = str(model_values[0]) if model_values else "open_meteo_unknown"
    notice = f"Open-Meteo archive: {len(out)} communes LGV consolidees sur {lookback} jours ({model_name})."
    if notices:
        notice = notice + " | " + " | ".join(notices[:2])
    return out, notice


def _build_source_metadata_table(stations_df: pd.DataFrame) -> pd.DataFrame:
    if stations_df.empty or "source" not in stations_df.columns:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for source in sorted(stations_df["source"].dropna().astype(str).unique().tolist()):
        sub = stations_df[stations_df["source"].astype(str) == str(source)].copy()
        age_h = pd.to_numeric(sub.get("obs_age_h"), errors="coerce")
        rel = pd.to_numeric(sub.get("reliability_score"), errors="coerce")
        dist = pd.to_numeric(sub.get("distance_to_lgv_km"), errors="coerce")
        if _is_infoclimat_source(source):
            data_type = "Stations observations (SYNOP/InfoClimat)"
            captation = "Mesure directe au sol via station SYNOP / station proche LGV."
            refresh = "Horaire (observation)"
            method = "Mesures station + calcul cumuls 24h/7j/30j puis controle de coherence locale."
            limits = "Densite de stations variable selon secteurs LGV."
            usage = "Affiner la lecture locale au droit de la plateforme et confirmer une alerte."
        elif str(source or "").strip() == METEO_FRANCE_SOURCE_LABEL:
            data_type = "Stations officielles (Meteo-France Portail API)"
            captation = "Observation officielle SYNOP au sol, reprojetee sur chaque commune LGV."
            refresh = "Tri-horaire (SYNOP)"
            method = "Observations SYNOP officielles projetees sur chaque commune LGV (station la plus proche)."
            limits = "La mesure est indirecte pour certaines communes (proxy station)."
            usage = "Support de reference officielle pour arbitrage exploitation / maintenance."
        elif _is_open_meteo_source(source):
            if "archive" in str(source).strip().lower():
                data_type = "Archive journaliere modele (Open-Meteo)"
                captation = "Precipitation journaliere archivee sur centroide ou point communal LGV."
                refresh = "Quotidienne"
                method = "Sommes journalieres archivees utilisees pour fiabiliser les cumuls 7j/30j/mois, alignees sur l'app monitoring LGV."
                limits = "Modele numerique homogene, mais pas une mesure directe de pluviometre."
                usage = "Reference corridor homogene pour saturation des sols, drainage et comparaison inter-communes."
            else:
                data_type = "Modele numerique (Open-Meteo)"
                captation = "Maille modele geolocalisee sur la LGV; court terme live puis archive journaliere pour les cumuls longs."
                refresh = "Horaire pour 12h/24h et prevision courte, quotidienne pour 7j/30j/mois"
                method = "Flux live Open-Meteo pour le court terme, puis recalage des cumuls longs sur l'archive journaliere utilisee dans l'app monitoring."
                limits = "Ce n'est pas une mesure directe de pluviometre."
                usage = "Garantir une couverture corridor exhaustive, meme sans station proche."
        else:
            data_type = "Source diverse"
            captation = "Captation heterogene, harmonisee dans le pipeline."
            refresh = "Selon source"
            method = "Harmonisation interne puis controle de coherence."
            limits = "Metadonnees limitees."
            usage = "Usage secondaire, a confirmer avant decision terrain."
        rows.append(
            {
                "source": source,
                "type_data": data_type,
                "captation": captation,
                "maj_typique": refresh,
                "nb_stations": int(len(sub)),
                "distance_mediane_lgv_km": round(float(dist.median()), 2) if dist.notna().any() else np.nan,
                "age_median_h": round(float(age_h.median()), 1) if age_h.notna().any() else np.nan,
                "fiabilite_mediane_100": round(float(rel.median()), 1) if rel.notna().any() else np.nan,
                "methodologie": method,
                "usage_ferroviaire": usage,
                "limites": limits,
            }
        )
    return pd.DataFrame(rows).sort_values(["fiabilite_mediane_100", "nb_stations"], ascending=[False, False], na_position="last")


def _source_capture_label(source: object) -> str:
    txt = str(source or "").strip()
    txt_norm = txt.lower()
    if _is_infoclimat_source(txt):
        return "Mesure station SYNOP / InfoClimat au sol"
    if txt == METEO_FRANCE_SOURCE_LABEL:
        return "Observation officielle SYNOP reprojetee sur commune LGV"
    if _is_open_meteo_source(txt):
        if "archive" in txt_norm:
            return "Archive journaliere modele sur commune LGV"
        return "Maille modele geolocalisee sur la LGV"
    return "Source harmonisee interne"


def _monthly_alert_level(month_mm: object) -> str:
    val = pd.to_numeric(month_mm, errors="coerce")
    if pd.isna(val):
        return "INCONNU"
    v = float(val)
    if v >= 180.0:
        return "CRITIQUE"
    if v >= 120.0:
        return "ELEVE"
    if v >= 70.0:
        return "MODERE"
    return "FAIBLE"


def _format_num(value: object, digits: int = 1, suffix: str = "") -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "N/A"
    return f"{float(num):.{digits}f}{suffix}"


def _display_rain_30d_text(row: pd.Series) -> str:
    if not bool(ENABLE_RAIN_30D):
        return "Indisponible (non fiabilise pour exploitation)"
    if not bool(row.get("rain_30d_is_reliable")):
        return "Indisponible (source longue non fiable)"
    value_txt = _format_num(row.get("rain_30d_mm"), 1, " mm")
    src = str(row.get("rain_30d_source_label") or "").strip()
    if src:
        return f"{value_txt} [{src}]"
    return value_txt


def _rain_30d_source_label(row: pd.Series) -> str:
    source = str(row.get("source") or "").strip()
    backfill = str(row.get("history_backfill_source") or "").strip()
    calc_method = str(row.get("rain_calc_method") or "").strip().lower()
    source_norm = source.lower()
    backfill_norm = backfill.lower()
    if _is_open_meteo_source(backfill):
        return backfill
    if _is_open_meteo_source(source):
        return source
    if backfill and (
        _is_infoclimat_source(backfill)
        or "archive" in backfill_norm
        or "meteo-france" in backfill_norm
        or "meteo france" in backfill_norm
    ):
        return backfill
    has_consolidated_history = any(
        token in calc_method
        for token in [
            "daily_history",
            "history_aggregate",
            "history_backfill",
            "archive_daily",
            "archive",
        ]
    )
    if has_consolidated_history and (
        _is_infoclimat_source(source)
        or "synop" in source_norm
        or "meteo-france" in source_norm
        or "meteo france" in source_norm
    ):
        return source or "Historique SYNOP consolide"
    history_span_h = pd.to_numeric(row.get("history_span_h"), errors="coerce")
    if pd.notna(history_span_h) and float(history_span_h) >= 25.0 * 24.0 and (
        _is_infoclimat_source(source)
        or "synop" in source_norm
        or "meteo-france" in source_norm
        or "meteo france" in source_norm
    ):
        return source or "Historique SYNOP consolide"
    return ""


def _apply_reliable_rain_30d_policy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "rain_30d_mm" not in out.columns:
        out["rain_30d_mm"] = np.nan
    out["rain_30d_mm"] = pd.to_numeric(out.get("rain_30d_mm"), errors="coerce")
    out["rain_30d_raw_mm"] = out["rain_30d_mm"]
    out["rain_30d_source_label"] = out.apply(_rain_30d_source_label, axis=1)
    out["rain_30d_is_reliable"] = out["rain_30d_source_label"].astype(str).str.len() > 0
    if bool(ENABLE_RAIN_30D):
        out.loc[~out["rain_30d_is_reliable"], "rain_30d_mm"] = np.nan
    else:
        out["rain_30d_mm"] = np.nan
        out["rain_30d_is_reliable"] = False
    return out


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = np.radians(float(lat1))
    p2 = np.radians(float(lat2))
    dlat = np.radians(float(lat2) - float(lat1))
    dlon = np.radians(float(lon2) - float(lon1))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def _ascii_norm(txt: object) -> str:
    raw = str(txt or "").strip().lower()
    if not raw:
        return ""
    norm = unicodedata.normalize("NFKD", raw)
    norm = "".join(ch for ch in norm if not unicodedata.combining(ch))
    norm = norm.replace("-", " ").replace("_", " ").replace("'", " ")
    return " ".join(norm.split())


def _is_unknown_commune(txt: object) -> bool:
    val = _ascii_norm(txt)
    return val in {"", "inconnue", "inconnu", "unknown", "na", "n a"}


def _priority_commune_from_row(station_id: object, station_name: object, commune_name: object) -> str:
    sid_digits = "".join(ch for ch in str(station_id or "") if ch.isdigit())
    sid_norm = sid_digits.zfill(5) if sid_digits else ""
    name_norm = _ascii_norm(station_name)
    commune_norm = _ascii_norm(commune_name)
    for item in INFOCLIMAT_PRIORITY_STATIONS:
        target_commune = str(item.get("commune") or item.get("name") or "").strip()
        aliases = [str(a) for a in item.get("aliases", [])] + [str(item.get("name") or "")]
        for alias in aliases:
            alias_digits = "".join(ch for ch in str(alias) if ch.isdigit())
            alias_norm = _ascii_norm(alias)
            if alias_digits and sid_norm and sid_norm == alias_digits.zfill(5):
                return target_commune
            if alias_norm and (alias_norm in name_norm or alias_norm in commune_norm):
                return target_commune
    return ""


def _nearest_lgv_commune_name(lat: float, lon: float, max_km: float = 40.0) -> str:
    catalog = _load_lgv_communes_catalog()
    if catalog.empty:
        return ""
    c_lat = pd.to_numeric(catalog.get("centroid_latitude"), errors="coerce").to_numpy(dtype=float)
    c_lon = pd.to_numeric(catalog.get("centroid_longitude"), errors="coerce").to_numpy(dtype=float)
    if c_lat.size == 0 or c_lon.size == 0:
        return ""

    r = 6371.0
    p1 = np.radians(float(lat))
    p2 = np.radians(c_lat)
    dlat = np.radians(c_lat - float(lat))
    dlon = np.radians(c_lon - float(lon))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    dist = 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    if dist.size <= 0:
        return ""
    best_idx = int(np.nanargmin(dist))
    best_dist = float(dist[best_idx])
    if best_dist > float(max_km):
        return ""
    return str(catalog.iloc[best_idx].get("commune_name") or "").strip()


@st.cache_data(show_spinner=False)
def _build_lgv_communes_pluvio_table(stations_df: pd.DataFrame) -> pd.DataFrame:
    communes = _load_lgv_communes_catalog()
    if communes.empty:
        return pd.DataFrame()

    work = stations_df.copy() if isinstance(stations_df, pd.DataFrame) else pd.DataFrame()
    if not work.empty:
        for col in [
            "latitude",
            "longitude",
            "rain_24h_mm",
            "rain_7d_mm",
            "rain_30d_mm",
            "rain_month_mm",
            "rain_12h_mm",
            "rain_instant_mm",
            "rain_forecast_mm",
        ]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        for col in [
            "station_id",
            "station_ref_id",
            "station_display",
            "station_commune_name",
            "source",
            "date_obs_raw",
            "history_backfill_source",
            "history_backfill_station",
            "history_backfill_obs_date",
        ]:
            if col not in work.columns:
                work[col] = ""
            work[col] = work[col].fillna("").astype(str)
        work["station_commune_norm"] = work["station_commune_name"].map(_normalize_commune_name)
    else:
        work = pd.DataFrame(
            columns=[
                "station_id",
                "station_ref_id",
                "station_display",
                "station_commune_name",
                "station_commune_norm",
                "source",
                "date_obs_raw",
                "history_backfill_source",
                "history_backfill_station",
                "history_backfill_obs_date",
                "latitude",
                "longitude",
                "rain_24h_mm",
                "rain_7d_mm",
                "rain_30d_mm",
                "rain_month_mm",
                "rain_12h_mm",
                "rain_instant_mm",
                "rain_forecast_mm",
            ]
        )

    out_rows: List[Dict[str, object]] = []
    for _, crow in communes.iterrows():
        code = str(crow.get("commune_code") or "").strip()
        name = str(crow.get("commune_name") or "").strip()
        clat = pd.to_numeric(crow.get("centroid_latitude"), errors="coerce")
        clon = pd.to_numeric(crow.get("centroid_longitude"), errors="coerce")
        picked = pd.Series(dtype=object)
        match_mode = "no_data"
        provider_dist = np.nan

        if not work.empty:
            code_mask = pd.Series(False, index=work.index)
            if code:
                code_mask = (
                    (work["station_ref_id"].astype(str).str.strip() == code)
                    | work["station_id"].astype(str).str.endswith(code)
                )
            code_match = work[code_mask].copy()
            if not code_match.empty:
                picked = code_match.sort_values("date_obs_raw", ascending=False).iloc[0]
                match_mode = "code_exact"
            else:
                comm_norm = _normalize_commune_name(name)
                name_match = work[work["station_commune_norm"] == comm_norm].copy()
                if not name_match.empty:
                    picked = name_match.sort_values("date_obs_raw", ascending=False).iloc[0]
                    match_mode = "commune_name"
                elif pd.notna(clat) and pd.notna(clon):
                    coords = work.dropna(subset=["latitude", "longitude"]).copy()
                    if not coords.empty:
                        dvals = coords.apply(
                            lambda r: _haversine_km(
                                float(clat),
                                float(clon),
                                float(pd.to_numeric(r.get("latitude"), errors="coerce")),
                                float(pd.to_numeric(r.get("longitude"), errors="coerce")),
                            ),
                            axis=1,
                        )
                        best_idx = int(dvals.idxmin())
                        picked = coords.loc[best_idx]
                        provider_dist = float(dvals.loc[best_idx])
                        match_mode = "nearest_station"

        if picked.empty:
            out_rows.append(
                {
                    "commune_code": code,
                    "commune_name": name,
                    "latitude": float(clat) if pd.notna(clat) else np.nan,
                    "longitude": float(clon) if pd.notna(clon) else np.nan,
                    "distance_to_lgv_km": 0.0,
                    "provider_station_id": "",
                    "provider_station": "",
                    "provider_source": "",
                    "provider_station_dist_km": np.nan,
                    "match_mode": match_mode,
                    "date_obs_raw": "",
                    "rain_24h_mm": np.nan,
                    "rain_7d_mm": np.nan,
                    "rain_30d_mm": np.nan,
                    "rain_30d_source_label": "",
                    "rain_30d_is_reliable": False,
                    "rain_month_mm": np.nan,
                    "rain_12h_mm": np.nan,
                    "rain_instant_mm": np.nan,
                    "rain_forecast_mm": np.nan,
                }
            )
            continue

        if pd.isna(provider_dist):
            plat = pd.to_numeric(picked.get("latitude"), errors="coerce")
            plon = pd.to_numeric(picked.get("longitude"), errors="coerce")
            if pd.notna(clat) and pd.notna(clon) and pd.notna(plat) and pd.notna(plon):
                provider_dist = _haversine_km(float(clat), float(clon), float(plat), float(plon))

        provider_source = str(picked.get("history_backfill_source") or picked.get("source") or "")
        provider_station = str(picked.get("history_backfill_station") or picked.get("station_display") or picked.get("station_id") or "")
        provider_obs_date = str(picked.get("history_backfill_obs_date") or picked.get("date_obs_raw") or "")
        rain_30d_source_label = _rain_30d_source_label(picked)
        rain_30d_is_reliable = bool(rain_30d_source_label)

        out_rows.append(
            {
                "commune_code": code,
                "commune_name": name,
                "latitude": float(clat) if pd.notna(clat) else float(pd.to_numeric(picked.get("latitude"), errors="coerce")),
                "longitude": float(clon) if pd.notna(clon) else float(pd.to_numeric(picked.get("longitude"), errors="coerce")),
                "distance_to_lgv_km": 0.0,
                "provider_station_id": str(picked.get("station_id") or ""),
                "provider_station": provider_station,
                "provider_source": provider_source,
                "provider_station_dist_km": round(float(provider_dist), 3) if pd.notna(provider_dist) else np.nan,
                "match_mode": match_mode,
                "date_obs_raw": provider_obs_date,
                "rain_24h_mm": pd.to_numeric(picked.get("rain_24h_mm"), errors="coerce"),
                "rain_7d_mm": pd.to_numeric(picked.get("rain_7d_mm"), errors="coerce"),
                "rain_30d_mm": pd.to_numeric(picked.get("rain_30d_mm"), errors="coerce"),
                "rain_30d_source_label": rain_30d_source_label,
                "rain_30d_is_reliable": rain_30d_is_reliable,
                "rain_month_mm": pd.to_numeric(picked.get("rain_month_mm"), errors="coerce"),
                "rain_12h_mm": pd.to_numeric(picked.get("rain_12h_mm"), errors="coerce"),
                "rain_instant_mm": pd.to_numeric(picked.get("rain_instant_mm"), errors="coerce"),
                "rain_forecast_mm": pd.to_numeric(picked.get("rain_forecast_mm"), errors="coerce"),
            }
        )

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    out = out.sort_values(["commune_name", "commune_code"], ascending=[True, True]).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def _build_commune_proxy_weather_df(
    stations_df: pd.DataFrame,
    source_label: str,
    selection_mode: str,
) -> pd.DataFrame:
    commune_df = _build_lgv_communes_pluvio_table(stations_df)
    if commune_df.empty:
        return pd.DataFrame()

    out = commune_df.copy()
    out["station_id"] = (
        "commune_proxy_"
        + out.get("commune_code", pd.Series("", index=out.index)).fillna("").astype(str).replace("", "na")
    )
    out["station_ref_id"] = out.get("commune_code", pd.Series("", index=out.index)).fillna("").astype(str)
    out["station_name"] = out.get("commune_name", pd.Series("Commune LGV", index=out.index)).fillna("Commune LGV").astype(str)
    out["station_commune_name"] = out["station_name"]
    out["station_display"] = out["station_name"].astype(str) + " (proxy commune LGV)"
    out["source"] = str(source_label)
    out["selection_mode"] = str(selection_mode)
    out["distance_to_lgv_km"] = 0.0
    out["provider_source"] = out.get("provider_source", pd.Series("", index=out.index)).fillna("").astype(str)
    out["rain_calc_method"] = (
        "lgv_commune_proxy_from_station_source="
        + out["provider_source"].replace("", "inconnue").astype(str)
    )
    out["meteo_model"] = out.get("provider_source", pd.Series("", index=out.index)).fillna("").astype(str)
    out["_obs_ts"] = pd.to_datetime(out.get("date_obs_raw", pd.Series("", index=out.index)), utc=True, errors="coerce")
    rain_cols = [c for c in ["rain_24h_mm", "rain_7d_mm", "rain_30d_mm", "rain_month_mm"] if c in out.columns]
    if rain_cols:
        data_mask = pd.Series(False, index=out.index)
        for col in rain_cols:
            data_mask = data_mask | pd.to_numeric(out[col], errors="coerce").notna()
        if "provider_station" in out.columns:
            data_mask = data_mask | out["provider_station"].fillna("").astype(str).str.len().gt(0)
        out = out[data_mask].copy()
    if out.empty:
        return pd.DataFrame()

    keep_cols = [
        "station_id",
        "station_ref_id",
        "station_name",
        "station_display",
        "station_commune_name",
        "date_obs_raw",
        "_obs_ts",
        "latitude",
        "longitude",
        "distance_to_lgv_km",
        "rain_24h_mm",
        "rain_7d_mm",
        "rain_30d_mm",
        "rain_month_mm",
        "rain_12h_mm",
        "rain_instant_mm",
        "rain_forecast_mm",
        "source",
        "selection_mode",
        "meteo_model",
        "provider_station",
        "provider_source",
        "provider_station_dist_km",
        "match_mode",
        "rain_calc_method",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    return out[keep_cols].copy()


@st.cache_data(show_spinner=False)
def _enrich_commune_weather_with_reference_history(
    base_df: pd.DataFrame,
    reference_weather_df: pd.DataFrame,
    replace_existing: bool = False,
    target_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, str]:
    if base_df.empty or reference_weather_df.empty:
        return base_df.copy(), ""
    cols_to_update = target_cols or ["rain_7d_mm", "rain_30d_mm", "rain_month_mm"]

    reference_communes = _build_lgv_communes_pluvio_table(reference_weather_df)
    if reference_communes.empty:
        return base_df.copy(), ""

    ref = reference_communes.copy()
    ref["commune_code_ref"] = ref.get("commune_code", pd.Series("", index=ref.index)).fillna("").astype(str).str.strip()
    ref["commune_name_ref"] = ref.get("commune_name", pd.Series("", index=ref.index)).fillna("").astype(str).str.strip()
    ref = ref.rename(
        columns={
            "rain_7d_mm": "ref_rain_7d_mm",
            "rain_30d_mm": "ref_rain_30d_mm",
            "rain_month_mm": "ref_rain_month_mm",
            "provider_source": "ref_provider_source",
            "provider_station": "ref_provider_station",
            "date_obs_raw": "ref_date_obs_raw",
            "match_mode": "ref_match_mode",
        }
    )
    ref_keep_cols = [
        "commune_code_ref",
        "commune_name_ref",
        "ref_rain_7d_mm",
        "ref_rain_30d_mm",
        "ref_rain_month_mm",
        "ref_provider_source",
        "ref_provider_station",
        "ref_date_obs_raw",
        "ref_match_mode",
    ]
    ref_keep_cols = [c for c in ref_keep_cols if c in ref.columns]
    ref = ref[ref_keep_cols].copy()

    out = base_df.copy()
    out["commune_code_ref"] = out.get("station_ref_id", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    out["commune_name_ref"] = out.get("station_commune_name", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    out = out.merge(ref, how="left", on=["commune_code_ref", "commune_name_ref"])

    updated_count = 0
    for target_col, ref_col in [
        ("rain_7d_mm", "ref_rain_7d_mm"),
        ("rain_30d_mm", "ref_rain_30d_mm"),
        ("rain_month_mm", "ref_rain_month_mm"),
    ]:
        if target_col not in cols_to_update:
            continue
        if target_col not in out.columns or ref_col not in out.columns:
            continue
        target_series = pd.to_numeric(out[target_col], errors="coerce")
        ref_series = pd.to_numeric(out[ref_col], errors="coerce")
        if replace_existing:
            fill_mask = ref_series.notna()
        else:
            fill_mask = target_series.isna() & ref_series.notna()
        updated_count += int(fill_mask.sum())
        out.loc[fill_mask, target_col] = ref_series.loc[fill_mask]

    ref_source = out.get("ref_provider_source", pd.Series("", index=out.index)).fillna("").astype(str)
    ref_station = out.get("ref_provider_station", pd.Series("", index=out.index)).fillna("").astype(str)
    ref_obs_date = out.get("ref_date_obs_raw", pd.Series("", index=out.index)).fillna("").astype(str)
    out["history_backfill_source"] = np.where(ref_source.str.len() > 0, ref_source, "")
    out["history_backfill_station"] = np.where(ref_station.str.len() > 0, ref_station, "")
    out["history_backfill_obs_date"] = np.where(ref_obs_date.str.len() > 0, ref_obs_date, "")

    if "rain_calc_method" not in out.columns:
        out["rain_calc_method"] = ""
    out["rain_calc_method"] = out["rain_calc_method"].fillna("").astype(str)
    hist_mask = out.get("history_backfill_source", pd.Series("", index=out.index)).fillna("").astype(str).str.len() > 0
    out.loc[hist_mask, "rain_calc_method"] = np.where(
        out.loc[hist_mask, "rain_calc_method"].astype(str).str.len() > 0,
        out.loc[hist_mask, "rain_calc_method"].astype(str) + ";history_backfill_commune",
        "history_backfill_commune",
    )

    if updated_count <= 0:
        return out.drop(
            columns=[
                c for c in [
                    "commune_code_ref",
                    "commune_name_ref",
                    "ref_rain_7d_mm",
                    "ref_rain_30d_mm",
                    "ref_rain_month_mm",
                    "ref_provider_source",
                    "ref_provider_station",
                    "ref_date_obs_raw",
                    "ref_match_mode",
                ] if c in out.columns
            ],
            errors="ignore",
        ), ""

    action_label = "remplaces" if replace_existing else "reconstitues"
    notice = f"Cumuls {action_label} pour {updated_count} champ(s) via reference communale."
    out = out.drop(
        columns=[
            c for c in [
                "commune_code_ref",
                "commune_name_ref",
                "ref_rain_7d_mm",
                "ref_rain_30d_mm",
                "ref_rain_month_mm",
                "ref_provider_source",
                "ref_provider_station",
                "ref_date_obs_raw",
                "ref_match_mode",
            ] if c in out.columns
        ],
        errors="ignore",
    )
    return out, notice


def _filter_infoclimat_nearest_lgv(
    stations_df: pd.DataFrame,
    max_distance_km: float,
    max_stations: int,
) -> pd.DataFrame:
    if stations_df.empty:
        return stations_df.copy()

    work = stations_df.copy()
    work["distance_to_lgv_km"] = pd.to_numeric(work.get("distance_to_lgv_km"), errors="coerce")
    work["latitude"] = pd.to_numeric(work.get("latitude"), errors="coerce")
    work["longitude"] = pd.to_numeric(work.get("longitude"), errors="coerce")
    work = work.dropna(subset=["latitude", "longitude"]).copy()
    if work.empty:
        return work

    work = work[work["distance_to_lgv_km"].fillna(9999.0) <= float(max_distance_km)].copy()
    if work.empty:
        return work
    work = work.sort_values("distance_to_lgv_km", ascending=True).reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    work["priority_name"] = ""
    work["priority_match_km"] = np.nan

    # Match user-priority stations by nearest coordinate and by alias in station labels.
    station_id_norm = work.get("station_id", pd.Series("", index=work.index)).fillna("").astype(str).str.zfill(5)
    station_name_norm = work.get("station_name", pd.Series("", index=work.index)).fillna("").map(_ascii_norm)
    commune_name_norm = work.get("station_commune_name", pd.Series("", index=work.index)).fillna("").map(_ascii_norm)

    priority_idx: set[int] = set()
    for item in INFOCLIMAT_PRIORITY_STATIONS:
        lat = float(item["lat"])
        lon = float(item["lon"])
        name = str(item["name"])
        commune = str(item.get("commune") or name)
        aliases = [str(a) for a in item.get("aliases", [])]

        # 1) Spatial nearest match.
        dists = work.apply(
            lambda r: _haversine_km(lat, lon, float(r["latitude"]), float(r["longitude"])),
            axis=1,
        )
        if not dists.empty and dists.notna().any():
            best_idx = int(dists.idxmin())
            best_dist = float(dists.loc[best_idx])
            if best_dist <= float(INFOCLIMAT_PRIORITY_MATCH_KM):
                priority_idx.add(best_idx)
                work.loc[best_idx, "priority_name"] = name
                work.loc[best_idx, "priority_match_km"] = best_dist
                work.loc[best_idx, "station_name"] = name
                work.loc[best_idx, "station_commune_name"] = commune

        # 2) Alias textual/id match.
        for alias in aliases:
            alias_norm = _ascii_norm(alias)
            alias_digits = "".join(ch for ch in alias if ch.isdigit())
            mask = pd.Series(False, index=work.index)
            if alias_norm:
                mask = mask | station_name_norm.str.contains(alias_norm, regex=False) | commune_name_norm.str.contains(alias_norm, regex=False)
            if alias_digits:
                mask = mask | (station_id_norm == alias_digits.zfill(5))
            hits = work[mask].index.tolist()
            for h in hits:
                priority_idx.add(int(h))
                if not str(work.loc[h, "priority_name"]).strip():
                    work.loc[h, "priority_name"] = name
                work.loc[h, "station_name"] = name
                if _is_unknown_commune(work.loc[h, "station_commune_name"]):
                    work.loc[h, "station_commune_name"] = commune

    # Keep all stations within distance threshold (no top-N truncation).
    _ = max_stations  # kept for backward compatibility in call sites
    nearest_idx = set(work.index.tolist())
    keep_idx = sorted(nearest_idx.union(priority_idx))
    out = work.loc[keep_idx].copy()
    out["selection_mode"] = out.get("selection_mode", pd.Series("", index=out.index)).fillna("").astype(str)
    out["selection_mode"] = np.where(
        out["selection_mode"].str.len() > 0,
        out["selection_mode"] + ";nearest_lgv_priority",
        "nearest_lgv_priority",
    )
    out["station_name"] = out.get("station_name", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    out["station_commune_name"] = out.get("station_commune_name", pd.Series("", index=out.index)).fillna("Inconnue").astype(str).str.strip()
    out["station_id"] = out.get("station_id", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    same_commune = out["station_name"].str.lower() == out["station_commune_name"].str.lower()
    out["station_display"] = np.where(
        same_commune,
        out["station_commune_name"] + " (" + out["station_id"] + ")",
        out["station_name"] + " (" + out["station_commune_name"] + " - " + out["station_id"] + ")",
    )
    out = out.sort_values(["distance_to_lgv_km", "priority_match_km"], ascending=[True, True], na_position="last")
    return out.drop(columns=["_orig_idx"], errors="ignore").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _select_infoclimat_monitoring_set(
    stations_df: pd.DataFrame,
    preferred_radius_km: float = INFOCLIMAT_STRICT_RADIUS_KM,
    min_stations: int = INFOCLIMAT_MIN_STATIONS_COVERAGE,
) -> Tuple[pd.DataFrame, float, bool]:
    if stations_df.empty:
        return pd.DataFrame(), float(preferred_radius_km), False

    radii: List[float] = []
    for r in INFOCLIMAT_ADAPTIVE_RADII_KM:
        rv = float(r)
        if rv not in radii:
            radii.append(rv)
    if float(preferred_radius_km) not in radii:
        radii.insert(0, float(preferred_radius_km))
    radii = sorted(radii)

    selected = pd.DataFrame()
    used_radius = float(preferred_radius_km)
    adaptive_used = False
    best_non_empty = pd.DataFrame()
    best_radius = float(preferred_radius_km)

    for radius in radii:
        candidate = _filter_infoclimat_nearest_lgv(
            stations_df=stations_df,
            max_distance_km=float(radius),
            max_stations=9999,
        )
        if not candidate.empty:
            best_non_empty = candidate.copy()
            best_radius = float(radius)
        if len(candidate) >= int(min_stations):
            selected = candidate.copy()
            used_radius = float(radius)
            adaptive_used = float(radius) > float(preferred_radius_km)
            break

    if selected.empty and not best_non_empty.empty:
        selected = best_non_empty.copy()
        used_radius = float(best_radius)
        adaptive_used = float(best_radius) > float(preferred_radius_km)

    return selected, used_radius, adaptive_used


def _source_reliability_note(source: object) -> float:
    txt = str(source or "").strip().upper()
    if not txt:
        return 68.0
    if "SYNOP" in txt:
        return SOURCE_RELIABILITY_HINTS["SYNOP"]
    if "INFOCLIMAT" in txt:
        return SOURCE_RELIABILITY_HINTS["INFOCLIMAT"]
    if "METEOFRANCE" in txt:
        return SOURCE_RELIABILITY_HINTS["METEOFRANCE"]
    if ("OPEN" in txt) and ("METEO" in txt):
        return SOURCE_RELIABILITY_HINTS["OPEN_METEO"]
    return 70.0


def _freshness_note(hours: float) -> float:
    h = float(hours)
    if h <= 6.0:
        return 100.0
    if h <= 12.0:
        return 92.0
    if h <= 24.0:
        return 84.0
    if h <= 48.0:
        return 60.0
    return 35.0


def _reliability_class(note: float) -> str:
    val = float(note)
    if val >= 85.0:
        return "FIABLE"
    if val >= 65.0:
        return "SURVEILLER"
    return "A_VERIFIER"


@st.cache_data(show_spinner=False, ttl=60)
def _build_proximity_quality(
    stations_df: pd.DataFrame,
    snapshot_ts: pd.Timestamp | None,
    metric_col: str,
    compare_radius_km: float,
    min_neighbors: int,
) -> pd.DataFrame:
    if stations_df.empty:
        return stations_df.copy()
    work = stations_df.copy()
    for col in ["latitude", "longitude", metric_col, "rain_24h_mm", "rain_7d_mm", "rain_30d_mm", "rain_month_mm"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "_obs_ts" not in work.columns:
        work["_obs_ts"] = pd.to_datetime(work.get("date_obs_raw"), utc=True, errors="coerce")
    obs_ts = pd.to_datetime(work["_obs_ts"], utc=True, errors="coerce")
    now_utc = pd.Timestamp.now(tz="UTC")
    if snapshot_ts is not None and not pd.isna(snapshot_ts):
        reference_ts = max(pd.Timestamp(snapshot_ts), now_utc)
    else:
        reference_ts = now_utc
    obs_age_h = (reference_ts - obs_ts).dt.total_seconds() / 3600.0
    work["obs_age_h"] = pd.to_numeric(obs_age_h, errors="coerce").fillna(999.0).clip(lower=0.0)
    work["source_note"] = work.get("source", pd.Series("", index=work.index)).map(_source_reliability_note)
    work["freshness_note"] = work["obs_age_h"].map(_freshness_note)

    metrics_for_consistency = [
        c for c in ["rain_24h_mm", "rain_7d_mm", "rain_month_mm"]
        if c in work.columns
    ]
    near_count_list: List[int] = []
    near_med_metric_list: List[float] = []
    near_delta_metric_mm_list: List[float] = []
    near_delta_metric_pct_list: List[float] = []
    coherence_note_list: List[float] = []

    for idx, row in work.iterrows():
        lat = pd.to_numeric(row.get("latitude"), errors="coerce")
        lon = pd.to_numeric(row.get("longitude"), errors="coerce")
        station_id = str(row.get("station_id") or "")
        if pd.isna(lat) or pd.isna(lon):
            near_count_list.append(0)
            near_med_metric_list.append(np.nan)
            near_delta_metric_mm_list.append(np.nan)
            near_delta_metric_pct_list.append(np.nan)
            coherence_note_list.append(45.0)
            continue

        neighbors = work.copy()
        neighbors = neighbors[neighbors["station_id"].astype(str) != station_id].copy()
        if neighbors.empty:
            near_count_list.append(0)
            near_med_metric_list.append(np.nan)
            near_delta_metric_mm_list.append(np.nan)
            near_delta_metric_pct_list.append(np.nan)
            coherence_note_list.append(45.0)
            continue

        neighbors["_dist_station_km"] = neighbors.apply(
            lambda r: _haversine_km(
                float(lat),
                float(lon),
                float(pd.to_numeric(r.get("latitude"), errors="coerce")),
                float(pd.to_numeric(r.get("longitude"), errors="coerce")),
            )
            if (pd.notna(pd.to_numeric(r.get("latitude"), errors="coerce")) and pd.notna(pd.to_numeric(r.get("longitude"), errors="coerce")))
            else np.nan,
            axis=1,
        )
        neighbors = neighbors[
            pd.to_numeric(neighbors["_dist_station_km"], errors="coerce").fillna(9999.0) <= float(compare_radius_km)
        ].copy()
        neighbors = neighbors.dropna(subset=["_dist_station_km"])
        near_count = int(len(neighbors))
        near_count_list.append(near_count)
        if near_count <= 0:
            near_med_metric_list.append(np.nan)
            near_delta_metric_mm_list.append(np.nan)
            near_delta_metric_pct_list.append(np.nan)
            coherence_note_list.append(45.0)
            continue

        metric_val = pd.to_numeric(row.get(metric_col), errors="coerce")
        near_med_metric = pd.to_numeric(neighbors.get(metric_col), errors="coerce").median(skipna=True)
        near_med_metric_list.append(float(near_med_metric) if pd.notna(near_med_metric) else np.nan)
        if pd.isna(metric_val) or pd.isna(near_med_metric):
            near_delta_metric_mm_list.append(np.nan)
            near_delta_metric_pct_list.append(np.nan)
        else:
            delta_mm = abs(float(metric_val) - float(near_med_metric))
            delta_pct = 100.0 * delta_mm / max(5.0, float(near_med_metric))
            near_delta_metric_mm_list.append(delta_mm)
            near_delta_metric_pct_list.append(delta_pct)

        consistency_notes: List[float] = []
        for mcol in metrics_for_consistency:
            val = pd.to_numeric(row.get(mcol), errors="coerce")
            med = pd.to_numeric(neighbors.get(mcol), errors="coerce").median(skipna=True)
            if pd.isna(val) or pd.isna(med):
                continue
            rel = abs(float(val) - float(med)) / max(5.0, float(med))
            metric_note = max(0.0, 100.0 - rel * 120.0)
            consistency_notes.append(metric_note)

        if near_count < int(min_neighbors):
            base_consistency = float(np.mean(consistency_notes)) if consistency_notes else 58.0
            coherence_note_list.append(min(72.0, base_consistency))
        else:
            coherence_note_list.append(float(np.mean(consistency_notes)) if consistency_notes else 65.0)

    work["near_station_count"] = pd.Series(near_count_list, index=work.index, dtype=int)
    work["near_median_metric_mm"] = pd.Series(near_med_metric_list, index=work.index, dtype=float)
    work["near_delta_metric_mm"] = pd.Series(near_delta_metric_mm_list, index=work.index, dtype=float)
    work["near_delta_metric_pct"] = pd.Series(near_delta_metric_pct_list, index=work.index, dtype=float)
    work["coherence_note"] = pd.Series(coherence_note_list, index=work.index, dtype=float).clip(lower=0.0, upper=100.0)

    work["reliability_source_component"] = 0.42 * pd.to_numeric(work["source_note"], errors="coerce").fillna(70.0)
    work["reliability_freshness_component"] = 0.23 * pd.to_numeric(work["freshness_note"], errors="coerce").fillna(40.0)
    work["reliability_coherence_component"] = 0.35 * pd.to_numeric(work["coherence_note"], errors="coerce").fillna(55.0)
    work["reliability_score"] = (
        pd.to_numeric(work["reliability_source_component"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["reliability_freshness_component"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["reliability_coherence_component"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0, upper=100.0)
    stale_mask = pd.to_numeric(work["obs_age_h"], errors="coerce").fillna(999.0) > float(WEATHER_STALE_ALERT_H)
    work.loc[stale_mask, "reliability_score"] = (
        pd.to_numeric(work.loc[stale_mask, "reliability_score"], errors="coerce").fillna(0.0) - 25.0
    ).clip(lower=0.0, upper=100.0)
    work["reliability_class"] = work["reliability_score"].map(_reliability_class)
    work.loc[stale_mask, "reliability_class"] = "A_VERIFIER"
    work["reliability_reason"] = np.where(
        stale_mask,
        "Observation trop ancienne pour un usage exploitation",
        np.where(
            work["near_station_count"] < int(min_neighbors),
            "Peu de stations voisines pour confirmer la coherence",
            np.where(
                work["near_delta_metric_pct"].fillna(0.0) >= 60.0,
                "Ecart eleve vs mediane des stations proches",
                "Coherence locale satisfaisante",
            ),
        ),
    )
    return work


def _nearest_neighbors_for_station(
    stations_df: pd.DataFrame,
    station_id: str,
    metric_col: str,
    compare_radius_km: float,
) -> pd.DataFrame:
    if stations_df.empty:
        return pd.DataFrame()
    base = stations_df[stations_df["station_id"].astype(str) == str(station_id)].copy()
    if base.empty:
        return pd.DataFrame()
    ref = base.iloc[0]
    ref_lat = pd.to_numeric(ref.get("latitude"), errors="coerce")
    ref_lon = pd.to_numeric(ref.get("longitude"), errors="coerce")
    if pd.isna(ref_lat) or pd.isna(ref_lon):
        return pd.DataFrame()

    work = stations_df.copy()
    work = work[work["station_id"].astype(str) != str(station_id)].copy()
    if work.empty:
        return pd.DataFrame()
    work["distance_station_ref_km"] = work.apply(
        lambda r: _haversine_km(
            float(ref_lat),
            float(ref_lon),
            float(pd.to_numeric(r.get("latitude"), errors="coerce")),
            float(pd.to_numeric(r.get("longitude"), errors="coerce")),
        )
        if (pd.notna(pd.to_numeric(r.get("latitude"), errors="coerce")) and pd.notna(pd.to_numeric(r.get("longitude"), errors="coerce")))
        else np.nan,
        axis=1,
    )
    work = work.dropna(subset=["distance_station_ref_km"])
    work = work[
        pd.to_numeric(work["distance_station_ref_km"], errors="coerce").fillna(9999.0) <= float(compare_radius_km)
    ].copy()
    if work.empty:
        return work
    work[metric_col] = pd.to_numeric(work.get(metric_col), errors="coerce").fillna(0.0)
    ref_metric = pd.to_numeric(ref.get(metric_col), errors="coerce")
    if pd.isna(ref_metric):
        work["delta_metric_mm"] = np.nan
    else:
        work["delta_metric_mm"] = (work[metric_col] - float(ref_metric)).abs()
    work = work.sort_values(["distance_station_ref_km", "delta_metric_mm"], ascending=[True, True], na_position="last")
    return work


@st.cache_data(show_spinner=False)
def _build_openmeteo_vs_infoclimat_pairs(
    stations_df: pd.DataFrame,
    metric_col: str,
    max_pair_distance_km: float,
) -> pd.DataFrame:
    if stations_df.empty:
        return pd.DataFrame()
    work = stations_df.copy()
    work["latitude"] = pd.to_numeric(work.get("latitude"), errors="coerce")
    work["longitude"] = pd.to_numeric(work.get("longitude"), errors="coerce")
    work[metric_col] = pd.to_numeric(work.get(metric_col), errors="coerce")
    work = work.dropna(subset=["latitude", "longitude", metric_col]).copy()
    if work.empty:
        return pd.DataFrame()

    infoclimat_df = work[work["source"].map(_is_infoclimat_source)].copy()
    open_meteo_df = work[work["source"].map(_is_open_meteo_source)].copy()
    if infoclimat_df.empty or open_meteo_df.empty:
        return pd.DataFrame()

    pairs: List[Dict[str, object]] = []
    max_dist = float(max_pair_distance_km)
    for _, info_row in infoclimat_df.iterrows():
        info_lat = float(info_row["latitude"])
        info_lon = float(info_row["longitude"])
        candidates = open_meteo_df.copy()
        candidates["pair_dist_km"] = candidates.apply(
            lambda r: _haversine_km(info_lat, info_lon, float(r["latitude"]), float(r["longitude"])),
            axis=1,
        )
        candidates = candidates[candidates["pair_dist_km"] <= max_dist].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values("pair_dist_km", ascending=True)
        open_row = candidates.iloc[0]
        info_val = float(pd.to_numeric(info_row.get(metric_col), errors="coerce"))
        open_val = float(pd.to_numeric(open_row.get(metric_col), errors="coerce"))
        delta = open_val - info_val
        rel_pct = 100.0 * abs(delta) / max(5.0, info_val)
        pairs.append(
            {
                "infoclimat_station_id": str(info_row.get("station_id") or ""),
                "infoclimat_station": str(info_row.get("station_display") or info_row.get("station_id") or ""),
                "infoclimat_commune": str(info_row.get("station_commune_name") or ""),
                "infoclimat_source": str(info_row.get("source") or ""),
                "infoclimat_mm": info_val,
                "open_meteo_station_id": str(open_row.get("station_id") or ""),
                "open_meteo_station": str(open_row.get("station_display") or open_row.get("station_id") or ""),
                "open_meteo_commune": str(open_row.get("station_commune_name") or ""),
                "open_meteo_source": str(open_row.get("source") or ""),
                "open_meteo_mm": open_val,
                "pair_distance_km": float(pd.to_numeric(open_row.get("pair_dist_km"), errors="coerce")),
                "delta_open_minus_info_mm": delta,
                "delta_abs_pct": rel_pct,
            }
        )
    if not pairs:
        return pd.DataFrame()
    out = pd.DataFrame(pairs).sort_values("delta_abs_pct", ascending=False).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False, ttl=300)
def _load_snapshot_payload() -> Tuple[Dict[str, object], str]:
    errors: List[str] = []
    if SNAPSHOT_LATEST.exists():
        try:
            payload = pd.read_json(SNAPSHOT_LATEST, typ="series").to_dict()
            if isinstance(payload, dict):
                return payload, f"local:{SNAPSHOT_LATEST.as_posix()}"
        except Exception as exc:
            errors.append(f"local:{exc}")
    for url in REMOTE_SNAPSHOT_URLS:
        try:
            resp = _http_get_with_retry(url, params={}, timeout=25, max_attempts=2)
            if resp.status_code != 200:
                errors.append(f"{url}:HTTP{resp.status_code}")
                continue
            payload = resp.json()
            if isinstance(payload, dict):
                return payload, f"remote:{url}"
            errors.append(f"{url}:payload_non_dict")
        except Exception as exc:
            errors.append(f"{url}:{exc}")
    raise RuntimeError("Snapshot indisponible: " + " | ".join(errors))


@st.cache_data(show_spinner=False, ttl=21600)
def _fetch_open_meteo_history(
    lat: float,
    lon: float,
    start_day: str,
    end_day: str,
    source_label: str,
    model: str | None = None,
) -> Tuple[pd.DataFrame, str]:
    params: Dict[str, object] = {
        "latitude": f"{float(lat):.6f}",
        "longitude": f"{float(lon):.6f}",
        "start_date": str(start_day),
        "end_date": str(end_day),
        "daily": "precipitation_sum",
        "timezone": "UTC",
    }
    if model:
        params["models"] = str(model)
    try:
        resp = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30, max_attempts=2)
        if resp.status_code != 200:
            fallback_params = dict(params)
            fallback_params.pop("models", None)
            resp = _http_get_with_retry(OPEN_METEO_ARCHIVE_URL, params=fallback_params, timeout=30, max_attempts=2)
        if resp.status_code != 200:
            return pd.DataFrame(), f"{source_label}: HTTP {resp.status_code}"
        payload = resp.json()
        entry = payload[0] if isinstance(payload, list) and payload else payload
        daily = entry.get("daily") if isinstance(entry, dict) else {}
        times = daily.get("time") if isinstance(daily, dict) else []
        values = daily.get("precipitation_sum") if isinstance(daily, dict) else []
        out = pd.DataFrame({"date": times, "precip_mm": values})
        if out.empty:
            return out, f"{source_label}: vide"
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out["precip_mm"] = pd.to_numeric(out["precip_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
        out = out.dropna(subset=["date"]).copy()
        out["source"] = source_label
        return out[["date", "precip_mm", "source"]], ""
    except Exception as exc:
        return pd.DataFrame(), f"{source_label}: {exc}"


def _station_id_candidates(station_id: object) -> List[str]:
    raw = str(station_id or "").strip()
    if not raw:
        return []
    only_digits = "".join(ch for ch in raw if ch.isdigit())
    out: List[str] = []
    for val in [raw, only_digits, only_digits.zfill(5), only_digits.lstrip("0"), raw.lstrip("0")]:
        txt = str(val or "").strip()
        if txt and txt not in out:
            out.append(txt)
    return out


@st.cache_data(show_spinner=False, ttl=21600)
def _fetch_infoclimat_history_local(
    station_id: str,
    start_day: str,
    end_day: str,
    source_label: str = INFOCLIMAT_HISTORY_SOURCE,
) -> Tuple[pd.DataFrame, str]:
    sid = str(station_id or "").strip()
    if not sid:
        return pd.DataFrame(), f"{source_label}: station_id manquant."
    if sid.lower().startswith("openmeteo_ref_"):
        return pd.DataFrame(), f"{source_label}: historique non applicable pour station virtuelle Open-Meteo ({sid})."

    start_ts = pd.Timestamp(start_day, tz="UTC").normalize()
    end_ts = pd.Timestamp(end_day, tz="UTC").normalize()
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    preferred_year_file = Path(f"data/synop_cache/synop_{int(start_ts.year)}.csv.gz")
    if preferred_year_file.exists():
        synop_path = preferred_year_file
    else:
        synop_path = _find_latest_file(["data/synop_cache/synop_*.csv.gz", "data/synop_cache/synop.*.csv"])
    if synop_path is None:
        return pd.DataFrame(), f"{source_label}: aucun fichier synop_cache disponible."

    id_candidates = _station_id_candidates(sid)
    id_candidates_z5 = {c.zfill(5) for c in id_candidates if c}
    rows: List[pd.DataFrame] = []
    try:
        usecols_candidates = {"geo_id_wmo", "validity_time", "reference_time", "rr24", "rr12", "rr6", "rr3", "rr1"}
        stream = pd.read_csv(
            synop_path,
            sep=";",
            dtype=str,
            usecols=lambda c: c in usecols_candidates,
            chunksize=250000,
            low_memory=False,
            compression="infer",
        )
        for chunk in stream:
            if chunk.empty or "geo_id_wmo" not in chunk.columns:
                continue
            chunk_station = chunk["geo_id_wmo"].fillna("").astype(str).str.strip()
            chunk_station_z5 = chunk_station.str.zfill(5)
            sub = chunk[chunk_station_z5.isin(id_candidates_z5)].copy()
            if sub.empty:
                continue

            if "validity_time" in sub.columns:
                obs_dt = pd.to_datetime(sub["validity_time"], utc=True, errors="coerce")
            elif "reference_time" in sub.columns:
                obs_dt = pd.to_datetime(sub["reference_time"], utc=True, errors="coerce")
            else:
                continue

            sub["date"] = obs_dt.dt.normalize()
            sub = sub.dropna(subset=["date"])
            sub = sub[(sub["date"] >= start_ts) & (sub["date"] <= end_ts)]
            if sub.empty:
                continue

            rr24 = pd.to_numeric(sub["rr24"], errors="coerce") if "rr24" in sub.columns else pd.Series(np.nan, index=sub.index)
            fallback_cols = [c for c in ["rr12", "rr6", "rr3", "rr1"] if c in sub.columns]
            if fallback_cols:
                fallback_rr = sub[fallback_cols].apply(pd.to_numeric, errors="coerce").max(axis=1, skipna=True)
            else:
                fallback_rr = pd.Series(np.nan, index=sub.index)

            rain_obs = rr24.where(rr24.notna(), fallback_rr)
            rain_obs = pd.to_numeric(rain_obs, errors="coerce")
            rain_obs = rain_obs.where((rain_obs >= 0.0) & (rain_obs < 900.0))

            tmp = pd.DataFrame({"date": sub["date"], "precip_mm": rain_obs})
            tmp = tmp.dropna(subset=["date", "precip_mm"])
            if tmp.empty:
                continue
            rows.append(tmp)
    except Exception as exc:
        return pd.DataFrame(), f"{source_label}: lecture synop impossible ({exc})."

    if not rows:
        return pd.DataFrame(), f"{source_label}: aucune mesure disponible pour station {sid}."

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.normalize()
    out["precip_mm"] = pd.to_numeric(out["precip_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = (
        out.dropna(subset=["date"])
        .groupby("date", as_index=False)["precip_mm"]
        .max()
        .sort_values("date")
        .reset_index(drop=True)
    )
    out["source"] = str(source_label)
    station_norm = "".join(ch for ch in sid if ch.isdigit()).zfill(5)
    return (
        out[["date", "precip_mm", "source"]],
        f"{source_label}: {len(out)} jours charges pour station {station_norm} ({synop_path.name}).",
    )


def _load_history_multi_source(
    lat: float,
    lon: float,
    station_id: str,
    start_day: date,
    end_day: date,
    selected_sources: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    start_iso = str(start_day.isoformat())
    end_iso = str(end_day.isoformat())
    blocks: List[pd.DataFrame] = []
    notices: List[str] = []

    if INFOCLIMAT_HISTORY_SOURCE in selected_sources:
        df, note = _fetch_infoclimat_history_local(
            station_id=str(station_id or ""),
            start_day=start_iso,
            end_day=end_iso,
            source_label=INFOCLIMAT_HISTORY_SOURCE,
        )
        if not df.empty:
            blocks.append(df)
        if note:
            notices.append(note)

    if OPEN_METEO_SOURCE_LABEL in selected_sources:
        df, note = _fetch_open_meteo_history(
            lat=lat,
            lon=lon,
            start_day=start_iso,
            end_day=end_iso,
            source_label=OPEN_METEO_SOURCE_LABEL,
            model=OPEN_METEO_MODEL_METEOFRANCE,
        )
        if not df.empty:
            blocks.append(df)
        if note:
            notices.append(note)

    if "Open-Meteo Standard" in selected_sources:
        df, note = _fetch_open_meteo_history(
            lat=lat,
            lon=lon,
            start_day=start_iso,
            end_day=end_iso,
            source_label="Open-Meteo Standard",
            model=None,
        )
        if not df.empty:
            blocks.append(df)
        if note:
            notices.append(note)

    if not blocks:
        return pd.DataFrame(), notices
    history = pd.concat(blocks, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"], utc=True, errors="coerce")
    history["precip_mm"] = pd.to_numeric(history["precip_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
    history = history.dropna(subset=["date"])
    history = history.sort_values(["source", "date"]).reset_index(drop=True)
    return history, notices


@st.cache_data(show_spinner=False)
def _build_map(
    lgv_lines: List[List[Tuple[float, float]]],
    stations_df: pd.DataFrame,
    rain_col: str,
    map_style: str,
) -> folium.Map:
    center = [46.2, 0.2]
    all_pts = [pt for line in lgv_lines for pt in line]
    if all_pts:
        center = [float(np.mean([p[0] for p in all_pts])), float(np.mean([p[1] for p in all_pts]))]
    m = _create_base_map(location=center, zoom_start=7, map_style=map_style)

    for line in lgv_lines:
        folium.PolyLine(line, color="#1d4ed8", weight=4, opacity=0.85, tooltip="Trace LGV SEA").add_to(m)

    if stations_df.empty:
        return m

    plot_df = stations_df.copy()
    for col in [
        rain_col,
        "rain_24h_mm",
        "rain_7d_mm",
        "rain_30d_mm",
        "rain_month_mm",
        "rain_12h_mm",
        "rain_forecast_mm",
        "distance_to_lgv_km",
        "reliability_score",
        "obs_age_h",
        "near_station_count",
        "near_delta_metric_pct",
    ]:
        if col not in plot_df.columns:
            plot_df[col] = np.nan
        plot_df[col] = pd.to_numeric(plot_df.get(col), errors="coerce")
    plot_df["monthly_rank"] = plot_df["rain_month_mm"].rank(method="dense", ascending=False).astype("Int64")
    plot_df["monthly_alert_level"] = plot_df["rain_month_mm"].map(_monthly_alert_level)
    plot_df = plot_df.sort_values(
        ["rain_month_mm", rain_col, "reliability_score", "distance_to_lgv_km"],
        ascending=[True, True, True, False],
        na_position="last",
    )

    month_vals = pd.to_numeric(plot_df.get("rain_month_mm"), errors="coerce")
    month_vmax = float(month_vals.max()) if month_vals.notna().any() else 1.0
    month_vmax = max(month_vmax, 1.0)
    total_points = int(len(plot_df))

    for _, row in plot_df.iterrows():
        lat = pd.to_numeric(row.get("latitude"), errors="coerce")
        lon = pd.to_numeric(row.get("longitude"), errors="coerce")
        val = pd.to_numeric(row.get(rain_col), errors="coerce")
        month_val = pd.to_numeric(row.get("rain_month_mm"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon) or (pd.isna(val) and pd.isna(month_val)):
            continue
        month_ratio = max(0.0, min(1.0, float(0.0 if pd.isna(month_val) else month_val) / month_vmax))
        monthly_level = str(row.get("monthly_alert_level") or "INCONNU")
        fill_color = MONTHLY_ALERT_COLORS.get(monthly_level, MONTHLY_ALERT_COLORS["INCONNU"])
        rclass = str(row.get("reliability_class") or "")
        color = RELIABILITY_BORDER_COLORS.get(rclass, "#334155")
        rel_score = pd.to_numeric(row.get("reliability_score"), errors="coerce")
        near_delta_pct = pd.to_numeric(row.get("near_delta_metric_pct"), errors="coerce")
        obs_age_h = pd.to_numeric(row.get("obs_age_h"), errors="coerce")
        near_count = pd.to_numeric(row.get("near_station_count"), errors="coerce")
        monthly_rank = pd.to_numeric(row.get("monthly_rank"), errors="coerce")
        capture_label = _source_capture_label(row.get("source"))
        hist_backfill_source = str(row.get("history_backfill_source") or "").strip()
        hist_backfill_station = str(row.get("history_backfill_station") or "").strip()
        hist_backfill_obs_date = str(row.get("history_backfill_obs_date") or "").strip()
        hist_backfill_html = ""
        if hist_backfill_source:
            hist_backfill_html = (
                "<hr style='margin:6px 0;'>"
                "<b>Traçabilite cumul long</b><br>"
                f"7j/30j/mois consolides via: {hist_backfill_source}<br>"
                f"Reference: {hist_backfill_station or 'commune LGV'}<br>"
                f"Derniere date archive: {hist_backfill_obs_date or 'N/A'}<br>"
            )
        popup = (
            "<div style='font-size:13px;line-height:1.45;'>"
            f"<b>{row.get('station_display')}</b><br>"
            f"Commune LGV: {row.get('station_commune_name')}<br>"
            f"Source: {row.get('source')}<br>"
            f"Type de captation: {capture_label}<br>"
            "<hr style='margin:6px 0;'>"
            "<b>Priorisation ferroviaire pluie mensuelle</b><br>"
            f"Rang mensuel: #{int(monthly_rank) if pd.notna(monthly_rank) else 'N/A'} / {total_points}<br>"
            f"Niveau mensuel: {monthly_level}<br>"
            f"Cumul mensuel: {_format_num(month_val, 1, ' mm')}<br>"
            f"Cumul 30 jours: {_display_rain_30d_text(row)}<br>"
            f"Cumul 7 jours: {_format_num(row.get('rain_7d_mm'), 1, ' mm')}<br>"
            f"Cumul 24h: {_format_num(row.get('rain_24h_mm'), 1, ' mm')}<br>"
            f"Cumul 12h: {_format_num(row.get('rain_12h_mm'), 1, ' mm')}<br>"
            f"Prevision courte: {_format_num(row.get('rain_forecast_mm'), 1, ' mm')}<br>"
            f"Indicateur affiche: {rain_col} = {_format_num(val, 1, ' mm')}<br>"
            "<hr style='margin:6px 0;'>"
            "<b>Contexte d'exploitation</b><br>"
            f"Distance a la LGV: {_format_num(row.get('distance_to_lgv_km'), 1, ' km')}<br>"
            f"Date observation: {row.get('date_obs_raw') or 'N/A'}<br>"
            f"Age observation: {_format_num(obs_age_h, 1, ' h')}<br>"
            f"{hist_backfill_html}"
            "<hr style='margin:6px 0;'>"
            "<b>Fiabilite de la donnee</b><br>"
            f"Score: {_format_num(rel_score, 1, '/100')} ({row.get('reliability_class', 'N/A')})<br>"
            f"Voisins compares: {int(near_count) if pd.notna(near_count) else 0}<br>"
            f"Ecart vs voisins: {_format_num(near_delta_pct, 1, ' %')}<br>"
            f"Lecture qualite: {row.get('reliability_reason') or 'N/A'}"
            "</div>"
        )
        folium.CircleMarker(
            [float(lat), float(lon)],
            radius=5 + 7 * month_ratio,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(popup, max_width=420),
            tooltip=(
                f"#{int(monthly_rank) if pd.notna(monthly_rank) else '-'} "
                f"{row.get('station_display')} | mensuel {_format_num(month_val, 1, ' mm')}"
            ),
        ).add_to(m)

    return m


def _build_commune_map_input(commune_pluvio_df: pd.DataFrame) -> pd.DataFrame:
    if commune_pluvio_df.empty:
        return pd.DataFrame()

    out = commune_pluvio_df.copy()
    out["station_id"] = "commune_lgv_" + out.get("commune_code", pd.Series("", index=out.index)).fillna("").astype(str)
    out["station_name"] = out.get("commune_name", pd.Series("Commune LGV", index=out.index)).fillna("Commune LGV").astype(str)
    out["station_display"] = out["station_name"].astype(str) + " (commune LGV)"
    out["station_commune_name"] = out["station_name"].astype(str)
    out["source"] = out.get("provider_source", pd.Series("", index=out.index)).fillna("Sans source").astype(str)
    out["distance_to_lgv_km"] = 0.0
    out["rain_12h_mm"] = np.nan
    out["rain_forecast_mm"] = np.nan
    out["history_backfill_source"] = out.get("provider_source", pd.Series("", index=out.index)).fillna("").astype(str)
    out["history_backfill_station"] = out.get("provider_station", pd.Series("", index=out.index)).fillna("").astype(str)
    out["history_backfill_obs_date"] = out.get("date_obs_raw", pd.Series("", index=out.index)).fillna("").astype(str)
    if "rain_30d_source_label" not in out.columns:
        out["rain_30d_source_label"] = ""
    out["rain_30d_source_label"] = out.apply(_rain_30d_source_label, axis=1)
    out["rain_30d_is_reliable"] = out["rain_30d_source_label"].astype(str).str.len() > 0
    out["reliability_score"] = out["source"].map(_source_reliability_note).astype(float)
    out["reliability_class"] = out["reliability_score"].map(_reliability_class)
    out["reliability_reason"] = np.where(
        out["provider_source"].fillna("").astype(str).str.len() > 0,
        "Lecture corridor communale LGV",
        "Commune LGV sans source exploitable",
    )
    out["obs_age_h"] = np.nan
    out["near_station_count"] = 0
    out["near_delta_metric_pct"] = np.nan
    return out


st.set_page_config(page_title="LGV SEA Pluvio Stations Pro", page_icon=":umbrella:", layout="wide")
st.title("LGV SEA - Pluviometrie Stations Pro")
st.caption(
    "Version pro: selection automatique de la source la plus exploitable "
    "(Meteo-France Portail API si configure, sinon Open-Meteo), "
    "avec option InfoClimat proche LGV pour comparaison."
)
st.caption("Rendu graphique actif: Plotly (compatible Streamlit Cloud).")

st.subheader("Vigilance Meteo-France - departements LGV SEA")
public_vigilance_df, public_vigilance_notice = _fetch_public_vigilance_departemental()
official_vigilance_df, official_vigilance_notice = pd.DataFrame(), ""
if _meteo_france_portal_is_configured():
    vigilance_token, vigilance_token_note = _request_meteo_france_portal_token()
    official_vigilance_df, official_vigilance_notice = _fetch_meteo_france_vigilance(vigilance_token)
    official_vigilance_notice = f"{vigilance_token_note} | {official_vigilance_notice}"

vigilance_df = pd.concat([official_vigilance_df, public_vigilance_df], ignore_index=True)
if not vigilance_df.empty:
    vigilance_df = vigilance_df.drop_duplicates(subset=["domain_id", "phenomenon_id"], keep="first")
vigilance_notice = " | ".join([n for n in [official_vigilance_notice, public_vigilance_notice] if n])

if vigilance_df.empty:
    st.caption(f"Vigilance indisponible pour le moment. {vigilance_notice}")
else:
    vigilance_alerts = vigilance_df[pd.to_numeric(vigilance_df["color_id"], errors="coerce") >= 3]
    if not vigilance_alerts.empty:
        alert_lines = [
            f"- {row['department_name']}: {row['phenomenon_name']} - {row['color_name']}"
            for _, row in vigilance_alerts.sort_values(
                ["color_id", "department_name"], ascending=[False, True]
            ).iterrows()
        ]
        st.error("Vigilance active (orange/rouge) sur le trace LGV:\n\n" + "\n".join(alert_lines))
    else:
        st.success("Aucune vigilance orange/rouge en cours sur les departements traverses par la LGV SEA.")

    vigilance_col_order = [
        VIGILANCE_PHENOMENON_LABELS[k] for k in sorted(VIGILANCE_PHENOMENON_LABELS, key=int)
    ]
    vigilance_pivot = vigilance_df.pivot_table(
        index="department_name", columns="phenomenon_name", values="color_name", aggfunc="first"
    ).reindex(columns=vigilance_col_order)
    vigilance_color_by_label = {v: k for k, v in VIGILANCE_COLOR_LABELS.items()}

    def _style_vigilance_cell(val: object) -> str:
        color_id = vigilance_color_by_label.get(str(val), "")
        hex_color = VIGILANCE_COLOR_HEX.get(color_id, "#e2e8f0")
        text_color = "#1f2937" if color_id in ("1", "2", "") else "#ffffff"
        return f"background-color: {hex_color}; color: {text_color}; text-align: center;"

    st.dataframe(vigilance_pivot.style.map(_style_vigilance_cell), use_container_width=True)
    if "Crues" not in vigilance_df["phenomenon_name"].unique().tolist():
        st.caption(
            "Crues non affichees: cette donnee n'est disponible que via l'API officielle "
            "Meteo-France Vigilance (abonnement DPVigilance requis sur portail-api.meteofrance.fr)."
        )
    st.caption(vigilance_notice)

source_mode_options = [SOURCE_MODE_METEOFRANCE, SOURCE_MODE_OPEN, SOURCE_MODE_MIX]
default_source_mode = SOURCE_MODE_METEOFRANCE if _meteo_france_portal_is_configured() else SOURCE_MODE_OPEN
with st.sidebar:
    st.subheader("Sources")
    source_mode = st.selectbox(
        "Jeu de donnees",
        options=source_mode_options,
        index=source_mode_options.index(default_source_mode),
        help=(
            "Demarrage automatique sur la source la plus exploitable pour cet environnement. "
            "Meteo-France reste prioritaire quand ses secrets sont disponibles."
        ),
    )
    infoclimat_near_max_km = float(INFOCLIMAT_STRICT_RADIUS_KM)
    infoclimat_near_top_n = 9999
    st.caption(
        "Regle pro: toutes stations InfoClimat a <=10 km de la LGV SEA, "
        "avec elargissement automatique si couverture insuffisante."
    )

try:
    snapshot, snapshot_source = _load_snapshot_payload()
except Exception as exc:
    st.error(str(exc))
    st.stop()

lgv_lines = _extract_lgv_lines(snapshot)
snapshot_ts = pd.to_datetime(snapshot.get("timestamp_utc"), utc=True, errors="coerce")
raw_snapshot_weather = snapshot.get("weather", []) if isinstance(snapshot, dict) else []
raw_snapshot_weather = [row for row in raw_snapshot_weather if isinstance(row, dict)]
snapshot_weather_df = _safe_weather_df({"weather": raw_snapshot_weather})
if not snapshot_weather_df.empty:
    snapshot_weather_df = snapshot_weather_df.copy()
    snapshot_weather_df["selection_mode"] = (
        snapshot_weather_df.get("selection_mode", pd.Series("", index=snapshot_weather_df.index))
        .fillna("")
        .astype(str)
        .replace("", "snapshot_latest_weather")
    )
snapshot_commune_proxy_df = _build_commune_proxy_weather_df(
    snapshot_weather_df,
    source_label="Snapshot local proxy communes LGV",
    selection_mode="snapshot_commune_proxy_weather",
)

data_build_notices: List[str] = []
infoclimat_local_df, infoclimat_local_notice = _load_infoclimat_synop_local(max_distance_km=LOCAL_INFOCLIMAT_RADIUS_KM)
if infoclimat_local_notice:
    data_build_notices.append(infoclimat_local_notice)

# Base InfoClimat/SYNOP: snapshot + enrichissement local (seed/cache).
raw_weather_rows = [row for row in raw_snapshot_weather if _is_infoclimat_row(row)]
if not infoclimat_local_df.empty:
    known_communes = {
        str(row.get("station_id") or "").strip(): str(row.get("station_commune_name") or "").strip()
        for row in raw_snapshot_weather
        if str(row.get("station_id") or "").strip()
    }
    infoclimat_local_df = infoclimat_local_df.copy()
    infoclimat_local_df["station_commune_name"] = np.where(
        infoclimat_local_df["station_id"].astype(str).map(known_communes).fillna("").astype(str).str.len() > 0,
        infoclimat_local_df["station_id"].astype(str).map(known_communes).fillna("").astype(str),
        infoclimat_local_df.get("station_commune_name", pd.Series("Inconnue", index=infoclimat_local_df.index))
        .fillna("Inconnue")
        .astype(str),
    )
    raw_weather_rows = [row for row in raw_weather_rows if not _is_infoclimat_source(row.get("source"))]
    raw_weather_rows.extend(infoclimat_local_df.to_dict(orient="records"))

infoclimat_weather_df = _safe_weather_df({"weather": raw_weather_rows})
if not infoclimat_weather_df.empty:
    weather_sources = infoclimat_weather_df.get("source", pd.Series("", index=infoclimat_weather_df.index)).fillna("").astype(str)
    weather_selection_mode = infoclimat_weather_df.get("selection_mode", pd.Series("", index=infoclimat_weather_df.index)).fillna("").astype(str)
    weather_calc_method = infoclimat_weather_df.get("rain_calc_method", pd.Series("", index=infoclimat_weather_df.index)).fillna("").astype(str)
    info_mask = (
        weather_sources.map(_is_infoclimat_source)
        | weather_selection_mode.str.lower().str.contains("info_climat|infoclimat|synop", regex=True)
        | weather_calc_method.str.lower().str.contains("synop", regex=False)
    )
    infoclimat_weather_df = infoclimat_weather_df[info_mask].copy()

reference_history_blocks: List[pd.DataFrame] = []
if not snapshot_weather_df.empty:
    reference_history_blocks.append(snapshot_weather_df.copy())
if not infoclimat_weather_df.empty:
    reference_history_blocks.append(infoclimat_weather_df.copy())
reference_history_weather_df = (
    _safe_weather_df({"weather": pd.concat(reference_history_blocks, ignore_index=True, sort=False).to_dict(orient="records")})
    if reference_history_blocks
    else pd.DataFrame()
)

infoclimat_near_df = pd.DataFrame()
if not infoclimat_weather_df.empty:
    infoclimat_near_df, used_monitor_radius_km, adaptive_radius_used = _select_infoclimat_monitoring_set(
        stations_df=infoclimat_weather_df,
        preferred_radius_km=float(infoclimat_near_max_km),
        min_stations=int(INFOCLIMAT_MIN_STATIONS_COVERAGE),
    )
    if not infoclimat_near_df.empty:
        priority_count = int((infoclimat_near_df.get("priority_name", pd.Series("", index=infoclimat_near_df.index)).astype(str).str.len() > 0).sum())
        data_build_notices.append(
            "InfoClimat proche LGV: "
            + f"{len(infoclimat_near_df)} stations retenues "
            + f"(distance<={float(used_monitor_radius_km):.0f} km, toutes stations, prioritaires={priority_count})."
        )
        if adaptive_radius_used:
            data_build_notices.append(
                "Couverture adaptee: rayon elargi automatiquement pour eviter une surveillance mono-station."
            )
    else:
        data_build_notices.append(
            f"InfoClimat proche LGV: aucune station retenue (distance<={float(infoclimat_near_max_km):.0f} km)."
        )
infoclimat_near_commune_proxy_df = _build_commune_proxy_weather_df(
    infoclimat_near_df,
    source_label="SYNOP/InfoClimat proxy communes LGV",
    selection_mode="infoclimat_commune_proxy_weather",
)
infoclimat_full_commune_proxy_df = _build_commune_proxy_weather_df(
    infoclimat_weather_df,
    source_label="SYNOP/InfoClimat pack proxy communes LGV",
    selection_mode="infoclimat_full_commune_proxy_weather",
)
if source_mode == SOURCE_MODE_METEOFRANCE:
    data_build_notices.append(
        "Mode Meteo-France strict: donnees officielles SYNOP (Portail API) projetees sur toutes les communes LGV."
    )
elif source_mode == SOURCE_MODE_OPEN:
    data_build_notices.append(
        "Mode Open-Meteo strict: grille MeteoFrance calculee sur les communes traversees par la LGV SEA."
    )

mf_notice = ""
meteo_france_df = pd.DataFrame()
if source_mode in {SOURCE_MODE_METEOFRANCE, SOURCE_MODE_MIX}:
    meteo_france_df, mf_notice = _fetch_meteo_france_portal_commune_weather()
    if mf_notice:
        data_build_notices.append(mf_notice)
    if not meteo_france_df.empty:
        meteo_france_df = meteo_france_df.copy()
        meteo_france_df["source"] = METEO_FRANCE_SOURCE_LABEL
        if not reference_history_weather_df.empty:
            meteo_france_df, mf_hist_notice = _enrich_commune_weather_with_reference_history(
                meteo_france_df,
                reference_history_weather_df,
            )
            if mf_hist_notice:
                data_build_notices.append("Meteo-France Portail: " + mf_hist_notice)

open_meteo_ref_df = pd.DataFrame()
open_meteo_archive_ref_df = pd.DataFrame()
if source_mode in {SOURCE_MODE_METEOFRANCE, SOURCE_MODE_OPEN, SOURCE_MODE_MIX} or meteo_france_df.empty:
    ref_points_total = 0
    ref_points_key = _build_open_meteo_key_from_lgv_communes()
    ref_points_total = int(len(ref_points_key))
    if not ref_points_key:
        open_ref_base = infoclimat_near_df.copy()
        if open_ref_base.empty and not infoclimat_local_df.empty:
            open_ref_base, used_local_radius_km, adaptive_local_radius = _select_infoclimat_monitoring_set(
                stations_df=infoclimat_local_df,
                preferred_radius_km=float(infoclimat_near_max_km),
                min_stations=int(INFOCLIMAT_MIN_STATIONS_COVERAGE),
            )
            if adaptive_local_radius:
                data_build_notices.append(
                    "Couverture adaptee (fallback local): rayon elargi automatiquement "
                    + f"jusqu'a {float(used_local_radius_km):.0f} km."
                )
        ref_points_key = _build_open_meteo_reference_key(open_ref_base) if not open_ref_base.empty else tuple()
        ref_points_total = int(len(ref_points_key))
    if ref_points_key:
        open_meteo_ref_df, open_meteo_notice = _fetch_open_meteo_reference_points(
            ref_points_key,
            model=OPEN_METEO_MODEL_METEOFRANCE,
        )
        if open_meteo_notice:
            data_build_notices.append(open_meteo_notice)
        open_meteo_archive_ref_df, open_meteo_archive_notice = _fetch_open_meteo_archive_reference_points(
            ref_points_key,
            model=OPEN_METEO_MODEL_METEOFRANCE,
        )
        if open_meteo_archive_notice:
            data_build_notices.append(open_meteo_archive_notice)
    if not open_meteo_ref_df.empty:
        open_meteo_ref_df = open_meteo_ref_df.copy()
        open_meteo_ref_df["source"] = OPEN_METEO_SOURCE_LABEL
        data_build_notices.append(
            "Open-Meteo MeteoFrance: grille active sur les communes traversees par la LGV SEA."
        )
        if not open_meteo_archive_ref_df.empty:
            open_meteo_ref_df, open_long_notice = _enrich_commune_weather_with_reference_history(
                open_meteo_ref_df,
                open_meteo_archive_ref_df,
                replace_existing=True,
                target_cols=["rain_7d_mm", "rain_30d_mm", "rain_month_mm"],
            )
            if open_long_notice:
                data_build_notices.append("Open-Meteo (cumuls longs): " + open_long_notice)
        if not meteo_france_df.empty and not open_meteo_archive_ref_df.empty:
            meteo_france_df, mf_long_notice = _enrich_commune_weather_with_reference_history(
                meteo_france_df,
                open_meteo_archive_ref_df,
                replace_existing=True,
                target_cols=["rain_7d_mm", "rain_30d_mm", "rain_month_mm"],
            )
            if mf_long_notice:
                data_build_notices.append("Meteo-France Portail (cumuls longs): " + mf_long_notice)
    if open_meteo_ref_df.empty:
        if not open_meteo_archive_ref_df.empty:
            open_meteo_ref_df = open_meteo_archive_ref_df.copy()
            data_build_notices.append(
                "Fallback Open-Meteo: archive journaliere communale active, alignee avec l'app monitoring LGV."
            )
            if not meteo_france_df.empty:
                meteo_france_df, mf_long_notice = _enrich_commune_weather_with_reference_history(
                    meteo_france_df,
                    open_meteo_archive_ref_df,
                    replace_existing=True,
                    target_cols=["rain_7d_mm", "rain_30d_mm", "rain_month_mm"],
                )
                if mf_long_notice:
                    data_build_notices.append("Meteo-France Portail (cumuls longs): " + mf_long_notice)
        else:
            open_meteo_ref_df, open_grid_notice = _load_open_meteo_grid_local()
            if open_grid_notice:
                data_build_notices.append(open_grid_notice)
            if not open_meteo_ref_df.empty:
                open_meteo_ref_df = open_meteo_ref_df.copy()
                open_meteo_ref_df["source"] = OPEN_METEO_SOURCE_LABEL
                data_build_notices.append(
                    "Fallback Open-Meteo: grille locale chargee (seed/cache) pour garantir la couverture LGV."
                )
                if not open_meteo_archive_ref_df.empty:
                    open_meteo_ref_df, open_long_notice = _enrich_commune_weather_with_reference_history(
                        open_meteo_ref_df,
                        open_meteo_archive_ref_df,
                        replace_existing=True,
                        target_cols=["rain_7d_mm", "rain_30d_mm", "rain_month_mm"],
                    )
                    if open_long_notice:
                        data_build_notices.append("Open-Meteo (cumuls longs): " + open_long_notice)
                if not meteo_france_df.empty and not open_meteo_archive_ref_df.empty:
                    meteo_france_df, mf_long_notice = _enrich_commune_weather_with_reference_history(
                        meteo_france_df,
                        open_meteo_archive_ref_df,
                        replace_existing=True,
                        target_cols=["rain_7d_mm", "rain_30d_mm", "rain_month_mm"],
                    )
                    if mf_long_notice:
                        data_build_notices.append("Meteo-France Portail (cumuls longs): " + mf_long_notice)

    if not open_meteo_ref_df.empty and ref_points_total > 0:
        coverage_col = "station_ref_id" if "station_ref_id" in open_meteo_ref_df.columns else "station_commune_name"
        covered = int(open_meteo_ref_df.get(coverage_col, pd.Series(dtype=str)).dropna().astype(str).str.strip().replace("", np.nan).dropna().nunique())
        coverage_pct = round(100.0 * covered / max(1, ref_points_total), 1)
        data_build_notices.append(
            "Couverture communes LGV (Open-Meteo): "
            + f"{covered}/{ref_points_total} ({coverage_pct}%)."
        )
    else:
        data_build_notices.append("Open-Meteo reference: aucun point geolocalise disponible.")

weather_blocks: List[pd.DataFrame] = []
if source_mode in {SOURCE_MODE_METEOFRANCE, SOURCE_MODE_MIX} and not meteo_france_df.empty:
    weather_blocks.append(meteo_france_df)
if source_mode in {SOURCE_MODE_OPEN, SOURCE_MODE_MIX} and not open_meteo_ref_df.empty:
    weather_blocks.append(open_meteo_ref_df)
if source_mode == SOURCE_MODE_MIX and not infoclimat_near_df.empty:
    weather_blocks.append(infoclimat_near_df)

if source_mode == SOURCE_MODE_METEOFRANCE and meteo_france_df.empty:
    if not open_meteo_ref_df.empty:
        weather_blocks.append(open_meteo_ref_df)
        data_build_notices.append("Fallback: Meteo-France indisponible, bascule Open-Meteo corridor.")
    elif not infoclimat_near_commune_proxy_df.empty:
        weather_blocks.append(infoclimat_near_commune_proxy_df)
        data_build_notices.append("Fallback: Meteo-France indisponible, bascule proxy communes LGV depuis SYNOP/InfoClimat proche.")
    elif not infoclimat_full_commune_proxy_df.empty:
        weather_blocks.append(infoclimat_full_commune_proxy_df)
        data_build_notices.append("Fallback: Meteo-France indisponible, utilisation du dernier pack SYNOP/InfoClimat reprojete sur communes LGV.")
    elif not snapshot_commune_proxy_df.empty:
        weather_blocks.append(snapshot_commune_proxy_df)
        data_build_notices.append("Fallback ultime: Meteo-France indisponible, utilisation du dernier snapshot reprojete sur communes LGV.")

if source_mode == SOURCE_MODE_OPEN and open_meteo_ref_df.empty:
    if not infoclimat_near_commune_proxy_df.empty:
        weather_blocks.append(infoclimat_near_commune_proxy_df)
        data_build_notices.append("Fallback: Open-Meteo indisponible, proxy communes LGV depuis InfoClimat proche.")
    elif not infoclimat_full_commune_proxy_df.empty:
        weather_blocks.append(infoclimat_full_commune_proxy_df)
        data_build_notices.append("Fallback: Open-Meteo indisponible, dernier pack SYNOP/InfoClimat reprojete sur communes LGV.")
    elif not snapshot_commune_proxy_df.empty:
        weather_blocks.append(snapshot_commune_proxy_df)
        data_build_notices.append("Fallback ultime: Open-Meteo indisponible, snapshot reprojete sur communes LGV.")

if source_mode == SOURCE_MODE_MIX and not weather_blocks and not snapshot_commune_proxy_df.empty:
    weather_blocks.append(snapshot_commune_proxy_df)
    data_build_notices.append("Fallback ultime: mix indisponible en live, utilisation du dernier snapshot reprojete sur communes LGV.")

if weather_blocks:
    merged_rows = pd.concat(weather_blocks, ignore_index=True, sort=False).to_dict(orient="records")
    weather_df = _safe_weather_df({"weather": merged_rows})
else:
    weather_df = _safe_weather_df({"weather": []})

if weather_df.empty:
    if not snapshot_commune_proxy_df.empty:
        weather_df = snapshot_commune_proxy_df.copy()
        data_build_notices.append(
            "Secours d'urgence: aucune source live exploitable, la page utilise le dernier snapshot reprojete sur communes LGV."
        )
    else:
        st.warning(
            f"Aucune donnee disponible pour le mode source '{source_mode}'. "
            + "Ni source primaire, ni fallback exploitable n'ont pu etre charges."
        )
        st.stop()

weather_df = _apply_reliable_rain_30d_policy(weather_df)
loaded_sources_all = sorted(weather_df.get("source", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
loaded_sources_label = ", ".join(loaded_sources_all) if loaded_sources_all else "Aucune"

with st.sidebar:
    st.subheader("Filtres stations")
    if st.button("Rafraichir", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    metric_label = st.selectbox("Indicateur pluvio", list(RAIN_METRICS.keys()), index=1)
    metric_col = RAIN_METRICS[metric_label]
    map_style = st.selectbox("Fond de carte", list(MAP_TILE_STYLES.keys()), index=0)

    max_distance_km = st.slider(
        "Distance max a la LGV (km)",
        min_value=1.0,
        max_value=float(LOCAL_INFOCLIMAT_RADIUS_KM),
        value=min(40.0, float(LOCAL_INFOCLIMAT_RADIUS_KM)),
        step=0.5,
    )
    compare_radius_km = st.slider(
        "Rayon comparaison stations proches (km)",
        min_value=2.0,
        max_value=80.0,
        value=15.0,
        step=0.5,
    )
    min_neighbors = st.slider(
        "Voisins min pour fiabiliser",
        min_value=1,
        max_value=8,
        value=2,
        step=1,
    )
    incoherence_alert_pct = st.slider(
        "Seuil alerte incoherence (%)",
        min_value=10,
        max_value=150,
        value=45,
        step=5,
        help="Ecart relatif (vs mediane des stations proches) au-dela duquel la station est marquee incoherente.",
    )
    source_options = sorted(weather_df.get("source", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    selected_sources = _multiselect_all("Sources stations", source_options, key="plv_sources")
    st.caption(f"Mode demande: {source_mode}")
    st.caption(f"Sources effectivement chargees: {loaded_sources_label}")

    commune_options = sorted(weather_df.get("station_commune_name", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    selected_communes = _multiselect_all("Communes stations", commune_options, key="plv_communes")

filtered_stations = weather_df.copy()
if not filtered_stations.empty and "distance_to_lgv_km" in filtered_stations.columns:
    filtered_stations = filtered_stations[
        pd.to_numeric(filtered_stations["distance_to_lgv_km"], errors="coerce").fillna(9999.0) <= float(max_distance_km)
    ]
if not filtered_stations.empty and selected_sources:
    filtered_stations = filtered_stations[filtered_stations["source"].astype(str).isin(selected_sources)]
if not filtered_stations.empty and selected_communes:
    filtered_stations = filtered_stations[filtered_stations["station_commune_name"].astype(str).isin(selected_communes)]

station_options = filtered_stations.get("station_display", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
station_options = sorted(station_options)
with st.sidebar:
    selected_stations = _multiselect_all("Stations (nom commune)", station_options, key="plv_station_display")
if not filtered_stations.empty and selected_stations:
    filtered_stations = filtered_stations[filtered_stations["station_display"].astype(str).isin(selected_stations)]

if filtered_stations.empty:
    by_source = (
        weather_df.get("source", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .value_counts()
        .to_dict()
    )
    st.warning(
        "Aucune station sur ce filtre. "
        + f"Stations chargees={int(len(weather_df))}. "
        + f"Repartition sources={by_source}. "
        + "Elargis la distance LGV ou remets les filtres communes/stations/sources sur 'Tout'."
    )
    st.stop()

if metric_col not in filtered_stations.columns:
    filtered_stations[metric_col] = 0.0
filtered_stations[metric_col] = pd.to_numeric(filtered_stations.get(metric_col), errors="coerce").fillna(0.0).clip(lower=0.0)
filtered_stations = _build_proximity_quality(
    stations_df=filtered_stations,
    snapshot_ts=snapshot_ts,
    metric_col=metric_col,
    compare_radius_km=float(compare_radius_km),
    min_neighbors=int(min_neighbors),
)
filtered_stations["rain_month_mm"] = pd.to_numeric(filtered_stations.get("rain_month_mm"), errors="coerce")
filtered_stations["monthly_rank"] = filtered_stations["rain_month_mm"].rank(method="dense", ascending=False).astype("Int64")
filtered_stations["monthly_alert_level"] = filtered_stations["rain_month_mm"].map(_monthly_alert_level)
filtered_stations["incoherence_flag"] = (
    pd.to_numeric(filtered_stations.get("near_delta_metric_pct"), errors="coerce").fillna(0.0) >= float(incoherence_alert_pct)
)
filtered_stations = filtered_stations.sort_values(
    [metric_col, "rain_month_mm", "distance_to_lgv_km"],
    ascending=[False, False, True],
    na_position="last",
)

top_n_max = max(1, int(len(filtered_stations)))
top_n_key = f"plv_top_n_{int(top_n_max)}"
top_n_default = int(min(25, top_n_max))
with st.sidebar:
    top_n = int(
        st.number_input(
            "Top stations (graphe)",
            min_value=1,
            max_value=top_n_max,
            value=top_n_default,
            step=1,
            key=top_n_key,
        )
    )
    st.markdown("---")
    st.subheader("Historique (depuis 2026)")
    history_reference_mode = "Commune LGV (aligne app PRO)"
    st.caption("Reference historique: centroide communal LGV, aligne sur l'app monitoring.")
    history_station_default = str(filtered_stations.iloc[0]["station_display"])
    history_station_options = sorted(filtered_stations["station_display"].astype(str).unique().tolist())
    history_station_display = st.selectbox(
        "Station historique",
        options=history_station_options,
        index=history_station_options.index(history_station_default),
    )
    history_source_options = [INFOCLIMAT_HISTORY_SOURCE, OPEN_METEO_SOURCE_LABEL]
    if source_mode in {SOURCE_MODE_OPEN, SOURCE_MODE_METEOFRANCE}:
        history_default_sources = [OPEN_METEO_SOURCE_LABEL]
    else:
        history_default_sources = [INFOCLIMAT_HISTORY_SOURCE, OPEN_METEO_SOURCE_LABEL]
    history_sources = st.multiselect(
        "Sources historique",
        options=history_source_options,
        default=history_default_sources,
    )
    st.caption("Historique compare: Open-Meteo archive et/ou InfoClimat local.")
    today_utc = datetime.now(timezone.utc).date()
    history_start = st.date_input(
        "Date debut",
        value=HISTORY_MIN_DATE,
        min_value=HISTORY_MIN_DATE,
        max_value=today_utc,
    )
    history_end = st.date_input(
        "Date fin",
        value=today_utc,
        min_value=HISTORY_MIN_DATE,
        max_value=today_utc,
    )

if history_end < history_start:
    history_start, history_end = history_end, history_start

selected_station_df = filtered_stations[filtered_stations["station_display"].astype(str) == str(history_station_display)].copy()
selected_station = selected_station_df.iloc[0].to_dict() if not selected_station_df.empty else filtered_stations.iloc[0].to_dict()
history_station_id = str(selected_station.get("station_id") or "")

commune_source_df = weather_df.copy()
if not commune_source_df.empty and selected_sources:
    commune_source_df = commune_source_df[commune_source_df["source"].astype(str).isin(selected_sources)]
commune_pluvio_df = _build_lgv_communes_pluvio_table(commune_source_df)
map_points_df = _build_commune_map_input(commune_pluvio_df) if not commune_pluvio_df.empty else filtered_stations.copy()
map_points_count = int(len(map_points_df))
map_reliability_series = pd.to_numeric(map_points_df.get("reliability_score"), errors="coerce").fillna(0.0)
map_reliability_class = map_points_df.get("reliability_class", pd.Series("A_VERIFIER", index=map_points_df.index)).astype(str)
map_metric_series = pd.to_numeric(map_points_df.get(metric_col), errors="coerce")

reliability_series = pd.to_numeric(filtered_stations.get("reliability_score"), errors="coerce").fillna(0.0)
reliability_class = filtered_stations.get("reliability_class", pd.Series("A_VERIFIER", index=filtered_stations.index)).astype(str)
incoherence_count = int(filtered_stations.get("incoherence_flag", pd.Series(False, index=filtered_stations.index)).fillna(False).sum())

k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
k1.metric("Points carte LGV", map_points_count, help="Nombre de points (communes ou stations) affiches sur la carte.")
k2.metric(
    "Communes",
    int(len(commune_pluvio_df)) if not commune_pluvio_df.empty else int(filtered_stations["station_commune_name"].astype(str).nunique()),
    help="Nombre de communes LGV distinctes couvertes par les donnees actuelles.",
)
k3.metric(
    f"Max {metric_label}",
    f"{float(map_metric_series.max()):.1f} mm" if map_metric_series.notna().any() else "N/A",
    help=f"Plus forte valeur de {metric_label.lower()} observee parmi les points affiches.",
)
k4.metric(
    f"Moyenne {metric_label}",
    f"{float(map_metric_series.mean()):.1f} mm" if map_metric_series.notna().any() else "N/A",
    help=f"Valeur moyenne de {metric_label.lower()} sur l'ensemble des points affiches.",
)
k5.metric("Distance max filtre", f"{float(max_distance_km):.1f} km", help="Rayon de filtrage applique autour de la LGV pour les stations InfoClimat.")
k6.metric(
    "Fiabilite moyenne",
    f"{float(map_reliability_series.mean()):.1f}/100" if not map_reliability_series.empty else "N/A",
    help="Score moyen de fiabilite des donnees (source + fraicheur + coherence locale), sur 100.",
)
k7.metric("Points FIABLE", int((map_reliability_class == "FIABLE").sum()), help="Nombre de points juges pleinement fiables (voir detail plus bas).")
k8.metric("Incoherences proximite", incoherence_count, help="Stations dont la mesure s'ecarte fortement de leurs voisines proches (a verifier en priorite).")

st.subheader("Alertes pluviometrie - communes LGV SEA")
_rain_alert_rank = {"CRITIQUE": 3, "ELEVE": 2, "MODERE": 1, "FAIBLE": 0, "INCONNU": -1}
rain_watch_df = map_points_df.copy()
rain_watch_df["monthly_alert_level"] = pd.to_numeric(rain_watch_df.get("rain_month_mm"), errors="coerce").map(
    _monthly_alert_level
)
rain_watch_df["_alert_rank"] = rain_watch_df["monthly_alert_level"].map(_rain_alert_rank).fillna(-1)
rain_watch_df = rain_watch_df[rain_watch_df["_alert_rank"] >= 2].sort_values(
    ["_alert_rank", "rain_month_mm"], ascending=[False, False]
)
if rain_watch_df.empty:
    st.success(
        "Aucune commune LGV en alerte pluie ELEVE ou CRITIQUE ce mois-ci "
        "(seuils: ELEVE >=120 mm, CRITIQUE >=180 mm cumules dans le mois)."
    )
else:
    st.error(f"{len(rain_watch_df)} commune(s) LGV en alerte pluie ELEVE ou CRITIQUE ce mois-ci.")
    rain_watch_display = pd.DataFrame(
        {
            "Commune": rain_watch_df["station_commune_name"].astype(str),
            "Cumul mensuel": rain_watch_df.get("rain_month_mm").map(lambda v: _format_num(v, 1, " mm")),
            "Niveau": rain_watch_df["monthly_alert_level"],
            "Fiabilite": rain_watch_df.get("reliability_class", pd.Series("", index=rain_watch_df.index)).astype(str),
        }
    )
    st.dataframe(rain_watch_display, use_container_width=True, hide_index=True)

stale_count = int(
    (pd.to_numeric(filtered_stations.get("obs_age_h"), errors="coerce").fillna(0.0) > float(WEATHER_STALE_ALERT_H)).sum()
)
unreliable_count = int(
    (filtered_stations.get("reliability_class", pd.Series("", index=filtered_stations.index)).astype(str) == "A_VERIFIER").sum()
)
if stale_count or unreliable_count:
    st.warning(
        f"Suivi qualite des donnees: {stale_count} station(s) avec observation vieille de plus de "
        f"{int(WEATHER_STALE_ALERT_H)}h, {unreliable_count} station(s) en fiabilite 'A verifier'. "
        "Voir le detail dans 'Fiabilite & Metadonnees sources' ci-dessous."
    )

with st.expander("Comment lire la carte et les alertes", expanded=False):
    st.markdown(
        "- **Couleur de remplissage des points sur la carte** = niveau de pluie du mois en cours "
        "(vert = faible, olive = modere, orange = eleve, rouge = critique).\n"
        "- **Couleur du contour des points** = fiabilite de la donnee "
        "(bleu-vert = fiable, marron = a surveiller, rouge fonce = a verifier).\n"
        "- **Taille du point** = proportionnelle au cumul mensuel de pluie.\n"
        "- **Alerte pluie** ci-dessus = communes dont le cumul mensuel depasse 120 mm (eleve) ou 180 mm (critique).\n"
        "- **Vigilance Meteo-France** en haut de page = risques meteo officiels par departement "
        "(orages/foudre, crues, vent, canicule, etc.), independants du niveau de pluie mesure."
    )

if snapshot_ts is not None and not pd.isna(snapshot_ts):
    st.caption(f"Snapshot: {snapshot_source} | timestamp_utc={snapshot_ts.isoformat()}")
else:
    st.caption(f"Snapshot: {snapshot_source} | timestamp inconnu")
if data_build_notices:
    st.caption(" | ".join([n for n in data_build_notices if n][:3]))
if bool(ENABLE_RAIN_30D):
    st.caption(
        "Le cumul 30 jours est affiche uniquement s'il provient d'une source longue fiabilisee "
        "(Open-Meteo archive communal, backfill communal equivalent, ou historique SYNOP consolide)."
    )
else:
    st.warning("Le cumul 30 jours est temporairement retire des usages exploitation car il n'est pas assez fiable.")

st.subheader("Fiabilite & Metadonnees sources")
st.caption("Score fiabilite station = 0.42*note_source + 0.23*fraicheur_obs + 0.35*coherence_locale.")
source_meta_df = _build_source_metadata_table(filtered_stations)
if source_meta_df.empty:
    st.info("Metadonnees sources indisponibles sur ce filtre.")
else:
    st.dataframe(source_meta_df, use_container_width=True, hide_index=True)
with st.expander("Detail du calcul de fiabilite par station", expanded=False):
    rel_cols = [
        "station_display",
        "source",
        "source_note",
        "freshness_note",
        "coherence_note",
        "reliability_source_component",
        "reliability_freshness_component",
        "reliability_coherence_component",
        "reliability_score",
        "reliability_class",
        "reliability_reason",
        "obs_age_h",
        "near_station_count",
        "near_delta_metric_pct",
    ]
    rel_cols = [c for c in rel_cols if c in filtered_stations.columns]
    st.dataframe(
        filtered_stations[rel_cols].sort_values("reliability_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Carte pluvio des communes LGV SEA")
map_obj = _build_map(lgv_lines, map_points_df, metric_col, map_style=map_style)
st_folium(
    map_obj,
    height=640,
    use_container_width=True,
    key="lgv_pluvio_map",
    returned_objects=[],
)
source_counts = (
    map_points_df.get("source", pd.Series(dtype=str))
    .dropna()
    .astype(str)
    .value_counts()
    .to_dict()
)
st.caption(f"Fond de carte actif: {map_style} | Points visibles: {map_points_count} | Sources: {source_counts}")
st.caption(
    "Lecture carte orientee gestionnaire de ligne: remplissage du point = pression pluvio mensuelle, "
    "contour = fiabilite de la donnee, ordre d'affichage = stations les plus arrosees sur le mois."
)
st.markdown("**Priorisation mensuelle des points a surveiller**")
monthly_priority_cols = [
    "monthly_rank",
    "monthly_alert_level",
    "station_display",
    "station_commune_name",
    "source",
    "rain_month_mm",
    "rain_7d_mm",
    "rain_24h_mm",
    "distance_to_lgv_km",
    "reliability_class",
    "reliability_score",
]
monthly_priority_cols = [c for c in monthly_priority_cols if c in filtered_stations.columns]
monthly_priority_df = (
    filtered_stations[monthly_priority_cols]
    .sort_values(["rain_month_mm", "reliability_score"], ascending=[False, False], na_position="last")
    .head(20)
)
st.dataframe(monthly_priority_df, use_container_width=True, hide_index=True)
with st.expander("Localisation des stations (lat/lon)", expanded=False):
    loc_cols = [
        "station_display",
        "station_id",
        "station_commune_name",
        "priority_name",
        "priority_match_km",
        "source",
        "latitude",
        "longitude",
        "distance_to_lgv_km",
        "date_obs_raw",
    ]
    loc_cols = [c for c in loc_cols if c in filtered_stations.columns]
    st.dataframe(
        filtered_stations[loc_cols].sort_values(["distance_to_lgv_km", "station_display"], ascending=[True, True]),
        use_container_width=True,
        hide_index=True,
    )

open_loc_df = filtered_stations[
    filtered_stations.get("source", pd.Series("", index=filtered_stations.index)).astype(str).map(_is_open_meteo_source)
].copy()
st.subheader("Localisation Open-Meteo par commune")
if open_loc_df.empty:
    st.info("Aucune station Open-Meteo visible sur ce filtre.")
else:
    open_loc_cols = [
        "station_display",
        "station_commune_name",
        "latitude",
        "longitude",
        "distance_to_lgv_km",
        metric_col,
        "date_obs_raw",
    ]
    open_loc_cols = [c for c in open_loc_cols if c in open_loc_df.columns]
    st.dataframe(
        open_loc_df[open_loc_cols].sort_values(["distance_to_lgv_km", "station_commune_name"], ascending=[True, True]),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Pluviometrie - Toutes les communes traversees par la LGV SEA")
if commune_pluvio_df.empty:
    st.info("Impossible de construire la vue commune LGV (catalogue ou donnees indisponibles).")
else:
    active_sources_label = ", ".join(selected_sources) if selected_sources else "Toutes sources"
    commune_pluvio_df = commune_pluvio_df.copy()
    commune_pluvio_df["match_mode_label"] = (
        commune_pluvio_df.get("match_mode", pd.Series("", index=commune_pluvio_df.index))
        .fillna("")
        .astype(str)
        .map(lambda v: COMMUNE_MATCH_MODE_LABELS.get(str(v), str(v) or "Inconnu"))
    )
    commune_pluvio_df["provider_source_label"] = (
        commune_pluvio_df.get("provider_source", pd.Series("", index=commune_pluvio_df.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "Sans source")
    )
    metric_series = pd.to_numeric(commune_pluvio_df.get(metric_col), errors="coerce")
    provider_dist_series = pd.to_numeric(commune_pluvio_df.get("provider_station_dist_km"), errors="coerce")
    n_total_communes = int(len(commune_pluvio_df))
    n_with_data = int(metric_series.notna().sum())
    coverage_pct = round(100.0 * n_with_data / max(1, n_total_communes), 1)
    missing_df = commune_pluvio_df[metric_series.isna()].copy()
    covered_df = commune_pluvio_df[metric_series.notna()].copy()
    max_metric_val = float(metric_series.max()) if metric_series.notna().any() else np.nan
    median_provider_dist = float(provider_dist_series.dropna().median()) if provider_dist_series.notna().any() else np.nan

    dominant_source = "N/A"
    dominant_source_counts = covered_df["provider_source_label"].value_counts() if not covered_df.empty else pd.Series(dtype=int)
    if not dominant_source_counts.empty:
        dominant_source = f"{dominant_source_counts.index[0]} ({int(dominant_source_counts.iloc[0])})"

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Couverture LGV", f"{n_with_data}/{n_total_communes}", delta=f"{coverage_pct:.1f}%")
    s2.metric("Communes sans mesure", int(len(missing_df)))
    s3.metric(f"Max {metric_label}", f"{float(max_metric_val):.1f} mm" if pd.notna(max_metric_val) else "N/A")
    s4.metric("Source dominante", dominant_source)
    s5.metric("Distance ref mediane", f"{float(median_provider_dist):.1f} km" if pd.notna(median_provider_dist) else "N/A")

    st.caption(f"Sources actives: {active_sources_label}.")

    summary_left, summary_right = st.columns([1.15, 0.85])
    with summary_left:
        source_summary = (
            commune_pluvio_df.groupby("provider_source_label", dropna=False)
            .agg(
                communes=("commune_code", "nunique"),
                communes_couvertes=(metric_col, lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
                pluie_moy_mm=(metric_col, lambda s: round(float(pd.to_numeric(s, errors="coerce").mean()), 1) if pd.to_numeric(s, errors="coerce").notna().any() else np.nan),
            )
            .reset_index()
            .sort_values(["communes_couvertes", "communes"], ascending=[False, False])
        )
        source_chart = px.bar(
            source_summary,
            x="communes_couvertes",
            y="provider_source_label",
            orientation="h",
            color="provider_source_label",
            labels={
                "communes_couvertes": "Communes avec mesure",
                "provider_source_label": "Source",
            },
            title="Couverture par source active",
            hover_data=["communes", "pluie_moy_mm"],
        )
        source_chart.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
        st.plotly_chart(source_chart, use_container_width=True)

    with summary_right:
        match_summary = (
            commune_pluvio_df.groupby("match_mode_label", dropna=False)
            .agg(communes=("commune_code", "nunique"))
            .reset_index()
            .sort_values("communes", ascending=False)
        )
        match_chart = px.bar(
            match_summary,
            x="communes",
            y="match_mode_label",
            orientation="h",
            color="match_mode_label",
            labels={
                "communes": "Communes",
                "match_mode_label": "Mode d'appairage",
            },
            title="Qualite d'appairage commune/station",
        )
        match_chart.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
        st.plotly_chart(match_chart, use_container_width=True)

    if not missing_df.empty:
        missing_cols = [c for c in ["commune_code", "commune_name", "match_mode_label"] if c in missing_df.columns]
        with st.expander(f"Communes sans mesure sur le filtre courant ({int(len(missing_df))})", expanded=False):
            st.dataframe(
                missing_df[missing_cols].sort_values(["commune_name", "commune_code"], ascending=[True, True]),
                use_container_width=True,
                hide_index=True,
            )

    review_queue = commune_pluvio_df[
        metric_series.isna()
        | (
            commune_pluvio_df.get("match_mode", pd.Series("", index=commune_pluvio_df.index))
            .fillna("")
            .astype(str)
            .eq("nearest_station")
        )
        | (provider_dist_series.fillna(0.0) > 10.0)
    ].copy()
    if not review_queue.empty:
        review_cols = [
            "commune_code",
            "commune_name",
            "provider_source_label",
            "provider_station",
            "provider_station_dist_km",
            "match_mode_label",
            metric_col,
            "date_obs_raw",
        ]
        review_cols = [c for c in review_cols if c in review_queue.columns]
        with st.expander("Communes a controler en priorite", expanded=False):
            st.dataframe(
                review_queue[review_cols].sort_values(
                    ["provider_station_dist_km", "commune_name"],
                    ascending=[False, True],
                    na_position="last",
                ),
                use_container_width=True,
                hide_index=True,
            )

    rain_cols_order = [metric_col] + [
        c for c in ["rain_24h_mm", "rain_7d_mm", "rain_month_mm"]
        if c != metric_col
    ]
    commune_view_cols = [
        "commune_code",
        "commune_name",
        *[c for c in rain_cols_order if c in commune_pluvio_df.columns],
        "rain_30d_source_label",
        "provider_source",
        "provider_station",
        "provider_station_dist_km",
        "match_mode_label",
        "date_obs_raw",
    ]
    commune_view_cols = [c for c in commune_view_cols if c in commune_pluvio_df.columns]
    commune_view = commune_pluvio_df[commune_view_cols].copy()
    if metric_col in commune_view.columns:
        commune_view = commune_view.sort_values([metric_col, "commune_name"], ascending=[False, True], na_position="last")
    commune_export_name = f"lgv_communes_pluvio_{metric_col}.csv"
    st.download_button(
        "Exporter le tableau communes LGV (CSV)",
        data=commune_view.to_csv(index=False).encode("utf-8-sig"),
        file_name=commune_export_name,
        mime="text/csv",
        use_container_width=False,
    )
    st.dataframe(commune_view, use_container_width=True, hide_index=True)

st.subheader(f"Top {int(top_n)} stations - {metric_label}")
top_df = filtered_stations.head(int(top_n)).copy()
top_df["station_label"] = top_df.get("station_display", top_df["station_id"].astype(str))
top_bar = px.bar(
    top_df.sort_values(metric_col, ascending=True),
    x=metric_col,
    y="station_label",
    color="source",
    orientation="h",
    title=f"Top stations - {metric_label}",
    labels={metric_col: f"Pluie {metric_label} (mm)", "station_label": "Station"},
    hover_data=["station_id", "station_commune_name", "distance_to_lgv_km", "date_obs_raw"],
)
top_bar.update_layout(height=460, margin=dict(l=10, r=10, t=45, b=10))
st.plotly_chart(top_bar, use_container_width=True)

st.subheader("Comparaison inter-stations de proximite (mode pro)")
pro_view = filtered_stations.copy()
pro_view["metric_station_mm"] = pd.to_numeric(pro_view.get(metric_col), errors="coerce")
pro_view["metric_mediane_voisins_mm"] = pd.to_numeric(pro_view.get("near_median_metric_mm"), errors="coerce")
pro_view["ecart_voisins_mm"] = pd.to_numeric(pro_view.get("near_delta_metric_mm"), errors="coerce")
pro_view["ecart_voisins_pct"] = pd.to_numeric(pro_view.get("near_delta_metric_pct"), errors="coerce")
pro_view["fiabilite_100"] = pd.to_numeric(pro_view.get("reliability_score"), errors="coerce").fillna(0.0)
pro_view["nb_voisins"] = pd.to_numeric(pro_view.get("near_station_count"), errors="coerce").fillna(0).astype(int)
pro_view["incoherent"] = pro_view.get("incoherence_flag", pd.Series(False, index=pro_view.index)).fillna(False).astype(bool)

scatter_df = pro_view.dropna(subset=["metric_station_mm", "metric_mediane_voisins_mm"]).copy()
if scatter_df.empty:
    st.info("Comparaison proximite indisponible sur ce filtre.")
else:
    scatter = px.scatter(
        scatter_df,
        x="metric_mediane_voisins_mm",
        y="metric_station_mm",
        color="reliability_class",
        symbol="incoherent",
        hover_data=[
            "station_display",
            "source",
            "distance_to_lgv_km",
            "ecart_voisins_mm",
            "ecart_voisins_pct",
            "nb_voisins",
            "fiabilite_100",
            "reliability_reason",
        ],
        labels={
            "metric_mediane_voisins_mm": f"Mediane voisins proches - {metric_label} (mm)",
            "metric_station_mm": f"Station - {metric_label} (mm)",
        },
        title="Coherence station vs voisins proches",
    )
    scatter.update_layout(height=430, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(scatter, use_container_width=True)

worst_df = pro_view.sort_values(["incoherent", "ecart_voisins_pct", "fiabilite_100"], ascending=[False, False, True], na_position="last").head(30).copy()
if not worst_df.empty:
    worst_df["station_label"] = worst_df.get("station_display", worst_df["station_id"].astype(str))
    worst_chart = px.bar(
        worst_df.sort_values("ecart_voisins_pct", ascending=True),
        x="ecart_voisins_pct",
        y="station_label",
        color="reliability_class",
        orientation="h",
        hover_data=["source", "ecart_voisins_mm", "nb_voisins", "fiabilite_100", "reliability_reason"],
        labels={"ecart_voisins_pct": f"Ecart relatif vs mediane voisins ({metric_label}, %)", "station_label": "Station"},
        title="Stations les plus en ecart avec leurs voisines",
    )
    worst_chart.update_layout(height=480, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(worst_chart, use_container_width=True)

st.caption(f"Comparatif multi-sources actif. Mode courant: {source_mode}.")
if source_mode == SOURCE_MODE_MIX:
    pair_df = _build_openmeteo_vs_infoclimat_pairs(
        stations_df=filtered_stations,
        metric_col=metric_col,
        max_pair_distance_km=15.0,
    )
    st.markdown("**Fiabilisation Open-Meteo vs InfoClimat (stations proches)**")
    if pair_df.empty:
        st.info("Comparatif Open-Meteo vs InfoClimat indisponible sur ce filtre.")
    else:
        pair_view_cols = [
            "infoclimat_station",
            "open_meteo_station",
            "pair_distance_km",
            "infoclimat_mm",
            "open_meteo_mm",
            "delta_open_minus_info_mm",
            "delta_abs_pct",
        ]
        pair_view_cols = [c for c in pair_view_cols if c in pair_df.columns]
        st.dataframe(pair_df[pair_view_cols].head(30), use_container_width=True, hide_index=True)
        delta_pct = pd.to_numeric(pair_df.get("delta_abs_pct"), errors="coerce")
        if delta_pct.notna().any():
            st.caption(
                "Ecart median Open-Meteo vs InfoClimat sur paires proches: "
                + f"{float(delta_pct.median()):.1f}% (distance max paire 15 km)."
            )

neighbor_df = _nearest_neighbors_for_station(
    stations_df=filtered_stations,
    station_id=str(history_station_id),
    metric_col=metric_col,
    compare_radius_km=float(compare_radius_km),
)
st.markdown(f"**Voisins de comparaison pour la station {selected_station.get('station_display')}**")
if neighbor_df.empty:
    st.info("Aucune station voisine dans le rayon de comparaison.")
else:
    ncols = [
        "station_display",
        "station_id",
        "station_commune_name",
        "source",
        "distance_station_ref_km",
        "distance_to_lgv_km",
        metric_col,
        "delta_metric_mm",
        "reliability_score",
        "reliability_class",
        "near_delta_metric_pct",
    ]
    ncols = [c for c in ncols if c in neighbor_df.columns]
    st.dataframe(neighbor_df[ncols].head(40), use_container_width=True, hide_index=True)

st.subheader("Table stations filtrees")
station_cols = [
    "station_display",
    "station_name",
    "station_id",
    "source",
    "station_commune_name",
    "priority_name",
    "priority_match_km",
    "distance_to_lgv_km",
    "latitude",
    "longitude",
    "rain_24h_mm",
    "rain_7d_mm",
    "rain_30d_mm",
    "rain_30d_source_label",
    "rain_month_mm",
    "near_station_count",
    "near_median_metric_mm",
    "near_delta_metric_mm",
    "near_delta_metric_pct",
    "source_note",
    "freshness_note",
    "coherence_note",
    "reliability_source_component",
    "reliability_freshness_component",
    "reliability_coherence_component",
    "reliability_score",
    "reliability_class",
    "reliability_reason",
    "history_backfill_source",
    "history_backfill_station",
    "history_backfill_obs_date",
    "date_obs_raw",
]
station_cols = [c for c in station_cols if c in filtered_stations.columns]
st.dataframe(filtered_stations[station_cols], use_container_width=True, hide_index=True)

st.subheader(f"Historique station: {selected_station.get('station_display')}")
if not history_sources:
    st.info("Selectionne au moins une source historique.")
else:
    ref_lat, ref_lon, ref_label, ref_notice = _resolve_history_reference_point(
        station_row=selected_station,
        reference_mode=history_reference_mode,
    )
    if ref_lat is None or ref_lon is None:
        st.warning("Coordonnees invalides pour charger l'historique.")
    else:
        st.caption(
            f"Reference historique: {ref_label} | station_id={str(history_station_id)} | "
            + f"lat={float(ref_lat):.5f}, lon={float(ref_lon):.5f} | timezone=UTC"
        )
        if ref_notice:
            st.caption(ref_notice)
        hist_df, hist_notices = _load_history_multi_source(
            lat=float(ref_lat),
            lon=float(ref_lon),
            station_id=str(history_station_id),
            start_day=history_start,
            end_day=history_end,
            selected_sources=history_sources,
        )
        if hist_notices:
            st.caption(" | ".join(hist_notices[:4]))
        if hist_df.empty:
            st.warning("Historique indisponible sur cette periode/source.")
        else:
            hist_df = hist_df.copy()
            hist_df["date"] = pd.to_datetime(hist_df["date"], utc=True, errors="coerce").dt.normalize()
            hist_df["precip_mm"] = pd.to_numeric(hist_df["precip_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
            hist_df = hist_df.dropna(subset=["date"])
            hist_df = (
                hist_df.groupby(["source", "date"], as_index=False)["precip_mm"]
                .max()
                .sort_values(["source", "date"])
                .reset_index(drop=True)
            )

            roll_df = hist_df.copy()
            roll_df["rolling_7d_mm"] = roll_df.groupby("source")["precip_mm"].transform(
                lambda s: s.rolling(window=7, min_periods=1).sum()
            )

            expected_dates = pd.date_range(start=history_start, end=history_end, freq="D", tz="UTC")
            expected_total_days = int(len(expected_dates))
            expected_days_map: Dict[pd.Timestamp, int] = {}
            if expected_total_days > 0:
                expected_df = pd.DataFrame({"date": expected_dates})
                expected_df["month_start"] = pd.to_datetime(
                    expected_df["date"].dt.strftime("%Y-%m-01"),
                    utc=True,
                    errors="coerce",
                )
                expected_days_map = expected_df.groupby("month_start").size().to_dict()

            monthly = hist_df.copy()
            monthly["month_start"] = pd.to_datetime(
                monthly["date"].dt.strftime("%Y-%m-01"),
                utc=True,
                errors="coerce",
            )
            monthly = (
                monthly.groupby(["source", "month_start"], as_index=False)
                .agg(
                    monthly_mm=("precip_mm", "sum"),
                    observed_days=("date", "nunique"),
                )
            )
            monthly["expected_days"] = monthly["month_start"].map(expected_days_map).fillna(0).astype(int)
            monthly["coverage_pct"] = np.where(
                monthly["expected_days"] > 0,
                100.0 * pd.to_numeric(monthly["observed_days"], errors="coerce").fillna(0.0)
                / pd.to_numeric(monthly["expected_days"], errors="coerce").replace(0, np.nan),
                np.nan,
            )
            monthly["ym"] = monthly["month_start"].dt.strftime("%Y-%m")

            daily_chart = px.line(
                hist_df,
                x="date",
                y="precip_mm",
                color="source",
                color_discrete_map=HISTORY_SOURCE_COLORS,
                labels={"date": "Date", "precip_mm": "Pluie journaliere (mm)", "source": "Source historique"},
                title="Historique journalier multi-sources",
            )
            daily_chart.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(daily_chart, use_container_width=True)

            roll_chart = px.line(
                roll_df,
                x="date",
                y="rolling_7d_mm",
                color="source",
                color_discrete_map=HISTORY_SOURCE_COLORS,
                labels={"date": "Date", "rolling_7d_mm": "Cumul glissant 7 jours (mm)", "source": "Source historique"},
                title="Cumul glissant 7 jours",
            )
            roll_chart.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(roll_chart, use_container_width=True)

            monthly_chart = px.bar(
                monthly,
                x="month_start",
                y="monthly_mm",
                color="source",
                barmode="group",
                color_discrete_map=HISTORY_SOURCE_COLORS,
                labels={"month_start": "Mois", "monthly_mm": "Cumul mensuel (mm)", "source": "Source historique"},
                title="Cumuls mensuels par source",
            )
            monthly_chart.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            monthly_chart.update_xaxes(dtick="M1", tickformat="%Y-%m", ticklabelmode="period")
            st.plotly_chart(monthly_chart, use_container_width=True)

            summary = (
                hist_df.groupby("source", as_index=False)
                .agg(
                    jours=("date", "nunique"),
                    total_mm=("precip_mm", "sum"),
                    moyenne_mm_j=("precip_mm", "mean"),
                    max_journalier_mm=("precip_mm", "max"),
                )
                .sort_values("total_mm", ascending=False)
            )
            summary["jours_attendus"] = expected_total_days
            summary["couverture_pct"] = np.where(
                pd.to_numeric(summary["jours_attendus"], errors="coerce") > 0,
                100.0
                * pd.to_numeric(summary["jours"], errors="coerce").fillna(0.0)
                / pd.to_numeric(summary["jours_attendus"], errors="coerce").replace(0, np.nan),
                np.nan,
            ).round(1)
            st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Metadata")
meta_method_tab, meta_source_tab, meta_fiability_tab = st.tabs(
    ["Methodologie", "Sources & captation", "Fiabilite operationnelle"]
)

with meta_method_tab:
    st.markdown(
        """
        **Lecture metier orientee gestionnaire de ligne ferroviaire**

        1. La carte sert d'abord a prioriser les rondes, visites plateforme, verification drainage et points sensibles GC.
        2. Le point cartographique represente une station reelle ou un point geolocalise proxy rattache a une commune LGV.
        3. Le classement visuel est base sur la **pluie mensuelle**, plus pertinente pour suivre la saturation des sols,
           les remblais/deblais, les talus et la persistance d'un contexte humide.
        4. Les indicateurs courts (12h/24h et prevision proche) servent a reperer un episode brutal; les cumuls `7j/30j/mois`
           servent a lire un contexte de saturation progressive critique pour talus, drainage, plateformes et acces maintenance.
        5. Les cumuls longs Open-Meteo sont maintenant calcules sur **archive journaliere par commune LGV**, selon la meme logique
           que l'app monitoring LGV, pour eviter les cumuls glissants incoherents ou uniformes.
        6. Une source sans station proche est acceptee si elle ameliore la couverture corridor, mais sa lecture doit etre
           nuancee par l'indicateur de fiabilite, la fraicheur et le type de captation.
        """
    )
    if data_build_notices:
        notices_df = pd.DataFrame({"etape_pipeline": [str(n) for n in data_build_notices if str(n).strip()]})
        st.dataframe(notices_df, use_container_width=True, hide_index=True)

with meta_source_tab:
    st.markdown(
        """
        **Types de donnees meteo utilises**

        - `Meteo-France Portail API` : observations officielles SYNOP, robustes pour le reporting et l'arbitrage exploitation.
        - `InfoClimat / SYNOP` : stations au sol proches de la LGV, utiles pour confirmer localement une alerte.
        - `Open-Meteo MeteoFrance` : grille modele geolocalisee utile pour le court terme et la couverture integrale des 111 communes.
        - `Open-Meteo archive (communes LGV)` : precipitation journaliere archivee sur point communal, utilisee pour fiabiliser les cumuls longs.

        **Methodologie appliquee a date**

        - `12h / 24h / prevision courte` : lecture court terme prioritaire pour l'exploitation, issue du flux live quand disponible.
        - `7 jours / 30 jours / mois courant` : cumul fiabilise sur archive journaliere Open-Meteo, aligne sur l'app `lgv-sea-monitoring-meteo-gc`.
        - `Projection communale` : chaque point est rattache a une commune LGV pour raisonner comme un gestionnaire de ligne, secteur par secteur.
        - `Fiabilite` : la decision ne repose pas sur la source seule, mais sur la source, la fraicheur et la coherence avec les voisins.
        """
    )
    if source_meta_df.empty:
        st.info("Metadonnees sources indisponibles sur ce filtre.")
    else:
        source_meta_view_cols = [
            "source",
            "type_data",
            "captation",
            "maj_typique",
            "nb_stations",
            "distance_mediane_lgv_km",
            "age_median_h",
            "fiabilite_mediane_100",
            "methodologie",
            "usage_ferroviaire",
            "limites",
        ]
        source_meta_view_cols = [c for c in source_meta_view_cols if c in source_meta_df.columns]
        st.dataframe(source_meta_df[source_meta_view_cols], use_container_width=True, hide_index=True)

with meta_fiability_tab:
    st.markdown(
        """
        **Comment la fiabilite est calculee**

        - `42% note_source` : confiance initiale selon la nature de la source.
        - `23% fraicheur_obs` : penalite si la derniere observation devient ancienne.
        - `35% coherence_locale` : comparaison avec les stations voisines dans le rayon choisi.

        **Classes de lecture**

        - `FIABLE` : donnee exploitable directement pour prioriser la surveillance.
        - `SURVEILLER` : donnee utile mais a croiser avec le contexte local.
        - `A_VERIFIER` : donnee a confirmer avant decision terrain ou restriction d'exploitation.
        """
    )
    st.caption(
        "Sur la carte, le contour du point represente la fiabilite; dans les popups, "
        "le gestionnaire voit aussi l'age de l'observation, le nombre de voisins compares et l'ecart a la mediane locale."
    )
    fiability_summary = pd.DataFrame(
        [
            {
                "classe": "FIABLE",
                "score_min": 85,
                "usage_recommande": "Pilotage direct des rondes et priorisation plateforme / talus.",
            },
            {
                "classe": "SURVEILLER",
                "score_min": 65,
                "usage_recommande": "Confirmer avec une seconde source, l'historique et le contexte terrain.",
            },
            {
                "classe": "A_VERIFIER",
                "score_min": 0,
                "usage_recommande": "Ne pas engager seul une decision d'exploitation sans verification complementaire.",
            },
        ]
    )
    st.dataframe(fiability_summary, use_container_width=True, hide_index=True)
