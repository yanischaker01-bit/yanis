from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

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

HISTORY_MIN_DATE = date(2026, 1, 1)
RAIN_METRICS = {
    "24h": "rain_24h_mm",
    "7 jours": "rain_7d_mm",
    "30 jours": "rain_30d_mm",
    "Mois courant": "rain_month_mm",
}
HISTORY_SOURCES = [
    "Open-Meteo MeteoFrance",
]
HISTORY_SOURCE_COLORS = {
    "Open-Meteo MeteoFrance": "#1d4ed8",
}
SOURCE_RELIABILITY_HINTS = {
    "SYNOP": 95.0,
    "INFOCLIMAT": 95.0,
    "METEOFRANCE": 93.0,
    "OPEN_METEO": 84.0,
}
LOCAL_INFOCLIMAT_RADIUS_KM = 120.0
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


def _http_get_with_retry(url: str, params: Dict[str, object], timeout: int = 30, max_attempts: int = 2) -> requests.Response:
    last_exc: Exception | None = None
    session = requests.Session()
    session.trust_env = False
    for _ in range(max(1, int(max_attempts))):
        try:
            return session.get(url, params=params, timeout=int(timeout))
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
        return df
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
    synop_path = _find_latest_file(["data/synop_all_stations_*.csv"])
    if synop_path is None:
        return pd.DataFrame(), "InfoClimat/SYNOP local: aucun fichier synop_all_stations_*.csv disponible."
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
                        "selection_mode": "open_meteo_at_infoclimat_station",
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
    notice = f"Open-Meteo reference: {len(out)} points calcules sur stations InfoClimat ({model_name})."
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
            refresh = "Horaire (observation)"
            method = "Mesures station + calcul cumuls 24h/7j/30j puis controle de coherence locale."
            limits = "Densite de stations variable selon secteurs LGV."
        elif _is_open_meteo_source(source):
            data_type = "Modele numerique (Open-Meteo)"
            refresh = "Horaire (reanalyse + prevision courte)"
            method = "Interpolation modele sur coordonnees station puis cumuls glissants."
            limits = "Ce n'est pas une mesure directe de pluviometre."
        else:
            data_type = "Source diverse"
            refresh = "Selon source"
            method = "Harmonisation interne puis controle de coherence."
            limits = "Metadonnees limitees."
        rows.append(
            {
                "source": source,
                "type_data": data_type,
                "maj_typique": refresh,
                "nb_stations": int(len(sub)),
                "distance_mediane_lgv_km": round(float(dist.median()), 2) if dist.notna().any() else np.nan,
                "age_median_h": round(float(age_h.median()), 1) if age_h.notna().any() else np.nan,
                "fiabilite_mediane_100": round(float(rel.median()), 1) if rel.notna().any() else np.nan,
                "methodologie": method,
                "limites": limits,
            }
        )
    return pd.DataFrame(rows).sort_values(["fiabilite_mediane_100", "nb_stations"], ascending=[False, False], na_position="last")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = np.radians(float(lat1))
    p2 = np.radians(float(lat2))
    dlat = np.radians(float(lat2) - float(lat1))
    dlon = np.radians(float(lon2) - float(lon1))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


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
    if snapshot_ts is not None and not pd.isna(snapshot_ts):
        obs_age_h = (snapshot_ts - obs_ts).dt.total_seconds() / 3600.0
    else:
        obs_age_h = pd.Series([np.nan] * len(work), index=work.index, dtype=float)
    work["obs_age_h"] = pd.to_numeric(obs_age_h, errors="coerce").fillna(999.0).clip(lower=0.0)
    work["source_note"] = work.get("source", pd.Series("", index=work.index)).map(_source_reliability_note)
    work["freshness_note"] = work["obs_age_h"].map(_freshness_note)

    metrics_for_consistency = [c for c in ["rain_24h_mm", "rain_7d_mm", "rain_30d_mm", "rain_month_mm"] if c in work.columns]
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
    work["reliability_class"] = work["reliability_score"].map(_reliability_class)
    work["reliability_reason"] = np.where(
        work["near_station_count"] < int(min_neighbors),
        "Peu de stations voisines pour confirmer la coherence",
        np.where(
            work["near_delta_metric_pct"].fillna(0.0) >= 60.0,
            "Ecart eleve vs mediane des stations proches",
            "Coherence locale satisfaisante",
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
            return pd.DataFrame(), f"{source_label}: HTTP {resp.status_code}"
        payload = resp.json()
        daily = payload.get("daily") if isinstance(payload, dict) else {}
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


def _load_history_multi_source(
    lat: float,
    lon: float,
    start_day: date,
    end_day: date,
    selected_sources: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    start_iso = str(start_day.isoformat())
    end_iso = str(end_day.isoformat())
    blocks: List[pd.DataFrame] = []
    notices: List[str] = []

    if "Open-Meteo MeteoFrance" in selected_sources:
        df, note = _fetch_open_meteo_history(
            lat=lat,
            lon=lon,
            start_day=start_iso,
            end_day=end_iso,
            source_label="Open-Meteo MeteoFrance",
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

    metric_vals = pd.to_numeric(stations_df.get(rain_col), errors="coerce")
    vmax = float(metric_vals.max()) if metric_vals.notna().any() else 1.0
    vmax = max(vmax, 1.0)
    for _, row in stations_df.iterrows():
        lat = pd.to_numeric(row.get("latitude"), errors="coerce")
        lon = pd.to_numeric(row.get("longitude"), errors="coerce")
        val = pd.to_numeric(row.get(rain_col), errors="coerce")
        if pd.isna(lat) or pd.isna(lon) or pd.isna(val):
            continue
        ratio = max(0.0, min(1.0, float(val) / vmax))
        rclass = str(row.get("reliability_class") or "")
        if rclass == "A_VERIFIER":
            color = "#7f1d1d"
        elif rclass == "SURVEILLER":
            color = "#b45309"
        else:
            color = "#16a34a" if ratio < 0.33 else ("#ea580c" if ratio < 0.66 else "#dc2626")
        rel_score = pd.to_numeric(row.get("reliability_score"), errors="coerce")
        near_delta_pct = pd.to_numeric(row.get("near_delta_metric_pct"), errors="coerce")
        popup = (
            f"<b>Station:</b> {row.get('station_display')}<br>"
            f"<b>Commune:</b> {row.get('station_commune_name')}<br>"
            f"<b>Source:</b> {row.get('source')}<br>"
            f"<b>Distance LGV:</b> {row.get('distance_to_lgv_km')} km<br>"
            f"<b>{rain_col}:</b> {float(val):.1f} mm<br>"
            f"<b>Score fiabilite:</b> {0.0 if pd.isna(rel_score) else float(rel_score):.1f}/100 ({row.get('reliability_class', 'N/A')})<br>"
            f"<b>Ecart vs stations proches:</b> {0.0 if pd.isna(near_delta_pct) else float(near_delta_pct):.1f}%<br>"
            f"<b>Obs:</b> {row.get('date_obs_raw')}"
        )
        folium.CircleMarker(
            [float(lat), float(lon)],
            radius=5 + 6 * ratio,
            color=color,
            fill=True,
            fill_opacity=0.85,
            weight=1,
            popup=folium.Popup(popup, max_width=380),
            tooltip=f"{row.get('station_display')} | {float(val):.1f} mm",
        ).add_to(m)

    return m


st.set_page_config(page_title="LGV SEA Pluvio Stations Pro", page_icon=":umbrella:", layout="wide")
st.title("LGV SEA - Pluviometrie Stations Pro")
st.caption(
    "Version pro: Open-Meteo MeteoFrance uniquement, fiabilisation des mesures, suivi stations et carte operative."
)
st.caption("Rendu graphique actif: Plotly (compatible Streamlit Cloud).")

try:
    snapshot, snapshot_source = _load_snapshot_payload()
except Exception as exc:
    st.error(str(exc))
    st.stop()

lgv_lines = _extract_lgv_lines(snapshot)
snapshot_ts = pd.to_datetime(snapshot.get("timestamp_utc"), utc=True, errors="coerce")
raw_snapshot_weather = snapshot.get("weather", []) if isinstance(snapshot, dict) else []
raw_snapshot_weather = [row for row in raw_snapshot_weather if isinstance(row, dict)]

data_build_notices: List[str] = []
infoclimat_local_df, infoclimat_local_notice = _load_infoclimat_synop_local(max_distance_km=LOCAL_INFOCLIMAT_RADIUS_KM)
if infoclimat_local_notice:
    data_build_notices.append(infoclimat_local_notice)

raw_weather_rows = list(raw_snapshot_weather)
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

open_meteo_ref_df = pd.DataFrame()
if not infoclimat_local_df.empty:
    ref_points_key = _build_open_meteo_reference_key(infoclimat_local_df)
    open_meteo_ref_df, open_meteo_ref_notice = _fetch_open_meteo_reference_points(ref_points_key)
    if open_meteo_ref_notice:
        data_build_notices.append(open_meteo_ref_notice)
    if not open_meteo_ref_df.empty:
        raw_weather_rows = [row for row in raw_weather_rows if not _is_open_meteo_source(row.get("source"))]
        raw_weather_rows.extend(open_meteo_ref_df.to_dict(orient="records"))

# Mode force: Open-Meteo MeteoFrance uniquement.
open_meteo_only_rows = [
    row
    for row in raw_weather_rows
    if _is_open_meteo_source(row.get("source"))
    and ("meteofrance" in str(row.get("meteo_model") or "").strip().lower())
]
weather_df = _safe_weather_df({"weather": open_meteo_only_rows})
if weather_df.empty:
    st.warning(
        "Aucune donnee Open-Meteo MeteoFrance disponible pour le moment."
    )
    st.stop()

with st.sidebar:
    st.subheader("Filtres stations")
    if st.button("Rafraichir", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    metric_label = st.selectbox("Indicateur pluvio", list(RAIN_METRICS.keys()), index=1)
    metric_col = RAIN_METRICS[metric_label]
    map_style = st.selectbox("Fond de carte", list(MAP_TILE_STYLES.keys()), index=0)

    max_distance_km = st.slider("Distance max a la LGV (km)", min_value=1.0, max_value=80.0, value=25.0, step=0.5)
    compare_radius_km = st.slider(
        "Rayon comparaison stations proches (km)",
        min_value=2.0,
        max_value=40.0,
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
    selected_sources = sorted(weather_df.get("source", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    st.caption("Source active: Open-Meteo MeteoFrance uniquement")

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
    st.warning("Aucune station sur ce filtre.")
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
filtered_stations["incoherence_flag"] = (
    pd.to_numeric(filtered_stations.get("near_delta_metric_pct"), errors="coerce").fillna(0.0) >= float(incoherence_alert_pct)
)
filtered_stations = filtered_stations.sort_values([metric_col, "distance_to_lgv_km"], ascending=[False, True], na_position="last")

top_n_max = max(1, int(len(filtered_stations)))
with st.sidebar:
    top_n = st.slider("Top stations (graphe)", min_value=1, max_value=top_n_max, value=min(25, top_n_max), step=1)
    st.markdown("---")
    st.subheader("Historique (depuis 2026)")
    history_station_default = str(filtered_stations.iloc[0]["station_display"])
    history_station_options = sorted(filtered_stations["station_display"].astype(str).unique().tolist())
    history_station_display = st.selectbox(
        "Station historique",
        options=history_station_options,
        index=history_station_options.index(history_station_default),
    )
    history_sources = list(HISTORY_SOURCES)
    st.caption("Historique: Open-Meteo MeteoFrance")
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

reliability_series = pd.to_numeric(filtered_stations.get("reliability_score"), errors="coerce").fillna(0.0)
reliability_class = filtered_stations.get("reliability_class", pd.Series("A_VERIFIER", index=filtered_stations.index)).astype(str)
incoherence_count = int(filtered_stations.get("incoherence_flag", pd.Series(False, index=filtered_stations.index)).fillna(False).sum())

k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
k1.metric("Stations filtrees", int(len(filtered_stations)))
k2.metric("Communes", int(filtered_stations["station_commune_name"].astype(str).nunique()))
k3.metric(f"Max {metric_label}", f"{float(filtered_stations[metric_col].max()):.1f} mm")
k4.metric(f"Moyenne {metric_label}", f"{float(filtered_stations[metric_col].mean()):.1f} mm")
k5.metric("Distance max filtre", f"{float(max_distance_km):.1f} km")
k6.metric("Fiabilite moyenne", f"{float(reliability_series.mean()):.1f}/100")
k7.metric("Stations FIABLE", int((reliability_class == "FIABLE").sum()))
k8.metric("Incoherences proximite", incoherence_count)

if snapshot_ts is not None and not pd.isna(snapshot_ts):
    st.caption(f"Snapshot: {snapshot_source} | timestamp_utc={snapshot_ts.isoformat()}")
else:
    st.caption(f"Snapshot: {snapshot_source} | timestamp inconnu")
if data_build_notices:
    st.caption(" | ".join([n for n in data_build_notices if n][:3]))

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

st.subheader("Carte stations pluvio autour de la LGV SEA")
map_obj = _build_map(lgv_lines, filtered_stations, metric_col, map_style=map_style)
st_folium(map_obj, height=640, use_container_width=True)
st.caption(f"Fond de carte actif: {map_style}")

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

st.caption("Comparatif Open-Meteo vs InfoClimat desactive: application verrouillee sur Open-Meteo MeteoFrance.")

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
    "distance_to_lgv_km",
    "latitude",
    "longitude",
    "rain_24h_mm",
    "rain_7d_mm",
    "rain_30d_mm",
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
    "date_obs_raw",
]
station_cols = [c for c in station_cols if c in filtered_stations.columns]
st.dataframe(filtered_stations[station_cols], use_container_width=True, hide_index=True)

st.subheader(f"Historique station: {selected_station.get('station_display')}")
if not history_sources:
    st.info("Selectionne au moins une source historique.")
else:
    station_lat = pd.to_numeric(selected_station.get("latitude"), errors="coerce")
    station_lon = pd.to_numeric(selected_station.get("longitude"), errors="coerce")
    if pd.isna(station_lat) or pd.isna(station_lon):
        st.warning("Coordonnees station invalides pour charger l'historique.")
    else:
        hist_df, hist_notices = _load_history_multi_source(
            lat=float(station_lat),
            lon=float(station_lon),
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
            hist_df["date"] = pd.to_datetime(hist_df["date"], utc=True, errors="coerce")
            hist_df["precip_mm"] = pd.to_numeric(hist_df["precip_mm"], errors="coerce").fillna(0.0).clip(lower=0.0)
            hist_df = hist_df.dropna(subset=["date"]).sort_values(["source", "date"])

            roll_df = hist_df.copy()
            roll_df["rolling_7d_mm"] = roll_df.groupby("source")["precip_mm"].transform(
                lambda s: s.rolling(window=7, min_periods=1).sum()
            )

            monthly = hist_df.copy()
            monthly["ym"] = monthly["date"].dt.strftime("%Y-%m")
            monthly = (
                monthly.groupby(["source", "ym"], as_index=False)["precip_mm"]
                .sum()
                .rename(columns={"precip_mm": "monthly_mm"})
            )

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
                x="ym",
                y="monthly_mm",
                color="source",
                barmode="group",
                color_discrete_map=HISTORY_SOURCE_COLORS,
                labels={"ym": "Mois", "monthly_mm": "Cumul mensuel (mm)", "source": "Source historique"},
                title="Cumuls mensuels par source",
            )
            monthly_chart.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(monthly_chart, use_container_width=True)

            summary = (
                hist_df.groupby("source", as_index=False)
                .agg(
                    jours=("date", "count"),
                    total_mm=("precip_mm", "sum"),
                    moyenne_mm_j=("precip_mm", "mean"),
                    max_journalier_mm=("precip_mm", "max"),
                )
                .sort_values("total_mm", ascending=False)
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
