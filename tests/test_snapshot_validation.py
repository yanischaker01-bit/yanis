from __future__ import annotations

import io
import json
import math
import os
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

SNAPSHOT_URLS = [
    "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json",
    "https://yanischaker01-bit.github.io/yanis/dashboard/reports/streamlit_snapshot_latest.json",
]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

ALIASES = {
    "commune": "commune_name",
    "nom_commune": "commune_name",
    "libelle_commune": "commune_name",
    "lat": "latitude",
    "y": "latitude",
    "latitude_wgs84": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "long": "longitude",
    "x": "longitude",
    "longitude_wgs84": "longitude",
    "pk": "pk_km",
    "pk_gps": "pk_km",
    "pk_theorique": "pk_km",
    "pk_m": "pk_km",
}
REQUIRED = ["commune_name", "latitude", "longitude", "pk_km"]


def normalize_sector_dataframe(df):
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_") for c in out.columns]
    for src, dst in ALIASES.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    if "pk_km" not in out.columns and "pk_m" in out.columns:
        out["pk_km"] = pd.to_numeric(out["pk_m"], errors="coerce") / 1000.0
    for col in ("latitude", "longitude", "pk_km"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def extract_sector_records(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for root in ("sectors", "data", "snapshot"):
        value = payload.get(root)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for inner in ("sectors", "records", "items", "features"):
                val = value.get(inner)
                if isinstance(val, list):
                    return val
    return []


def snapshot_usable(payload):
    records = extract_sector_records(payload)
    if not records:
        return False, "aucun secteur", pd.DataFrame()
    frame = normalize_sector_dataframe(pd.DataFrame(records))
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        return False, f"colonnes manquantes : {', '.join(missing)}", frame
    if not all(col in frame.columns for col in ("latitude", "longitude", "pk_km")):
        return False, "colonnes manquantes : latitude, longitude, pk_km", frame
    localized = frame.dropna(subset=["latitude", "longitude", "pk_km"]).copy()
    if localized.empty:
        return False, "aucun secteur exploitable", frame
    return True, "ok", localized


@st.cache_data(ttl=900)
def _fetch_snapshot_raw() -> dict:
    errors = []
    for url in SNAPSHOT_URLS:
        try:
            response = requests.get(
                url,
                timeout=(10, 30),
                headers={
                    "Accept": "application/json",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                },
                params={"v": int(datetime.now(timezone.utc).timestamp() // 300)},
            )
            response.raise_for_status()
            payload = response.json()
            ok, reason, _ = snapshot_usable(payload)
            if ok:
                return payload
            errors.append(f"{url}: {reason}")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(" | ".join(errors))


def load_snapshot() -> dict:
    try:
        return _fetch_snapshot_raw()
    except Exception as exc:
        return {"_error": str(exc)}


# Original file continues below.
