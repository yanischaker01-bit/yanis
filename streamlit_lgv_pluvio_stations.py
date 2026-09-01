import io
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

SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# NASA FIRMS (Fire Information for Resource Management System) — détections satellite quasi temps réel
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{day_range}/{date}"
FIRMS_SOURCES  = ["VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"]  # résolution ~375 m
FIRMS_BBOX     = "-0.7,44.75,1.0,47.5"  # west,south,east,north — corridor LGV SEA + marge
FIRMS_RADIUS_KM = 0.5
FIRMS_MAX_DAY_RANGE = 10   # limite de l'API area (une seule requête par source)
FIRMS_MAX_LOOKBACK_DAYS = 60  # au-delà, le NRT n'est plus garanti disponible

# Vigilance météo officielle Météo-France, republiée en open data (sans clé) sur Opendatasoft —
# mêmes bulletins que vigilance.meteofrance.fr, source de référence utilisée par SNCF/préfectures.
MF_VIGILANCE_URL     = "https://public.opendatasoft.com/api/records/1.0/search/"
MF_VIGILANCE_DATASET = "weatherref-france-vigilance-meteo-departement"
MF_PHENOMENON_LABELS = {
    "pluie":              ("🌧️", "Pluie-inondation"),
    "orages":             ("⛈️", "Orages"),
    "canicule":           ("🌡️", "Canicule"),
    "vent":               ("💨", "Vent violent"),
    "neige / verglas":    ("❄️", "Neige-verglas"),
    "vagues submersion":  ("🌊", "Vagues-submersion"),
}
MF_COLOR_TO_LEVEL = {"vert": "VERT", "jaune": "JAUNE", "orange": "ORANGE", "rouge": "ROUGE"}

DEPS = {
    "37": {"nom": "Indre-et-Loire",   "lat": 47.38, "lon":  0.69},
    "86": {"nom": "Vienne",            "lat": 46.58, "lon":  0.34},
    "79": {"nom": "Deux-Sèvres",       "lat": 46.32, "lon": -0.46},
    "16": {"nom": "Charente",           "lat": 45.65, "lon":  0.16},
    "17": {"nom": "Charente-Maritime", "lat": 45.75, "lon": -0.63},
    "33": {"nom": "Gironde",            "lat": 44.84, "lon": -0.58},
}

ALERT_CFG = {
    "ORAGE":      ("⛈️",  "Orage"),
    "CANICULE":   ("🌡️",  "Canicule"),
    "INCENDIE":   ("🔥",  "Incendie"),
    "INONDATION": ("🌊",  "Inondation"),
    "VENT":       ("💨",  "Vent violent"),
    "VIGICRUE":   ("🏞️",  "Vigilance crue"),
    "FEU_FIRMS":  ("🔥",  "Détection incendie FIRMS"),
    "MF_VIGILANCE": ("🛡️", "Vigilance officielle Météo-France"),
}
LEVEL_COLOR = {"ROUGE":"#dc2626","ORANGE":"#ea580c","JAUNE":"#eab308","VERT":"#16a34a","INFO":"#3b82f6"}
LEVEL_RANK  = {"ROUGE":4,"ORANGE":3,"JAUNE":2,"VERT":1,"INFO":0}
LEVEL_LABEL = {"ROUGE":"Rouge","ORANGE":"Orange","JAUNE":"Jaune","VERT":"Vert","INFO":"Info"}
# st.badge only accepts this fixed palette — map our 5 severity levels onto it.
LEVEL_BADGE = {"ROUGE":"red","ORANGE":"orange","JAUNE":"yellow","VERT":"green","INFO":"gray"}


def rain_risk(max_mm: float):
    if max_mm >= 60: return "ROUGE",  "#dc2626", "🔴"
    if max_mm >= 30: return "ORANGE", "#ea580c", "🟠"
    if max_mm >= 10: return "JAUNE",  "#eab308", "🟡"
    return               "VERT",   "#16a34a", "🟢"

def rain_color_mm(mm: float) -> str:
    if mm >= 60: return "#dc2626"
    if mm >= 30: return "#ea580c"
    if mm >= 10: return "#3b82f6"
    return "#93c5fd"


@st.cache_data(ttl=900)
def _fetch_snapshot_raw() -> dict:
    r = requests.get(SNAPSHOT_URL, timeout=20)
    r.raise_for_status()
    return r.json()


def load_snapshot() -> dict:
    """Seules les réponses réussies sont mises en cache (via _fetch_snapshot_raw) :
    un échec réseau ponctuel n'immobilise pas l'appli 15 min, il est retenté au
    prochain rerun au lieu de rester figé jusqu'à expiration du TTL."""
    try:
        return _fetch_snapshot_raw()
    except Exception as e:
        return {"_error": str(e)}


@st.cache_data(ttl=1800)
def _fetch_mf_vigilance_raw(dep_codes: tuple) -> list:
    q = " OR ".join(f"domain_id:{d}" for d in dep_codes)
    r = requests.get(MF_VIGILANCE_URL, params={
        "dataset": MF_VIGILANCE_DATASET, "q": q, "rows": 100,
    }, timeout=15)
    r.raise_for_status()
    return r.json().get("records", [])


def load_meteofrance_vigilance() -> tuple[list, bool]:
    """Vigilance officielle Météo-France (aujourd'hui J / demain J1) par département,
    republiée en open data sans clé — mêmes bulletins que vigilance.meteofrance.fr.
    Retourne (alertes non-vertes, ok). ok=False seulement si l'API n'a pas pu être
    interrogée du tout (à ne pas confondre avec une vigilance verte légitime)."""
    try:
        records = _fetch_mf_vigilance_raw(tuple(sorted(DEPS.keys())))
    except Exception:
        return [], False

    alerts = []
    for rec in records:
        f = rec.get("fields", {})
        dep = f.get("domain_id")
        if dep not in DEPS:
            continue
        level = MF_COLOR_TO_LEVEL.get((f.get("color") or "").lower())
        if not level or level == "VERT":
            continue
        phen = f.get("phenomenon", "")
        icon, label = MF_PHENOMENON_LABELS.get(phen, ("", phen.capitalize() or "Phénomène"))
        jour = "aujourd'hui" if f.get("echeance") == "J" else "demain"
        alerts.append(dict(
            dep=dep, type="MF_VIGILANCE", level=level, phenomenon=phen, icon=icon,
            msg=f"Dép.{dep} ({DEPS[dep]['nom']}) — {label} : vigilance {level.lower()} ({jour})",
        ))

    alerts.sort(key=lambda a: (-LEVEL_RANK.get(a["level"], 0), a["dep"]))
    return alerts, True


@st.cache_data(ttl=1800)
def _fetch_dept_forecast_raw(dep: str) -> dict:
    info = DEPS[dep]
    r = requests.get(FORECAST_URL, params={
        "latitude": info["lat"], "longitude": info["lon"],
        "daily": ("precipitation_sum,temperature_2m_max,"
                  "weathercode,wind_speed_10m_max"),
        "forecast_days": 7, "timezone": "Europe/Paris",
    }, timeout=15)
    r.raise_for_status()
    return r.json().get("daily", {})


def load_weather_alerts_all() -> tuple[list, int, int]:
    """Alertes dérivées des prévisions Open-Meteo pour les 6 départements LGV SEA.
    Retourne (alertes, nb_dept_ok, nb_dept_total) : si nb_dept_ok == 0, l'appelant
    doit afficher un avertissement plutôt qu'un silencieux "aucune alerte"
    (chaque département étant récupéré via une fonction cachée séparée, l'échec
    de l'un n'empêche pas les autres d'être servis depuis leur propre cache)."""
    alerts = []
    ok_count = 0
    for dep in DEPS:
        try:
            daily = _fetch_dept_forecast_raw(dep)
            ok_count += 1
        except Exception:
            continue

        dates   = daily.get("time", [])
        precips = daily.get("precipitation_sum",        [0]*7)
        tmaxes  = daily.get("temperature_2m_max",       [0]*7)
        wcodes  = daily.get("weathercode",              [0]*7)
        winds   = daily.get("wind_speed_10m_max",       [0]*7)
        rain7   = sum(p or 0 for p in precips)

        seen_fire = False
        for i, date in enumerate(dates):
            p = precips[i] or 0
            t = tmaxes[i]  or 0
            w = wcodes[i]  or 0
            v = winds[i]   or 0
            d_str = date[5:]  # MM-DD

            # ⛈️ Orages (WMO codes 80-82 averses, 95-99 orages)
            if w >= 99:
                alerts.append(dict(dep=dep, date=date, type="ORAGE", level="ROUGE",
                    msg=f"Dép.{dep} le {d_str} — Orages violents avec grêle"))
            elif w >= 95:
                alerts.append(dict(dep=dep, date=date, type="ORAGE", level="ORANGE",
                    msg=f"Dép.{dep} le {d_str} — Orages"))
            elif w in (80, 81, 82):
                alerts.append(dict(dep=dep, date=date, type="ORAGE", level="JAUNE",
                    msg=f"Dép.{dep} le {d_str} — Averses orageuses"))

            # 🌊 Inondation (précipitations intenses)
            if p >= 60:
                alerts.append(dict(dep=dep, date=date, type="INONDATION", level="ROUGE",
                    msg=f"Dép.{dep} le {d_str} — Pluies diluviennes : {p:.0f} mm"))
            elif p >= 30:
                alerts.append(dict(dep=dep, date=date, type="INONDATION", level="ORANGE",
                    msg=f"Dép.{dep} le {d_str} — Pluies intenses : {p:.0f} mm"))
            elif p >= 15:
                alerts.append(dict(dep=dep, date=date, type="INONDATION", level="JAUNE",
                    msg=f"Dép.{dep} le {d_str} — Pluies soutenues : {p:.0f} mm"))

            # 🌡️ Canicule
            if t >= 40:
                alerts.append(dict(dep=dep, date=date, type="CANICULE", level="ROUGE",
                    msg=f"Dép.{dep} le {d_str} — Canicule extrême : {t:.0f}°C"))
            elif t >= 36:
                alerts.append(dict(dep=dep, date=date, type="CANICULE", level="ROUGE",
                    msg=f"Dép.{dep} le {d_str} — Canicule : {t:.0f}°C"))
            elif t >= 33:
                alerts.append(dict(dep=dep, date=date, type="CANICULE", level="ORANGE",
                    msg=f"Dép.{dep} le {d_str} — Forte chaleur : {t:.0f}°C"))

            # 💨 Vent violent
            if v >= 100:
                alerts.append(dict(dep=dep, date=date, type="VENT", level="ROUGE",
                    msg=f"Dép.{dep} le {d_str} — Vents très violents : {v:.0f} km/h"))
            elif v >= 80:
                alerts.append(dict(dep=dep, date=date, type="VENT", level="ORANGE",
                    msg=f"Dép.{dep} le {d_str} — Vents violents : {v:.0f} km/h"))
            elif v >= 60:
                alerts.append(dict(dep=dep, date=date, type="VENT", level="JAUNE",
                    msg=f"Dép.{dep} le {d_str} — Vents forts : {v:.0f} km/h"))

            # 🔥 Risque incendie (chaleur + sécheresse + vent)
            if not seen_fire and t >= 30 and rain7 < 10 and v >= 25:
                lvl = "ROUGE" if (t >= 35 and rain7 < 5 and v >= 35) else "ORANGE"
                alerts.append(dict(dep=dep, date=date, type="INCENDIE", level=lvl,
                    msg=f"Dép.{dep} — Risque incendie : {t:.0f}°C, "
                        f"vent {v:.0f} km/h, pluie 7j {rain7:.0f} mm"))
                seen_fire = True

    alerts.sort(key=lambda x: (-LEVEL_RANK.get(x["level"], 0), x["date"]))
    return alerts, ok_count, len(DEPS)


def _normalize(s: str) -> str:
    """Lowercase + strip accents for robust name matching."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )

# Cours d'eau traversés ou longés par la LGV SEA Tours→Bordeaux
_RIVERS_RAW = [
    "vienne", "clain", "charente", "boutonne", "seugne", "touvre",
    "dronne", "isle", "dordogne", "garonne", "thouet", "sevre",
    "indre", "cher", "creuse", "ciron", "jalles", "estey",
    "leyre", "midouze", "brion", "anglin",
]
RIVERS_LGV = [_normalize(r) for r in _RIVERS_RAW]

# Departments along LGV SEA (filter unrelated rivers with same name elsewhere)
_DEP_OK = {"37","86","79","16","17","33","24","47","40","49","85","36"}


@st.cache_data(ttl=1800)
def load_vigicrue_rivers() -> tuple[list, bool]:
    """Charge la vigilance crues officielle pour les cours d'eau LGV SEA.

    L'ancien appel ``/services/2/InfoVigiCrue.xml`` n'est pas un endpoint
    public valide. L'API documentee expose le referentiel en JSON sous
    ``/services/v1.1/TerEntVigiCru.json``. On charge la liste des territoires,
    puis leur detail afin de recuperer les troncons et leur niveau de vigilance.
    """
    api_url = "https://www.vigicrues.gouv.fr/services/v1.1/TerEntVigiCru.json"
    headers = {
        "Accept": "application/json",
        "User-Agent": "LGV-PluvioStations/1.0 (+https://lgvpluviostations.streamlit.app/)",
    }
    level_by_number = {1: "VERT", 2: "JAUNE", 3: "ORANGE", 4: "ROUGE"}
    level_by_name = {
        "vert": "VERT", "green": "VERT",
        "jaune": "JAUNE", "yellow": "JAUNE",
        "orange": "ORANGE",
        "rouge": "ROUGE", "red": "ROUGE",
    }

    def walk(value):
        """Parcourt tous les dictionnaires d'une reponse JSON imbriquee."""
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    def first(item: dict, keys: tuple[str, ...]):
        for key in keys:
            value = item.get(key)
            if value not in (None, ""):
                return value
        return None

    def vigilance_level(item: dict) -> str | None:
        raw = first(item, (
            "NivVigiCru", "NivVigiCruHydro", "NivVig",
            "NiveauVigilance", "CdCouleur", "Couleur", "couleur",
        ))
        if raw is None:
            return None
        normalized = _normalize(str(raw)).strip()
        if normalized in level_by_name:
            return level_by_name[normalized]
        try:
            return level_by_number.get(int(float(normalized)))
        except (TypeError, ValueError):
            return None

    results: list = []
    successful_responses = 0

    try:
        response = requests.get(api_url, headers=headers, timeout=(5, 25))
        response.raise_for_status()
        territories_payload = response.json()
        successful_responses += 1
    except (requests.RequestException, ValueError):
        return [], False

    territories = territories_payload.get("ListEntVigiCru", [])
    if not isinstance(territories, list):
        return [], False

    for territory in territories:
        if not isinstance(territory, dict):
            continue
        code = territory.get("CdEntVigiCru")
        entity_type = territory.get("TypEntVigiCru", "5")
        if not code:
            continue
        try:
            response = requests.get(
                api_url,
                params={"CdEntVigiCru": code, "TypEntVigiCru": entity_type},
                headers=headers,
                timeout=(5, 25),
            )
            response.raise_for_status()
            payload = response.json()
            successful_responses += 1
        except (requests.RequestException, ValueError):
            continue

        for item in walk(payload):
            name = first(item, (
                "LbEntVigiCru", "LibEntVigiCru", "NomEntVigiCru",
                "LibTroncon", "NomTroncon", "NomCoursDeau", "Nom", "lib",
            ))
            level = vigilance_level(item)
            if not name or not level:
                continue

            name = str(name).strip()
            name_norm = _normalize(name)
            if not any(river in name_norm for river in RIVERS_LGV):
                continue

            dep_raw = first(item, ("CdDep", "CDDep", "CodeDepartement"))
            dep = str(dep_raw).zfill(2) if dep_raw not in (None, "") else ""
            if dep and dep not in _DEP_OK:
                continue

            results.append({
                "riviere": name,
                "dep": dep,
                "level": level,
                "type": "VIGICRUE",
                "msg": f"{name} — vigilance {level.lower()}",
            })

    # Une reponse API valide sans alerte LGV est un resultat legitime.
    parsed_ok = successful_responses > 0

    seen: set = set()
    dedup: list = []
    for item in results:
        key = (_normalize(item["riviere"]), item["level"])
        if key not in seen:
            seen.add(key)
            dedup.append(item)

    dedup.sort(key=lambda x: (-LEVEL_RANK.get(x["level"], 0), x["riviere"]))
    return dedup, parsed_ok


def get_firms_map_key() -> str | None:
    """Clé API FIRMS : st.secrets['FIRMS_MAP_KEY'] ou variable d'env FIRMS_MAP_KEY.
    Clé gratuite : https://firms.modaps.eosdis.nasa.gov/api/map_key/"""
    try:
        key = st.secrets.get("FIRMS_MAP_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("FIRMS_MAP_KEY")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


@st.cache_data(ttl=3600)
def build_lgv_pk_polyline(_lgv_lines) -> list[tuple[float, float, float]]:
    """[(lat, lon, pk_km cumulé), ...] à partir du 1er tracé LGV du snapshot.
    Le cumul de distance haversine le long du tracé colle au pk_km réel (écart < 100 m)."""
    if not _lgv_lines:
        return []
    seg = _lgv_lines[0]
    pts = [(p["lat"], p["lon"]) for p in seg if isinstance(p, dict) and "lat" in p and "lon" in p]
    if len(pts) < 2:
        return []
    out = [(pts[0][0], pts[0][1], 0.0)]
    cum = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(pts, pts[1:]):
        cum += _haversine_km(lat1, lon1, lat2, lon2)
        out.append((lat2, lon2, cum))
    return out


def pk_and_distance(lat: float, lon: float, polyline: list) -> tuple[float | None, float | None]:
    """Retourne (pk_km, distance_km) du point le plus proche du tracé LGV.
    Projection sur chaque segment en repère métrique local (correction cos(latitude))."""
    if len(polyline) < 2:
        return None, None
    best_dist2 = None
    best_pk    = None
    for (lat1, lon1, pk1), (lat2, lon2, pk2) in zip(polyline, polyline[1:]):
        lat_mid = (lat1 + lat2) / 2.0
        kx = 111.320 * math.cos(math.radians(lat_mid))
        ky = 111.320
        x1, y1 = lon1 * kx, lat1 * ky
        x2, y2 = lon2 * kx, lat2 * ky
        xp, yp = lon * kx, lat * ky
        dx, dy = x2 - x1, y2 - y1
        seg_len2 = dx * dx + dy * dy
        t = 0.0 if seg_len2 == 0 else max(0.0, min(1.0, ((xp - x1) * dx + (yp - y1) * dy) / seg_len2))
        cx, cy = x1 + t * dx, y1 + t * dy
        dist2 = (xp - cx) ** 2 + (yp - cy) ** 2
        if best_dist2 is None or dist2 < best_dist2:
            best_dist2 = dist2
            best_pk = pk1 + t * (pk2 - pk1)
    return best_pk, math.sqrt(best_dist2) if best_dist2 is not None else None


@st.cache_data(ttl=300)
def _fetch_firms_source_raw(key: str, source: str, day_range: int, date_str: str) -> pd.DataFrame:
    url = FIRMS_AREA_URL.format(key=key, source=source, area=FIRMS_BBOX,
                                 day_range=day_range, date=date_str)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    txt = r.text.strip()
    if "invalid" in txt.lower()[:200]:
        raise ValueError("invalid_key")
    if not txt or "<html" in txt.lower()[:200]:
        raise RuntimeError("unexpected_response")
    df = pd.read_csv(io.StringIO(txt))
    if "latitude" not in df.columns:
        raise RuntimeError("unexpected_response")
    df["source"] = source
    return df


def load_firms_hotspots(day_range: int = 1, end_date=None) -> tuple[pd.DataFrame, str | None]:
    """Détections FIRMS (VIIRS NRT) sur la bbox du corridor LGV SEA, sur `day_range`
    jours se terminant le `end_date` (par défaut aujourd'hui — permet de remonter
    dans le temps, jusqu'à `FIRMS_MAX_LOOKBACK_DAYS` en arrière).
    Distingue explicitement « aucune détection » (chaque source a répondu, 0 résultat)
    de « échec de la vérification » (aucune source n'a pu être interrogée) — sans quoi
    une panne réseau/API afficherait à tort un statut "aucun feu" rassurant mais faux."""
    key = get_firms_map_key()
    if not key:
        return pd.DataFrame(), "missing_key"

    date_str = (end_date or datetime.now(timezone.utc).date()).isoformat()
    frames = []
    ok_count = 0
    for source in FIRMS_SOURCES:
        try:
            df = _fetch_firms_source_raw(key, source, day_range, date_str)
            ok_count += 1
            if not df.empty:
                frames.append(df)
        except ValueError:
            return pd.DataFrame(), "invalid_key"
        except Exception:
            continue

    if ok_count == 0:
        return pd.DataFrame(), "fetch_failed"
    if not frames:
        return pd.DataFrame(), None
    return pd.concat(frames, ignore_index=True), None


FIRMS_CONF_LABELS = {"l": "Faible", "n": "Nominale", "h": "Élevée"}


def _firms_conf_label(raw) -> str:
    """VIIRS renvoie un code lettre (l/n/h) peu lisible tel quel — MODIS renvoie un %
    numérique déjà clair. On ne traduit que le premier cas, l'autre passe inchangé."""
    s = str(raw).strip().lower()
    return FIRMS_CONF_LABELS.get(s, str(raw))


@st.cache_data(ttl=300)
def load_firms_alerts(_polyline: list, day_range: int = 1, end_date=None,
                       radius_km: float = FIRMS_RADIUS_KM) -> tuple[list, str | None]:
    """Détections FIRMS filtrées à moins de `radius_km` de la LGV SEA, avec PK et distance."""
    df, err = load_firms_hotspots(day_range, end_date)
    if err:
        return [], err
    if df.empty or not _polyline:
        return [], None

    alerts: list = []
    seen: set = set()
    for _, row in df.iterrows():
        try:
            lat = float(row["latitude"]); lon = float(row["longitude"])
        except Exception:
            continue
        key = (round(lat, 4), round(lon, 4))
        if key in seen:
            continue
        pk, dist_km = pk_and_distance(lat, lon, _polyline)
        if pk is None or dist_km is None or dist_km > radius_km:
            continue
        seen.add(key)

        acq_date = str(row.get("acq_date", "") or "")
        acq_time = str(row.get("acq_time", "") or "").zfill(4)
        heure    = f"{acq_time[:2]}:{acq_time[2:]}" if len(acq_time) == 4 else acq_time
        conf     = _firms_conf_label(row.get("confidence", ""))
        frp      = row.get("frp", None)
        dist_m   = dist_km * 1000

        alerts.append(dict(
            type="FEU_FIRMS", level="ROUGE", lat=lat, lon=lon,
            pk_km=round(pk, 1), distance_m=round(dist_m),
            confidence=conf, frp=frp,
            satellite=row.get("satellite", ""), source=row.get("source", ""),
            date=acq_date, heure=heure,
            msg=(f"PK {pk:.1f} km — à {dist_m:.0f} m de la LGV SEA — "
                 f"détecté le {acq_date} à {heure} UTC (confiance {conf})"),
        ))

    alerts.sort(key=lambda a: a["pk_km"])
    return alerts, None


@st.cache_data(ttl=3600)
def _fetch_commune_daily_series_raw(lat: float, lon: float, days: int) -> dict:
    end   = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days - 1)
    r = requests.get(ARCHIVE_URL, params={
        "latitude": lat, "longitude": lon,
        "start_date": str(start), "end_date": str(end),
        "daily": "precipitation_sum", "timezone": "Europe/Paris",
    }, timeout=20)
    r.raise_for_status()
    return r.json().get("daily", {})


def load_commune_daily_series(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    """Série journalière de pluie sur les `days` derniers jours — API archive ERA5
    (réanalyse), pas l'API prévision : son paramètre `past_days` renvoie pour les
    1-2 derniers jours une sortie modèle non recalée sur l'observé, constatée jusqu'à
    11x plus élevée que l'ERA5 le même jour (45,6 mm vs 4,0 mm), ce qui gonflait
    artificiellement les cumuls du classement TOP 20."""
    try:
        daily = _fetch_commune_daily_series_raw(lat, lon, days)
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame({"date": daily.get("time", []),
                        "pluie_mm": daily.get("precipitation_sum", [])})
    df["pluie_mm"] = pd.to_numeric(df["pluie_mm"], errors="coerce").fillna(0.0)
    return df


@st.cache_data(ttl=3600)
def load_all_communes_daily_rain(_sectors_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Série journalière de pluie pour chaque commune du corridor (1 requête/commune, cache 1h)."""
    if _sectors_df.empty or "commune_name" not in _sectors_df.columns:
        return pd.DataFrame()
    coords = (_sectors_df.dropna(subset=["latitude", "longitude"])
              .groupby("commune_name")[["latitude", "longitude"]].mean())
    frames = []
    for commune, row in coords.iterrows():
        df = load_commune_daily_series(round(float(row["latitude"]), 4),
                                        round(float(row["longitude"]), 4), days)
        if df.empty:
            continue
        df = df.copy()
        df["commune_name"] = commune
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=900)
def load_commune_rain_ometo(lat: float, lon: float, periode: str) -> float:
    """Cumul pluie.
    24h  → API prévision Open-Meteo, AROME Météo-France 1,3 km si dispo, sinon
           blend — la veille est un cycle déjà bouclé, pas une sortie modèle
           non recalée (vérifié : identique à l'ERA5 sur un cas testé).
    7j+  → API archive ERA5 (réanalyse), pas l'API prévision : son paramètre
           `past_days` a été mesuré jusqu'à 2x plus élevé que l'ERA5 sur le
           même point (46,4 mm vs 21,9 mm sur 7 j ; 71,1 mm vs 54,6 mm sur 30 j),
           le même défaut que celui corrigé pour le TOP 20.
    """
    today = datetime.now(timezone.utc).date()
    lat_r, lon_r = round(lat, 4), round(lon, 4)

    if periode == "24h":
        for model in ["meteofrance_arome_france", None]:
            try:
                params = {
                    "latitude": lat_r, "longitude": lon_r,
                    "daily": "precipitation_sum",
                    "past_days": 1, "forecast_days": 0,
                    "timezone": "Europe/Paris",
                }
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

    if periode == "7 jours":
        days = 7
    elif periode == "30 jours":
        days = 30
    else:
        days = today.day - 1
    if days <= 0:
        return 0.0

    start = today - timedelta(days=days)
    end   = today - timedelta(days=1)
    try:
        r = requests.get(ARCHIVE_URL, params={
            "latitude": lat_r, "longitude": lon_r,
            "start_date": str(start), "end_date": str(end),
            "daily": "precipitation_sum", "timezone": "Europe/Paris",
        }, timeout=20)
        r.raise_for_status()
        vals = r.json()["daily"]["precipitation_sum"]
        if vals:
            return round(sum(v for v in vals if v is not None), 1)
    except Exception:
        pass
    return float("nan")


@st.cache_data(ttl=3600)
def _fetch_forecast_dep_raw(dep: str) -> dict:
    d = DEPS[dep]
    r = requests.get(FORECAST_URL, params={
        "latitude": d["lat"], "longitude": d["lon"],
        "daily": "precipitation_sum,weathercode",
        "forecast_days": 7, "timezone": "Europe/Paris",
    }, timeout=15)
    r.raise_for_status()
    return r.json()


def load_forecast_dep(dep: str) -> dict:
    try:
        return _fetch_forecast_dep_raw(dep)
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def load_forecast_coord(lat: float, lon: float) -> pd.DataFrame:
    try:
        r = requests.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max",
            "forecast_days": 7, "timezone": "Europe/Paris",
        }, timeout=15)
        r.raise_for_status()
        daily = r.json().get("daily", {})
        return pd.DataFrame({
            "date":     daily.get("time", []),
            "pluie_mm": daily.get("precipitation_sum", []),
            "proba_%":  daily.get("precipitation_probability_max", []),
            "tmax":     daily.get("temperature_2m_max", []),
        })
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_monthly_rain(lat: float, lon: float) -> pd.DataFrame:
    end   = datetime.now(timezone.utc).date()
    start = (end.replace(day=1) - timedelta(days=365)).replace(day=1)
    try:
        r = requests.get(ARCHIVE_URL, params={
            "latitude": lat, "longitude": lon,
            "start_date": str(start), "end_date": str(end),
            "daily": "precipitation_sum", "timezone": "Europe/Paris",
        }, timeout=20)
        r.raise_for_status()
        data  = r.json()
        dates = data["daily"]["time"]
        rain  = data["daily"]["precipitation_sum"]
        monthly: dict = defaultdict(float)
        for d, v in zip(dates, rain):
            if v is not None:
                monthly[d[:7]] += v
        return pd.DataFrame([{"mois": m, "pluie_mm": round(v, 1)}
                              for m, v in sorted(monthly.items())])
    except Exception:
        return pd.DataFrame()




# Charte graphique commune aux graphiques météo/pluviométrie.
# Cette couche de présentation est indépendante du traitement NASA FIRMS.
CHART_COLORS = {
    "blue": "#2563eb", "cyan": "#0891b2", "teal": "#0f766e",
    "orange": "#f97316", "red": "#dc2626", "slate": "#475569",
}
CHART_GRID = "#e2e8f0"
CHART_TEXT = "#334155"


def style_weather_chart(fig: go.Figure, *, height: int = 340,
                        hovermode: str = "x unified") -> go.Figure:
    """Applique une présentation homogène sans modifier les données du graphe."""
    fig.update_layout(
        height=height,
        hovermode=hovermode,
        font=dict(family="Arial, sans-serif", size=12, color=CHART_TEXT),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=48, b=48, l=56, r=44),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, bgcolor="rgba(255,255,255,0.85)",
        ),
        hoverlabel=dict(bgcolor="white", font_size=12, font_color="#0f172a"),
        transition=dict(duration=250),
    )
    fig.update_xaxes(
        showgrid=False, showline=True, linecolor="#cbd5e1",
        tickcolor="#cbd5e1", automargin=True, title=None,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=CHART_GRID, gridwidth=1,
        zeroline=True, zerolinecolor="#cbd5e1", automargin=True,
    )
    return fig


def show_weather_chart(fig: go.Figure, *, height: int = 340,
                       hovermode: str = "x unified") -> None:
    """Affiche un graphe météo responsive avec une barre d'outils allégée."""
    style_weather_chart(fig, height=height, hovermode=hovermode)
    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def safe_df(records) -> pd.DataFrame:
    if isinstance(records, list) and records:
        try:
            return pd.DataFrame(records)
        except Exception:
            pass
    return pd.DataFrame()


def alert_card(a: dict):
    """Carte d'alerte plus lisible. FIRMS conserve volontairement son rendu initial."""
    lvl   = a.get("level", "")
    atype = a.get("type", "")
    icon  = a.get("icon") or ALERT_CFG.get(atype, ("", ""))[0] or None

    # Ne pas modifier la présentation ni le comportement des alertes NASA FIRMS.
    if atype == "FEU_FIRMS":
        c_badge, c_msg = st.columns([1, 7], vertical_alignment="center")
        with c_badge:
            st.badge(LEVEL_LABEL.get(lvl, lvl or "Info"), color=LEVEL_BADGE.get(lvl, "gray"))  # type: ignore[arg-type]
        with c_msg:
            st.markdown(f"{icon}  {a.get('msg','')}" if icon else a.get("msg", ""))
        return

    label = ALERT_CFG.get(atype, ("", atype.replace("_", " ").title()))[1]
    color = LEVEL_COLOR.get(lvl, LEVEL_COLOR["INFO"])
    date_label = a.get("date", "")
    dep_label = f"Dép. {a['dep']}" if a.get("dep") else ""
    meta = " · ".join(str(v) for v in (label, dep_label, date_label) if v)

    with st.container(border=True):
        c_icon, c_body, c_level = st.columns([0.45, 6.4, 1.15], vertical_alignment="center")
        with c_icon:
            st.markdown(f"<div style='font-size:1.45rem;text-align:center'>{icon or 'ℹ️'}</div>", unsafe_allow_html=True)
        with c_body:
            st.markdown(f"**{meta or label}**")
            st.caption(str(a.get("msg", "")))
        with c_level:
            st.markdown(
                f"<div style='border-left:4px solid {color};padding-left:.55rem;font-weight:700;color:{color}'>"
                f"{LEVEL_LABEL.get(lvl, lvl or 'Info')}</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="LGV SEA – Pluviométrie", page_icon="🌧", layout="wide")
st.title("🌧 LGV SEA – Pluviométrie")

col_title, col_btn = st.columns([5, 1])
col_title.caption(
    "📡 Données météo : **Open-Meteo** (modèles ERA5 + ECMWF) · "
    "Crues : **Vigicrue** · Données non officielles — "
    "pour les alertes officielles : "
    "[vigilance.meteofrance.fr](https://vigilance.meteofrance.fr/) · "
    "[vigicrues.gouv.fr](https://www.vigicrues.gouv.fr/)"
)
if col_btn.button("🔄 Rafraîchir"):
    st.cache_data.clear()
    st.rerun()

snapshot = load_snapshot()
if "_error" in snapshot:
    st.error(f"Erreur snapshot : {snapshot['_error']}")
    st.stop()

_sec       = snapshot.get("sectors")
sectors_df = safe_df(_sec.get("sectors", []) if isinstance(_sec, dict) else [])
for col in ["weather_max_24h_mm","weather_max_7d_mm","weather_max_30d_mm",
            "weather_max_month_mm","latitude","longitude","pk_km"]:
    if col in sectors_df.columns:
        sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")

# Pre-compute dept forecast rain (used for map coloring + dept cards)
RAIN_LABELS = {
    "VERT":   "Peu de pluie",
    "JAUNE":  "Pluie modérée",
    "ORANGE": "Pluies fortes",
    "ROUGE":  "Pluies très fortes",
    "INDETERMINE": "Indisponible",
}
dep_rain_data: dict = {}   # dep -> {max, total, lvl, color, emoji, ok}
for _dep in DEPS:
    _fc    = load_forecast_dep(_dep)
    _ok    = bool(_fc)
    _rains = [v for v in _fc.get("daily", {}).get("precipitation_sum", []) if v is not None]
    _max   = max(_rains) if _rains else 0.0
    _total = sum(_rains)
    if _ok:
        _lvl, _color, _emoji = rain_risk(_max)
    else:
        # Échec de récupération distinct d'un "peu de pluie" légitime — sans ce
        # marqueur, une panne Open-Meteo afficherait à tort une carte verte rassurante.
        _lvl, _color, _emoji = "INDETERMINE", "#9ca3af", "❓"
    dep_rain_data[_dep] = {"max": _max, "total": _total,
                            "lvl": _lvl, "color": _color, "emoji": _emoji, "ok": _ok}


def nearest_dep(lat: float, lon: float) -> str:
    """Return dep code nearest to a lat/lon (for map coloring)."""
    return min(DEPS.keys(),
               key=lambda d: (DEPS[d]["lat"] - lat) ** 2 + (DEPS[d]["lon"] - lon) ** 2)


# ── 1. PLUIE PRÉVUE PAR DÉPARTEMENT ─────────────────────────────────────────
st.subheader("Pluie prévue 7 jours par département")
st.caption("Source : Open-Meteo")
dep_cols = st.columns(len(DEPS))
for col_w, (dep, info) in zip(dep_cols, DEPS.items()):
    d = dep_rain_data[dep]
    with col_w.container(border=True):
        st.caption(f"Dép. {dep} · {info['nom']}")
        if d["ok"]:
            lvl_label = RAIN_LABELS[d["lvl"]]
            st.badge(lvl_label, color=LEVEL_BADGE.get(d["lvl"], "gray"))  # type: ignore[arg-type]
            st.metric("Cumul 7 j", f"{d['total']:.0f} mm",
                      help=f"Maximum journalier : {d['max']:.0f} mm/j")
        else:
            st.badge("Indisponible", color="gray")
            st.caption("Open-Meteo injoignable")

# ── 1bis. VIGILANCE OFFICIELLE MÉTÉO-FRANCE ─────────────────────────────────
st.subheader("🛡️ Vigilance officielle Météo-France (aujourd'hui / demain)")

mf_alerts, mf_ok = load_meteofrance_vigilance()

if not mf_ok:
    st.warning("Vigilance Météo-France injoignable — statut **non vérifié** "
               "(réessaie dans quelques minutes).")
elif mf_alerts:
    st.error(f"{len(mf_alerts)} vigilance(s) active(s) sur les départements du corridor LGV SEA.")
    for a in mf_alerts:
        alert_card(a)
else:
    st.success("Vigilance verte sur les 6 départements du corridor (aujourd'hui et demain).")

st.caption(
    "Source officielle : [vigilance.meteofrance.fr](https://vigilance.meteofrance.fr/) — "
    "données republiées en open data (sans clé), mêmes bulletins que l'appli officielle. "
    "Les « indicateurs météo » ci-dessous sont une estimation complémentaire sur 7 jours, "
    "pas une vigilance officielle."
)

st.divider()

# ── 2. INDICATEURS MÉTÉO ─────────────────────────────────────────────────────
st.subheader("📊 Indicateurs météo indicatifs — 7 prochains jours")

met_alerts, met_ok, met_total = load_weather_alerts_all()
vc_alerts, vc_ok = load_vigicrue_rivers()

active_met = [a for a in met_alerts if a["level"] in ("ROUGE","ORANGE","JAUNE")]
by_met: dict = defaultdict(list)
for a in active_met:
    by_met[a["type"]].append(a)

if met_ok == 0:
    st.warning("Open-Meteo injoignable — indicateurs météo **non vérifiés** actuellement "
               "(ce n'est pas un « aucune alerte », réessaie dans quelques minutes).")
elif met_ok < met_total:
    st.caption(f"ℹ️ Prévisions récupérées pour {met_ok}/{met_total} départements — "
               "le reste sera réessayé au prochain rafraîchissement.")

if active_met:
    badge_cols = st.columns(len(by_met))
    for col, (atype, alist) in zip(badge_cols, by_met.items()):
        worst = max(alist, key=lambda x: LEVEL_RANK.get(x["level"], 0))
        icon, label = ALERT_CFG.get(atype, ("", atype))
        col.badge(f"{label} ({len(alist)})", icon=icon or None,
                  color=LEVEL_BADGE.get(worst["level"], "gray"))  # type: ignore[arg-type]
elif met_ok > 0:
    st.success("Aucun indicateur météo significatif sur les 7 prochains jours.")

tab_labels: list = []
tab_data:   list = []
for atype in ("ORAGE", "INONDATION", "CANICULE", "INCENDIE", "VENT"):
    if atype in by_met:
        icon, label = ALERT_CFG[atype]
        tab_labels.append(f"{icon} {label} ({len(by_met[atype])})")
        tab_data.append(by_met[atype])

vc_active = [a for a in vc_alerts if a["level"] in ("ROUGE","ORANGE","JAUNE")]
if not vc_ok:
    vc_label = "🏞️ Vigicrue ❓"
elif vc_active:
    vc_label = f"🏞️ Vigicrue ⚠ {len(vc_active)}"
else:
    vc_label = "🏞️ Vigicrue ✅"
tab_labels.append(vc_label)
tab_data.append(vc_alerts)

if tab_labels:
    tabs = st.tabs(tab_labels)
    for tab, alist in zip(tabs, tab_data):
        with tab:
            if tab is tabs[-1] and not vc_ok:
                st.warning("API Vigicrue injoignable — statut crues **non vérifié** "
                           "(réessaie dans quelques minutes).")
            elif not alist:
                st.info("Aucune crue en vigilance actuellement sur ces cours d'eau."
                         if tab is tabs[-1] else "Aucune donnée.")
            for a in alist:
                alert_card(a)

st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
communes = (sorted(sectors_df["commune_name"].dropna().unique())
            if "commune_name" in sectors_df.columns else [])

# Communes par défaut : répartition géographique nord→sud sur la LGV SEA
def _find_commune(kw: str) -> str | None:
    k = unicodedata.normalize("NFD", kw.lower()).encode("ascii", "ignore").decode()
    return next((c for c in communes
                 if k in unicodedata.normalize("NFD", c.lower()).encode("ascii", "ignore").decode()), None)

_DEFAULT_KW = ["nouatre", "fontaine", "poitier", "biard", "villognon", "clerac", "ambares"]
_default_communes: list = []
for _kw in _DEFAULT_KW:
    _m = _find_commune(_kw)
    if _m and _m not in _default_communes:
        _default_communes.append(_m)
_default_communes = _default_communes[:6] or (communes[:6] if len(communes) >= 6 else communes)

with st.sidebar:
    st.subheader("📍 Communes")
    selected_multi = st.multiselect("Comparer communes", communes,
                                     default=_default_communes)
    selected_one   = st.selectbox("Commune principale", ["— Toutes —"] + list(communes))
    periode  = st.selectbox("📅 Période pluvio", ["24h","7 jours","30 jours","Mois courant"])

    st.subheader("🔥 Incendies FIRMS")
    _today_date = datetime.now(timezone.utc).date()
    firms_end_date = st.date_input(
        "Jusqu'au", value=_today_date,
        min_value=_today_date - timedelta(days=FIRMS_MAX_LOOKBACK_DAYS),
        max_value=_today_date,
        help="Choisis une date passée pour consulter l'historique FIRMS au lieu du temps réel.",
    )
    firms_day_range = st.slider("Fenêtre (jours avant cette date)", 1, FIRMS_MAX_DAY_RANGE, 1)
    firms_is_live = firms_end_date == _today_date
    if not firms_is_live:
        st.caption("📜 Mode historique — le rafraîchissement automatique ne changera rien "
                   "à une date passée, seul un rerun manuel peut en changer les données.")

    st.subheader("🔄 Actualisation")
    auto_refresh = st.checkbox("Rafraîchissement automatique", value=True)
    if auto_refresh:
        _refresh_label = st.selectbox("Intervalle", ["5 min", "10 min", "15 min"], index=1)
        refresh_seconds = {"5 min": 300, "10 min": 600, "15 min": 900}[_refresh_label]
        st.caption("⚠️ Recharge la page entière (réinitialise filtres/sélections). "
                   "La fraîcheur réelle des données reste bornée par leur propre cache "
                   "(15-60 min selon la source), pas par cet intervalle.")

if auto_refresh:
    st.markdown(f'<meta http-equiv="refresh" content="{refresh_seconds}">', unsafe_allow_html=True)

# ── 2bis. INCENDIES — NASA FIRMS ────────────────────────────────────────────
_firms_start_date = firms_end_date - timedelta(days=firms_day_range - 1)
if firms_is_live and firms_day_range <= 3:
    firms_window_label = f"dernières {firms_day_range * 24} h"
elif firms_is_live:
    firms_window_label = f"derniers {firms_day_range} jours"
elif _firms_start_date == firms_end_date:
    firms_window_label = firms_end_date.strftime('%d/%m/%Y')
else:
    firms_window_label = (f"{_firms_start_date.strftime('%d/%m/%Y')} → "
                           f"{firms_end_date.strftime('%d/%m/%Y')}")

_firms_title = ("temps réel" if firms_is_live
                else f"historique · {firms_window_label}")
st.subheader(f"🔥 Détections incendie {_firms_title} — NASA FIRMS "
             f"(rayon {FIRMS_RADIUS_KM*1000:.0f} m autour de la LGV SEA)")

lgv_polyline = build_lgv_pk_polyline(snapshot.get("lgv_lines"))
firms_alerts, firms_err = load_firms_alerts(lgv_polyline, day_range=firms_day_range,
                                             end_date=firms_end_date)

if firms_err == "missing_key":
    st.warning(
        "Clé FIRMS manquante. Crée une clé gratuite sur "
        "[firms.modaps.eosdis.nasa.gov/api/map_key](https://firms.modaps.eosdis.nasa.gov/api/map_key/), "
        "puis renseigne-la dans `.streamlit/secrets.toml` (`FIRMS_MAP_KEY = \"...\"`) "
        "ou la variable d'environnement `FIRMS_MAP_KEY`."
    )
elif firms_err == "invalid_key":
    st.error("Clé FIRMS invalide — vérifie `FIRMS_MAP_KEY`.")
elif firms_err == "fetch_failed":
    st.warning("FIRMS injoignable actuellement (réseau/API) — statut incendie **non vérifié** "
               "(ce n'est pas un « aucun feu détecté », réessaie dans quelques minutes).")
elif firms_alerts:
    st.error(f"{len(firms_alerts)} détection(s) FIRMS à moins de "
             f"{FIRMS_RADIUS_KM*1000:.0f} m de la LGV SEA ({firms_window_label}).")
    df_firms = pd.DataFrame(firms_alerts).rename(columns={
        "pk_km": "PK (km)", "distance_m": "Distance LGV (m)",
        "date": "Date", "heure": "Heure (UTC)",
        "confidence": "Confiance", "frp": "FRP (MW)", "satellite": "Satellite",
    })
    st.dataframe(
        df_firms[["PK (km)", "Distance LGV (m)", "Date", "Heure (UTC)",
                  "Confiance", "FRP (MW)", "Satellite"]],
        use_container_width=True, hide_index=True,
    )
else:
    st.success(f"Aucune détection FIRMS à moins de {FIRMS_RADIUS_KM*1000:.0f} m "
               f"de la LGV SEA ({firms_window_label}).")

_firms_freshness = (
    "Vérifié toutes les 5 min ici, mais un satellite ne survole un même point que "
    "2 fois par jour environ, avec 1 à 3 h de traitement avant publication : "
    "actualiser plus souvent ne fait pas apparaître une détection plus tôt que ça."
    if firms_is_live else
    "Requête historique figée sur la période choisie — les résultats ne changeront "
    "pas au fil des rafraîchissements."
)
st.caption(
    "Source : NASA FIRMS · VIIRS NOAA-20/NOAA-21/S-NPP (résolution ~375 m) · "
    "[firms.modaps.eosdis.nasa.gov/map](https://firms.modaps.eosdis.nasa.gov/map) — "
    "PK et distance calculés par projection sur le tracé LGV SEA. " + _firms_freshness
)

st.divider()

# ── 2ter. POINTS DE VIGILANCE — SYNTHÈSE LGV SEA ────────────────────────────
st.subheader("🧭 Points de vigilance — surveillance ligne LGV SEA")

_firms_unverified = firms_err in ("fetch_failed", "invalid_key", "missing_key")

vig_cols = st.columns(4)
vig_cols[0].badge(
    "Vigilance MF — non vérifié" if not mf_ok else f"Vigilance MF ({len(mf_alerts)})",
    icon="🛡️", color="gray" if not mf_ok else ("red" if mf_alerts else "green"))  # type: ignore[arg-type]
vig_cols[1].badge(
    "Météo — non vérifié" if met_ok == 0 else f"Météo ({len(active_met)})",
    icon="🌦️", color="gray" if met_ok == 0 else ("orange" if active_met else "green"))  # type: ignore[arg-type]
vig_cols[2].badge(
    "Vigicrue — non vérifié" if not vc_ok else f"Vigicrue ({len(vc_active)})",
    icon="🏞️", color="gray" if not vc_ok else ("blue" if vc_active else "green"))  # type: ignore[arg-type]
vig_cols[3].badge(
    "FIRMS — non vérifié" if _firms_unverified else f"FIRMS ({len(firms_alerts)})",
    icon="🔥", color="gray" if _firms_unverified else ("red" if firms_alerts else "green"))  # type: ignore[arg-type]

if mf_alerts or active_met or vc_active or firms_alerts:
    with st.expander("⚠️ Détail des alertes actives (vigilance MF, météo, crues, incendie)", expanded=True):
        for a in mf_alerts:
            alert_card(a)
        for a in sorted(active_met, key=lambda x: -LEVEL_RANK.get(x["level"], 0)):
            alert_card(a)
        for a in vc_active:
            alert_card(a)
        for a in firms_alerts:
            alert_card(a)
elif not mf_ok or met_ok == 0 or not vc_ok or _firms_unverified:
    st.warning("Au moins une source (vigilance MF, météo, crues ou incendie) n'a pas pu être vérifiée "
               "— voir le détail ci-dessus. Ne pas interpréter comme « aucune alerte ».")
else:
    st.success("Aucune alerte météo, crue ou incendie active actuellement.")

st.divider()

# ── 2quater. TOP 20 COMMUNES — CUMUL DE PRÉCIPITATION ───────────────────────
st.subheader("🌧 TOP 20 communes — cumul de précipitation le plus élevé (30 derniers jours)")

with st.spinner("Chargement pluvio 30 jours pour toutes les communes du corridor…"):
    all_rain_df = load_all_communes_daily_rain(sectors_df, days=30)

_n_communes_total = (sectors_df["commune_name"].dropna().nunique()
                      if "commune_name" in sectors_df.columns else 0)
_n_communes_ok = all_rain_df["commune_name"].nunique() if not all_rain_df.empty else 0

if all_rain_df.empty:
    st.info("Données pluvio indisponibles pour le classement.")
else:
    if _n_communes_total and _n_communes_ok < _n_communes_total:
        st.caption(f"⚠️ Données récupérées pour {_n_communes_ok}/{_n_communes_total} communes du corridor "
                   "— le classement ci-dessous ne porte que sur les communes disponibles.")
    all_rain_df = all_rain_df.copy()
    all_rain_df["date"] = pd.to_datetime(all_rain_df["date"]).dt.date
    d_min, d_max = all_rain_df["date"].min(), all_rain_df["date"].max()

    date_range = st.slider(
        "📅 Filtrer par période (filtre appliqué sur les données déjà chargées — aucun rechargement)",
        min_value=d_min, max_value=d_max, value=(d_min, d_max), format="DD/MM",
    )
    filtered_rain = all_rain_df[(all_rain_df["date"] >= date_range[0]) &
                                 (all_rain_df["date"] <= date_range[1])]

    totals = (filtered_rain.groupby("commune_name")["pluie_mm"].sum()
              .sort_values(ascending=False).head(20))

    if totals.empty:
        st.info("Aucune donnée sur la période sélectionnée.")
    else:
        top20_communes = totals.index.tolist()

        bar_df = totals.sort_values(ascending=True).reset_index()
        bar_df["color"] = bar_df["pluie_mm"].apply(rain_color_mm)
        fig_top = go.Figure(go.Bar(
            x=bar_df["pluie_mm"], y=bar_df["commune_name"], orientation="h",
            marker_color=bar_df["color"].tolist(),
            text=bar_df["pluie_mm"].apply(lambda v: f"{v:.0f} mm"),
            textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Cumul : %{x:.1f} mm<extra></extra>",
        ))
        fig_top.update_layout(
            height=max(320, len(bar_df) * 26 + 60),
            xaxis=dict(title="Cumul pluie (mm)", zeroline=True),
            yaxis=dict(tickfont=dict(size=11)),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=30, l=10, r=90),
        )
        show_weather_chart(fig_top, height=420, hovermode="closest")

        # Courbes journalières : top 3 en couleur (identité), le reste du TOP 20 en gris (contexte)
        HIGHLIGHT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
        fig_curve = go.Figure()
        for commune in reversed(top20_communes):
            s = filtered_rain[filtered_rain["commune_name"] == commune].sort_values("date")
            rank = top20_communes.index(commune)
            is_top3 = rank < 3
            fig_curve.add_scatter(
                x=s["date"], y=s["pluie_mm"], mode="lines", name=commune,
                line=dict(color=HIGHLIGHT_COLORS[rank] if is_top3 else "#c3c2b7",
                          width=2.5 if is_top3 else 1),
                opacity=1.0 if is_top3 else 0.45,
                showlegend=is_top3,
                hovertemplate=f"<b>{commune}</b><br>%{{x|%d/%m}} : %{{y:.1f}} mm<extra></extra>",
            )
        fig_curve.update_layout(
            height=340, xaxis=dict(title=""), yaxis=dict(title="Pluie/jour (mm)"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.15),
            margin=dict(t=35, b=20, l=20, r=20),
        )
        show_weather_chart(fig_curve, height=390)
        st.caption("Courbes : les 3 communes les plus arrosées de ce TOP 20 sont mises en évidence "
                   "(couleur + légende) · les 17 autres apparaissent en gris clair pour le contexte.")

        with st.expander("📋 Table du TOP 20"):
            st.dataframe(
                totals.reset_index().rename(columns={"commune_name": "Commune", "pluie_mm": "Cumul (mm)"}),
                use_container_width=True, hide_index=True,
            )
    st.caption("Source : Open-Meteo ERA5 (réanalyse, série journalière 30 jours) · "
               "une requête par commune, mise en cache 1h.")

st.divider()

# ── 3. COMPARAISON COMMUNES ─────────────────────────────────────────────────
if selected_multi and not sectors_df.empty:
    _today = datetime.now(timezone.utc).date()
    if periode == "24h":
        _titre_periode = f"hier {(_today - timedelta(days=1)).strftime('%d/%m/%Y')}"
    elif periode == "7 jours":
        _titre_periode = f"7 derniers jours (jusqu'au {(_today - timedelta(days=1)).strftime('%d/%m/%Y')})"
    elif periode == "30 jours":
        _titre_periode = f"30 derniers jours (jusqu'au {(_today - timedelta(days=1)).strftime('%d/%m/%Y')})"
    else:
        _titre_periode = f"mois de {_today.strftime('%B %Y')} (jusqu'au {(_today - timedelta(days=1)).strftime('%d/%m')})"
    st.subheader(f"📊 Comparaison communes — cumul pluie {_titre_periode}")

    if len(selected_multi) > 12:
        st.warning("Sélectionne 12 communes max pour la comparaison.")
    else:
        rows = []
        with st.spinner("Chargement données pluvio Open-Meteo…"):
            for commune in selected_multi:
                loc = (sectors_df[sectors_df["commune_name"] == commune]
                       .dropna(subset=["latitude","longitude"]))
                if loc.empty:
                    continue
                lat = round(float(loc["latitude"].mean()), 4)
                lon = round(float(loc["longitude"].mean()), 4)
                rain = load_commune_rain_ometo(lat, lon, periode)
                rows.append({"commune_name": commune, "rain_mm": rain})

        df_cmp = (pd.DataFrame(rows)
                  .dropna(subset=["rain_mm"])
                  .sort_values("rain_mm", ascending=True))

        if df_cmp.empty:
            st.info("Pas de données pour les communes sélectionnées.")
        else:
            today = datetime.now(timezone.utc).date()
            if periode == "24h":
                date_str = f"hier {(today - timedelta(days=1)).strftime('%d/%m/%Y')}"
            elif periode == "7 jours":
                date_str = f"{(today - timedelta(days=7)).strftime('%d/%m')} → {(today - timedelta(days=1)).strftime('%d/%m/%Y')}"
            elif periode == "30 jours":
                date_str = f"{(today - timedelta(days=30)).strftime('%d/%m')} → {(today - timedelta(days=1)).strftime('%d/%m/%Y')}"
            else:
                date_str = f"1er → {(today - timedelta(days=1)).strftime('%d/%m/%Y')}"

            df_cmp["label"] = df_cmp["rain_mm"].apply(
                lambda v: "Sec" if v == 0 else f"{v:.1f} mm")
            df_cmp["color"] = df_cmp["rain_mm"].apply(
                lambda v: "#d1d5db" if v == 0 else rain_color_mm(v))

            fig = go.Figure(go.Bar(
                x=df_cmp["rain_mm"],
                y=df_cmp["commune_name"],
                orientation="h",
                marker_color=df_cmp["color"].tolist(),
                text=df_cmp["label"],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Cumul : %{x:.1f} mm<extra></extra>",
            ))
            height = max(260, len(df_cmp) * 34 + 60)
            fig.update_layout(
                xaxis=dict(title=f"Cumul pluie (mm)", zeroline=True),
                yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
                height=height,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=30, l=10, r=90),
            )
            show_weather_chart(fig, height=360, hovermode="closest")
            _src = "AROME Météo-France 1,3 km" if periode == "24h" else "ERA5 near real-time"
            st.caption(f"Source : Open-Meteo {_src} · {date_str}")

            _missing = sorted(set(selected_multi) - set(df_cmp["commune_name"]))
            if _missing:
                st.warning(f"Pas de donnée pluvio récupérée pour : {', '.join(_missing)} "
                           "— absentes du graphique ci-dessus, pas de valeur à 0 supposée.")

# ── 4. PRÉVISIONS 7J ────────────────────────────────────────────────────────
comm_df = (sectors_df if selected_one == "— Toutes —"
           else sectors_df[sectors_df["commune_name"] == selected_one])
map_df = (comm_df.dropna(subset=["latitude","longitude"])
          if "latitude" in sectors_df.columns and "longitude" in sectors_df.columns
          else pd.DataFrame())
lat_c = float(map_df["latitude"].mean())  if not map_df.empty else 46.2
lon_c = float(map_df["longitude"].mean()) if not map_df.empty else 0.2

label_loc = "LGV SEA" if selected_one == "— Toutes —" else selected_one
st.subheader(f"🔮 Prévisions 7 jours — {label_loc}")
fc_df = load_forecast_coord(lat_c, lon_c)
if not fc_df.empty:
    fc_df["pluie_mm"] = pd.to_numeric(fc_df["pluie_mm"], errors="coerce").fillna(0)
    fc_df["tmax"]     = pd.to_numeric(fc_df["tmax"],     errors="coerce").fillna(0)
    fc_df["color"]    = fc_df["pluie_mm"].apply(rain_color_mm)
    fig2 = go.Figure()
    fig2.add_bar(x=fc_df["date"], y=fc_df["pluie_mm"],
                 marker_color=fc_df["color"].tolist(),
                 text=fc_df["pluie_mm"].apply(lambda v: f"{v:.0f}"),
                 textposition="outside", name="Pluie (mm)")
    if "proba_%" in fc_df.columns:
        fig2.add_scatter(x=fc_df["date"], y=fc_df["proba_%"],
                         mode="lines+markers", name="Proba pluie %",
                         yaxis="y2", line=dict(color="#6366f1", dash="dot"),
                         marker=dict(size=5))
    if "tmax" in fc_df.columns:
        fig2.add_scatter(x=fc_df["date"], y=fc_df["tmax"],
                         mode="lines+markers", name="T° max (°C)",
                         yaxis="y3", line=dict(color="#f97316", width=2),
                         marker=dict(size=5, symbol="diamond"))
    fig2.update_layout(
        yaxis=dict(title="Pluie (mm)", side="left"),
        yaxis2=dict(title="Proba %",   overlaying="y", side="right",
                    range=[0, 110], showgrid=False),
        yaxis3=dict(title="T°C",       overlaying="y", side="right",
                    position=0.92,     showgrid=False, anchor="free"),
        legend=dict(orientation="h", y=1.12), height=290,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=20, l=20, r=80), xaxis=dict(tickangle=-20),
    )
    show_weather_chart(fig2, height=380)
else:
    st.info("Prévisions indisponibles.")

# ── 5. HISTORIQUE MENSUEL ───────────────────────────────────────────────────
st.subheader("📅 Historique mensuel — 12 mois")
monthly_df = load_monthly_rain(lat_c, lon_c)
if not monthly_df.empty:
    fig3 = px.bar(
        monthly_df, x="mois", y="pluie_mm", color="pluie_mm",
        color_continuous_scale=["#bfdbfe","#3b82f6","#1d4ed8","#1e3a8a"],
        labels={"mois":"Mois","pluie_mm":"Pluie (mm)"}, text="pluie_mm",
    )
    fig3.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    monthly_mean = float(monthly_df["pluie_mm"].mean())
    fig3.add_hline(
        y=monthly_mean, line_dash="dash", line_color=CHART_COLORS["teal"],
        annotation_text=f"Moyenne : {monthly_mean:.0f} mm",
        annotation_position="top left",
    )
    fig3.update_layout(coloraxis_showscale=False, height=260,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       margin=dict(t=20,b=20,l=20,r=20), xaxis=dict(tickangle=-30))
    show_weather_chart(fig3, height=350, hovermode="closest")
else:
    st.info("Historique indisponible.")

# ── 6. CARTE ────────────────────────────────────────────────────────────────
st.subheader("🗺 Carte des secteurs LGV SEA")
if not map_df.empty:
    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=8 if selected_one == "— Toutes —" else 11,
        tiles="CartoDB positron", control_scale=True,
    )

    if selected_one != "— Toutes —":
        # Single commune: fetch Open-Meteo rain and color markers
        loc_single = map_df.dropna(subset=["latitude","longitude"])
        if not loc_single.empty:
            lat_s = round(float(loc_single["latitude"].mean()), 4)
            lon_s = round(float(loc_single["longitude"].mean()), 4)
            rain_s = load_commune_rain_ometo(lat_s, lon_s, periode)
            rain_label_s = f"{rain_s:.1f} mm" if not pd.isna(rain_s) else "N/A"
            col_s = rain_color_mm(rain_s) if not pd.isna(rain_s) else "#9ca3af"
        for _, row in map_df.dropna(subset=["latitude","longitude"]).iterrows():
            folium.CircleMarker(
                [float(row["latitude"]), float(row["longitude"])],
                radius=7, color=col_s, fill=True, fill_opacity=0.85, weight=1.5,
                tooltip=f"{row.get('commune_name','')} PK {row.get('pk_km','')} km — {rain_label_s} ({periode})",
                popup=folium.Popup(
                    f"<b>{row.get('commune_name','')} — PK {row.get('pk_km','')} km</b><br>"
                    f"Cumul {periode} : <b>{rain_label_s}</b><br>"
                    f"<small>Source : Open-Meteo AROME 1,3 km</small>", max_width=250),
            ).add_to(m)
    else:
        # All communes: color by nearest dept's forecast (no per-sector API calls)
        for _, row in map_df.dropna(subset=["latitude","longitude"]).iterrows():
            rlat = float(row["latitude"]); rlon = float(row["longitude"])
            dep  = nearest_dep(rlat, rlon)
            d    = dep_rain_data[dep]
            folium.CircleMarker(
                [rlat, rlon], radius=5,
                color=d["color"], fill=True, fill_opacity=0.75, weight=1.2,
                tooltip=(f"{row.get('commune_name','')} (PK {row.get('pk_km','')}) — "
                         f"Dép.{dep} : {d['total']:.0f} mm prévu 7j"),
                popup=folium.Popup(
                    f"<b>{row.get('commune_name','')} — PK {row.get('pk_km','')} km</b><br>"
                    f"Dép. {dep} — {d['total']:.0f} mm prévu sur 7j<br>"
                    f"Max journalier : {d['max']:.0f} mm/j<br>"
                    f"<small>Source : Open-Meteo prévision 7j</small>", max_width=260),
            ).add_to(m)

    for seg in (snapshot.get("lgv_lines") or []):
        if isinstance(seg, list):
            pts = []
            for p in seg:
                if isinstance(p, dict) and "lat" in p and "lon" in p:
                    pts.append([p["lat"], p["lon"]])
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append([p[0], p[1]])
            if pts:
                folium.PolyLine(pts, color="#cc0000", weight=2.5, opacity=0.7).add_to(m)

    for a in firms_alerts:
        folium.Marker(
            [a["lat"], a["lon"]],
            icon=folium.Icon(color="red", icon="fire", prefix="fa"),
            tooltip=f"🔥 PK {a['pk_km']} km — {a['distance_m']} m de la LGV SEA",
            popup=folium.Popup(
                f"<b>🔥 Détection FIRMS</b><br>"
                f"PK {a['pk_km']} km — {a['distance_m']} m de la LGV SEA<br>"
                f"{a['date']} {a['heure']} UTC · confiance {a['confidence']}<br>"
                f"Satellite : {a['satellite']}", max_width=250),
        ).add_to(m)

    st_folium(m, use_container_width=True, height=450, returned_objects=[])
    if selected_one == "— Toutes —":
        st.caption("Couleur = prévision pluie 7j par département (Open-Meteo). "
                   "Sélectionner une commune pour voir son cumul mesuré.")
else:
    st.info("Pas de données de localisation.")

# ── 7. TABLEAU ──────────────────────────────────────────────────────────────
st.subheader("📋 Secteurs LGV SEA")
show_cols = [c for c in ["commune_name","pk_km"] if c in comm_df.columns]
disp = comm_df[show_cols].rename(columns={"commune_name":"Commune","pk_km":"PK (km)"})

if selected_one != "— Toutes —" and not disp.empty:
    # Add Open-Meteo rain for the selected commune
    loc_t = comm_df.dropna(subset=["latitude","longitude"]) if "latitude" in comm_df.columns else pd.DataFrame()
    if not loc_t.empty:
        lat_t = round(float(loc_t["latitude"].mean()), 4)
        lon_t = round(float(loc_t["longitude"].mean()), 4)
        rain_t = load_commune_rain_ometo(lat_t, lon_t, periode)
        disp[f"Cumul {periode} (mm)"] = rain_t if not pd.isna(rain_t) else None
        _src_lbl = "AROME 1,3 km" if periode == "24h" else "ERA5 near real-time"
        st.caption(f"Pluie : Open-Meteo {_src_lbl} · cumul {periode}")
elif not disp.empty:
    st.caption("ℹ️ Voir **Comparaison communes** ci-dessus pour les données pluvio fiables (Open-Meteo).")

if not disp.empty:
    st.dataframe(disp.sort_values("PK (km)") if "PK (km)" in disp.columns else disp,
                 use_container_width=True, hide_index=True, height=300)
utilise la bonne api vigicruefrom __future__ import annotations

import io
import math
import os
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# NASA FIRMS (Fire Information for Resource Management System) — détections satellite quasi temps réel
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{day_range}/{date}"
FIRMS_SOURCES  = ["VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"]  # résolution ~375 m
FIRMS_BBOX     = "-0.7,44.75,1.0,47.5"  # west,south,east,north — corridor LGV SEA + marge
FIRMS_RADIUS_KM = 0.5
FIRMS_MAX_DAY_RANGE = 10   # limite de l'API area (une seule requête par source)
FIRMS_MAX_LOOKBACK_DAYS = 60  # au-delà, le NRT n'est plus garanti disponible

# Vigilance météo officielle Météo-France, republiée en open data (sans clé) sur Opendatasoft —
# mêmes bulletins que vigilance.meteofrance.fr, source de référence utilisée par SNCF/préfectures.
MF_VIGILANCE_URL     = "https://public.opendatasoft.com/api/records/1.0/search/"
MF_VIGILANCE_DATASET = "weatherref-france-vigilance-meteo-departement"
MF_PHENOMENON_LABELS = {
    "pluie":              ("🌧️", "Pluie-inondation"),
    "orages":             ("⛈️", "Orages"),
    "canicule":           ("🌡️", "Canicule"),
    "vent":               ("💨", "Vent violent"),
    "neige / verglas":    ("❄️", "Neige-verglas"),
    "vagues submersion":  ("🌊", "Vagues-submersion"),
}
MF_COLOR_TO_LEVEL = {"vert": "VERT", "jaune": "JAUNE", "orange": "ORANGE", "rouge": "ROUGE"}

DEPS = {
    "37": {"nom": "Indre-et-Loire",   "lat": 47.38, "lon":  0.69},
    "86": {"nom": "Vienne",            "lat": 46.58, "lon":  0.34},
    "79": {"nom": "Deux-Sèvres",       "lat": 46.32, "lon": -0.46},
    "16": {"nom": "Charente",           "lat": 45.65, "lon":  0.16},
    "17": {"nom": "Charente-Maritime", "lat": 45.75, "lon": -0.63},
    "33": {"nom": "Gironde",            "lat": 44.84, "lon": -0.58},
}

ALERT_CFG = {
    "ORAGE":      ("⛈️",  "Orage"),
    "CANICULE":   ("🌡️",  "Canicule"),
    "INCENDIE":   ("🔥",  "Incendie"),
    "INONDATION": ("🌊",  "Inondation"),
    "VENT":       ("💨",  "Vent violent"),
    "VIGICRUE":   ("🏞️",  "Vigilance crue"),
    "FEU_FIRMS":  ("🔥",  "Détection incendie FIRMS"),
    "MF_VIGILANCE": ("🛡️", "Vigilance officielle Météo-France"),
}
LEVEL_COLOR = {"ROUGE":"#dc2626","ORANGE":"#ea580c","JAUNE":"#eab308","VERT":"#16a34a","INFO":"#3b82f6"}
LEVEL_RANK  = {"ROUGE":4,"ORANGE":3,"JAUNE":2,"VERT":1,"INFO":0}
LEVEL_LABEL = {"ROUGE":"Rouge","ORANGE":"Orange","JAUNE":"Jaune","VERT":"Vert","INFO":"Info"}
# st.badge only accepts this fixed palette — map our 5 severity levels onto it.
LEVEL_BADGE = {"ROUGE":"red","ORANGE":"orange","JAUNE":"yellow","VERT":"green","INFO":"gray"}


def rain_risk(max_mm: float):
    if max_mm >= 60: return "ROUGE",  "#dc2626", "🔴"
    if max_mm >= 30: return "ORANGE", "#ea580c", "🟠"
    if max_mm >= 10: return "JAUNE",  "#eab308", "🟡"
    return               "VERT",   "#16a34a", "🟢"

def rain_color_mm(mm: float) -> str:
    if mm >= 60: return "#dc2626"
    if mm >= 30: return "#ea580c"
    if mm >= 10: return "#3b82f6"
    return "#93c5fd"


@st.cache_data(ttl=900)
def _fetch_snapshot_raw() -> dict:
    r = requests.get(SNAPSHOT_URL, timeout=20)
    r.raise_for_status()
    return r.json()


def load_snapshot() -> dict:
    """Seules les réponses réussies sont mises en cache (via _fetch_snapshot_raw) :
    un échec réseau ponctuel n'immobilise pas l'appli 15 min, il est retenté au
    prochain rerun au lieu de rester figé jusqu'à expiration du TTL."""
    try:
        return _fetch_snapshot_raw()
    except Exception as e:
        return {"_error": str(e)}


@st.cache_data(ttl=1800)
def _fetch_mf_vigilance_raw(dep_codes: tuple) -> list:
    q = " OR ".join(f"domain_id:{d}" for d in dep_codes)
    r = requests.get(MF_VIGILANCE_URL, params={
        "dataset": MF_VIGILANCE_DATASET, "q": q, "rows": 100,
    }, timeout=15)
    r.raise_for_status()
    return r.json().get("records", [])


def load_meteofrance_vigilance() -> tuple[list, bool]:
    """Vigilance officielle Météo-France (aujourd'hui J / demain J1) par département,
    republiée en open data sans clé — mêmes bulletins que vigilance.meteofrance.fr.
    Retourne (alertes non-vertes, ok). ok=False seulement si l'API n'a pas pu être
    interrogée du tout (à ne pas confondre avec une vigilance verte légitime)."""
    try:
        records = _fetch_mf_vigilance_raw(tuple(sorted(DEPS.keys())))
    except Exception:
        return [], False

    alerts = []
    for rec in records:
        f = rec.get("fields", {})
        dep = f.get("domain_id")
        if dep not in DEPS:
            continue
        level = MF_COLOR_TO_LEVEL.get((f.get("color") or "").lower())
        if not level or level == "VERT":
            continue
        phen = f.get("phenomenon", "")
        icon, label = MF_PHENOMENON_LABELS.get(phen, ("", phen.capitalize() or "Phénomène"))
        jour = "aujourd'hui" if f.get("echeance") == "J" else "demain"
        alerts.append(dict(
            dep=dep, type="MF_VIGILANCE", level=level, phenomenon=phen, icon=icon,
            msg=f"Dép.{dep} ({DEPS[dep]['nom']}) — {label} : vigilance {level.lower()} ({jour})",
        ))

    alerts.sort(key=lambda a: (-LEVEL_RANK.get(a["level"], 0), a["dep"]))
    return alerts, True


@st.cache_data(ttl=1800)
def _fetch_dept_forecast_raw(dep: str) -> dict:
    info = DEPS[dep]
    r = requests.get(FORECAST_URL, params={
        "latitude": info["lat"], "longitude": info["lon"],
        "daily": ("precipitation_sum,temperature_2m_max,"
                  "weathercode,wind_speed_10m_max"),
        "forecast_days": 7, "timezone": "Europe/Paris",
    }, timeout=15)
    r.raise_for_status()
    return r.json().get("daily", {})


def load_weather_alerts_all() -> tuple[list, int, int]:
    """Alertes dérivées des prévisions Open-Meteo pour les 6 départements LGV SEA.
    Retourne (alertes, nb_dept_ok, nb_dept_total) : si nb_dept_ok == 0, l'appelant
    doit afficher un avertissement plutôt qu'un silencieux "aucune alerte"
    (chaque département étant récupéré via une fonction cachée séparée, l'échec
    de l'un n'empêche pas les autres d'être servis depuis leur propre cache)."""
    alerts = []
    ok_count = 0
    for dep in DEPS:
        try:
            daily = _fetch_dept_forecast_raw(dep)
            ok_count += 1
        except Exception:
            continue

        dates   = daily.get("time", [])
        precips = daily.get("precipitation_sum",        [0]*7)
        tmaxes  = daily.get("temperature_2m_max",       [0]*7)
        wcodes  = daily.get("weathercode",              [0]*7)
        winds   = daily.get("wind_speed_10m_max",       [0]*7)
        rain7   = sum(p or 0 for p in precips)

        seen_fire = False
        for i, date in enumerate(dates):
            p = precips[i] or 0
            t = tmaxes[i]  or 0
            w = wcodes[i]  or 0
            v = winds[i]   or 0
            d_str = date[5:]  # MM-DD

            # ⛈️ Orages (WMO codes 80-82 averses, 95-99 orages)
            if w >= 99:
                alerts.append(dict(dep=dep, date=date, type="ORAGE", level="ROUGE",
                    msg=f"Dép.{dep} le {d_str} — Orages violents avec grêle"))
            elif w >= 95:
                alerts.append(dict(dep=dep, date=date, type="ORAGE", level="ORANGE",
                    msg=f"Dép.{dep} le {d_str} — Orages"))
            elif w in (80, 81, 82):
                alerts.append(dict(dep=dep, date=date, type="ORAGE", level="JAUNE",
                    msg=f"Dép.{dep} le {d_str} — Averses orageuses"))

            # 🌊 Inondation (précipitations intenses)
            if p >= 60:
                alerts.append(dict(dep=dep, date=date, type="INONDATION", level="ROUGE",
                    msg=f"Dép.{dep} le {d_str} — Pluies diluviennes : {p:.0f} mm"))
            elif p >= 30:
                alerts.append(dict(dep=dep, date=date, type="INONDATION", level="ORANGE",
                    msg=f"Dép.{dep} le {d_str} — Pluies intenses : {p:.0f} mm"))
            elif p >= 15:
                alerts.append(dict(dep=dep, date=date, type="INONDATION", level="JAUNE",
                    msg=f"Dép.{dep} le {d_str} — Pluies soutenues : {p:.0f} mm"))

            # 🌡️ Canicule
            if t >= 40:
                alerts.append(dict(dep=dep, date=date, type="CANICULE", level="ROUGE",
                    msg=f"Dép.{dep} le {d_str} — Canicule extrême : {t:.0f}°C"))
            elif t >= 36:
                alerts.append(dict(dep=dep, date=date, type="CANICULE", level="ROUGE",
                    msg=f"Dép.{dep} le {d_str} — Canicule : {t:.0f}°C"))
            elif t >= 33:
                alerts.append(dict(dep=dep, date=date, type="CANICULE", level="ORANGE",
                    msg=f"Dép.{dep} le {d_str} — Forte chaleur : {t:.0f}°C"))

            # 💨 Vent violent
            if v >= 100:
                alerts.append(dict(dep=dep, date=date, type="VENT", level="ROUGE",
                    msg=f"Dép.{dep} le {d_str} — Vents très violents : {v:.0f} km/h"))
            elif v >= 80:
                alerts.append(dict(dep=dep, date=date, type="VENT", level="ORANGE",
                    msg=f"Dép.{dep} le {d_str} — Vents violents : {v:.0f} km/h"))
            elif v >= 60:
                alerts.append(dict(dep=dep, date=date, type="VENT", level="JAUNE",
                    msg=f"Dép.{dep} le {d_str} — Vents forts : {v:.0f} km/h"))

            # 🔥 Risque incendie (chaleur + sécheresse + vent)
            if not seen_fire and t >= 30 and rain7 < 10 and v >= 25:
                lvl = "ROUGE" if (t >= 35 and rain7 < 5 and v >= 35) else "ORANGE"
                alerts.append(dict(dep=dep, date=date, type="INCENDIE", level=lvl,
                    msg=f"Dép.{dep} — Risque incendie : {t:.0f}°C, "
                        f"vent {v:.0f} km/h, pluie 7j {rain7:.0f} mm"))
                seen_fire = True

    alerts.sort(key=lambda x: (-LEVEL_RANK.get(x["level"], 0), x["date"]))
    return alerts, ok_count, len(DEPS)


def _normalize(s: str) -> str:
    """Lowercase + strip accents for robust name matching."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )

# Cours d'eau traversés ou longés par la LGV SEA Tours→Bordeaux
_RIVERS_RAW = [
    "vienne", "clain", "charente", "boutonne", "seugne", "touvre",
    "dronne", "isle", "dordogne", "garonne", "thouet", "sevre",
    "indre", "cher", "creuse", "ciron", "jalles", "estey",
    "leyre", "midouze", "brion", "anglin",
]
RIVERS_LGV = [_normalize(r) for r in _RIVERS_RAW]

# Departments along LGV SEA (filter unrelated rivers with same name elsewhere)
_DEP_OK = {"37","86","79","16","17","33","24","47","40","49","85","36"}


VIGICRUES_GEOJSON_URL = "https://www.vigicrues.gouv.fr/services/vigicrues.geojson/"
VIGICRUES_HEADERS = {
    "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.1",
    "User-Agent": "LGV-PluvioStations/1.1 (https://lgvpluviostations.streamlit.app/)",
}


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_vigicrues_geojson_raw() -> dict:
    """Télécharge le flux public Vigicrues. Aucune clé API n'est requise."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                VIGICRUES_GEOJSON_URL,
                headers=VIGICRUES_HEADERS,
                timeout=(8, 35),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Réponse Vigicrues inattendue.")
            features = payload.get("features")
            if not isinstance(features, list):
                raise RuntimeError("Le flux Vigicrues ne contient pas de liste 'features'.")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < 2:
                import time
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Vigicrues indisponible après 3 tentatives : {last_error}")


def load_vigicrue_rivers() -> tuple[list, bool]:
    """Retourne les vigilances des tronçons liés au corridor LGV SEA.

    Le flux GeoJSON officiel contient directement le nom du tronçon et le niveau
    ``NivSituVigiCruEnt`` (1 vert, 2 jaune, 3 orange, 4 rouge). Une réponse valide
    sans tronçon correspondant reste un contrôle réussi. Aucune clé API Vigicrues
    n'est nécessaire.
    """
    try:
        payload = _fetch_vigicrues_geojson_raw()
    except Exception:
        return [], False

    level_by_number = {1: "VERT", 2: "JAUNE", 3: "ORANGE", 4: "ROUGE"}
    results: list[dict] = []

    for feature in payload.get("features", []):
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        if not isinstance(props, dict):
            continue

        name = (
            props.get("NomEntVigiCru")
            or props.get("LbEntVigiCru")
            or props.get("name")
            or props.get("nom")
        )
        if not name:
            continue

        name = str(name).strip()
        name_norm = _normalize(name)
        if not any(river in name_norm for river in RIVERS_LGV):
            continue

        raw_level = (
            props.get("NivSituVigiCruEnt")
            or props.get("NivVigiCru")
            or props.get("niveau")
        )
        try:
            level = level_by_number.get(int(float(raw_level)))
        except (TypeError, ValueError):
            level = None
        if level is None:
            continue

        code = str(props.get("CdEntVigiCru") or "").strip()
        results.append({
            "riviere": name,
            "dep": "",
            "code": code,
            "level": level,
            "type": "VIGICRUE",
            "msg": f"{name} — vigilance {level.lower()}",
        })

    seen: set[tuple[str, str]] = set()
    dedup: list[dict] = []
    for item in results:
        key = (item.get("code") or _normalize(item["riviere"]), item["level"])
        if key not in seen:
            seen.add(key)
            dedup.append(item)

    dedup.sort(key=lambda x: (-LEVEL_RANK.get(x["level"], 0), x["riviere"]))
    return dedup, True


def get_firms_map_key() -> str | None:
    """Clé API FIRMS : st.secrets['FIRMS_MAP_KEY'] ou variable d'env FIRMS_MAP_KEY.
    Clé gratuite : https://firms.modaps.eosdis.nasa.gov/api/map_key/"""
    try:
        key = st.secrets.get("FIRMS_MAP_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("FIRMS_MAP_KEY")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


@st.cache_data(ttl=3600)
def build_lgv_pk_polyline(_lgv_lines) -> list[tuple[float, float, float]]:
    """[(lat, lon, pk_km cumulé), ...] à partir du 1er tracé LGV du snapshot.
    Le cumul de distance haversine le long du tracé colle au pk_km réel (écart < 100 m)."""
    if not _lgv_lines:
        return []
    seg = _lgv_lines[0]
    pts = [(p["lat"], p["lon"]) for p in seg if isinstance(p, dict) and "lat" in p and "lon" in p]
    if len(pts) < 2:
        return []
    out = [(pts[0][0], pts[0][1], 0.0)]
    cum = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(pts, pts[1:]):
        cum += _haversine_km(lat1, lon1, lat2, lon2)
        out.append((lat2, lon2, cum))
    return out


def pk_and_distance(lat: float, lon: float, polyline: list) -> tuple[float | None, float | None]:
    """Retourne (pk_km, distance_km) du point le plus proche du tracé LGV.
    Projection sur chaque segment en repère métrique local (correction cos(latitude))."""
    if len(polyline) < 2:
        return None, None
    best_dist2 = None
    best_pk    = None
    for (lat1, lon1, pk1), (lat2, lon2, pk2) in zip(polyline, polyline[1:]):
        lat_mid = (lat1 + lat2) / 2.0
        kx = 111.320 * math.cos(math.radians(lat_mid))
        ky = 111.320
        x1, y1 = lon1 * kx, lat1 * ky
        x2, y2 = lon2 * kx, lat2 * ky
        xp, yp = lon * kx, lat * ky
        dx, dy = x2 - x1, y2 - y1
        seg_len2 = dx * dx + dy * dy
        t = 0.0 if seg_len2 == 0 else max(0.0, min(1.0, ((xp - x1) * dx + (yp - y1) * dy) / seg_len2))
        cx, cy = x1 + t * dx, y1 + t * dy
        dist2 = (xp - cx) ** 2 + (yp - cy) ** 2
        if best_dist2 is None or dist2 < best_dist2:
            best_dist2 = dist2
            best_pk = pk1 + t * (pk2 - pk1)
    return best_pk, math.sqrt(best_dist2) if best_dist2 is not None else None


@st.cache_data(ttl=300)
def _fetch_firms_source_raw(key: str, source: str, day_range: int, date_str: str) -> pd.DataFrame:
    url = FIRMS_AREA_URL.format(key=key, source=source, area=FIRMS_BBOX,
                                 day_range=day_range, date=date_str)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    txt = r.text.strip()
    if "invalid" in txt.lower()[:200]:
        raise ValueError("invalid_key")
    if not txt or "<html" in txt.lower()[:200]:
        raise RuntimeError("unexpected_response")
    df = pd.read_csv(io.StringIO(txt))
    if "latitude" not in df.columns:
        raise RuntimeError("unexpected_response")
    df["source"] = source
    return df


def load_firms_hotspots(day_range: int = 1, end_date=None) -> tuple[pd.DataFrame, str | None]:
    """Détections FIRMS (VIIRS NRT) sur la bbox du corridor LGV SEA, sur `day_range`
    jours se terminant le `end_date` (par défaut aujourd'hui — permet de remonter
    dans le temps, jusqu'à `FIRMS_MAX_LOOKBACK_DAYS` en arrière).
    Distingue explicitement « aucune détection » (chaque source a répondu, 0 résultat)
    de « échec de la vérification » (aucune source n'a pu être interrogée) — sans quoi
    une panne réseau/API afficherait à tort un statut "aucun feu" rassurant mais faux."""
    key = get_firms_map_key()
    if not key:
        return pd.DataFrame(), "missing_key"

    date_str = (end_date or datetime.now(timezone.utc).date()).isoformat()
    frames = []
    ok_count = 0
    for source in FIRMS_SOURCES:
        try:
            df = _fetch_firms_source_raw(key, source, day_range, date_str)
            ok_count += 1
            if not df.empty:
                frames.append(df)
        except ValueError:
            return pd.DataFrame(), "invalid_key"
        except Exception:
            continue

    if ok_count == 0:
        return pd.DataFrame(), "fetch_failed"
    if not frames:
        return pd.DataFrame(), None
    return pd.concat(frames, ignore_index=True), None


FIRMS_CONF_LABELS = {"l": "Faible", "n": "Nominale", "h": "Élevée"}


def _firms_conf_label(raw) -> str:
    """VIIRS renvoie un code lettre (l/n/h) peu lisible tel quel — MODIS renvoie un %
    numérique déjà clair. On ne traduit que le premier cas, l'autre passe inchangé."""
    s = str(raw).strip().lower()
    return FIRMS_CONF_LABELS.get(s, str(raw))


@st.cache_data(ttl=300)
def load_firms_alerts(_polyline: list, day_range: int = 1, end_date=None,
                       radius_km: float = FIRMS_RADIUS_KM) -> tuple[list, str | None]:
    """Détections FIRMS filtrées à moins de `radius_km` de la LGV SEA, avec PK et distance."""
    df, err = load_firms_hotspots(day_range, end_date)
    if err:
        return [], err
    if df.empty or not _polyline:
        return [], None

    alerts: list = []
    seen: set = set()
    for _, row in df.iterrows():
        try:
            lat = float(row["latitude"]); lon = float(row["longitude"])
        except Exception:
            continue
        key = (round(lat, 4), round(lon, 4))
        if key in seen:
            continue
        pk, dist_km = pk_and_distance(lat, lon, _polyline)
        if pk is None or dist_km is None or dist_km > radius_km:
            continue
        seen.add(key)

        acq_date = str(row.get("acq_date", "") or "")
        acq_time = str(row.get("acq_time", "") or "").zfill(4)
        heure    = f"{acq_time[:2]}:{acq_time[2:]}" if len(acq_time) == 4 else acq_time
        conf     = _firms_conf_label(row.get("confidence", ""))
        frp      = row.get("frp", None)
        dist_m   = dist_km * 1000

        alerts.append(dict(
            type="FEU_FIRMS", level="ROUGE", lat=lat, lon=lon,
            pk_km=round(pk, 1), distance_m=round(dist_m),
            confidence=conf, frp=frp,
            satellite=row.get("satellite", ""), source=row.get("source", ""),
            date=acq_date, heure=heure,
            msg=(f"PK {pk:.1f} km — à {dist_m:.0f} m de la LGV SEA — "
                 f"détecté le {acq_date} à {heure} UTC (confiance {conf})"),
        ))

    alerts.sort(key=lambda a: a["pk_km"])
    return alerts, None


@st.cache_data(ttl=3600)
def _fetch_commune_daily_series_raw(lat: float, lon: float, days: int) -> dict:
    end   = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days - 1)
    r = requests.get(ARCHIVE_URL, params={
        "latitude": lat, "longitude": lon,
        "start_date": str(start), "end_date": str(end),
        "daily": "precipitation_sum", "timezone": "Europe/Paris",
    }, timeout=20)
    r.raise_for_status()
    return r.json().get("daily", {})


def load_commune_daily_series(lat: float, lon: float, days: int = 30) -> pd.DataFrame:
    """Série journalière de pluie sur les `days` derniers jours — API archive ERA5
    (réanalyse), pas l'API prévision : son paramètre `past_days` renvoie pour les
    1-2 derniers jours une sortie modèle non recalée sur l'observé, constatée jusqu'à
    11x plus élevée que l'ERA5 le même jour (45,6 mm vs 4,0 mm), ce qui gonflait
    artificiellement les cumuls du classement TOP 20."""
    try:
        daily = _fetch_commune_daily_series_raw(lat, lon, days)
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame({"date": daily.get("time", []),
                        "pluie_mm": daily.get("precipitation_sum", [])})
    df["pluie_mm"] = pd.to_numeric(df["pluie_mm"], errors="coerce").fillna(0.0)
    return df


@st.cache_data(ttl=3600)
def load_all_communes_daily_rain(_sectors_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Série journalière de pluie pour chaque commune du corridor (1 requête/commune, cache 1h)."""
    if _sectors_df.empty or "commune_name" not in _sectors_df.columns:
        return pd.DataFrame()
    coords = (_sectors_df.dropna(subset=["latitude", "longitude"])
              .groupby("commune_name")[["latitude", "longitude"]].mean())
    frames = []
    for commune, row in coords.iterrows():
        df = load_commune_daily_series(round(float(row["latitude"]), 4),
                                        round(float(row["longitude"]), 4), days)
        if df.empty:
            continue
        df = df.copy()
        df["commune_name"] = commune
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=900)
def load_commune_rain_ometo(lat: float, lon: float, periode: str) -> float:
    """Cumul pluie.
    24h  → API prévision Open-Meteo, AROME Météo-France 1,3 km si dispo, sinon
           blend — la veille est un cycle déjà bouclé, pas une sortie modèle
           non recalée (vérifié : identique à l'ERA5 sur un cas testé).
    7j+  → API archive ERA5 (réanalyse), pas l'API prévision : son paramètre
           `past_days` a été mesuré jusqu'à 2x plus élevé que l'ERA5 sur le
           même point (46,4 mm vs 21,9 mm sur 7 j ; 71,1 mm vs 54,6 mm sur 30 j),
           le même défaut que celui corrigé pour le TOP 20.
    """
    today = datetime.now(timezone.utc).date()
    lat_r, lon_r = round(lat, 4), round(lon, 4)

    if periode == "24h":
        for model in ["meteofrance_arome_france", None]:
            try:
                params = {
                    "latitude": lat_r, "longitude": lon_r,
                    "daily": "precipitation_sum",
                    "past_days": 1, "forecast_days": 0,
                    "timezone": "Europe/Paris",
                }
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

    if periode == "7 jours":
        days = 7
    elif periode == "30 jours":
        days = 30
    else:
        days = today.day - 1
    if days <= 0:
        return 0.0

    start = today - timedelta(days=days)
    end   = today - timedelta(days=1)
    try:
        r = requests.get(ARCHIVE_URL, params={
            "latitude": lat_r, "longitude": lon_r,
            "start_date": str(start), "end_date": str(end),
            "daily": "precipitation_sum", "timezone": "Europe/Paris",
        }, timeout=20)
        r.raise_for_status()
        vals = r.json()["daily"]["precipitation_sum"]
        if vals:
            return round(sum(v for v in vals if v is not None), 1)
    except Exception:
        pass
    return float("nan")


@st.cache_data(ttl=3600)
def _fetch_forecast_dep_raw(dep: str) -> dict:
    d = DEPS[dep]
    r = requests.get(FORECAST_URL, params={
        "latitude": d["lat"], "longitude": d["lon"],
        "daily": "precipitation_sum,weathercode",
        "forecast_days": 7, "timezone": "Europe/Paris",
    }, timeout=15)
    r.raise_for_status()
    return r.json()


def load_forecast_dep(dep: str) -> dict:
    try:
        return _fetch_forecast_dep_raw(dep)
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def load_forecast_coord(lat: float, lon: float) -> pd.DataFrame:
    try:
        r = requests.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon,
            "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max",
            "forecast_days": 7, "timezone": "Europe/Paris",
        }, timeout=15)
        r.raise_for_status()
        daily = r.json().get("daily", {})
        return pd.DataFrame({
            "date":     daily.get("time", []),
            "pluie_mm": daily.get("precipitation_sum", []),
            "proba_%":  daily.get("precipitation_probability_max", []),
            "tmax":     daily.get("temperature_2m_max", []),
        })
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_monthly_rain(lat: float, lon: float) -> pd.DataFrame:
    end   = datetime.now(timezone.utc).date()
    start = (end.replace(day=1) - timedelta(days=365)).replace(day=1)
    try:
        r = requests.get(ARCHIVE_URL, params={
            "latitude": lat, "longitude": lon,
            "start_date": str(start), "end_date": str(end),
            "daily": "precipitation_sum", "timezone": "Europe/Paris",
        }, timeout=20)
        r.raise_for_status()
        data  = r.json()
        dates = data["daily"]["time"]
        rain  = data["daily"]["precipitation_sum"]
        monthly: dict = defaultdict(float)
        for d, v in zip(dates, rain):
            if v is not None:
                monthly[d[:7]] += v
        return pd.DataFrame([{"mois": m, "pluie_mm": round(v, 1)}
                              for m, v in sorted(monthly.items())])
    except Exception:
        return pd.DataFrame()




# Charte graphique commune aux graphiques météo/pluviométrie.
# Cette couche de présentation est indépendante du traitement NASA FIRMS.
CHART_COLORS = {
    "blue": "#2563eb", "cyan": "#0891b2", "teal": "#0f766e",
    "orange": "#f97316", "red": "#dc2626", "slate": "#475569",
}
CHART_GRID = "#e2e8f0"
CHART_TEXT = "#334155"


def style_weather_chart(fig: go.Figure, *, height: int = 340,
                        hovermode: str = "x unified") -> go.Figure:
    """Applique une présentation homogène sans modifier les données du graphe."""
    fig.update_layout(
        height=height,
        hovermode=hovermode,
        font=dict(family="Arial, sans-serif", size=12, color=CHART_TEXT),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=48, b=48, l=56, r=44),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, bgcolor="rgba(255,255,255,0.85)",
        ),
        hoverlabel=dict(bgcolor="white", font_size=12, font_color="#0f172a"),
        transition=dict(duration=250),
    )
    fig.update_xaxes(
        showgrid=False, showline=True, linecolor="#cbd5e1",
        tickcolor="#cbd5e1", automargin=True, title=None,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=CHART_GRID, gridwidth=1,
        zeroline=True, zerolinecolor="#cbd5e1", automargin=True,
    )
    return fig


def show_weather_chart(fig: go.Figure, *, height: int = 340,
                       hovermode: str = "x unified") -> None:
    """Affiche un graphe météo responsive avec une barre d'outils allégée."""
    style_weather_chart(fig, height=height, hovermode=hovermode)
    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def safe_df(records) -> pd.DataFrame:
    if isinstance(records, list) and records:
        try:
            return pd.DataFrame(records)
        except Exception:
            pass
    return pd.DataFrame()


def alert_card(a: dict):
    """Carte d'alerte plus lisible. FIRMS conserve volontairement son rendu initial."""
    lvl   = a.get("level", "")
    atype = a.get("type", "")
    icon  = a.get("icon") or ALERT_CFG.get(atype, ("", ""))[0] or None

    # Ne pas modifier la présentation ni le comportement des alertes NASA FIRMS.
    if atype == "FEU_FIRMS":
        c_badge, c_msg = st.columns([1, 7], vertical_alignment="center")
        with c_badge:
            st.badge(LEVEL_LABEL.get(lvl, lvl or "Info"), color=LEVEL_BADGE.get(lvl, "gray"))  # type: ignore[arg-type]
        with c_msg:
            st.markdown(f"{icon}  {a.get('msg','')}" if icon else a.get("msg", ""))
        return

    label = ALERT_CFG.get(atype, ("", atype.replace("_", " ").title()))[1]
    color = LEVEL_COLOR.get(lvl, LEVEL_COLOR["INFO"])
    date_label = a.get("date", "")
    dep_label = f"Dép. {a['dep']}" if a.get("dep") else ""
    meta = " · ".join(str(v) for v in (label, dep_label, date_label) if v)

    with st.container(border=True):
        c_icon, c_body, c_level = st.columns([0.45, 6.4, 1.15], vertical_alignment="center")
        with c_icon:
            st.markdown(f"<div style='font-size:1.45rem;text-align:center'>{icon or 'ℹ️'}</div>", unsafe_allow_html=True)
        with c_body:
            st.markdown(f"**{meta or label}**")
            st.caption(str(a.get("msg", "")))
        with c_level:
            st.markdown(
                f"<div style='border-left:4px solid {color};padding-left:.55rem;font-weight:700;color:{color}'>"
                f"{LEVEL_LABEL.get(lvl, lvl or 'Info')}</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="LGV SEA – Pluviométrie", page_icon="🌧", layout="wide")
st.title("🌧 LGV SEA – Pluviométrie")

col_title, col_btn = st.columns([5, 1])
col_title.caption(
    "📡 Données météo : **Open-Meteo** (modèles ERA5 + ECMWF) · "
    "Crues : **Vigicrue** · Données non officielles — "
    "pour les alertes officielles : "
    "[vigilance.meteofrance.fr](https://vigilance.meteofrance.fr/) · "
    "[vigicrues.gouv.fr](https://www.vigicrues.gouv.fr/)"
)
if col_btn.button("🔄 Rafraîchir"):
    st.cache_data.clear()
    st.rerun()

snapshot = load_snapshot()
if "_error" in snapshot:
    st.error(f"Erreur snapshot : {snapshot['_error']}")
    st.stop()

_sec       = snapshot.get("sectors")
sectors_df = safe_df(_sec.get("sectors", []) if isinstance(_sec, dict) else [])
for col in ["weather_max_24h_mm","weather_max_7d_mm","weather_max_30d_mm",
            "weather_max_month_mm","latitude","longitude","pk_km"]:
    if col in sectors_df.columns:
        sectors_df[col] = pd.to_numeric(sectors_df[col], errors="coerce")

# Pre-compute dept forecast rain (used for map coloring + dept cards)
RAIN_LABELS = {
    "VERT":   "Peu de pluie",
    "JAUNE":  "Pluie modérée",
    "ORANGE": "Pluies fortes",
    "ROUGE":  "Pluies très fortes",
    "INDETERMINE": "Indisponible",
}
dep_rain_data: dict = {}   # dep -> {max, total, lvl, color, emoji, ok}
for _dep in DEPS:
    _fc    = load_forecast_dep(_dep)
    _ok    = bool(_fc)
    _rains = [v for v in _fc.get("daily", {}).get("precipitation_sum", []) if v is not None]
    _max   = max(_rains) if _rains else 0.0
    _total = sum(_rains)
    if _ok:
        _lvl, _color, _emoji = rain_risk(_max)
    else:
        # Échec de récupération distinct d'un "peu de pluie" légitime — sans ce
        # marqueur, une panne Open-Meteo afficherait à tort une carte verte rassurante.
        _lvl, _color, _emoji = "INDETERMINE", "#9ca3af", "❓"
    dep_rain_data[_dep] = {"max": _max, "total": _total,
                            "lvl": _lvl, "color": _color, "emoji": _emoji, "ok": _ok}


def nearest_dep(lat: float, lon: float) -> str:
    """Return dep code nearest to a lat/lon (for map coloring)."""
    return min(DEPS.keys(),
               key=lambda d: (DEPS[d]["lat"] - lat) ** 2 + (DEPS[d]["lon"] - lon) ** 2)


# ── 1. PLUIE PRÉVUE PAR DÉPARTEMENT ─────────────────────────────────────────
st.subheader("Pluie prévue 7 jours par département")
st.caption("Source : Open-Meteo")
dep_cols = st.columns(len(DEPS))
for col_w, (dep, info) in zip(dep_cols, DEPS.items()):
    d = dep_rain_data[dep]
    with col_w.container(border=True):
        st.caption(f"Dép. {dep} · {info['nom']}")
        if d["ok"]:
            lvl_label = RAIN_LABELS[d["lvl"]]
            st.badge(lvl_label, color=LEVEL_BADGE.get(d["lvl"], "gray"))  # type: ignore[arg-type]
            st.metric("Cumul 7 j", f"{d['total']:.0f} mm",
                      help=f"Maximum journalier : {d['max']:.0f} mm/j")
        else:
            st.badge("Indisponible", color="gray")
            st.caption("Open-Meteo injoignable")

# ── 1bis. VIGILANCE OFFICIELLE MÉTÉO-FRANCE ─────────────────────────────────
st.subheader("🛡️ Vigilance officielle Météo-France (aujourd'hui / demain)")

mf_alerts, mf_ok = load_meteofrance_vigilance()

if not mf_ok:
    st.warning("Vigilance Météo-France injoignable — statut **non vérifié** "
               "(réessaie dans quelques minutes).")
elif mf_alerts:
    st.error(f"{len(mf_alerts)} vigilance(s) active(s) sur les départements du corridor LGV SEA.")
    for a in mf_alerts:
        alert_card(a)
else:
    st.success("Vigilance verte sur les 6 départements du corridor (aujourd'hui et demain).")

st.caption(
    "Source officielle : [vigilance.meteofrance.fr](https://vigilance.meteofrance.fr/) — "
    "données republiées en open data (sans clé), mêmes bulletins que l'appli officielle. "
    "Les « indicateurs météo » ci-dessous sont une estimation complémentaire sur 7 jours, "
    "pas une vigilance officielle."
)

st.divider()

# ── 2. INDICATEURS MÉTÉO ─────────────────────────────────────────────────────
st.subheader("📊 Indicateurs météo indicatifs — 7 prochains jours")

met_alerts, met_ok, met_total = load_weather_alerts_all()
vc_alerts, vc_ok = load_vigicrue_rivers()

active_met = [a for a in met_alerts if a["level"] in ("ROUGE","ORANGE","JAUNE")]
by_met: dict = defaultdict(list)
for a in active_met:
    by_met[a["type"]].append(a)

if met_ok == 0:
    st.warning("Open-Meteo injoignable — indicateurs météo **non vérifiés** actuellement "
               "(ce n'est pas un « aucune alerte », réessaie dans quelques minutes).")
elif met_ok < met_total:
    st.caption(f"ℹ️ Prévisions récupérées pour {met_ok}/{met_total} départements — "
               "le reste sera réessayé au prochain rafraîchissement.")

if active_met:
    badge_cols = st.columns(len(by_met))
    for col, (atype, alist) in zip(badge_cols, by_met.items()):
        worst = max(alist, key=lambda x: LEVEL_RANK.get(x["level"], 0))
        icon, label = ALERT_CFG.get(atype, ("", atype))
        col.badge(f"{label} ({len(alist)})", icon=icon or None,
                  color=LEVEL_BADGE.get(worst["level"], "gray"))  # type: ignore[arg-type]
elif met_ok > 0:
    st.success("Aucun indicateur météo significatif sur les 7 prochains jours.")

tab_labels: list = []
tab_data:   list = []
for atype in ("ORAGE", "INONDATION", "CANICULE", "INCENDIE", "VENT"):
    if atype in by_met:
        icon, label = ALERT_CFG[atype]
        tab_labels.append(f"{icon} {label} ({len(by_met[atype])})")
        tab_data.append(by_met[atype])

vc_active = [a for a in vc_alerts if a["level"] in ("ROUGE","ORANGE","JAUNE")]
if not vc_ok:
    vc_label = "🏞️ Vigicrue ❓"
elif vc_active:
    vc_label = f"🏞️ Vigicrue ⚠ {len(vc_active)}"
else:
    vc_label = "🏞️ Vigicrue ✅"
tab_labels.append(vc_label)
tab_data.append(vc_alerts)

if tab_labels:
    tabs = st.tabs(tab_labels)
    for tab, alist in zip(tabs, tab_data):
        with tab:
            if tab is tabs[-1] and not vc_ok:
                st.warning("API Vigicrue injoignable — statut crues **non vérifié** "
                           "(réessaie dans quelques minutes).")
            elif not alist:
                st.info("Aucune crue en vigilance actuellement sur ces cours d'eau."
                         if tab is tabs[-1] else "Aucune donnée.")
            for a in alist:
                alert_card(a)

st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
communes = (sorted(sectors_df["commune_name"].dropna().unique())
            if "commune_name" in sectors_df.columns else [])

# Communes par défaut : répartition géographique nord→sud sur la LGV SEA
def _find_commune(kw: str) -> str | None:
    k = unicodedata.normalize("NFD", kw.lower()).encode("ascii", "ignore").decode()
    return next((c for c in communes
                 if k in unicodedata.normalize("NFD", c.lower()).encode("ascii", "ignore").decode()), None)

_DEFAULT_KW = ["nouatre", "fontaine", "poitier", "biard", "villognon", "clerac", "ambares"]
_default_communes: list = []
for _kw in _DEFAULT_KW:
    _m = _find_commune(_kw)
    if _m and _m not in _default_communes:
        _default_communes.append(_m)
_default_communes = _default_communes[:6] or (communes[:6] if len(communes) >= 6 else communes)

with st.sidebar:
    st.subheader("📍 Communes")
    selected_multi = st.multiselect("Comparer communes", communes,
                                     default=_default_communes)
    selected_one   = st.selectbox("Commune principale", ["— Toutes —"] + list(communes))
    periode  = st.selectbox("📅 Période pluvio", ["24h","7 jours","30 jours","Mois courant"])

    st.subheader("🔥 Incendies FIRMS")
    _today_date = datetime.now(timezone.utc).date()
    firms_end_date = st.date_input(
        "Jusqu'au", value=_today_date,
        min_value=_today_date - timedelta(days=FIRMS_MAX_LOOKBACK_DAYS),
        max_value=_today_date,
        help="Choisis une date passée pour consulter l'historique FIRMS au lieu du temps réel.",
    )
    firms_day_range = st.slider("Fenêtre (jours avant cette date)", 1, FIRMS_MAX_DAY_RANGE, 1)
    firms_is_live = firms_end_date == _today_date
    if not firms_is_live:
        st.caption("📜 Mode historique — le rafraîchissement automatique ne changera rien "
                   "à une date passée, seul un rerun manuel peut en changer les données.")

    st.subheader("🔄 Actualisation")
    auto_refresh = st.checkbox("Rafraîchissement automatique", value=True)
    if auto_refresh:
        _refresh_label = st.selectbox("Intervalle", ["5 min", "10 min", "15 min"], index=1)
        refresh_seconds = {"5 min": 300, "10 min": 600, "15 min": 900}[_refresh_label]
        st.caption("⚠️ Recharge la page entière (réinitialise filtres/sélections). "
                   "La fraîcheur réelle des données reste bornée par leur propre cache "
                   "(15-60 min selon la source), pas par cet intervalle.")

if auto_refresh:
    st.markdown(f'<meta http-equiv="refresh" content="{refresh_seconds}">', unsafe_allow_html=True)

# ── 2bis. INCENDIES — NASA FIRMS ────────────────────────────────────────────
_firms_start_date = firms_end_date - timedelta(days=firms_day_range - 1)
if firms_is_live and firms_day_range <= 3:
    firms_window_label = f"dernières {firms_day_range * 24} h"
elif firms_is_live:
    firms_window_label = f"derniers {firms_day_range} jours"
elif _firms_start_date == firms_end_date:
    firms_window_label = firms_end_date.strftime('%d/%m/%Y')
else:
    firms_window_label = (f"{_firms_start_date.strftime('%d/%m/%Y')} → "
                           f"{firms_end_date.strftime('%d/%m/%Y')}")

_firms_title = ("temps réel" if firms_is_live
                else f"historique · {firms_window_label}")
st.subheader(f"🔥 Détections incendie {_firms_title} — NASA FIRMS "
             f"(rayon {FIRMS_RADIUS_KM*1000:.0f} m autour de la LGV SEA)")

lgv_polyline = build_lgv_pk_polyline(snapshot.get("lgv_lines"))
firms_alerts, firms_err = load_firms_alerts(lgv_polyline, day_range=firms_day_range,
                                             end_date=firms_end_date)

if firms_err == "missing_key":
    st.warning(
        "Clé FIRMS manquante. Crée une clé gratuite sur "
        "[firms.modaps.eosdis.nasa.gov/api/map_key](https://firms.modaps.eosdis.nasa.gov/api/map_key/), "
        "puis renseigne-la dans `.streamlit/secrets.toml` (`FIRMS_MAP_KEY = \"...\"`) "
        "ou la variable d'environnement `FIRMS_MAP_KEY`."
    )
elif firms_err == "invalid_key":
    st.error("Clé FIRMS invalide — vérifie `FIRMS_MAP_KEY`.")
elif firms_err == "fetch_failed":
    st.warning("FIRMS injoignable actuellement (réseau/API) — statut incendie **non vérifié** "
               "(ce n'est pas un « aucun feu détecté », réessaie dans quelques minutes).")
elif firms_alerts:
    st.error(f"{len(firms_alerts)} détection(s) FIRMS à moins de "
             f"{FIRMS_RADIUS_KM*1000:.0f} m de la LGV SEA ({firms_window_label}).")
    df_firms = pd.DataFrame(firms_alerts).rename(columns={
        "pk_km": "PK (km)", "distance_m": "Distance LGV (m)",
        "date": "Date", "heure": "Heure (UTC)",
        "confidence": "Confiance", "frp": "FRP (MW)", "satellite": "Satellite",
    })
    st.dataframe(
        df_firms[["PK (km)", "Distance LGV (m)", "Date", "Heure (UTC)",
                  "Confiance", "FRP (MW)", "Satellite"]],
        use_container_width=True, hide_index=True,
    )
else:
    st.success(f"Aucune détection FIRMS à moins de {FIRMS_RADIUS_KM*1000:.0f} m "
               f"de la LGV SEA ({firms_window_label}).")

_firms_freshness = (
    "Vérifié toutes les 5 min ici, mais un satellite ne survole un même point que "
    "2 fois par jour environ, avec 1 à 3 h de traitement avant publication : "
    "actualiser plus souvent ne fait pas apparaître une détection plus tôt que ça."
    if firms_is_live else
    "Requête historique figée sur la période choisie — les résultats ne changeront "
    "pas au fil des rafraîchissements."
)
st.caption(
    "Source : NASA FIRMS · VIIRS NOAA-20/NOAA-21/S-NPP (résolution ~375 m) · "
    "[firms.modaps.eosdis.nasa.gov/map](https://firms.modaps.eosdis.nasa.gov/map) — "
    "PK et distance calculés par projection sur le tracé LGV SEA. " + _firms_freshness
)

st.divider()

# ── 2ter. POINTS DE VIGILANCE — SYNTHÈSE LGV SEA ────────────────────────────
st.subheader("🧭 Points de vigilance — surveillance ligne LGV SEA")

_firms_unverified = firms_err in ("fetch_failed", "invalid_key", "missing_key")

vig_cols = st.columns(4)
vig_cols[0].badge(
    "Vigilance MF — non vérifié" if not mf_ok else f"Vigilance MF ({len(mf_alerts)})",
    icon="🛡️", color="gray" if not mf_ok else ("red" if mf_alerts else "green"))  # type: ignore[arg-type]
vig_cols[1].badge(
    "Météo — non vérifié" if met_ok == 0 else f"Météo ({len(active_met)})",
    icon="🌦️", color="gray" if met_ok == 0 else ("orange" if active_met else "green"))  # type: ignore[arg-type]
vig_cols[2].badge(
    "Vigicrue — non vérifié" if not vc_ok else f"Vigicrue ({len(vc_active)})",
    icon="🏞️", color="gray" if not vc_ok else ("blue" if vc_active else "green"))  # type: ignore[arg-type]
vig_cols[3].badge(
    "FIRMS — non vérifié" if _firms_unverified else f"FIRMS ({len(firms_alerts)})",
    icon="🔥", color="gray" if _firms_unverified else ("red" if firms_alerts else "green"))  # type: ignore[arg-type]

if mf_alerts or active_met or vc_active or firms_alerts:
    with st.expander("⚠️ Détail des alertes actives (vigilance MF, météo, crues, incendie)", expanded=True):
        for a in mf_alerts:
            alert_card(a)
        for a in sorted(active_met, key=lambda x: -LEVEL_RANK.get(x["level"], 0)):
            alert_card(a)
        for a in vc_active:
            alert_card(a)
        for a in firms_alerts:
            alert_card(a)
elif not mf_ok or met_ok == 0 or not vc_ok or _firms_unverified:
    st.warning("Au moins une source (vigilance MF, météo, crues ou incendie) n'a pas pu être vérifiée "
               "— voir le détail ci-dessus. Ne pas interpréter comme « aucune alerte ».")
else:
    st.success("Aucune alerte météo, crue ou incendie active actuellement.")

st.divider()

# ── 2quater. TOP 20 COMMUNES — CUMUL DE PRÉCIPITATION ───────────────────────
st.subheader("🌧 TOP 20 communes — cumul de précipitation le plus élevé (30 derniers jours)")

with st.spinner("Chargement pluvio 30 jours pour toutes les communes du corridor…"):
    all_rain_df = load_all_communes_daily_rain(sectors_df, days=30)

_n_communes_total = (sectors_df["commune_name"].dropna().nunique()
                      if "commune_name" in sectors_df.columns else 0)
_n_communes_ok = all_rain_df["commune_name"].nunique() if not all_rain_df.empty else 0

if all_rain_df.empty:
    st.info("Données pluvio indisponibles pour le classement.")
else:
    if _n_communes_total and _n_communes_ok < _n_communes_total:
        st.caption(f"⚠️ Données récupérées pour {_n_communes_ok}/{_n_communes_total} communes du corridor "
                   "— le classement ci-dessous ne porte que sur les communes disponibles.")
    all_rain_df = all_rain_df.copy()
    all_rain_df["date"] = pd.to_datetime(all_rain_df["date"]).dt.date
    d_min, d_max = all_rain_df["date"].min(), all_rain_df["date"].max()

    date_range = st.slider(
        "📅 Filtrer par période (filtre appliqué sur les données déjà chargées — aucun rechargement)",
        min_value=d_min, max_value=d_max, value=(d_min, d_max), format="DD/MM",
    )
    filtered_rain = all_rain_df[(all_rain_df["date"] >= date_range[0]) &
                                 (all_rain_df["date"] <= date_range[1])]

    totals = (filtered_rain.groupby("commune_name")["pluie_mm"].sum()
              .sort_values(ascending=False).head(20))

    if totals.empty:
        st.info("Aucune donnée sur la période sélectionnée.")
    else:
        top20_communes = totals.index.tolist()

        bar_df = totals.sort_values(ascending=True).reset_index()
        bar_df["color"] = bar_df["pluie_mm"].apply(rain_color_mm)
        fig_top = go.Figure(go.Bar(
            x=bar_df["pluie_mm"], y=bar_df["commune_name"], orientation="h",
            marker_color=bar_df["color"].tolist(),
            text=bar_df["pluie_mm"].apply(lambda v: f"{v:.0f} mm"),
            textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Cumul : %{x:.1f} mm<extra></extra>",
        ))
        fig_top.update_layout(
            height=max(320, len(bar_df) * 26 + 60),
            xaxis=dict(title="Cumul pluie (mm)", zeroline=True),
            yaxis=dict(tickfont=dict(size=11)),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=30, l=10, r=90),
        )
        show_weather_chart(fig_top, height=420, hovermode="closest")

        # Courbes journalières : top 3 en couleur (identité), le reste du TOP 20 en gris (contexte)
        HIGHLIGHT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
        fig_curve = go.Figure()
        for commune in reversed(top20_communes):
            s = filtered_rain[filtered_rain["commune_name"] == commune].sort_values("date")
            rank = top20_communes.index(commune)
            is_top3 = rank < 3
            fig_curve.add_scatter(
                x=s["date"], y=s["pluie_mm"], mode="lines", name=commune,
                line=dict(color=HIGHLIGHT_COLORS[rank] if is_top3 else "#c3c2b7",
                          width=2.5 if is_top3 else 1),
                opacity=1.0 if is_top3 else 0.45,
                showlegend=is_top3,
                hovertemplate=f"<b>{commune}</b><br>%{{x|%d/%m}} : %{{y:.1f}} mm<extra></extra>",
            )
        fig_curve.update_layout(
            height=340, xaxis=dict(title=""), yaxis=dict(title="Pluie/jour (mm)"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.15),
            margin=dict(t=35, b=20, l=20, r=20),
        )
        show_weather_chart(fig_curve, height=390)
        st.caption("Courbes : les 3 communes les plus arrosées de ce TOP 20 sont mises en évidence "
                   "(couleur + légende) · les 17 autres apparaissent en gris clair pour le contexte.")

        with st.expander("📋 Table du TOP 20"):
            st.dataframe(
                totals.reset_index().rename(columns={"commune_name": "Commune", "pluie_mm": "Cumul (mm)"}),
                use_container_width=True, hide_index=True,
            )
    st.caption("Source : Open-Meteo ERA5 (réanalyse, série journalière 30 jours) · "
               "une requête par commune, mise en cache 1h.")

st.divider()

# ── 3. COMPARAISON COMMUNES ─────────────────────────────────────────────────
if selected_multi and not sectors_df.empty:
    _today = datetime.now(timezone.utc).date()
    if periode == "24h":
        _titre_periode = f"hier {(_today - timedelta(days=1)).strftime('%d/%m/%Y')}"
    elif periode == "7 jours":
        _titre_periode = f"7 derniers jours (jusqu'au {(_today - timedelta(days=1)).strftime('%d/%m/%Y')})"
    elif periode == "30 jours":
        _titre_periode = f"30 derniers jours (jusqu'au {(_today - timedelta(days=1)).strftime('%d/%m/%Y')})"
    else:
        _titre_periode = f"mois de {_today.strftime('%B %Y')} (jusqu'au {(_today - timedelta(days=1)).strftime('%d/%m')})"
    st.subheader(f"📊 Comparaison communes — cumul pluie {_titre_periode}")

    if len(selected_multi) > 12:
        st.warning("Sélectionne 12 communes max pour la comparaison.")
    else:
        rows = []
        with st.spinner("Chargement données pluvio Open-Meteo…"):
            for commune in selected_multi:
                loc = (sectors_df[sectors_df["commune_name"] == commune]
                       .dropna(subset=["latitude","longitude"]))
                if loc.empty:
                    continue
                lat = round(float(loc["latitude"].mean()), 4)
                lon = round(float(loc["longitude"].mean()), 4)
                rain = load_commune_rain_ometo(lat, lon, periode)
                rows.append({"commune_name": commune, "rain_mm": rain})

        df_cmp = (pd.DataFrame(rows)
                  .dropna(subset=["rain_mm"])
                  .sort_values("rain_mm", ascending=True))

        if df_cmp.empty:
            st.info("Pas de données pour les communes sélectionnées.")
        else:
            today = datetime.now(timezone.utc).date()
            if periode == "24h":
                date_str = f"hier {(today - timedelta(days=1)).strftime('%d/%m/%Y')}"
            elif periode == "7 jours":
                date_str = f"{(today - timedelta(days=7)).strftime('%d/%m')} → {(today - timedelta(days=1)).strftime('%d/%m/%Y')}"
            elif periode == "30 jours":
                date_str = f"{(today - timedelta(days=30)).strftime('%d/%m')} → {(today - timedelta(days=1)).strftime('%d/%m/%Y')}"
            else:
                date_str = f"1er → {(today - timedelta(days=1)).strftime('%d/%m/%Y')}"

            df_cmp["label"] = df_cmp["rain_mm"].apply(
                lambda v: "Sec" if v == 0 else f"{v:.1f} mm")
            df_cmp["color"] = df_cmp["rain_mm"].apply(
                lambda v: "#d1d5db" if v == 0 else rain_color_mm(v))

            fig = go.Figure(go.Bar(
                x=df_cmp["rain_mm"],
                y=df_cmp["commune_name"],
                orientation="h",
                marker_color=df_cmp["color"].tolist(),
                text=df_cmp["label"],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Cumul : %{x:.1f} mm<extra></extra>",
            ))
            height = max(260, len(df_cmp) * 34 + 60)
            fig.update_layout(
                xaxis=dict(title=f"Cumul pluie (mm)", zeroline=True),
                yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
                height=height,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=30, l=10, r=90),
            )
            show_weather_chart(fig, height=360, hovermode="closest")
            _src = "AROME Météo-France 1,3 km" if periode == "24h" else "ERA5 near real-time"
            st.caption(f"Source : Open-Meteo {_src} · {date_str}")

            _missing = sorted(set(selected_multi) - set(df_cmp["commune_name"]))
            if _missing:
                st.warning(f"Pas de donnée pluvio récupérée pour : {', '.join(_missing)} "
                           "— absentes du graphique ci-dessus, pas de valeur à 0 supposée.")

# ── 4. PRÉVISIONS 7J ────────────────────────────────────────────────────────
comm_df = (sectors_df if selected_one == "— Toutes —"
           else sectors_df[sectors_df["commune_name"] == selected_one])
map_df = (comm_df.dropna(subset=["latitude","longitude"])
          if "latitude" in sectors_df.columns and "longitude" in sectors_df.columns
          else pd.DataFrame())
lat_c = float(map_df["latitude"].mean())  if not map_df.empty else 46.2
lon_c = float(map_df["longitude"].mean()) if not map_df.empty else 0.2

label_loc = "LGV SEA" if selected_one == "— Toutes —" else selected_one
st.subheader(f"🔮 Prévisions 7 jours — {label_loc}")
fc_df = load_forecast_coord(lat_c, lon_c)
if not fc_df.empty:
    fc_df["pluie_mm"] = pd.to_numeric(fc_df["pluie_mm"], errors="coerce").fillna(0)
    fc_df["tmax"]     = pd.to_numeric(fc_df["tmax"],     errors="coerce").fillna(0)
    fc_df["color"]    = fc_df["pluie_mm"].apply(rain_color_mm)
    fig2 = go.Figure()
    fig2.add_bar(x=fc_df["date"], y=fc_df["pluie_mm"],
                 marker_color=fc_df["color"].tolist(),
                 text=fc_df["pluie_mm"].apply(lambda v: f"{v:.0f}"),
                 textposition="outside", name="Pluie (mm)")
    if "proba_%" in fc_df.columns:
        fig2.add_scatter(x=fc_df["date"], y=fc_df["proba_%"],
                         mode="lines+markers", name="Proba pluie %",
                         yaxis="y2", line=dict(color="#6366f1", dash="dot"),
                         marker=dict(size=5))
    if "tmax" in fc_df.columns:
        fig2.add_scatter(x=fc_df["date"], y=fc_df["tmax"],
                         mode="lines+markers", name="T° max (°C)",
                         yaxis="y3", line=dict(color="#f97316", width=2),
                         marker=dict(size=5, symbol="diamond"))
    fig2.update_layout(
        yaxis=dict(title="Pluie (mm)", side="left"),
        yaxis2=dict(title="Proba %",   overlaying="y", side="right",
                    range=[0, 110], showgrid=False),
        yaxis3=dict(title="T°C",       overlaying="y", side="right",
                    position=0.92,     showgrid=False, anchor="free"),
        legend=dict(orientation="h", y=1.12), height=290,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=20, l=20, r=80), xaxis=dict(tickangle=-20),
    )
    show_weather_chart(fig2, height=380)
else:
    st.info("Prévisions indisponibles.")

# ── 5. HISTORIQUE MENSUEL ───────────────────────────────────────────────────
st.subheader("📅 Historique mensuel — 12 mois")
monthly_df = load_monthly_rain(lat_c, lon_c)
if not monthly_df.empty:
    fig3 = px.bar(
        monthly_df, x="mois", y="pluie_mm", color="pluie_mm",
        color_continuous_scale=["#bfdbfe","#3b82f6","#1d4ed8","#1e3a8a"],
        labels={"mois":"Mois","pluie_mm":"Pluie (mm)"}, text="pluie_mm",
    )
    fig3.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    monthly_mean = float(monthly_df["pluie_mm"].mean())
    fig3.add_hline(
        y=monthly_mean, line_dash="dash", line_color=CHART_COLORS["teal"],
        annotation_text=f"Moyenne : {monthly_mean:.0f} mm",
        annotation_position="top left",
    )
    fig3.update_layout(coloraxis_showscale=False, height=260,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       margin=dict(t=20,b=20,l=20,r=20), xaxis=dict(tickangle=-30))
    show_weather_chart(fig3, height=350, hovermode="closest")
else:
    st.info("Historique indisponible.")

# ── 6. CARTE ────────────────────────────────────────────────────────────────
st.subheader("🗺 Carte des secteurs LGV SEA")
if not map_df.empty:
    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=8 if selected_one == "— Toutes —" else 11,
        tiles="CartoDB positron", control_scale=True,
    )

    if selected_one != "— Toutes —":
        # Single commune: fetch Open-Meteo rain and color markers
        loc_single = map_df.dropna(subset=["latitude","longitude"])
        if not loc_single.empty:
            lat_s = round(float(loc_single["latitude"].mean()), 4)
            lon_s = round(float(loc_single["longitude"].mean()), 4)
            rain_s = load_commune_rain_ometo(lat_s, lon_s, periode)
            rain_label_s = f"{rain_s:.1f} mm" if not pd.isna(rain_s) else "N/A"
            col_s = rain_color_mm(rain_s) if not pd.isna(rain_s) else "#9ca3af"
        for _, row in map_df.dropna(subset=["latitude","longitude"]).iterrows():
            folium.CircleMarker(
                [float(row["latitude"]), float(row["longitude"])],
                radius=7, color=col_s, fill=True, fill_opacity=0.85, weight=1.5,
                tooltip=f"{row.get('commune_name','')} PK {row.get('pk_km','')} km — {rain_label_s} ({periode})",
                popup=folium.Popup(
                    f"<b>{row.get('commune_name','')} — PK {row.get('pk_km','')} km</b><br>"
                    f"Cumul {periode} : <b>{rain_label_s}</b><br>"
                    f"<small>Source : Open-Meteo AROME 1,3 km</small>", max_width=250),
            ).add_to(m)
    else:
        # All communes: color by nearest dept's forecast (no per-sector API calls)
        for _, row in map_df.dropna(subset=["latitude","longitude"]).iterrows():
            rlat = float(row["latitude"]); rlon = float(row["longitude"])
            dep  = nearest_dep(rlat, rlon)
            d    = dep_rain_data[dep]
            folium.CircleMarker(
                [rlat, rlon], radius=5,
                color=d["color"], fill=True, fill_opacity=0.75, weight=1.2,
                tooltip=(f"{row.get('commune_name','')} (PK {row.get('pk_km','')}) — "
                         f"Dép.{dep} : {d['total']:.0f} mm prévu 7j"),
                popup=folium.Popup(
                    f"<b>{row.get('commune_name','')} — PK {row.get('pk_km','')} km</b><br>"
                    f"Dép. {dep} — {d['total']:.0f} mm prévu sur 7j<br>"
                    f"Max journalier : {d['max']:.0f} mm/j<br>"
                    f"<small>Source : Open-Meteo prévision 7j</small>", max_width=260),
            ).add_to(m)

    for seg in (snapshot.get("lgv_lines") or []):
        if isinstance(seg, list):
            pts = []
            for p in seg:
                if isinstance(p, dict) and "lat" in p and "lon" in p:
                    pts.append([p["lat"], p["lon"]])
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append([p[0], p[1]])
            if pts:
                folium.PolyLine(pts, color="#cc0000", weight=2.5, opacity=0.7).add_to(m)

    for a in firms_alerts:
        folium.Marker(
            [a["lat"], a["lon"]],
            icon=folium.Icon(color="red", icon="fire", prefix="fa"),
            tooltip=f"🔥 PK {a['pk_km']} km — {a['distance_m']} m de la LGV SEA",
            popup=folium.Popup(
                f"<b>🔥 Détection FIRMS</b><br>"
                f"PK {a['pk_km']} km — {a['distance_m']} m de la LGV SEA<br>"
                f"{a['date']} {a['heure']} UTC · confiance {a['confidence']}<br>"
                f"Satellite : {a['satellite']}", max_width=250),
        ).add_to(m)

    st_folium(m, use_container_width=True, height=450, returned_objects=[])
    if selected_one == "— Toutes —":
        st.caption("Couleur = prévision pluie 7j par département (Open-Meteo). "
                   "Sélectionner une commune pour voir son cumul mesuré.")
else:
    st.info("Pas de données de localisation.")

# ── 7. TABLEAU ──────────────────────────────────────────────────────────────
st.subheader("📋 Secteurs LGV SEA")
show_cols = [c for c in ["commune_name","pk_km"] if c in comm_df.columns]
disp = comm_df[show_cols].rename(columns={"commune_name":"Commune","pk_km":"PK (km)"})

if selected_one != "— Toutes —" and not disp.empty:
    # Add Open-Meteo rain for the selected commune
    loc_t = comm_df.dropna(subset=["latitude","longitude"]) if "latitude" in comm_df.columns else pd.DataFrame()
    if not loc_t.empty:
        lat_t = round(float(loc_t["latitude"].mean()), 4)
        lon_t = round(float(loc_t["longitude"].mean()), 4)
        rain_t = load_commune_rain_ometo(lat_t, lon_t, periode)
        disp[f"Cumul {periode} (mm)"] = rain_t if not pd.isna(rain_t) else None
        _src_lbl = "AROME 1,3 km" if periode == "24h" else "ERA5 near real-time"
        st.caption(f"Pluie : Open-Meteo {_src_lbl} · cumul {periode}")
elif not disp.empty:
    st.caption("ℹ️ Voir **Comparaison communes** ci-dessus pour les données pluvio fiables (Open-Meteo).")

if not disp.empty:
    st.dataframe(disp.sort_values("PK (km)") if "PK (km)" in disp.columns else disp,
                 use_container_width=True, hide_index=True, height=300)
