from __future__ import annotations

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
}
LEVEL_COLOR = {"ROUGE":"#dc2626","ORANGE":"#ea580c","JAUNE":"#eab308","VERT":"#16a34a","INFO":"#3b82f6"}
LEVEL_EMOJI = {"ROUGE":"🔴","ORANGE":"🟠","JAUNE":"🟡","VERT":"🟢","INFO":"🔵"}
LEVEL_RANK  = {"ROUGE":4,"ORANGE":3,"JAUNE":2,"VERT":1,"INFO":0}


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
def load_snapshot():
    try:
        r = requests.get(SNAPSHOT_URL, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


@st.cache_data(ttl=1800)
def load_weather_alerts_all() -> list:
    """Alertes dérivées des prévisions Open-Meteo pour les 6 départements LGV SEA."""
    alerts = []
    for dep, info in DEPS.items():
        try:
            r = requests.get(FORECAST_URL, params={
                "latitude": info["lat"], "longitude": info["lon"],
                "daily": ("precipitation_sum,temperature_2m_max,"
                          "weathercode,wind_speed_10m_max"),
                "forecast_days": 7, "timezone": "Europe/Paris",
            }, timeout=15)
            r.raise_for_status()
            daily   = r.json().get("daily", {})
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

        except Exception:
            continue

    return sorted(alerts, key=lambda x: (-LEVEL_RANK.get(x["level"], 0), x["date"]))


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
def load_vigicrue_rivers() -> list:
    """Vigilance crues des cours d'eau à proximité de la LGV SEA (API Vigicrue XML)."""
    VC_LEVEL = {1: "VERT", 2: "JAUNE", 3: "ORANGE", 4: "ROUGE"}
    results: list = []

    # TypEntVigiCru=3 = tronçons (sections de cours d'eau) — principal format Vigicrue v2
    for params in [{"TypEntVigiCru": 3}, {}]:
        try:
            r = requests.get(
                "https://www.vigicrues.gouv.fr/services/2/InfoVigiCrue.xml",
                params=params, timeout=15,
                headers={"Accept": "application/xml,text/xml,*/*"},
            )
            if r.status_code != 200:
                continue
            content = r.content.strip()
            if not content.startswith(b"<"):
                continue

            root = ET.fromstring(content)
            found = False

            for elem in root.iter():
                # Name: Vigicrue v2 uses LibTroncon or NomTroncon; fallback to others
                name = ""
                for attr in ("LibTroncon", "NomTroncon", "NomEntVigiCru",
                             "LibEntVigiCru", "NomCoursDeau", "Nom", "lib"):
                    name = elem.get(attr, "")
                    if name:
                        break
                if not name:
                    continue

                # Vigilance level: NivVigiCru is standard in v2
                niv_raw = ""
                for attr in ("NivVigiCru", "NivVigiCruHydro", "NivVig", "couleur"):
                    niv_raw = elem.get(attr, "")
                    if niv_raw:
                        break
                if not niv_raw:
                    continue
                try:
                    niv = int(float(niv_raw))
                except ValueError:
                    continue

                dep_raw = (elem.get("CdDep") or elem.get("CdEntVigiCru")
                           or elem.get("CDDep") or "")
                name_norm = _normalize(name)
                is_lgv = any(rv in name_norm for rv in RIVERS_LGV)
                dep_ok = (not dep_raw) or (dep_raw in _DEP_OK)

                if is_lgv and dep_ok:
                    lvl = VC_LEVEL.get(niv, "VERT")
                    results.append(dict(
                        riviere=name, dep=dep_raw, level=lvl, type="VIGICRUE",
                        msg=f"{name} — vigilance {lvl.lower()}",
                    ))
                    found = True

            if found:
                break
        except Exception:
            continue

    # Deduplicate
    seen: set = set()
    dedup: list = []
    for item in results:
        key = (_normalize(item["riviere"])[:25], item["level"])
        if key not in seen:
            seen.add(key)
            dedup.append(item)

    # Sort: highest level first, then by name
    dedup.sort(key=lambda x: (-LEVEL_RANK.get(x["level"], 0), x["riviere"]))

    if not dedup:
        dedup.append(dict(riviere="", dep="", level="INFO", type="VIGICRUE",
            msg="API Vigicrue indisponible ou aucune crue en cours sur les cours d'eau LGV SEA"))
    return dedup


@st.cache_data(ttl=900)
def load_commune_rain_ometo(lat: float, lon: float, periode: str) -> float:
    """Cumul pluie via Open-Meteo.
    24h  → AROME Météo-France 1,3 km (orages locaux), sinon ERA5.
    7j+  → ERA5 seamless (couverture longue durée).
    """
    today = datetime.now(timezone.utc).date()
    if periode == "24h":
        past_days = 1
    elif periode == "7 jours":
        past_days = 7
    elif periode == "30 jours":
        past_days = 30
    else:
        past_days = today.day - 1
    if past_days <= 0:
        return 0.0

    base = {
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "daily": "precipitation_sum",
        "past_days": past_days, "forecast_days": 0,
        "timezone": "Europe/Paris",
    }
    # For 24h: try AROME first (1.3 km, hourly update — captures local storms)
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


@st.cache_data(ttl=3600)
def load_forecast_dep(dep: str) -> dict:
    d = DEPS[dep]
    try:
        r = requests.get(FORECAST_URL, params={
            "latitude": d["lat"], "longitude": d["lon"],
            "daily": "precipitation_sum,weathercode",
            "forecast_days": 7, "timezone": "Europe/Paris",
        }, timeout=15)
        r.raise_for_status()
        return r.json()
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




def safe_df(records) -> pd.DataFrame:
    if isinstance(records, list) and records:
        try:
            return pd.DataFrame(records)
        except Exception:
            pass
    return pd.DataFrame()


def alert_card(a: dict):
    lvl   = a.get("level", "")
    atype = a.get("type", "")
    color = LEVEL_COLOR.get(lvl, "#6b7280")
    icon  = ALERT_CFG.get(atype, ("⚠️", ""))[0]
    prefix = LEVEL_EMOJI.get(lvl, "")
    if lvl == "VERT":
        prefix = "✅"
    st.markdown(
        f'<div style="padding:7px 14px;border-radius:6px;border-left:4px solid {color};'
        f'background:{color}12;margin-bottom:5px;font-size:13px">'
        f'{prefix} {icon} <b>[{lvl}]</b> {a.get("msg","")}'
        f'</div>', unsafe_allow_html=True)


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
    "VERT":   ("Peu de pluie",    "#16a34a", "🟢"),
    "JAUNE":  ("Pluie modérée",   "#eab308", "🟡"),
    "ORANGE": ("Pluies fortes",   "#ea580c", "🟠"),
    "ROUGE":  ("Pluies très fortes","#dc2626","🔴"),
}
dep_rain_data: dict = {}   # dep -> {max, total, lvl, color, emoji}
for _dep in DEPS:
    _fc    = load_forecast_dep(_dep)
    _rains = [v for v in _fc.get("daily", {}).get("precipitation_sum", []) if v is not None]
    _max   = max(_rains) if _rains else 0.0
    _total = sum(_rains)
    _lvl, _color, _emoji = rain_risk(_max)
    dep_rain_data[_dep] = {"max": _max, "total": _total,
                            "lvl": _lvl, "color": _color, "emoji": _emoji}


def nearest_dep(lat: float, lon: float) -> str:
    """Return dep code nearest to a lat/lon (for map coloring)."""
    return min(DEPS.keys(),
               key=lambda d: (DEPS[d]["lat"] - lat) ** 2 + (DEPS[d]["lon"] - lon) ** 2)


# ── 1. PLUIE PRÉVUE PAR DÉPARTEMENT ─────────────────────────────────────────
st.subheader("📡 Pluie prévue 7 jours par département (Open-Meteo)")
dep_cols = st.columns(len(DEPS))
for col_w, (dep, info) in zip(dep_cols, DEPS.items()):
    d = dep_rain_data[dep]
    lvl_label, color, emoji = RAIN_LABELS[d["lvl"]]
    col_w.markdown(
        f'<div style="padding:10px 6px;border-radius:8px;border:2px solid {color};'
        f'text-align:center;background:{color}14">'
        f'<div style="font-size:22px">{emoji}</div>'
        f'<b style="font-size:13px">Dép. {dep}</b><br>'
        f'<span style="font-size:11px;color:#666">{info["nom"]}</span><br>'
        f'<b style="color:{color};font-size:12px">{lvl_label}</b><br>'
        f'<span style="font-size:12px">max {d["max"]:.0f} mm/j · {d["total"]:.0f} mm</span>'
        f'</div>', unsafe_allow_html=True)

# ── 2. INDICATEURS MÉTÉO ─────────────────────────────────────────────────────
st.subheader("📊 Indicateurs météo — 7 prochains jours")

met_alerts = load_weather_alerts_all()
vc_alerts  = load_vigicrue_rivers()

active_met = [a for a in met_alerts if a["level"] in ("ROUGE","ORANGE","JAUNE")]
by_met: dict = defaultdict(list)
for a in active_met:
    by_met[a["type"]].append(a)

if active_met:
    chips = ""
    for atype, alist in by_met.items():
        worst = max(alist, key=lambda x: LEVEL_RANK.get(x["level"], 0))
        color = LEVEL_COLOR.get(worst["level"], "#6b7280")
        icon, label = ALERT_CFG.get(atype, ("⚠️", ""))
        chips += (f'<span style="display:inline-block;margin:3px 4px;padding:3px 10px;'
                  f'border-radius:20px;background:{color};color:white;font-size:12px;font-weight:600">'
                  f'{icon} {label} ({len(alist)})</span>')
    st.markdown(chips, unsafe_allow_html=True)
else:
    st.success("✅ Aucun indicateur météo significatif sur les 7 prochains jours.")

tab_labels: list = []
tab_data:   list = []
for atype in ("ORAGE", "INONDATION", "CANICULE", "INCENDIE", "VENT"):
    if atype in by_met:
        icon, label = ALERT_CFG[atype]
        tab_labels.append(f"{icon} {label} ({len(by_met[atype])})")
        tab_data.append(by_met[atype])

vc_active = [a for a in vc_alerts if a["level"] in ("ROUGE","ORANGE","JAUNE")]
vc_label  = "🏞️ Vigicrue" + (f" ⚠ {len(vc_active)}" if vc_active else " ✅")
tab_labels.append(vc_label)
tab_data.append(vc_alerts)

if tab_labels:
    tabs = st.tabs(tab_labels)
    for tab, alist in zip(tabs, tab_data):
        with tab:
            if not alist:
                st.info("Aucune donnée.")
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
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=10, b=30, l=10, r=90),
            )
            st.plotly_chart(fig, use_container_width=True)
            _src = "AROME Météo-France 1,3 km" if periode == "24h" else "ERA5 near real-time"
            st.caption(f"Source : Open-Meteo {_src} · {date_str}")

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
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=30, b=20, l=20, r=80), xaxis=dict(tickangle=-20),
    )
    st.plotly_chart(fig2, use_container_width=True)
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
    fig3.update_layout(coloraxis_showscale=False, height=260,
                       plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(t=20,b=20,l=20,r=20), xaxis=dict(tickangle=-30))
    st.plotly_chart(fig3, use_container_width=True)
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
            pts = [[p[0], p[1]] for p in seg if isinstance(p, (list, tuple)) and len(p) >= 2]
            if pts:
                folium.PolyLine(pts, color="#cc0000", weight=2.5, opacity=0.7).add_to(m)
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
