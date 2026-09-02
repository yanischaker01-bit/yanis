from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

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
SNAPSHOT_URL = "https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PIEZO_STATIONS_URL = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations"
PIEZO_CHRONIQUES_URL = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques"
DEPARTEMENTS = ["37", "86", "79", "16", "17", "33"]
LEVEL_RANK = {"VERT": 0, "JAUNE": 1, "ORANGE": 2, "ROUGE": 3, "INDETERMINE": -1}
LEVEL_COLOR = {"VERT": "#16a34a", "JAUNE": "#eab308", "ORANGE": "#ea580c", "ROUGE": "#dc2626", "INDETERMINE": "#64748b"}
LEVEL_ACTION = {
    "VERT": "Surveillance normale",
    "JAUNE": "Contrôle renforcé des données et de leur fraîcheur",
    "ORANGE": "Inspection ciblée du secteur à programmer",
    "ROUGE": "Contrôle prioritaire et remontée opérationnelle",
    "INDETERMINE": "Données insuffisantes, statut non vérifié",
}

st.set_page_config(page_title="LGV SEA - Météo et piézométrie", page_icon="🌧️", layout="wide")

# =============================================================================
# OUTILS GEOGRAPHIQUES
# =============================================================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def build_lgv_polyline(lines) -> list[tuple[float, float, float]]:
    best: list[tuple[float, float]] = []
    for segment in lines or []:
        pts = []
        for p in segment if isinstance(segment, list) else []:
            if isinstance(p, dict) and p.get("lat") is not None and p.get("lon") is not None:
                pts.append((float(p["lat"]), float(p["lon"])))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
        if len(pts) > len(best):
            best = pts
    if len(best) < 2:
        return []
    out = [(best[0][0], best[0][1], 0.0)]
    cumul = 0.0
    for p1, p2 in zip(best, best[1:]):
        cumul += haversine_km(*p1, *p2)
        out.append((p2[0], p2[1], cumul))
    return out


def pk_and_distance(lat: float, lon: float, polyline) -> tuple[float | None, float | None]:
    if len(polyline) < 2:
        return None, None
    best_d2, best_pk = None, None
    for (lat1, lon1, pk1), (lat2, lon2, pk2) in zip(polyline, polyline[1:]):
        latm = (lat1 + lat2) / 2
        kx, ky = 111.320 * math.cos(math.radians(latm)), 111.320
        x1, y1, x2, y2, xp, yp = lon1*kx, lat1*ky, lon2*kx, lat2*ky, lon*kx, lat*ky
        dx, dy = x2-x1, y2-y1
        den = dx*dx + dy*dy
        t = 0.0 if den == 0 else max(0.0, min(1.0, ((xp-x1)*dx + (yp-y1)*dy)/den))
        cx, cy = x1+t*dx, y1+t*dy
        d2 = (xp-cx)**2 + (yp-cy)**2
        if best_d2 is None or d2 < best_d2:
            best_d2, best_pk = d2, pk1 + t*(pk2-pk1)
    return best_pk, math.sqrt(best_d2) if best_d2 is not None else None

# =============================================================================
# CHARGEMENT DES DONNEES
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_snapshot() -> dict:
    r = requests.get(SNAPSHOT_URL, timeout=(5, 25))
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather(lat: float, lon: float) -> dict:
    today = datetime.now(timezone.utc).date()
    past_start, past_end = today - timedelta(days=30), today - timedelta(days=1)
    hist = requests.get(ARCHIVE_URL, params={
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "start_date": past_start.isoformat(), "end_date": past_end.isoformat(),
        "daily": "precipitation_sum", "timezone": "Europe/Paris",
    }, timeout=(5, 25))
    hist.raise_for_status()
    fc = requests.get(FORECAST_URL, params={
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "hourly": "precipitation,soil_moisture_0_to_7cm,wind_gusts_10m",
        "daily": "precipitation_sum,precipitation_probability_max,wind_gusts_10m_max,weather_code",
        "forecast_days": 7, "timezone": "Europe/Paris",
    }, timeout=(5, 25))
    fc.raise_for_status()
    return {"history": hist.json(), "forecast": fc.json()}


def weather_assessment(payload: dict) -> dict:
    hvals = pd.to_numeric(pd.Series(payload["history"].get("daily", {}).get("precipitation_sum", [])), errors="coerce").fillna(0)
    fday = payload["forecast"].get("daily", {})
    fvals = pd.to_numeric(pd.Series(fday.get("precipitation_sum", [])), errors="coerce").fillna(0)
    hourly = payload["forecast"].get("hourly", {})
    hp = pd.to_numeric(pd.Series(hourly.get("precipitation", [])), errors="coerce").fillna(0)
    soil = pd.to_numeric(pd.Series(hourly.get("soil_moisture_0_to_7cm", [])), errors="coerce").dropna()
    gust = pd.to_numeric(pd.Series(hourly.get("wind_gusts_10m", [])), errors="coerce").fillna(0)
    rain24 = float(hp.iloc[:24].sum()) if len(hp) else 0.0
    rain6 = float(max((hp.rolling(6, min_periods=1).sum()).max(), 0)) if len(hp) else 0.0
    past3, past7, past30 = float(hvals.tail(3).sum()), float(hvals.tail(7).sum()), float(hvals.sum())
    forecast7, max_day = float(fvals.sum()), float(fvals.max()) if len(fvals) else 0.0
    soil_now = float(soil.iloc[0]) if len(soil) else float("nan")
    gust_max = float(gust.max()) if len(gust) else 0.0

    score = 0
    score += min(25, rain6 / 40 * 25)
    score += min(20, rain24 / 60 * 20)
    score += min(15, past7 / 100 * 15)
    score += min(10, past30 / 200 * 10)
    score += min(20, forecast7 / 100 * 20)
    if not math.isnan(soil_now):
        score += min(10, max(0, (soil_now - 0.20) / 0.25 * 10))
    score = round(min(100, score), 1)
    level = "ROUGE" if score >= 75 else "ORANGE" if score >= 50 else "JAUNE" if score >= 25 else "VERT"
    return {"score": score, "level": level, "rain6": rain6, "rain24": rain24, "past3": past3,
            "past7": past7, "past30": past30, "forecast7": forecast7, "max_day": max_day,
            "soil": soil_now, "gust": gust_max, "daily": fday}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_piezo_stations() -> pd.DataFrame:
    frames = []
    for dep in DEPARTEMENTS:
        try:
            r = requests.get(PIEZO_STATIONS_URL, params={"code_departement": dep, "size": 20000, "format": "json"}, timeout=(5, 40))
            r.raise_for_status()
            frames.append(pd.DataFrame(r.json().get("data", [])))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True).drop_duplicates("code_bss") if frames else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_piezo_history(code_bss: str, start: date, end: date) -> pd.DataFrame:
    rows, page = [], 1
    while page <= 20:
        r = requests.get(PIEZO_CHRONIQUES_URL, params={
            "code_bss": code_bss, "date_debut_mesure": start.isoformat(),
            "date_fin_mesure": end.isoformat(), "size": 20000, "page": page,
        }, timeout=(5, 45))
        r.raise_for_status()
        payload = r.json()
        batch = payload.get("data", [])
        rows.extend(batch)
        if not batch or not payload.get("next"):
            break
        page += 1
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df.get("date_mesure"), errors="coerce")
    df["niveau_ngf"] = pd.to_numeric(df.get("niveau_nappe_eau"), errors="coerce")
    return df.dropna(subset=["date", "niveau_ngf"]).sort_values("date").drop_duplicates("date")

# =============================================================================
# SEUILS ET EVENEMENTS
# =============================================================================
def monthly_thresholds(df: pd.DataFrame, selected_month: int) -> dict:
    ref = df[df["date"].dt.month == selected_month]["niveau_ngf"].dropna()
    if len(ref) < 20:
        ref = df["niveau_ngf"].dropna()
    if len(ref) < 10:
        return {}
    return {"JAUNE": float(ref.quantile(.90)), "ORANGE": float(ref.quantile(.95)), "ROUGE": float(ref.quantile(.99))}


def classify(value: float, thresholds: dict) -> str:
    if not thresholds:
        return "INDETERMINE"
    if value >= thresholds["ROUGE"]: return "ROUGE"
    if value >= thresholds["ORANGE"]: return "ORANGE"
    if value >= thresholds["JAUNE"]: return "JAUNE"
    return "VERT"


def threshold_events(df: pd.DataFrame, threshold_history: pd.DataFrame) -> pd.DataFrame:
    if df.empty or threshold_history.empty:
        return pd.DataFrame()
    d = df.copy().reset_index(drop=True)
    d["niveau_surveillance"] = [classify(v, t) for v, t in zip(d["niveau_ngf"], threshold_history.to_dict("records"))]
    d["active"] = d["niveau_surveillance"] != "VERT"
    d["groupe"] = (d["active"] != d["active"].shift()).cumsum()
    events = []
    for _, g in d[d["active"]].groupby("groupe"):
        imax = g["niveau_ngf"].idxmax()
        worst = max(g["niveau_surveillance"], key=lambda x: LEVEL_RANK.get(x, -1))
        events.append({"Début": g["date"].min(), "Fin": g["date"].max(),
                       "Durée (j)": max(1, (g["date"].max()-g["date"].min()).days+1),
                       "Seuil max": worst, "Maximum (m NGF)": round(float(g["niveau_ngf"].max()), 3),
                       "Date du max": d.loc[imax, "date"]})
    return pd.DataFrame(events).sort_values("Début", ascending=False) if events else pd.DataFrame()

# =============================================================================
# INTERFACE
# =============================================================================
st.title("🌧️ LGV SEA - Surveillance météo et piézométrique")
st.caption("Aide à la surveillance. Les seuils statistiques proposés ne remplacent ni les seuils métier validés ni les consignes d'exploitation.")

try:
    snapshot = load_snapshot()
except Exception as exc:
    st.error(f"Snapshot LGV indisponible : {exc}")
    st.stop()

polyline = build_lgv_polyline(snapshot.get("lgv_lines"))
sec = snapshot.get("sectors", {})
sectors = pd.DataFrame(sec.get("sectors", []) if isinstance(sec, dict) else [])
for c in ["latitude", "longitude", "pk_km"]:
    if c in sectors: sectors[c] = pd.to_numeric(sectors[c], errors="coerce")
if sectors.empty or not polyline:
    st.error("Le snapshot ne contient pas les secteurs ou le tracé LGV nécessaires.")
    st.stop()

sectors = sectors.dropna(subset=["latitude", "longitude"])
sectors["zone_pk"] = sectors["pk_km"].apply(lambda x: f"PK {int(x//10)*10:03d}-{int(x//10)*10+10:03d}" if pd.notna(x) else "Zone inconnue")

with st.sidebar:
    st.header("Paramètres")
    zone = st.selectbox("Zone LGV", ["Toutes les zones"] + sorted(sectors["zone_pk"].unique().tolist()))
    station_radius = st.slider("Distance maximale des piézomètres à la LGV", 1, 30, 10, 1, format="%d km")
    history_years = st.slider("Historique pour les seuils", 2, 15, 8)
    st.caption("Seuils automatiques saisonniers : quantiles 90 %, 95 % et 99 % par mois.")

zone_df = sectors if zone == "Toutes les zones" else sectors[sectors["zone_pk"] == zone]
lat_c, lon_c = float(zone_df["latitude"].mean()), float(zone_df["longitude"].mean())

st.subheader(f"🌦️ Diagnostic météo - {zone}")
try:
    weather = weather_assessment(fetch_weather(lat_c, lon_c))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score météo", f"{weather['score']}/100")
    c2.metric("Pluie max glissante 6 h", f"{weather['rain6']:.1f} mm")
    c3.metric("Pluie prévue 24 h", f"{weather['rain24']:.1f} mm")
    c4.metric("Cumul antérieur 7 j", f"{weather['past7']:.1f} mm")
    c5.metric("Prévision 7 j", f"{weather['forecast7']:.1f} mm")
    st.markdown(f"**Niveau {weather['level']}** : {LEVEL_ACTION[weather['level']]}")
    daily = pd.DataFrame(weather["daily"])
    if not daily.empty:
        figw = go.Figure(go.Bar(x=daily["time"], y=daily["precipitation_sum"], marker_color="#2563eb", name="Pluie"))
        figw.update_layout(height=300, yaxis_title="Pluie (mm)", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(figw, use_container_width=True, config={"displayModeBar": False})
except Exception as exc:
    weather = None
    st.warning(f"Météo non vérifiée : {exc}")

st.divider()
st.subheader("💧 Piézomètres proches de la LGV avec PK")
with st.spinner("Chargement des stations Hub'Eau et calcul des PK..."):
    stations = fetch_piezo_stations()

if stations.empty:
    st.warning("Aucune station piézométrique récupérée.")
    st.stop()

lat_col = next((c for c in ["latitude", "y"] if c in stations.columns), None)
lon_col = next((c for c in ["longitude", "x"] if c in stations.columns), None)
if not lat_col or not lon_col:
    st.error("Les coordonnées des stations ne sont pas présentes dans la réponse Hub'Eau.")
    st.stop()
stations["latitude"] = pd.to_numeric(stations[lat_col], errors="coerce")
stations["longitude"] = pd.to_numeric(stations[lon_col], errors="coerce")
stations = stations.dropna(subset=["latitude", "longitude"]).copy()
calc = stations.apply(lambda r: pk_and_distance(float(r["latitude"]), float(r["longitude"]), polyline), axis=1)
stations[["pk_km", "distance_km"]] = pd.DataFrame(calc.tolist(), index=stations.index)
stations = stations[stations["distance_km"] <= station_radius].copy()
stations["zone_pk"] = stations["pk_km"].apply(lambda x: f"PK {int(x//10)*10:03d}-{int(x//10)*10+10:03d}")
if zone != "Toutes les zones": stations = stations[stations["zone_pk"] == zone]

if stations.empty:
    st.info("Aucun piézomètre dans la zone et le rayon sélectionnés.")
    st.stop()

stations["station_label"] = stations.apply(lambda r: f"{r.get('libelle_pe') or r.get('nom_commune') or 'Station'} | {r['code_bss']} | PK {r['pk_km']:.1f}", axis=1)
show = stations[["code_bss", "libelle_pe", "nom_commune", "pk_km", "distance_km", "zone_pk"]].copy()
show.columns = ["Code BSS", "Station", "Commune", "PK (km)", "Distance LGV (km)", "Zone"]
st.dataframe(show.sort_values("PK (km)"), use_container_width=True, hide_index=True)

selected_label = st.selectbox("Choisir une station pour l'historique", stations.sort_values("pk_km")["station_label"].tolist())
station = stations.loc[stations["station_label"] == selected_label].iloc[0]
end = datetime.now(timezone.utc).date()
start = end - timedelta(days=365 * history_years)
with st.spinner("Chargement de la chronique piézométrique..."):
    try:
        hist = fetch_piezo_history(str(station["code_bss"]), start, end)
    except Exception as exc:
        st.error(f"Chronique indisponible : {exc}")
        st.stop()

if hist.empty:
    st.info("Aucune mesure disponible sur la période.")
    st.stop()

# Seuil saisonnier pour chaque observation, calculé sur le mois correspondant.
month_thresholds = {m: monthly_thresholds(hist, m) for m in range(1, 13)}
th_rows = []
for dt in hist["date"]:
    t = month_thresholds.get(dt.month, {})
    th_rows.append({k: t.get(k, np.nan) for k in ["JAUNE", "ORANGE", "ROUGE"]})
th_df = pd.DataFrame(th_rows, index=hist.index)
current_thresholds = month_thresholds.get(hist["date"].iloc[-1].month, {})
current_level = classify(float(hist["niveau_ngf"].iloc[-1]), current_thresholds)

c1, c2, c3, c4 = st.columns(4)
c1.metric("PK station", f"{station['pk_km']:.1f} km")
c2.metric("Distance LGV", f"{station['distance_km']*1000:.0f} m")
c3.metric("Dernier niveau", f"{hist['niveau_ngf'].iloc[-1]:.3f} m NGF")
c4.metric("Surveillance", current_level)
st.caption(f"Dernière mesure : {hist['date'].iloc[-1]:%d/%m/%Y %H:%M} | {LEVEL_ACTION[current_level]}")

st.subheader("📈 Historique, seuils saisonniers et dates des maxima")
view_start, view_end = st.date_input("Période affichée", value=(max(start, end-timedelta(days=730)), end), min_value=start, max_value=end)
view = hist[(hist["date"].dt.date >= view_start) & (hist["date"].dt.date <= view_end)].copy()
view_th = th_df.loc[view.index]
fig = go.Figure()
fig.add_scatter(x=view["date"], y=view["niveau_ngf"], name="Niveau nappe", line=dict(color="#2563eb", width=2))
for lvl in ["JAUNE", "ORANGE", "ROUGE"]:
    fig.add_scatter(x=view["date"], y=view_th[lvl], name=f"Seuil {lvl.lower()}", line=dict(color=LEVEL_COLOR[lvl], dash="dash", width=1.5))
if not view.empty:
    imax = view["niveau_ngf"].idxmax()
    fig.add_scatter(x=[view.loc[imax, "date"]], y=[view.loc[imax, "niveau_ngf"]], mode="markers+text",
                    text=[f"Max {view.loc[imax, 'niveau_ngf']:.3f} m"], textposition="top center",
                    marker=dict(color="#7f1d1d", size=11, symbol="diamond"), name="Maximum période")
fig.update_layout(height=430, yaxis_title="Niveau (m NGF)", hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

events = threshold_events(view, view_th)
min_level = st.selectbox("Afficher les franchissements à partir du seuil", ["JAUNE", "ORANGE", "ROUGE"])
if not events.empty:
    events = events[events["Seuil max"].map(LEVEL_RANK) >= LEVEL_RANK[min_level]]
if events.empty:
    st.success("Aucun franchissement correspondant sur la période choisie.")
else:
    st.dataframe(events, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Exporter les franchissements CSV", events.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"seuils_{station['code_bss'].replace('/', '_')}.csv", mime="text/csv")

st.divider()
st.subheader("🗺️ Carte satellite")
m = folium.Map(location=[lat_c, lon_c], zoom_start=9 if zone == "Toutes les zones" else 11, tiles=None, control_scale=True)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    name="Satellite Esri", overlay=False, control=True, max_zoom=19,
).add_to(m)
for seg in snapshot.get("lgv_lines") or []:
    pts = [[p["lat"], p["lon"]] for p in seg if isinstance(p, dict) and "lat" in p and "lon" in p]
    if pts: folium.PolyLine(pts, color="#ef4444", weight=3, opacity=.9, tooltip="LGV SEA").add_to(m)
for _, s in stations.iterrows():
    selected = s["code_bss"] == station["code_bss"]
    folium.CircleMarker([s["latitude"], s["longitude"]], radius=8 if selected else 5,
        color="#facc15" if selected else "#0ea5e9", fill=True, fill_opacity=.9,
        tooltip=f"{s['code_bss']} | PK {s['pk_km']:.1f} | {s['distance_km']*1000:.0f} m",
        popup=folium.Popup(f"<b>{s.get('libelle_pe') or s.get('nom_commune') or 'Piézomètre'}</b><br>Code BSS : {s['code_bss']}<br>PK : {s['pk_km']:.1f} km<br>Distance LGV : {s['distance_km']*1000:.0f} m", max_width=300)).add_to(m)
folium.LayerControl(collapsed=True).add_to(m)
st_folium(m, use_container_width=True, height=520, returned_objects=[])

st.caption("Sources : snapshot LGV SEA, Open-Meteo, Hub'Eau/ADES. Seuils automatiques à valider avec les référentiels métier et le retour d'expérience maintenance.")
