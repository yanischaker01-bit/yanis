from __future__ import annotations



import io

@@ -14,6 +13,7 @@

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go


from plotly.subplots import make_subplots

import requests

import streamlit as st

from streamlit_folium import st_folium

@@ -22,16 +22,15 @@

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"




# NASA FIRMS (Fire Information for Resource Management System) — détections satellite quasi temps réel


# NASA FIRMS

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{day_range}/{date}"


FIRMS_SOURCES  = ["VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"]  # résolution ~375 m


FIRMS_BBOX     = "-0.7,44.75,1.0,47.5"  # west,south,east,north — corridor LGV SEA + marge


FIRMS_SOURCES  = ["VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"]


FIRMS_BBOX     = "-0.7,44.75,1.0,47.5"

FIRMS_RADIUS_KM = 0.5


FIRMS_MAX_DAY_RANGE = 10   # limite de l'API area (une seule requête par source)


FIRMS_MAX_LOOKBACK_DAYS = 60  # au-delà, le NRT n'est plus garanti disponible


FIRMS_MAX_DAY_RANGE = 10


FIRMS_MAX_LOOKBACK_DAYS = 60




# Vigilance météo officielle Météo-France, republiée en open data (sans clé) sur Opendatasoft —


# mêmes bulletins que vigilance.meteofrance.fr, source de référence utilisée par SNCF/préfectures.


# Météo-France vigilance

MF_VIGILANCE_URL     = "https://public.opendatasoft.com/api/records/1.0/search/"

MF_VIGILANCE_DATASET = "weatherref-france-vigilance-meteo-departement"

MF_PHENOMENON_LABELS = {

@@ -66,7 +65,6 @@

LEVEL_COLOR = {"ROUGE":"#dc2626","ORANGE":"#ea580c","JAUNE":"#eab308","VERT":"#16a34a","INFO":"#3b82f6"}

LEVEL_RANK  = {"ROUGE":4,"ORANGE":3,"JAUNE":2,"VERT":1,"INFO":0}

LEVEL_LABEL = {"ROUGE":"Rouge","ORANGE":"Orange","JAUNE":"Jaune","VERT":"Vert","INFO":"Info"}


# st.badge only accepts this fixed palette — map our 5 severity levels onto it.

LEVEL_BADGE = {"ROUGE":"red","ORANGE":"orange","JAUNE":"yellow","VERT":"green","INFO":"gray"}





@@ -91,9 +89,6 @@





def load_snapshot() -> dict:


    """Seules les réponses réussies sont mises en cache (via _fetch_snapshot_raw) :


    un échec réseau ponctuel n'immobilise pas l'appli 15 min, il est retenté au


    prochain rerun au lieu de rester figé jusqu'à expiration du TTL."""

    try:

        return _fetch_snapshot_raw()

    except Exception as e:

@@ -111,10 +106,6 @@





def load_meteofrance_vigilance() -> tuple[list, bool]:


    """Vigilance officielle Météo-France (aujourd'hui J / demain J1) par département,


    republiée en open data sans clé — mêmes bulletins que vigilance.meteofrance.fr.


    Retourne (alertes non-vertes, ok). ok=False seulement si l'API n'a pas pu être


    interrogée du tout (à ne pas confondre avec une vigilance verte légitime)."""

    try:

        records = _fetch_mf_vigilance_raw(tuple(sorted(DEPS.keys())))

    except Exception:

@@ -155,11 +146,6 @@





def load_weather_alerts_all() -> tuple[list, int, int]:


    """Alertes dérivées des prévisions Open-Meteo pour les 6 départements LGV SEA.


    Retourne (alertes, nb_dept_ok, nb_dept_total) : si nb_dept_ok == 0, l'appelant


    doit afficher un avertissement plutôt qu'un silencieux "aucune alerte"


    (chaque département étant récupéré via une fonction cachée séparée, l'échec


    de l'un n'empêche pas les autres d'être servis depuis leur propre cache)."""

    alerts = []

    ok_count = 0

    for dep in DEPS:

@@ -182,9 +168,8 @@

            t = tmaxes[i]  or 0

            w = wcodes[i]  or 0

            v = winds[i]   or 0


            d_str = date[5:]  # MM-DD


            d_str = date[5:]




            # ⛈️ Orages (WMO codes 80-82 averses, 95-99 orages)

            if w >= 99:

                alerts.append(dict(dep=dep, date=date, type="ORAGE", level="ROUGE",

                    msg=f"Dép.{dep} le {d_str} — Orages violents avec grêle"))

@@ -195,7 +180,6 @@

                alerts.append(dict(dep=dep, date=date, type="ORAGE", level="JAUNE",

                    msg=f"Dép.{dep} le {d_str} — Averses orageuses"))




            # 🌊 Inondation (précipitations intenses)

            if p >= 60:

                alerts.append(dict(dep=dep, date=date, type="INONDATION", level="ROUGE",

                    msg=f"Dép.{dep} le {d_str} — Pluies diluviennes : {p:.0f} mm"))

@@ -206,7 +190,6 @@

                alerts.append(dict(dep=dep, date=date, type="INONDATION", level="JAUNE",

                    msg=f"Dép.{dep} le {d_str} — Pluies soutenues : {p:.0f} mm"))




            # 🌡️ Canicule

            if t >= 40:

                alerts.append(dict(dep=dep, date=date, type="CANICULE", level="ROUGE",

                    msg=f"Dép.{dep} le {d_str} — Canicule extrême : {t:.0f}°C"))

@@ -217,7 +200,6 @@

                alerts.append(dict(dep=dep, date=date, type="CANICULE", level="ORANGE",

                    msg=f"Dép.{dep} le {d_str} — Forte chaleur : {t:.0f}°C"))




            # 💨 Vent violent

            if v >= 100:

                alerts.append(dict(dep=dep, date=date, type="VENT", level="ROUGE",

                    msg=f"Dép.{dep} le {d_str} — Vents très violents : {v:.0f} km/h"))

@@ -228,7 +210,6 @@

                alerts.append(dict(dep=dep, date=date, type="VENT", level="JAUNE",

                    msg=f"Dép.{dep} le {d_str} — Vents forts : {v:.0f} km/h"))




            # 🔥 Risque incendie (chaleur + sécheresse + vent)

            if not seen_fire and t >= 30 and rain7 < 10 and v >= 25:

                lvl = "ROUGE" if (t >= 35 and rain7 < 5 and v >= 35) else "ORANGE"

                alerts.append(dict(dep=dep, date=date, type="INCENDIE", level=lvl,

@@ -241,34 +222,23 @@





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

@@ -283,7 +253,6 @@

    }



    def walk(value):


        """Parcourt tous les dictionnaires d'une reponse JSON imbriquee."""

        if isinstance(value, dict):

            yield value

            for child in value.values():

@@ -376,7 +345,6 @@

                "msg": f"{name} — vigilance {level.lower()}",

            })




    # Une reponse API valide sans alerte LGV est un resultat legitime.

    parsed_ok = successful_responses > 0



    seen: set = set()

@@ -392,32 +360,24 @@





def get_firms_map_key() -> str | None:


    """Récupère la clé FIRMS depuis les secrets Streamlit ou les variables d'environnement.


    Essaie plusieurs emplacements possibles."""


    # 1. Secrets Streamlit (racine)

    try:

        key = st.secrets.get("FIRMS_MAP_KEY")

        if key:

            return key

    except Exception:

        pass




    # 2. Secrets Streamlit (section 'firms')

    try:

        key = st.secrets.get("firms", {}).get("FIRMS_MAP_KEY")

        if key:

            return key

    except Exception:

        pass




    # 3. Variables d'environnement

    key = os.environ.get("FIRMS_MAP_KEY") or os.environ.get("FIRMS_KEY")

    if key:

        return key




    # 4. (optionnel) Si vous utilisez un fichier .env non standard


    #    Vous pouvez ajouter une lecture depuis un fichier de config.




    return None





@@ -432,8 +392,6 @@



@st.cache_data(ttl=3600)

def build_lgv_pk_polyline(_lgv_lines) -> list[tuple[float, float, float]]:


    """[(lat, lon, pk_km cumulé), ...] à partir du 1er tracé LGV du snapshot.


    Le cumul de distance haversine le long du tracé colle au pk_km réel (écart < 100 m)."""

    if not _lgv_lines:

        return []

    seg = _lgv_lines[0]

@@ -449,8 +407,6 @@





def pk_and_distance(lat: float, lon: float, polyline: list) -> tuple[float | None, float | None]:


    """Retourne (pk_km, distance_km) du point le plus proche du tracé LGV.


    Projection sur chaque segment en repère métrique local (correction cos(latitude))."""

    if len(polyline) < 2:

        return None, None

    best_dist2 = None

@@ -494,18 +450,11 @@

        except (requests.RequestException, ValueError, RuntimeError) as e:

            if attempt == 2:

                raise


            time.sleep(2 ** attempt)  # 1s, 2s, 4s


    # fallback (ne devrait pas arriver)


            time.sleep(2 ** attempt)

    raise RuntimeError("Failed after retries")





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

@@ -535,16 +484,13 @@





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

@@ -601,11 +547,6 @@





def load_commune_daily_series(lat: float, lon: float, days: int = 30) -> pd.DataFrame:


    """Série journalière de pluie sur les `days` derniers jours — API archive ERA5


    (réanalyse), pas l'API prévision : son paramètre `past_days` renvoie pour les


    1-2 derniers jours une sortie modèle non recalée sur l'observé, constatée jusqu'à


    11x plus élevée que l'ERA5 le même jour (45,6 mm vs 4,0 mm), ce qui gonflait


    artificiellement les cumuls du classement TOP 20."""

    try:

        daily = _fetch_commune_daily_series_raw(lat, lon, days)

    except Exception:

@@ -618,7 +559,6 @@



@st.cache_data(ttl=3600)

def load_all_communes_daily_rain(_sectors_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:


    """Série journalière de pluie pour chaque commune du corridor (1 requête/commune, cache 1h)."""

    if _sectors_df.empty or "commune_name" not in _sectors_df.columns:

        return pd.DataFrame()

    coords = (_sectors_df.dropna(subset=["latitude", "longitude"])

@@ -639,7 +579,8 @@



@st.cache_data(ttl=900)

def load_commune_rain_ometo(lat: float, lon: float, periode: str) -> float:


    """Cumul pluie.


    """


    Cumul pluie.

    24h  → API prévision Open-Meteo, AROME Météo-France 1,3 km si dispo, sinon

           blend — la veille est un cycle déjà bouclé, pas une sortie modèle

           non recalée (vérifié : identique à l'ERA5 sur un cas testé).

@@ -760,8 +701,6 @@

        return pd.DataFrame()






# Charte graphique commune aux graphiques météo/pluviométrie.


# Cette couche de présentation est indépendante du traitement NASA FIRMS.

CHART_COLORS = {

    "blue": "#2563eb", "cyan": "#0891b2", "teal": "#0f766e",

    "orange": "#f97316", "red": "#dc2626", "slate": "#475569",

@@ -772,7 +711,6 @@



def style_weather_chart(fig: go.Figure, *, height: int = 340,

                        hovermode: str = "x unified") -> go.Figure:


    """Applique une présentation homogène sans modifier les données du graphe."""

    fig.update_layout(

        height=height,

        hovermode=hovermode,

@@ -800,7 +738,6 @@



def show_weather_chart(fig: go.Figure, *, height: int = 340,

                       hovermode: str = "x unified") -> None:


    """Affiche un graphe météo responsive avec une barre d'outils allégée."""

    style_weather_chart(fig, height=height, hovermode=hovermode)

    st.plotly_chart(

        fig, use_container_width=True,

@@ -818,16 +755,14 @@





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


            st.badge(LEVEL_LABEL.get(lvl, lvl or "Info"), color=LEVEL_BADGE.get(lvl, "gray"))

        with c_msg:

            st.markdown(f"{icon}  {a.get('msg','')}" if icon else a.get("msg", ""))

        return

@@ -881,15 +816,14 @@

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


dep_rain_data: dict = {}

for _dep in DEPS:

    _fc    = load_forecast_dep(_dep)

    _ok    = bool(_fc)

@@ -899,15 +833,12 @@

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



@@ -922,7 +853,7 @@

        st.caption(f"Dép. {dep} · {info['nom']}")

        if d["ok"]:

            lvl_label = RAIN_LABELS[d["lvl"]]


            st.badge(lvl_label, color=LEVEL_BADGE.get(d["lvl"], "gray"))  # type: ignore[arg-type]


            st.badge(lvl_label, color=LEVEL_BADGE.get(d["lvl"], "gray"))

            st.metric("Cumul 7 j", f"{d['total']:.0f} mm",

                      help=f"Maximum journalier : {d['max']:.0f} mm/j")

        else:

@@ -977,7 +908,7 @@

        worst = max(alist, key=lambda x: LEVEL_RANK.get(x["level"], 0))

        icon, label = ALERT_CFG.get(atype, ("", atype))

        col.badge(f"{label} ({len(alist)})", icon=icon or None,


                  color=LEVEL_BADGE.get(worst["level"], "gray"))  # type: ignore[arg-type]


                  color=LEVEL_BADGE.get(worst["level"], "gray"))

elif met_ok > 0:

    st.success("Aucun indicateur météo significatif sur les 7 prochains jours.")



@@ -1018,7 +949,6 @@

communes = (sorted(sectors_df["commune_name"].dropna().unique())

            if "commune_name" in sectors_df.columns else [])




# Communes par défaut : répartition géographique nord→sud sur la LGV SEA

def _find_commune(kw: str) -> str | None:

    k = unicodedata.normalize("NFD", kw.lower()).encode("ascii", "ignore").decode()

    return next((c for c in communes

@@ -1140,16 +1070,16 @@

vig_cols = st.columns(4)

vig_cols[0].badge(

    "Vigilance MF — non vérifié" if not mf_ok else f"Vigilance MF ({len(mf_alerts)})",


    icon="🛡️", color="gray" if not mf_ok else ("red" if mf_alerts else "green"))  # type: ignore[arg-type]


    icon="🛡️", color="gray" if not mf_ok else ("red" if mf_alerts else "green"))

vig_cols[1].badge(

    "Météo — non vérifié" if met_ok == 0 else f"Météo ({len(active_met)})",


    icon="🌦️", color="gray" if met_ok == 0 else ("orange" if active_met else "green"))  # type: ignore[arg-type]


    icon="🌦️", color="gray" if met_ok == 0 else ("orange" if active_met else "green"))

vig_cols[2].badge(

    "Vigicrue — non vérifié" if not vc_ok else f"Vigicrue ({len(vc_active)})",


    icon="🏞️", color="gray" if not vc_ok else ("blue" if vc_active else "green"))  # type: ignore[arg-type]


    icon="🏞️", color="gray" if not vc_ok else ("blue" if vc_active else "green"))

vig_cols[3].badge(

    "FIRMS — non vérifié" if _firms_unverified else f"FIRMS ({len(firms_alerts)})",


    icon="🔥", color="gray" if _firms_unverified else ("red" if firms_alerts else "green"))  # type: ignore[arg-type]


    icon="🔥", color="gray" if _firms_unverified else ("red" if firms_alerts else "green"))



if mf_alerts or active_met or vc_active or firms_alerts:

    with st.expander("⚠️ Détail des alertes actives (vigilance MF, météo, crues, incendie)", expanded=True):

@@ -1222,7 +1152,6 @@

        )

        show_weather_chart(fig_top, height=420, hovermode="closest")




        # Courbes journalières : top 3 en couleur (identité), le reste du TOP 20 en gris (contexte)

        HIGHLIGHT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

        fig_curve = go.Figure()

        for commune in reversed(top20_communes):

@@ -1349,28 +1278,31 @@

if not fc_df.empty:

    fc_df["pluie_mm"] = pd.to_numeric(fc_df["pluie_mm"], errors="coerce").fillna(0)

    fc_df["tmax"]     = pd.to_numeric(fc_df["tmax"],     errors="coerce").fillna(0)


    fc_df["proba_%"]  = pd.to_numeric(fc_df["proba_%"],  errors="coerce").fillna(0)

    fc_df["color"]    = fc_df["pluie_mm"].apply(rain_color_mm)


    fig2 = go.Figure()





    # Création d'une figure avec sous-grille pour axes secondaires


    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

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


                 textposition="outside", name="Pluie (mm)",


                 secondary_y=False)


    fig2.add_scatter(x=fc_df["date"], y=fc_df["proba_%"],


                     mode="lines+markers", name="Proba pluie %",


                     line=dict(color="#6366f1", dash="dot"),


                     marker=dict(size=5),


                     secondary_y=True)


    fig2.add_scatter(x=fc_df["date"], y=fc_df["tmax"],


                     mode="lines+markers", name="T° max (°C)",


                     line=dict(color="#f97316", width=2),


                     marker=dict(size=5, symbol="diamond"),


                     secondary_y=True)





    fig2.update_yaxes(title_text="Pluie (mm)", secondary_y=False, rangemode="tozero")


    fig2.update_yaxes(title_text="Probabilité / T°C", secondary_y=True, rangemode="tozero")




    fig2.update_layout(


        yaxis=dict(title="Pluie (mm)", side="left"),


        yaxis2=dict(title="Proba %",   overlaying="y", side="right",


                    range=[0, 110], showgrid=False),


        yaxis3=dict(title="T°C",       overlaying="y", side="right",


                    position=0.92,     showgrid=False, anchor="free"),

        legend=dict(orientation="h", y=1.12), height=290,

        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",

        margin=dict(t=30, b=20, l=20, r=80), xaxis=dict(tickangle=-20),

@@ -1412,7 +1344,6 @@

    )



    if selected_one != "— Toutes —":


        # Single commune: fetch Open-Meteo rain and color markers

        loc_single = map_df.dropna(subset=["latitude","longitude"])

        if not loc_single.empty:

            lat_s = round(float(loc_single["latitude"].mean()), 4)

@@ -1431,7 +1362,6 @@

                    f"<small>Source : Open-Meteo AROME 1,3 km</small>", max_width=250),

            ).add_to(m)

    else:


        # All communes: color by nearest dept's forecast (no per-sector API calls)

        for _, row in map_df.dropna(subset=["latitude","longitude"]).iterrows():

            rlat = float(row["latitude"]); rlon = float(row["longitude"])

            dep  = nearest_dep(rlat, rlon)

@@ -1484,18 +1414,17 @@

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
