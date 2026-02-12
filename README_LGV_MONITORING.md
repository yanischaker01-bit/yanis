# LGV SEA Monitoring (local)

Outil local pour suivre:
- la pluviometrie a proximite de la LGV SEA Bordeaux-Tours
- les niveaux d'eau des principaux cours d'eau traversant la ligne

## 1) Installation

```powershell
pip install -r requirements_lgv_monitoring.txt
```

## 2) Lancer une collecte unique

```powershell
python run_collection.py --once --max-distance-km 25
```

La base locale est `data/lgv_monitoring.db`.

## 3) Lancer la collecte continue

```powershell
python run_collection.py --interval-min 60 --max-distance-km 25
```

## 4) Lancer l'interface web cliquable

```powershell
streamlit run streamlit_app.py
```

## 5) Lancer le nouveau rapport Streamlit Pro

```powershell
python run_streamlit_pro.py
```

ou

```powershell
streamlit run streamlit_lgv_pro.py
```

Fonctions:
- filtres interactifs (periode pluvio, niveau de risque, source, communes)
- carte dynamique multicouche (LGV, meteo, secteurs, hydro, piezometres, geotech)
- classement professionnel par commune (note GC /100)
- tableaux d'alertes et recommandations
- suivi historique meteo mensuel multi-annees par secteur
- onglet Metadata (logique stations, scores GC, fonctionnement global)

## 6) Publier sur Streamlit Community Cloud (lien public)

1. Ouvre `https://share.streamlit.io/`.
2. Connecte ton compte GitHub.
3. Clique `New app`.
4. Selectionne:
   - Repository: `yanischaker01-bit/yanis`
   - Branch: `main`
   - Main file path: `streamlit_lgv_pro.py`
5. Clique `Deploy`.

Sur Streamlit Cloud, l'app lit en priorite le snapshot publie par GitHub Pages:
`https://yanischaker01-bit.github.io/yanis/reports/streamlit_snapshot_latest.json`.
Le projet force `streamlit==1.54.0` et `altair==5.5.0` pour compatibilite Cloud recente.

## Fonctionnement

- La collecte pluvio lit SYNOP Meteo-France et stocke les observations en SQLite.
- L'interface affiche les stations pluvio, la ligne LGV, et les points cours d'eau.
- Le filtre d'affichage des stations est reglable; valeur cible demandee: `1 km`.
- Pour chaque cours d'eau, renseigne `station_code` (HubEau `code_entite`) depuis le panneau droit.

## Limites importantes

- Avec un filtre strict a 1 km, il peut n'y avoir aucune station SYNOP.
- Tant que `station_code` est vide pour un cours d'eau, l'hydrometrie reste non disponible.
- Les distances sont calculees avec une approximation plane (suffisant pour un MVP operationnel).
