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

## Fonctionnement

- La collecte pluvio lit SYNOP Meteo-France et stocke les observations en SQLite.
- L'interface affiche les stations pluvio, la ligne LGV, et les points cours d'eau.
- Le filtre d'affichage des stations est reglable; valeur cible demandee: `1 km`.
- Pour chaque cours d'eau, renseigne `station_code` (HubEau `code_entite`) depuis le panneau droit.

## Limites importantes

- Avec un filtre strict a 1 km, il peut n'y avoir aucune station SYNOP.
- Tant que `station_code` est vide pour un cours d'eau, l'hydrometrie reste non disponible.
- Les distances sont calculees avec une approximation plane (suffisant pour un MVP operationnel).

