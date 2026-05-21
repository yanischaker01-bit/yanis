"""Génère un snapshot minimal valide quand la collecte complète a échoué."""
import datetime
import json
import os

os.makedirs("reports", exist_ok=True)
out = "reports/streamlit_snapshot_latest.json"
if os.path.exists(out):
    print(f"Snapshot already exists: {out}")
else:
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    snap = {
        "timestamp_utc": ts,
        "risk_level": "INDETERMINE",
        "score": 0,
        "weather_notice": "Donnees en cours de collecte (snapshot minimal)",
        "alerts": [],
        "recommendations": [],
        "details": {},
        "map_path": None,
        "lgv_lines": [],
        "weather": [],
        "rivers": [],
        "geotech": {},
        "piezometers": {},
        "hydro_network": {},
        "sectors": {},
        "lgv_communes": {},
        "fr_geography": {},
        "metadata": {
            "line_monitoring": {
                "line_name": "LGV SEA Bordeaux-Tours",
                "sector_length_km": 5.0,
            }
        },
        "commune_ranking": [],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False)
    print(f"Minimal snapshot created: {out}")
