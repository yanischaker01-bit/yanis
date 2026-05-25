#!/usr/bin/env python3
"""Convert LRS PK 1M shapefile to GeoJSON for carte_lgv.
Usage: python convert_pk_metric.py
Output: data/pk_metric.geojson
Then run: python geojson2pmtiles.py pk_metric
"""
import json, sys
from pathlib import Path

INPUT = Path(r"C:\Users\YCHAKER\Desktop\SIG\pk\LRS PK 1M.shp")
OUTPUT = Path(__file__).parent / "data" / "pk_metric.geojson"

try:
    import fiona
except ImportError:
    sys.exit("pip install fiona")

if not INPUT.exists():
    sys.exit(f"Fichier introuvable : {INPUT}")

features = []
skipped = 0
with fiona.open(str(INPUT)) as src:
    print(f"  CRS : {src.crs}")
    print(f"  Schema : {src.schema}")
    for feat in src:
        if not feat.get("geometry"):
            skipped += 1
            continue
        geom = dict(feat["geometry"])
        props = {k: v for k, v in (feat["properties"] or {}).items() if v is not None}
        # Normalise PK label pour la recherche
        for key in ("PK", "pk", "Pk", "LRS_PK", "chainage", "CHAINAGE"):
            if key in props:
                props["_pk_label"] = str(props[key])
                break
        features.append({"type": "Feature", "geometry": geom, "properties": props})

fc = {"type": "FeatureCollection", "features": features}
OUTPUT.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
print(f"\nDone: {len(features)} points  ({skipped} ignorés)  →  {OUTPUT.name}")
print("Prochaine étape : python geojson2pmtiles.py pk_metric")
