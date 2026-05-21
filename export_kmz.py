#!/usr/bin/env python3
"""
Export LGV SEA -> KMZ optimisé pour Google Maps (limite 5 Mo / fichier).
Simplification géométrique optionnelle (si shapely installé).
Usage : python export_kmz.py [--simplify 0.001] [--max-features 5000]
"""

import json, os, sys, argparse, math
from pathlib import Path

try:
    import simplekml
except ImportError:
    print("Erreur : simplekml non installé. Lancez : pip install simplekml")
    sys.exit(1)

# Tente d'importer shapely pour la simplification (optionnel)
try:
    from shapely.geometry import shape, mapping
    from shapely.ops import transform
    import pyproj
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("Info : shapely non installé. Pas de simplification géométrique.")

# Arguments
parser = argparse.ArgumentParser(description="Export LGV SEA vers KMZ compatible Google Maps")
parser.add_argument("--simplify", type=float, default=0.001,
                    help="Tolérance de simplification (degrés). 0 = désactivé. Ex: 0.0005 (environ 50m)")
parser.add_argument("--max-features", type=int, default=5000,
                    help="Nombre max d'entités par couche (0 = illimité)")
parser.add_argument("--max-size-mb", type=float, default=4.5,
                    help="Taille max recommandée par fichier KMZ (Mo). Au-delà, split.")
args = parser.parse_args()

DATA_DIR = "data"
OUT_DIR  = "export_kmz_google"
os.makedirs(OUT_DIR, exist_ok=True)

# ======================= FONCTIONS =======================
def kml_color(hex_col, alpha=200):
    """Convertit #RRGGBB + alpha (0-255) en format KML AABBGGRR."""
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return simplekml.Color.rgb(r, g, b, alpha)

def make_line_style(color, width=3):
    s = simplekml.Style()
    s.linestyle.color = kml_color(color)
    s.linestyle.width = width
    return s

def make_poly_style(line_color, fill_color=None, fill_alpha=80):
    s = simplekml.Style()
    s.linestyle.color = kml_color(line_color)
    s.linestyle.width = 1.5
    fill = fill_color or line_color
    s.polystyle.color = kml_color(fill, fill_alpha)
    return s

def make_point_style(color, scale=1.0):
    """Style simple pour Google Maps (icône par défaut sans URL externe)."""
    s = simplekml.Style()
    s.iconstyle.color = kml_color(color)
    s.iconstyle.scale = scale
    # Utilise une icône standard Google Maps (red-dot)
    s.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/red-dot.png"
    return s

def simplify_geometry(geom, tolerance):
    """Simplifie une géométrie GeoJSON avec shapely."""
    if not HAS_SHAPELY or tolerance <= 0:
        return geom
    try:
        shp = shape(geom)
        # Projection locale pour une simplification en mètres (approximatif)
        if shp.is_empty:
            return geom
        # Simplification en degrés (tolérance faible)
        simplified = shp.simplify(tolerance, preserve_topology=True)
        # Si le résultat est trop simplifié (ex: point au lieu de polygone), on garde l'original
        if simplified.is_empty or (geom['type'] != 'Point' and simplified.geom_type == 'Point'):
            return geom
        return mapping(simplified)
    except Exception as e:
        print(f"  ↳ Erreur simplification : {e}")
        return geom

def props_to_description(props, max_length=500):
    """Description courte pour éviter les popups trop lourds."""
    rows = []
    for k, v in (props or {}).items():
        if k in {"_ph_b64","_ph_path","_ph_url","_sp_url"}:
            continue
        sv = str(v) if v is not None else ""
        if sv in ("","nan","None","null","NaN"):
            continue
        rows.append(f"{k}: {sv}")
    desc = "\n".join(rows)
    if len(desc) > max_length:
        desc = desc[:max_length] + "..."
    return desc

def feat_name(props, layer_name):
    for k in ("CODE_OBJET","code_objet","NOM_ACCES","Name","name",
              "libelle","LIBELLE","sitename","cd_sig","CODE","Appellation","TYPE_OBJET"):
        v = props.get(k, "")
        if v and str(v) not in ("nan","None","null",""):
            return str(v)[:60]  # limite longueur nom
    return layer_name

def add_feature(folder, feat, style, tolerance, layer_name=""):
    geom = feat.get("geometry")
    if not geom:
        return
    props = feat.get("properties", {})
    name = feat_name(props, layer_name)
    desc = props_to_description(props)
    
    # Simplification éventuelle
    if tolerance > 0 and geom['type'] not in ('Point','MultiPoint'):
        geom = simplify_geometry(geom, tolerance)
    
    gtype = geom['type']
    coords = geom.get('coordinates', [])
    
    def apply(pm):
        pm.name = name
        pm.description = desc
        pm.style = style
    
    try:
        if gtype == "Point":
            pm = folder.newpoint(name=name, coords=[coords])
            apply(pm)
        elif gtype == "MultiPoint":
            for c in coords:
                pm = folder.newpoint(name=name, coords=[c])
                apply(pm)
        elif gtype == "LineString":
            ls = folder.newlinestring(name=name, coords=coords)
            apply(ls)
        elif gtype == "MultiLineString":
            for c in coords:
                ls = folder.newlinestring(name=name, coords=c)
                apply(ls)
        elif gtype == "Polygon":
            if coords:
                pol = folder.newpolygon(name=name, outerboundaryis=coords[0],
                                        innerboundaryis=coords[1:] if len(coords)>1 else [])
                apply(pol)
        elif gtype == "MultiPolygon":
            for ring in coords:
                if ring:
                    pol = folder.newpolygon(name=name, outerboundaryis=ring[0],
                                            innerboundaryis=ring[1:] if len(ring)>1 else [])
                    apply(pol)
    except Exception as e:
        print(f"    ⚠ Erreur ajout entité {name}: {e}")

def estimate_kmz_size(folder_path):
    """Estimation grossière de la taille d'un KMZ (non compressé)."""
    if not os.path.exists(folder_path):
        return 0
    # On ne peut pas estimer avant création, on se base sur le nombre d'entités
    return 0

# ======================= CONFIGURATION DES COUCHES =======================
LAYERS = [
    ("rail",  "Tracé LGV SEA",                "line",  "#cc0000", None,     4.0),
    ("pk",    "Points kilométriques",         "point", "#333333", None,     1.0),
    ("oa",    "OA – Ouvrages d'Art",          "point", "#00bcd4", None,     1.1),
    ("oh",    "OH – Ouvrages Hydrauliques",   "line",  "#ef6c00", None,     3.5),
    ("pam",   "PAM",                          "point", "#9c27b0", None,     1.0),
    ("acces", "Accès PTLO/PTLA",              "point", "#ff6600", None,     1.0),
    ("old",   "Zones OLD",                    "poly",  "#0d47a1", "#4f83ff",1.5),
    ("mc",    "Mesures compensatoires",       "poly",  "#7b1fa2", "#ab47bc",1.5),
    ("n2",    "Natura 2000",                  "poly",  "#1b5e20", "#2e7d32",1.5),
    ("eco",   "Ecopaturage",                  "poly",  "#2e7d32", "#81c784",1.5),
    ("bois",  "Boisements",                   "poly",  "#FFEB3B", "#388e3c",1.5),
]

# ======================= EXPORT =======================
print("Export LGV SEA → KMZ pour Google Maps")
print("=" * 60)
print(f"Simplification : {args.simplify} degrés" + ("" if args.simplify<=0 else " (activée)"))
print(f"Limite entités/couche : {args.max_features if args.max_features>0 else 'illimitée'}")
print()

total_features = 0
for (fname, display, gtype, line_col, fill_col, width) in LAYERS:
    path = os.path.join(DATA_DIR, fname + ".geojson")
    if not os.path.exists(path):
        print(f"⚠ {fname}.geojson introuvable – ignoré")
        continue
    
    with open(path, encoding="utf-8") as f:
        fc = json.load(f)
    feats = fc.get("features", [])
    nb_feats = len(feats)
    total_features += nb_feats
    
    # Limitation du nombre d'entités
    if args.max_features > 0 and nb_feats > args.max_features:
        print(f"  ⚠ {display}: {nb_feats} entités, tronqué à {args.max_features}")
        feats = feats[:args.max_features]
    
    print(f"  {display:35s} {len(feats):5d} entités")
    
    kml = simplekml.Kml(name=display)
    
    # Style selon type
    if gtype == "line":
        style = make_line_style(line_col, width)
    elif gtype == "point":
        style = make_point_style(line_col, width)
    else:
        style = make_poly_style(line_col, fill_col, fill_alpha=80)
    
    folder = kml.newfolder(name=display)
    for feat in feats:
        add_feature(folder, feat, style, args.simplify, display)
    
    out = os.path.join(OUT_DIR, fname + ".kmz")
    kml.savekmz(out)
    
    # Vérification taille fichier (approximative)
    size_mb = os.path.getsize(out) / (1024 * 1024)
    if size_mb > args.max_size_mb:
        print(f"    ⚠ Taille {size_mb:.1f} Mo (> {args.max_size_mb} Mo) – peut poser problème dans Google Maps")
    else:
        print(f"    ✓ Taille {size_mb:.1f} Mo")

# ======================= EXPORT GLOBAL (optionnel) =======================
if total_features < 20000:
    print("\n  Génération du KMZ global...")
    kml_all = simplekml.Kml(name="LGV SEA – Toutes couches")
    for (fname, display, gtype, line_col, fill_col, width) in LAYERS:
        path = os.path.join(DATA_DIR, fname + ".geojson")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            fc = json.load(f)
        feats = fc.get("features", [])
        if args.max_features > 0 and len(feats) > args.max_features:
            feats = feats[:args.max_features]
        if gtype == "line":
            style = make_line_style(line_col, width)
        elif gtype == "point":
            style = make_point_style(line_col, width)
        else:
            style = make_poly_style(line_col, fill_col, fill_alpha=80)
        folder = kml_all.newfolder(name=display)
        for feat in feats:
            add_feature(folder, feat, style, args.simplify, display)
    out_all = os.path.join(OUT_DIR, "lgv_sea_complet.kmz")
    kml_all.savekmz(out_all)
    size_all = os.path.getsize(out_all) / (1024 * 1024)
    if size_all > args.max_size_mb:
        print(f"  ⚠ Taille globale {size_all:.1f} Mo – risque de rejet par Google Maps")
    else:
        print(f"  ✓ Global : {size_all:.1f} Mo")
else:
    print(f"\n  Trop d'entités ({total_features}) – export global ignoré (risque de saturation)")

print(f"\nTerminé. Fichiers dans : {os.path.abspath(OUT_DIR)}/")
print("Pour Google Maps :")
print("  - Importez les .kmz un par un (limite ~5 Mo/fichier)")
print("  - Utilisez de préférence Google Earth Desktop pour les gros volumes")