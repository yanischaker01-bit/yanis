#!/usr/bin/env python3
"""
Export LGV SEA -> KMZ avec symbologie identique a la carte web.
Prerequis : pip install simplekml
Usage    : python export_kmz.py
Sortie   : dossier export_kmz/ avec un .kmz par couche + lgv_sea_complet.kmz
"""
import json, os, sys

try:
    import simplekml
except ImportError:
    print("Manque simplekml. Lancez : pip install simplekml")
    sys.exit(1)

DATA_DIR = "data"
OUT_DIR  = "export_kmz"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette identique à la carte web ───────────────────────────────────────
def kml_color(hex_col, alpha=200):
    """Convertit #RRGGBB + alpha en format KML AABBGGRR."""
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return simplekml.Color.rgb(r, g, b, alpha)

def make_line_style(kml, color, width=3):
    s = kml.newstyle()
    s.linestyle.color = kml_color(color)
    s.linestyle.width = width
    return s

def make_poly_style(kml, line_color, fill_color=None, fill_alpha=80, dash=False):
    s = kml.newstyle()
    s.linestyle.color = kml_color(line_color)
    s.linestyle.width = 1.5
    fill = fill_color or line_color
    s.polystyle.color = kml_color(fill, fill_alpha)
    return s

def make_point_style(kml, color, scale=1.1):
    s = kml.newstyle()
    s.iconstyle.color = kml_color(color)
    s.iconstyle.scale = scale
    s.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png"
    return s

# ── Description HTML depuis les propriétés ─────────────────────────────────
SKIP_KEYS = {"_ph_b64","_ph_path","_ph_url","_sp_url"}

def props_to_description(props):
    rows = []
    for k, v in (props or {}).items():
        if k in SKIP_KEYS: continue
        sv = str(v) if v is not None else ""
        if sv in ("","nan","None","null","NaN"): continue
        rows.append(f"<tr><td><b>{k}</b></td><td>{sv}</td></tr>")
    if not rows: return ""
    return "<table border='1' cellpadding='3'>"+"\n".join(rows)+"</table>"

def feat_name(props, layer_name):
    for k in ("CODE_OBJET","code_objet","NOM_ACCES","Name","name",
              "libelle","LIBELLE","sitename","cd_sig","CODE","Appellation","TYPE_OBJET"):
        v = props.get(k,"")
        if v and str(v) not in ("nan","None","null",""): return str(v)
    return layer_name

# ── Ajout d'une feature GeoJSON dans un dossier KML ─────────────────────────
def add_feature(folder, feat, style, layer_name=""):
    geom  = feat.get("geometry") or {}
    props = feat.get("properties") or {}
    gtype = geom.get("type","")
    coords= geom.get("coordinates",[])
    name  = feat_name(props, layer_name)
    desc  = props_to_description(props)

    def apply(pm):
        pm.description = desc
        pm.style = style

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
        pol = folder.newpolygon(name=name,
                                outerboundaryis=coords[0] if coords else [],
                                innerboundaryis=coords[1:] if len(coords)>1 else [])
        apply(pol)
    elif gtype == "MultiPolygon":
        for ring in coords:
            pol = folder.newpolygon(name=name,
                                    outerboundaryis=ring[0] if ring else [],
                                    innerboundaryis=ring[1:] if len(ring)>1 else [])
            apply(pol)

# ── Config couches : (fichier, nom affichage, type_geom, couleur_ligne, couleur_fill, epaisseur) ──
LAYERS = [
    ("rail",  "Tracé LGV SEA",                "line",  "#cc0000", None,     4.5),
    ("pk",    "Points kilométriques",          "point", "#333333", None,     1.0),
    ("oa",    "OA – Ouvrages d'Art",           "point", "#00bcd4", None,     1.1),
    ("oh",    "OH – Ouvrages Hydrauliques",    "line",  "#ef6c00", None,     3.5),
    ("pam",   "PAM",                           "point", "#9c27b0", None,     1.0),
    ("acces", "Accès PTLO/PTLA",               "point", "#ff6600", None,     1.0),
    ("old",   "Zones OLD",                     "poly",  "#0d47a1", "#4f83ff",1.5),
    ("mc",    "Mesures compensatoires",        "poly",  "#7b1fa2", "#ab47bc",1.5),
    ("n2",    "Natura 2000",                   "poly",  "#1b5e20", "#2e7d32",1.5),
    ("eco",   "Ecopaturage",                   "poly",  "#2e7d32", "#81c784",1.5),
    ("bois",  "Boisements",                    "poly",  "#FFEB3B", "#388e3c",1.5),
]

# ── Export individuel par couche ────────────────────────────────────────────
print("Export LGV SEA → KMZ")
print("=" * 50)

for (fname, display, gtype, line_col, fill_col, width) in LAYERS:
    path = os.path.join(DATA_DIR, fname+".geojson")
    if not os.path.exists(path):
        print(f"  ⚠  {fname}.geojson introuvable – ignoré")
        continue

    with open(path, encoding="utf-8") as f:
        fc = json.load(f)
    feats = fc.get("features") or []
    print(f"  {display:35s} {len(feats):5d} entités")

    kml = simplekml.Kml(name=display)

    if gtype == "line":
        style = make_line_style(kml, line_col, width)
    elif gtype == "point":
        style = make_point_style(kml, line_col, width)
    else:
        style = make_poly_style(kml, line_col, fill_col, fill_alpha=80)

    folder = kml.newfolder(name=display)
    for feat in feats:
        add_feature(folder, feat, style, display)

    out = os.path.join(OUT_DIR, fname+".kmz")
    kml.savekmz(out)

# ── Export global (toutes couches dans un seul KMZ) ─────────────────────────
print("\n  Génération du KMZ global...")
kml_all = simplekml.Kml(name="LGV SEA – Toutes couches")

for (fname, display, gtype, line_col, fill_col, width) in LAYERS:
    path = os.path.join(DATA_DIR, fname+".geojson")
    if not os.path.exists(path): continue
    with open(path, encoding="utf-8") as f:
        fc = json.load(f)
    feats = fc.get("features") or []

    if gtype == "line":
        style = make_line_style(kml_all, line_col, width)
    elif gtype == "point":
        style = make_point_style(kml_all, line_col, width)
    else:
        style = make_poly_style(kml_all, line_col, fill_col)

    folder = kml_all.newfolder(name=display)
    for feat in feats:
        add_feature(folder, feat, style, display)

out_all = os.path.join(OUT_DIR, "lgv_sea_complet.kmz")
kml_all.savekmz(out_all)
print(f"  → {out_all}")
print(f"\nTerminé. Fichiers dans : {os.path.abspath(OUT_DIR)}/")
print("Ouvrez les .kmz dans Google Earth, QGIS ou Google Maps.")
