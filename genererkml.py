import pandas as pd
import simplekml
import pyproj

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
excel_file = r"C:\Users\YCHAKER\Desktop\Liste_batiments_ENRICHIE.xlsx"
kml_out = r"C:\Users\YCHAKER\Desktop\Points_Attention.kml"

print("Chargement Excel...")
sheets = pd.read_excel(excel_file, sheet_name=None)

# -------------------------------------------------------
# KML
# -------------------------------------------------------
kml = simplekml.Kml()

folders = {
    "BAT": kml.newfolder(name="Points_Attention — Bâtiments"),
    "ACCES": kml.newfolder(name="Accès"),
    "PAM": kml.newfolder(name="PAM")
}

# -------------------------------------------------------
# PROJECTION Lambert93 → WGS84 (sécurité)
# -------------------------------------------------------
proj_l93_to_wgs = pyproj.Transformer.from_crs(2154, 4326, always_xy=True)

# -------------------------------------------------------
# DETECTION COLONNES (Accès / PAM)
# -------------------------------------------------------
def find_col(df, keywords):
    for c in df.columns:
        for k in keywords:
            if k in c.upper():
                return c
    return None

# -------------------------------------------------------
# AJOUT POINT KML
# -------------------------------------------------------
def add_point(folder, name, lon, lat, desc=""):
    if pd.notna(lon) and pd.notna(lat):
        p = folder.newpoint(name=str(name))
        p.coords = [(float(lon), float(lat))]
        p.description = desc

# -------------------------------------------------------
# PARCOURS FEUILLES
# -------------------------------------------------------
print("Création KML...")

for sheet_name, df in sheets.items():

    print("→", sheet_name)

    # -------------------------------
    # COLONNES BATIMENTS (FORCÉES)
    # -------------------------------
    lib_col = find_col(df, ["LIBELL"])
    lon_bat = "WGS_LONGITUDE_mat" if "WGS_LONGITUDE_mat" in df.columns else None
    lat_bat = "WGS_LATITUDE_mat" if "WGS_LATITUDE_mat" in df.columns else None

    # -------------------------------
    # ACCES
    # -------------------------------
    acc_lon = find_col(df, ["ACCES_LONG"])
    acc_lat = find_col(df, ["ACCES_LAT"])

    # -------------------------------
    # PAM
    # -------------------------------
    pam_lon = find_col(df, ["PAM_LONG"])
    pam_lat = find_col(df, ["PAM_LAT"])

    for _, row in df.iterrows():

        label = row[lib_col] if lib_col else "Point"

        # ------------------------------------------------
        # BATIMENTS
        # ------------------------------------------------
        if lon_bat and lat_bat:
            add_point(
                folders["BAT"],
                f"{sheet_name} - {label}",
                row.get(lon_bat),
                row.get(lat_bat),
                f"Feuille : {sheet_name}"
            )

        # ------------------------------------------------
        # ACCES
        # ------------------------------------------------
        add_point(
            folders["ACCES"],
            f"ACCES - {row.get('ACCES_NOM','')}",
            row.get(acc_lon),
            row.get(acc_lat),
            f"Feuille : {sheet_name}"
        )

        # ------------------------------------------------
        # PAM
        # ------------------------------------------------
        add_point(
            folders["PAM"],
            f"PAM - {row.get('PAM_NOM','')}",
            row.get(pam_lon),
            row.get(pam_lat),
            f"Feuille : {sheet_name}"
        )

# -------------------------------------------------------
# SAVE
# -------------------------------------------------------
kml.save(kml_out)

print("✅ KML généré :", kml_out)
