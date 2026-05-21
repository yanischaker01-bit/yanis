import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------
# 1. CHARGEMENT
# --------------------------------------------------------
print("Chargement...")

old_zs = gpd.read_file(r"C:\Users\YCHAKER\Desktop\TEMP\OLD_ZS_coord.shp")
oa_coord = gpd.read_file(r"C:\Users\YCHAKER\Desktop\TEMP\OA_coord.shp")
acces_layer = gpd.read_file(r"C:\Users\YCHAKER\Downloads\shp\Accès.shp")
pam_layer = gpd.read_file(r"C:\Users\YCHAKER\Downloads\shp\PAM.shp")

excel_path = r"C:\Users\YCHAKER\Downloads\Liste des bâtiments à finaliser.xlsx"
excel_sheets = pd.read_excel(excel_path, sheet_name=None)

# --------------------------------------------------------
# 2. NORMALISATION TEXTE
# --------------------------------------------------------
def clean(txt):
    if pd.isna(txt):
        return ""
    return str(txt).strip().upper()

old_zs["LIB_CLEAN"] = old_zs["libellé"].apply(clean)
oa_coord["LIB_CLEAN"] = oa_coord["LIBELLE"].apply(clean)

# --------------------------------------------------------
# 3. CRS
# --------------------------------------------------------
METRIC_CRS = 2154  # Lambert 93

old_zs_metric = old_zs.to_crs(METRIC_CRS)
oa_coord_metric = oa_coord.to_crs(METRIC_CRS)
acces_metric = acces_layer.to_crs(METRIC_CRS)
pam_metric = pam_layer.to_crs(METRIC_CRS)

old_zs_wgs = old_zs.to_crs(4326)
oa_coord_wgs = oa_coord.to_crs(4326)
acces_wgs = acces_layer.to_crs(4326)
pam_wgs = pam_layer.to_crs(4326)

# --------------------------------------------------------
# 4. LAMBERT UNIVERSAL
# --------------------------------------------------------
def get_lambert(row):
    for x in ["LAMBERT_X","Lambert_X","Lambert_x","lambert_x"]:
        if x in row and pd.notna(row[x]):
            lx = row[x]
            break
    else:
        lx = None

    for y in ["LAMBERT_Y","Lambert_Y","Lambert_y","lambert_y"]:
        if y in row and pd.notna(row[y]):
            ly = row[y]
            break
    else:
        ly = None

    return lx, ly

# --------------------------------------------------------
# 5. DICTIONNAIRES RAPIDES
# --------------------------------------------------------
old_dict = old_zs.set_index("LIB_CLEAN").to_dict("index")
oa_dict = oa_coord.set_index("LIB_CLEAN").to_dict("index")

# --------------------------------------------------------
# 6. PLUS PROCHE (METRIQUE)
# --------------------------------------------------------
def nearest_feature_metric(point_metric, layer_metric):
    distances = layer_metric.geometry.distance(point_metric)
    idx = distances.idxmin()
    return layer_metric.loc[idx], distances.min()

# --------------------------------------------------------
# 7. TRAITEMENT EXCEL
# --------------------------------------------------------
print("Traitement Excel...")

enriched_sheets = {}

for sheet_name, df in excel_sheets.items():

    print("→", sheet_name)

    out = df.copy()

    lib_cols = [c for c in out.columns if "libell" in c.lower()]
    if not lib_cols:
        enriched_sheets[sheet_name] = out
        continue

    lib_col = lib_cols[0]

    new_cols = [
        "WGS_LONGITUDE","WGS_LATITUDE",
        "Lambert_X","Lambert_Y",
        "OA_LIBELLE",
        "ACCES_NOM","ACCES_AXE","ACCES_PK",
        "ACCES_LONGITUDE","ACCES_LATITUDE",
        "DISTANCE_ACCES_M",
        "PAM_NOM","PAM_DISTANCE_M",
        "PAM_LONGITUDE","PAM_LATITUDE"
    ]

    for c in new_cols:
        if c not in out:
            out[c] = None

    # ----------------------------------------------------
    # BOUCLE
    # ----------------------------------------------------
    for i,row in out.iterrows():

        lib = clean(row[lib_col])

        # ==========================
        # VIA / PRA → OA
        # ==========================
        if "VIA" in sheet_name.upper() or "PRA" in sheet_name.upper():

            if lib in oa_dict:

                geom_metric = oa_coord_metric.loc[
                    oa_coord_metric["LIB_CLEAN"]==lib
                ].geometry.values[0]

                geom_wgs = oa_coord_wgs.loc[
                    oa_coord_wgs["LIB_CLEAN"]==lib
                ].geometry.values[0]

                geom_src = oa_coord.loc[
                    oa_coord["LIB_CLEAN"]==lib
                ].iloc[0]

                p_metric = geom_metric if geom_metric.geom_type=="Point" else geom_metric.centroid
                p_wgs = geom_wgs if geom_wgs.geom_type=="Point" else geom_wgs.centroid

                out.at[i,"OA_LIBELLE"] = lib
                out.at[i,"WGS_LONGITUDE"] = p_wgs.x
                out.at[i,"WGS_LATITUDE"] = p_wgs.y

                lx,ly = get_lambert(geom_src)
                out.at[i,"Lambert_X"] = lx
                out.at[i,"Lambert_Y"] = ly

                # ACCES
                acc, d = nearest_feature_metric(p_metric, acces_metric)
                acc_wgs = acces_wgs.loc[acc.name]

                out.at[i,"ACCES_NOM"] = acc.get("NOM_ACCES")
                out.at[i,"ACCES_AXE"] = acc.get("AXE")
                out.at[i,"ACCES_PK"] = acc.get("PK")
                out.at[i,"ACCES_LONGITUDE"] = acc_wgs.geometry.x
                out.at[i,"ACCES_LATITUDE"] = acc_wgs.geometry.y
                out.at[i,"DISTANCE_ACCES_M"] = round(d,2)

                # PAM
                pam, dp = nearest_feature_metric(p_metric, pam_metric)
                pam_wgs_row = pam_wgs.loc[pam.name]

                out.at[i,"PAM_NOM"] = pam.get("LIBELLE")
                out.at[i,"PAM_DISTANCE_M"] = round(dp,2)
                out.at[i,"PAM_LONGITUDE"] = pam_wgs_row.geometry.x
                out.at[i,"PAM_LATITUDE"] = pam_wgs_row.geometry.y

            continue

        # ==========================
        # NORMAL → OLD_ZS
        # ==========================
        if lib in old_dict:

            geom_metric = old_zs_metric.loc[
                old_zs_metric["LIB_CLEAN"]==lib
            ].geometry.values[0]

            geom_wgs = old_zs_wgs.loc[
                old_zs_wgs["LIB_CLEAN"]==lib
            ].geometry.values[0]

            geom_src = old_zs.loc[
                old_zs["LIB_CLEAN"]==lib
            ].iloc[0]

            p_metric = geom_metric if geom_metric.geom_type=="Point" else geom_metric.centroid
            p_wgs = geom_wgs if geom_wgs.geom_type=="Point" else geom_wgs.centroid

            out.at[i,"WGS_LONGITUDE"] = p_wgs.x
            out.at[i,"WGS_LATITUDE"] = p_wgs.y

            lx,ly = get_lambert(geom_src)
            out.at[i,"Lambert_X"] = lx
            out.at[i,"Lambert_Y"] = ly

            # ACCES
            acc, d = nearest_feature_metric(p_metric, acces_metric)
            acc_wgs = acces_wgs.loc[acc.name]

            out.at[i,"ACCES_NOM"] = acc.get("NOM_ACCES")
            out.at[i,"ACCES_AXE"] = acc.get("AXE")
            out.at[i,"ACCES_PK"] = acc.get("PK")
            out.at[i,"ACCES_LONGITUDE"] = acc_wgs.geometry.x
            out.at[i,"ACCES_LATITUDE"] = acc_wgs.geometry.y
            out.at[i,"DISTANCE_ACCES_M"] = round(d,2)

            # PAM
            pam, dp = nearest_feature_metric(p_metric, pam_metric)
            pam_wgs_row = pam_wgs.loc[pam.name]

            out.at[i,"PAM_NOM"] = pam.get("LIBELLE")
            out.at[i,"PAM_DISTANCE_M"] = round(dp,2)
            out.at[i,"PAM_LONGITUDE"] = pam_wgs_row.geometry.x
            out.at[i,"PAM_LATITUDE"] = pam_wgs_row.geometry.y

    enriched_sheets[sheet_name] = out

# --------------------------------------------------------
# EXPORT
# --------------------------------------------------------
print("Export Excel...")

out_path = r"C:\Users\YCHAKER\Desktop\Liste_batiments_ENRICHIE.xlsx"

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    for name, df in enriched_sheets.items():
        df.to_excel(writer, sheet_name=name, index=False)

print("OK :", out_path)
